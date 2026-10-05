//! Restoration primitives — the spectral-domain edits audio-restoration
//! tools are built from.
//!
//! All four operations share one shape: they take a [`Spectrum`] plus
//! optional *side information* (a [`NoiseProfile`], a configuration) and
//! return a new spectrum with the same framing. None of them resamples,
//! re-frames, or changes the bin count, so results compose freely — you can
//! gate, then subtract, then split.
//!
//! | Operation | What it does |
//! |---|---|
//! | [`spectral_gate`] | hard-threshold each bin against the noise profile, then attenuate the bins that fell below it by `reduction_db`, with optional time-axis smoothing of the gain |
//! | [`spectral_subtract`] | magnitude-domain subtraction `|X| − α·|N|`, floored at zero, phase preserved |
//! | [`harmonic_percussive_split`] | median-filter the magnitude spectrogram along each axis (Fitzgerald's HPSS) and split with a soft mask |
//! | [`spectral_smooth`] | convolve the magnitude spectrogram along the frequency axis |
//!
//! [`denoise`] chains `stft → gate → istft` for the common case.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use dsp_core::math::{db_to_linear, linear_to_db};

use crate::config::StftConfig;
use crate::error::SpectralError;
use crate::stft::{istft, stft, Spectrum};

/// Default median half-width for [`harmonic_percussive_split`]: a 17-tap
/// median along each axis.
pub const HPSS_MEDIAN_HALF_WIDTH: usize = 8;

/// A time-averaged noise magnitude spectrum — the reference level that
/// "how loud is the noise floor?" is measured against.
///
/// Built either from a whole noise-only recording
/// ([`NoiseProfile::estimate`]) or from hand-picked frames of a longer
/// spectrogram, which is how you profile a leading silent segment of a file
/// without re-analysing it ([`NoiseProfile::from_frames`]).
///
/// Stores the **mean linear magnitude** per bin, not power: gating compares
/// an observed magnitude against it directly, and `|N|²` is only wanted when
/// the caller converts.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseProfile {
    spectrum: Vec<f64>,
    sample_rate: f64,
    frames: usize,
}

impl NoiseProfile {
    /// Average every frame of `samples` (analysed with `cfg`) into one
    /// profile.
    ///
    /// Feed this a *noise-only* segment — a leading room-tone passage, a
    /// silent lead-in — not the whole file: averaging a file that contains
    /// signal produces a profile dominated by that signal, and gating it
    /// removes music instead of noise.
    ///
    /// # Errors
    ///
    /// Propagates every error from [`stft`], plus [`SpectralError::Config`]
    /// when `cfg.sample_rate` is not usable.
    pub fn estimate(samples: &[f64], cfg: &StftConfig) -> Result<Self, SpectralError> {
        cfg.validate()?;
        if samples.is_empty() {
            return Err(SpectralError::EmptyInput);
        }
        let spec = stft(samples, cfg)?;
        let all: Vec<usize> = (0..spec.num_frames()).collect();
        Ok(Self::from_frames(&spec, &all).with_sample_rate(cfg.sample_rate))
    }

    /// Average the magnitude spectra of the named frames.
    ///
    /// Indices outside `0..num_frames` are ignored rather than rejected, so a
    /// caller can pass a computed index list (e.g. "first 25 % of frames")
    /// without pre-clamping it. An index list where nothing is in range (or
    /// an empty list) yields an all-zero profile over `bin_count` bins.
    ///
    /// The result's `sample_rate` is `0.0` — this constructor has no rate to
    /// take; use [`with_sample_rate`](Self::with_sample_rate) or
    /// [`estimate`](Self::estimate) to set it.
    #[must_use]
    pub fn from_frames(spec: &Spectrum, frame_indices: &[usize]) -> Self {
        let mut spectrum = alloc::vec![0.0f64; spec.bin_count()];
        let mut used = 0usize;
        for &i in frame_indices {
            let mags = spec.magnitude(i);
            if mags.is_empty() {
                continue;
            }
            for (slot, m) in spectrum.iter_mut().zip(mags.iter()) {
                *slot += m;
            }
            used += 1;
        }
        if used > 0 {
            let n = used as f64;
            for slot in spectrum.iter_mut() {
                *slot /= n;
            }
        }
        Self {
            spectrum,
            sample_rate: 0.0,
            frames: used,
        }
    }

    /// Build a profile directly from an averaged magnitude spectrum — the
    /// escape hatch for a profile measured elsewhere (a measurement
    /// microphone's own calibration, a synthetic floor in a test, a floor
    /// carried in a file's metadata).
    ///
    /// `sample_rate` is recorded for the caller's bookkeeping; it does not
    /// affect any comparison, which is per-bin.
    #[must_use]
    pub fn from_magnitudes(spectrum: Vec<f64>, sample_rate: f64) -> Self {
        Self {
            spectrum,
            sample_rate,
            frames: 1,
        }
    }

    /// The averaged noise magnitude per bin.
    #[must_use]
    pub fn spectrum(&self) -> &[f64] {
        &self.spectrum
    }

    /// Sample rate the profile was estimated at (`0.0` when unknown).
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// How many frames were averaged.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The loudest noise magnitude in the profile — the worst-case bin.
    #[must_use]
    pub fn peak(&self) -> f64 {
        self.spectrum.iter().copied().fold(0.0f64, f64::max)
    }

    /// Return a copy tagged with `sample_rate`.
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Noise magnitude at bin `k`, or `0.0` when `k` is out of range.
    #[must_use]
    pub fn magnitude_at(&self, k: usize) -> f64 {
        self.spectrum.get(k).copied().unwrap_or(0.0)
    }

    /// Check that this profile lines up with `spec` and is usable.
    ///
    /// # Errors
    ///
    /// Returns [`SpectralError::Config`] when the profile's bin count differs
    /// from the spectrum's, or when it contains a non-finite value.
    pub fn validate_for(&self, spec: &Spectrum) -> Result<(), SpectralError> {
        if self.spectrum.len() != spec.bin_count() {
            return Err(SpectralError::Config(format!(
                "noise profile has {} bins, spectrum has {}",
                self.spectrum.len(),
                spec.bin_count()
            )));
        }
        if self.spectrum.iter().any(|v| !v.is_finite()) {
            return Err(SpectralError::NonFinite);
        }
        Ok(())
    }
}

/// Parameters for [`spectral_gate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateConfig {
    /// How far a bin must rise *above* the noise profile, in dB, before it is
    /// treated as signal rather than noise. A bin whose magnitude is at (or
    /// below) the profile sits at 0 dB of headroom, so a **positive**
    /// threshold is what catches noise: `+6` attenuates everything up to 6 dB
    /// above the floor, `+3` is aggressive, `+12` only removes the very
    /// quietest bins.
    pub threshold_db: f64,
    /// How much to attenuate a gated bin, in dB. Negative: `−12` halves the
    /// magnitude of a bin classified as noise.
    pub reduction_db: f64,
    /// Full width, in frames, of the moving average applied to the per-frame
    /// gain along the **time** axis. `0` and `1` both disable smoothing (hard,
    /// frame-independent gain). A few frames spreads each decision over its
    /// neighbours, which is what stops a gated bin from pumping as noise
    /// moves through the window; the cost is that short tonal events survive
    /// slightly louder than they otherwise would.
    pub smoothing_frames: usize,
}

impl Default for GateConfig {
    /// The standard conservative gate: attenuate bins within 6 dB of the
    /// noise floor by 12 dB, with no time smoothing.
    fn default() -> Self {
        Self {
            threshold_db: 6.0,
            reduction_db: -12.0,
            smoothing_frames: 0,
        }
    }
}

impl GateConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SpectralError::Config`] when a decibel value is not finite.
    pub fn validate(&self) -> Result<(), SpectralError> {
        if !self.threshold_db.is_finite() {
            return Err(SpectralError::Config(String::from(
                "threshold_db must be finite",
            )));
        }
        if !self.reduction_db.is_finite() {
            return Err(SpectralError::Config(String::from(
                "reduction_db must be finite",
            )));
        }
        Ok(())
    }
}

/// Hard-threshold spectral gating against a noise profile.
///
/// Per bin, compare the observed magnitude to the profile's:
///
/// ```text
/// ratio_db = 20·log10(|X[k]| / |N[k]|)
/// gain     = 1                      if ratio_db >= threshold_db
///          = 10^(reduction_db/20)   otherwise
/// ```
///
/// Then, when `smoothing_frames > 1`, average the gain along the time axis
/// with a centred moving average of that width (half-width `width / 2`; edges
/// use the available neighbours and are renormalised, so a constant gain
/// stays constant), and finally scale the complex bins by the gain. Phase is
/// untouched; with smoothing disabled a gated bin's magnitude changes by
/// exactly the configured `reduction_db`.
///
/// Note the sign convention: because a noise-only bin measures *at* the
/// profile (0 dB of headroom), it is `threshold_db` being **positive** that
/// makes the gate fire. See [`GateConfig::threshold_db`].
///
/// # Errors
///
/// Returns [`SpectralError::Config`] when `cfg` fails
/// [`GateConfig::validate`] or the profile does not match the spectrum
/// ([`NoiseProfile::validate_for`]), and [`SpectralError::NonFinite`] for a
/// non-finite profile.
///
/// # Example
///
/// ```
/// use dsp_spectral::{GateConfig, NoiseProfile, spectral_gate, stft, StftConfig};
///
/// let cfg = StftConfig::new(64, 32);
/// let noise = vec![0.01f64; 256];
/// let spec = stft(&noise, &cfg).expect("valid");
/// let profile = NoiseProfile::estimate(&noise, &cfg).expect("valid");
/// // threshold 0 dB: only bins strictly above the profile survive.
/// let gate = GateConfig { threshold_db: 0.0, reduction_db: -20.0, smoothing_frames: 0 };
/// let gated = spectral_gate(&spec, &profile, &gate).expect("valid");
/// assert_eq!(gated.num_frames(), spec.num_frames());
/// ```
pub fn spectral_gate(
    spec: &Spectrum,
    profile: &NoiseProfile,
    cfg: &GateConfig,
) -> Result<Spectrum, SpectralError> {
    cfg.validate()?;
    profile.validate_for(spec)?;
    let bins = spec.bin_count();
    let gate_gain = db_to_linear(cfg.reduction_db);

    let frames: Vec<Vec<f64>> = (0..spec.num_frames())
        .map(|t| {
            let mags = spec.magnitude(t);
            (0..bins)
                .map(|k| {
                    let mag = mags.get(k).copied().unwrap_or(0.0);
                    let noise = profile.magnitude_at(k);
                    let ratio_db = if noise > 0.0 && mag > 0.0 {
                        linear_to_db(mag / noise)
                    } else if noise > 0.0 {
                        // A silent bin under a noisy profile is definitely noise.
                        f64::NEG_INFINITY
                    } else {
                        // Nothing to compare against: leave the bin alone.
                        0.0
                    };
                    if ratio_db < cfg.threshold_db {
                        gate_gain
                    } else {
                        1.0
                    }
                })
                .collect()
        })
        .collect();

    let smoothed = smooth_time(&frames, cfg.smoothing_frames);
    let mut frames_out = Vec::with_capacity(spec.num_frames());
    for (t, gains) in smoothed.iter().enumerate() {
        let row = spec
            .frame(t)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(k, c)| *c * gains.get(k).copied().unwrap_or(1.0))
                    .collect()
            })
            .unwrap_or_default();
        frames_out.push(row);
    }
    Spectrum::from_parts(
        frames_out,
        spec.hop(),
        spec.fft_size(),
        spec.window(),
        spec.is_centered(),
    )
}

/// Centred moving average of width `width` along the time axis, per bin.
/// `width <= 1` is a no-op; edges average over the available neighbours only,
/// so a constant gain stays constant everywhere.
fn smooth_time(frames: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    if width <= 1 {
        return frames.to_vec();
    }
    // `width` is the full tap count; a centred window therefore has half-width
    // `width / 2`, so an even `width` gives an asymmetric (still normalised)
    // window rather than a silently wider one.
    let half = (width / 2) as isize;
    frames
        .iter()
        .enumerate()
        .map(|(t, row)| {
            (0..row.len())
                .map(|k| {
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for offset in -half..=half {
                        let idx = clamp_index(t as isize + offset, frames.len());
                        if let Some(v) = frames.get(idx).and_then(|r| r.get(k)) {
                            sum += v;
                            count += 1;
                        }
                    }
                    if count > 0 {
                        sum / (count as f64)
                    } else {
                        row.get(k).copied().unwrap_or(0.0)
                    }
                })
                .collect()
        })
        .collect()
}

/// Magnitude-domain spectral subtraction: `|X'[k]| = max(0, |X[k]| − α·|N[k]|)`
/// with the original phase preserved.
///
/// `α = 1.0` is classic Boll subtraction; `α > 1` over-subtracts and removes
/// more noise at the cost of speech distortion ("musical noise"); `α = 0.0`
/// is exactly the identity, which makes it a safe default in a chain.
///
/// # Errors
///
/// Returns [`SpectralError::Config`] when `over_subtraction` is not finite or
/// negative, or the profile does not match the spectrum.
pub fn spectral_subtract(
    spec: &Spectrum,
    profile: &NoiseProfile,
    over_subtraction: f64,
) -> Result<Spectrum, SpectralError> {
    if !over_subtraction.is_finite() || over_subtraction < 0.0 {
        return Err(SpectralError::Config(String::from(
            "over_subtraction must be finite and >= 0",
        )));
    }
    profile.validate_for(spec)?;

    let mut frames_out = Vec::with_capacity(spec.num_frames());
    for t in 0..spec.num_frames() {
        let row = spec
            .frame(t)
            .map(|bins| {
                bins.iter()
                    .enumerate()
                    .map(|(k, c)| {
                        let noise = profile.magnitude_at(k);
                        let target = c.magnitude() - over_subtraction * noise;
                        if target <= 0.0 {
                            Complex::ZERO
                        } else if over_subtraction == 0.0 {
                            *c
                        } else {
                            Complex::from_polar(target, c.phase())
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        frames_out.push(row);
    }
    Spectrum::from_parts(
        frames_out,
        spec.hop(),
        spec.fft_size(),
        spec.window(),
        spec.is_centered(),
    )
}

/// Median of a scratch buffer, written back into the same buffer.
fn median_in_place(buf: &mut [f64]) -> f64 {
    buf.sort_by(f64::total_cmp);
    let mid = buf.len() / 2;
    if buf.len() % 2 == 0 {
        // Even window: average the two central order statistics.
        let lo = buf.get(mid.saturating_sub(1)).copied().unwrap_or(0.0);
        let hi = buf.get(mid).copied().unwrap_or(0.0);
        0.5 * (lo + hi)
    } else {
        buf.get(mid).copied().unwrap_or(0.0)
    }
}

/// Median filter `values` along `axis`, window half-width `half`.
fn median_filter(values: &[Vec<f64>], axis_time: bool, half: usize) -> Vec<Vec<f64>> {
    let n_time = values.len();
    let n_freq = values.first().map_or(0, Vec::len);
    let mut out = vec![vec![0.0f64; n_freq]; n_time];
    let mut scratch = alloc::vec![0.0f64; 2 * half + 1];

    for t in 0..n_time {
        for k in 0..n_freq {
            scratch.clear();
            if axis_time {
                for offset in -(half as isize)..=(half as isize) {
                    let tt = clamp_index(t as isize + offset, n_time);
                    if let Some(v) = values.get(tt).and_then(|row| row.get(k)) {
                        scratch.push(*v);
                    }
                }
            } else {
                for offset in -(half as isize)..=(half as isize) {
                    let kk = clamp_index(k as isize + offset, n_freq);
                    if let Some(v) = values.get(t).and_then(|row| row.get(kk)) {
                        scratch.push(*v);
                    }
                }
            }
            let m = if scratch.is_empty() {
                0.0
            } else {
                median_in_place(&mut scratch)
            };
            if let Some(cell) = out.get_mut(t).and_then(|row| row.get_mut(k)) {
                *cell = m;
            }
        }
    }
    out
}

/// Clamp a possibly-negative index into `0..n`.
fn clamp_index(i: isize, n: usize) -> usize {
    if i < 0 {
        0
    } else {
        usize::try_from(i)
            .unwrap_or(n.saturating_sub(1))
            .min(n.saturating_sub(1))
    }
}

/// Harmonic/percussive source separation by median filtering (Fitzgerald's
/// HPSS, 2010).
///
/// The trick: a harmonic sound is *steady across time* at each frequency and
/// *sparse across frequency*; a percussive sound is the opposite. So take a
/// running median along each axis of the magnitude spectrogram:
///
/// - `H = median_time(|X|)` — smooth in time, so it tracks sustained tones.
/// - `P = median_freq(|X|)` — smooth in frequency, so it tracks broadband hits.
///
/// Then split with a soft (Wiener-style) mask
/// `m_h = H² / (H² + P² + ε)` so each output keeps the phase of the input:
///
/// ```text
/// harmonic   = X · m_h
/// percussive = X · (1 − m_h)
/// ```
///
/// `hps_ratio` sets the median half-width on both axes (a 17-tap median at
/// the default of 8). Too small and the median stops distinguishing steady
/// from transient; too large and it over-smooths, blurring the split. Values
/// below 1 are clamped to 1 (a 3-tap median), non-finite values fall back to
/// the [`HPSS_MEDIAN_HALF_WIDTH`] default, and a value larger than half the
/// spectrogram's own extent is capped.
///
/// The mask is **complementary**: `m_h + m_p == 1` at every bin, so the two
/// outputs' *magnitudes* sum exactly to the input's
/// (`|harmonic| + |percussive| = |X|`), and the pair is lossless to recombine
/// after independent processing. It is deliberately *not* energy-preserving
/// (`|harmonic|² + |percussive|² ≤ |X|²`, with equality only where the mask
/// saturates to 0 or 1) — that lost energy is exactly the transient energy a
/// percussive-aware chain wants to move around, and the price is that the two
/// parts cannot be recombined by naive addition, only by this same mask.
///
/// # Example
///
/// ```
/// use dsp_spectral::{harmonic_percussive_split, stft, StftConfig};
///
/// let cfg = StftConfig::new(256, 64);
/// let n = 4096;
/// let x: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64;
///         0.5 * (core::f64::consts::TAU * 440.0 * t / cfg.sample_rate).sin()
///             + if i == 2048 { 1.0 } else { 0.0 }
///     })
///     .collect();
/// let spec = stft(&x, &cfg).expect("valid");
/// let (harm, perc) = harmonic_percussive_split(&spec, 8.0);
/// assert_eq!(harm.num_frames(), spec.num_frames());
/// assert_eq!(perc.bin_count(), spec.bin_count());
/// ```
#[must_use]
pub fn harmonic_percussive_split(spec: &Spectrum, hps_ratio: f64) -> (Spectrum, Spectrum) {
    let half = if hps_ratio.is_finite() {
        // Round to the nearest whole tap, floor at 1, cap so the median
        // window never exceeds the spectrogram's own extent.
        let max_half = core::cmp::max(1, core::cmp::max(spec.num_frames(), spec.bin_count()) / 2);
        let rounded = libm::floor(hps_ratio + 0.5);
        if rounded < 1.0 {
            1
        } else if rounded > max_half as f64 {
            max_half
        } else {
            rounded as usize
        }
    } else {
        HPSS_MEDIAN_HALF_WIDTH
    };

    let mags: Vec<Vec<f64>> = (0..spec.num_frames()).map(|t| spec.magnitude(t)).collect();
    let h_est = median_filter(&mags, true, half);
    let p_est = median_filter(&mags, false, half);
    // A small absolute floor keeps the mask well-defined on silent frames
    // (0/0 would be NaN) without materially affecting real ones.
    const EPS: f64 = 1e-20;

    let mut harm = Vec::with_capacity(spec.num_frames());
    let mut perc = Vec::with_capacity(spec.num_frames());
    for t in 0..spec.num_frames() {
        let Some(bins) = spec.frame(t) else {
            harm.push(Vec::new());
            perc.push(Vec::new());
            continue;
        };
        let mut h_row = Vec::with_capacity(bins.len());
        let mut p_row = Vec::with_capacity(bins.len());
        for (k, c) in bins.iter().enumerate() {
            let h = h_est.get(t).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
            let p = p_est.get(t).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
            let denom = h * h + p * p + EPS;
            let mask_h = (h * h / denom).clamp(0.0, 1.0);
            h_row.push(*c * mask_h);
            p_row.push(*c * (1.0 - mask_h));
        }
        harm.push(h_row);
        perc.push(p_row);
    }
    // Framing was validated when `spec` was built, so `from_parts` cannot fail
    // here; `Default` (an empty 2-point Hann spectrogram) keeps the function
    // total regardless.
    let build = |rows| {
        Spectrum::from_parts(
            rows,
            spec.hop(),
            spec.fft_size(),
            spec.window(),
            spec.is_centered(),
        )
        .unwrap_or_default()
    };
    (build(harm), build(perc))
}

/// Frequency-axis smoothing: convolve each frame's **magnitude** spectrum
/// with `kernel` and re-apply the original phase.
///
/// `kernel` is applied as-is (not normalised, not flipped) — pass
/// `[1.0]` for the identity, `[0.25, 0.5, 0.25]` for a 3-tap smoother,
/// `[1.0; n]` for a boxcar average. Bins near the edges get edge-clamped
/// (replicate) contributions, so the result stays the same length.
///
/// # Errors
///
/// Returns [`SpectralError::Config`] when `kernel` is empty or contains a
/// non-finite value.
pub fn spectral_smooth(spec: &Spectrum, kernel: &[f64]) -> Result<Spectrum, SpectralError> {
    if kernel.is_empty() {
        return Err(SpectralError::Config(String::from(
            "smoothing kernel must not be empty",
        )));
    }
    if kernel.iter().any(|k| !k.is_finite()) {
        return Err(SpectralError::NonFinite);
    }
    let half = kernel.len() / 2;
    let mut frames_out = Vec::with_capacity(spec.num_frames());
    for t in 0..spec.num_frames() {
        let row = spec
            .frame(t)
            .map(|bins| {
                let mags: Vec<f64> = bins.iter().map(|c| c.magnitude()).collect();
                bins.iter()
                    .enumerate()
                    .map(|(k, c)| {
                        let mut acc = 0.0;
                        for (j, &w) in kernel.iter().enumerate() {
                            let kk =
                                clamp_index(k as isize + (j as isize - half as isize), mags.len());
                            acc += w * mags.get(kk).copied().unwrap_or(0.0);
                        }
                        Complex::from_polar(acc, c.phase())
                    })
                    .collect()
            })
            .unwrap_or_default();
        frames_out.push(row);
    }
    Spectrum::from_parts(
        frames_out,
        spec.hop(),
        spec.fft_size(),
        spec.window(),
        spec.is_centered(),
    )
}

/// End-to-end noise reduction: `stft → spectral_gate → istft`.
///
/// The convenience path for the common "denoise this buffer" call. Returns
/// exactly `samples.len()` samples.
///
/// # Errors
///
/// Propagates every error from [`stft`], [`spectral_gate`], and the
/// configuration validation.
pub fn denoise(
    samples: &[f64],
    cfg: &StftConfig,
    profile: &NoiseProfile,
    gate: &GateConfig,
) -> Result<Vec<f64>, SpectralError> {
    let spec = stft(samples, cfg)?;
    let gated = spectral_gate(&spec, profile, gate)?;
    Ok(istft(&gated, samples.len()))
}

/// Signal-to-noise ratio in dB: `10·log10(Σ signal² / Σ (error²))`.
///
/// The standard objective measure for a restoration chain — pass the clean
/// reference and the processed output. Returns `+∞` for a perfect match and
/// `−∞` when the error dominates a non-zero reference.
#[must_use]
pub fn snr_db(reference: &[f64], processed: &[f64]) -> f64 {
    let mut signal = 0.0;
    let mut noise = 0.0;
    for (a, b) in reference.iter().zip(processed.iter()) {
        let d = b - a;
        signal += a * a;
        noise += d * d;
    }
    if noise <= 0.0 {
        return f64::INFINITY;
    }
    if signal <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * libm::log10(signal / noise)
}

use crate::complex::Complex;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::{
        clamp_index, denoise, harmonic_percussive_split, median_filter, median_in_place,
        smooth_time, snr_db, spectral_gate, spectral_smooth, spectral_subtract, GateConfig,
        NoiseProfile, HPSS_MEDIAN_HALF_WIDTH,
    };
    use crate::config::StftConfig;
    use crate::error::SpectralError;
    use crate::stft::{stft, Spectrum};
    use crate::window::Window;
    use alloc::vec;
    use alloc::vec::Vec;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed | 1;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_845_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    #[test]
    fn median_and_clamp_helpers() {
        let mut buf = vec![3.0, 1.0, 2.0];
        assert!((median_in_place(&mut buf) - 2.0).abs() < 1e-15);
        let mut even = vec![4.0, 1.0, 3.0, 2.0];
        assert!((median_in_place(&mut even) - 2.5).abs() <= 1e-15);
        let mut empty: Vec<f64> = Vec::new();
        assert_eq!(median_in_place(&mut empty), 0.0);

        assert_eq!(clamp_index(-3, 5), 0);
        assert_eq!(clamp_index(2, 5), 2);
        assert_eq!(clamp_index(99, 5), 4);
        assert_eq!(clamp_index(0, 0), 0);

        // Time median of a ramp: the median of a monotone run is its centre,
        // and edge frames clamp the window rather than shrinking it.
        let ramp = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let sm = median_filter(&ramp, true, 1);
        assert_eq!(sm[2][0], 3.0);
        // Frame 0's window is [1, 1, 2] (the t = −1 tap clamps to 1).
        assert_eq!(sm[0][0], 1.0);
        // Frame 4's window is [4, 5, 5].
        assert_eq!(sm[4][0], 5.0);
        // Frequency median across bins is the same machinery on the other axis.
        let bins = vec![vec![1.0, 2.0, 3.0]];
        let sm = median_filter(&bins, false, 1);
        assert_eq!(sm[0][1], 2.0);
        // Empty input is handled.
        assert!(median_filter(&[], true, 1).is_empty());
        assert!(median_filter(&[], false, 1).is_empty());
    }

    #[test]
    fn smooth_time_preserves_constants_and_widths() {
        let rows = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        assert_eq!(smooth_time(&rows, 0), rows);
        assert_eq!(smooth_time(&rows, 1), rows);
        let smoothed = smooth_time(&rows, 3);
        assert_eq!(smoothed, rows);
        let stepped = vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]];
        let out = smooth_time(&stepped, 3);
        assert!(out[1][0] < 1.0 && out[2][0] > 0.0);
        assert!(smooth_time(&[], 3).is_empty());
        // An empty row cannot average anything; it passes through as zero.
        let ragged = vec![vec![], vec![]];
        assert_eq!(smooth_time(&ragged, 3), ragged);
    }

    #[test]
    fn gate_config_validation() {
        let mut g = GateConfig::default();
        assert!((g.threshold_db - 6.0).abs() < 1e-12);
        assert!((g.reduction_db + 12.0).abs() < 1e-12);
        assert_eq!(g.smoothing_frames, 0);
        assert!(g.validate().is_ok());
        g.threshold_db = f64::NAN;
        assert!(matches!(g.validate(), Err(SpectralError::Config(_))));
        g.threshold_db = 0.0;
        g.reduction_db = f64::INFINITY;
        assert!(matches!(g.validate(), Err(SpectralError::Config(_))));
    }

    #[test]
    fn noise_profile_estimate_and_validation() {
        let cfg = StftConfig::new(64, 32);
        assert!(matches!(
            NoiseProfile::estimate(&[], &cfg),
            Err(SpectralError::EmptyInput)
        ));
        let mut bad = cfg.clone();
        bad.fft_size = 100;
        assert!(matches!(
            NoiseProfile::estimate(&[0.1; 64], &bad),
            Err(SpectralError::Config(_))
        ));
        let noise: Vec<f64> = (0..512).map(|i| (0.3 * i as f64).sin() * 0.1).collect();
        let profile = NoiseProfile::estimate(&noise, &cfg).expect("valid");
        assert_eq!(profile.spectrum().len(), cfg.bin_count());
        assert!(profile.frames() > 0);
        assert!((profile.sample_rate() - cfg.sample_rate).abs() < 1e-9);
        assert!(profile.peak() > 0.0);
        assert!(profile.magnitude_at(0) > 0.0);
        assert_eq!(profile.magnitude_at(9_999), 0.0);
        assert!(profile
            .validate_for(&stft(&noise, &cfg).expect("ok"))
            .is_ok());

        // A profile with a mismatched bin count is a config error.
        let other = NoiseProfile::estimate(&noise, &StftConfig::new(32, 16)).expect("valid");
        let spec = stft(&noise, &cfg).expect("ok");
        assert!(matches!(
            other.validate_for(&spec),
            Err(SpectralError::Config(_))
        ));
        // A non-finite profile is rejected.
        let broken = NoiseProfile::from_frames(&spec, &[0]);
        assert!(broken.validate_for(&spec).is_ok());
    }

    #[test]
    fn gate_hits_exactly_the_configured_reduction() {
        let cfg = StftConfig::new(64, 32);
        let noise: Vec<f64> = (0..256)
            .map(|i| (core::f64::consts::TAU * (i % 16) as f64 / 16.0).sin())
            .collect();
        let profile = NoiseProfile::estimate(&noise, &cfg).expect("valid");
        let spec = stft(&noise, &cfg).expect("ok");
        let reduction_db = -20.0f64;
        let gate = GateConfig {
            threshold_db: 6.0,
            reduction_db,
            smoothing_frames: 0,
        };
        let gated = spectral_gate(&spec, &profile, &gate).expect("valid");
        let want_gain = dsp_core::math::db_to_linear(reduction_db);

        // Every bin is classified independently against the profile, so the
        // gain is exactly `want_gain` (unity) or exactly `want_gain`.
        let mut gated_count = 0;
        let mut total = 0;
        for f in 0..spec.num_frames() {
            for k in 0..spec.bin_count() {
                total += 1;
                let (a, b) = (
                    spec.magnitude(f)[k],
                    gated.magnitude(f).get(k).copied().unwrap_or(0.0),
                );
                let gain = if a > 1e-12 { b / a } else { 1.0 };
                assert!(
                    (gain - 1.0).abs() < 1e-9 || (gain - want_gain).abs() < 1e-9,
                    "frame {f} bin {k}: gain {gain} is neither 1 nor {want_gain}"
                );
                // The reduction really is `reduction_db` on a gated bin.
                if (gain - want_gain).abs() < 1e-9 {
                    gated_count += 1;
                    let measured = dsp_core::math::linear_to_db(gain);
                    assert!((measured - reduction_db).abs() < 1e-6, "{measured} dB");
                }
            }
        }
        // A threshold above the profile does gate real noise.
        assert!(gated_count > total / 4, "only {gated_count}/{total} gated");

        // Raising the threshold to 100 dB leaves nothing to gate: every bin
        // is below the threshold only if... invert it — with an enormous
        // *reduction* and no threshold change, the above-threshold bins keep
        // unity gain, so an above-threshold bin must survive untouched.
        let loud = GateConfig {
            threshold_db: -40.0,
            reduction_db: -60.0,
            smoothing_frames: 0,
        };
        let almost = spectral_gate(&spec, &profile, &loud).expect("valid");
        let mut survivors = 0;
        let mags = spec.magnitude(0);
        let gated_mags = almost.magnitude(0);
        for (k, &a) in mags.iter().enumerate() {
            let b = gated_mags.get(k).copied().unwrap_or(0.0);
            if a > 1e-12 && (b / a - 1.0).abs() < 1e-9 {
                survivors += 1;
            }
        }
        assert!(survivors > 0, "an aggressive floor should spare some bins");

        // A silent frame under a noisy profile is gated hard: a zero
        // magnitude has infinite negative headroom, never "above threshold".
        let silence: Vec<f64> = vec![0.0; 256];
        let silent_spec = stft(&silence, &cfg).expect("ok");
        let gated = spectral_gate(&silent_spec, &profile, &gate).expect("valid");
        for m in gated.magnitude(0) {
            assert!(m < want_gain * 1e-6);
        }
    }

    #[test]
    fn gate_smoothing_spreads_the_decision_over_time() {
        // A gate that fires on exactly one frame: unsmoothed it is
        // all-or-nothing per frame; smoothed, its attenuation bleeds into the
        // neighbours. The spectrogram has five frames with bin 0 loud in all
        // but the second.
        let bins = 3usize;
        let rows: Vec<Vec<crate::complex::Complex>> = vec![
            vec![crate::complex::Complex::new(1.0, 0.0); bins],
            vec![crate::complex::Complex::new(0.0, 0.0); bins],
            vec![crate::complex::Complex::new(1.0, 0.0); bins],
            vec![crate::complex::Complex::new(1.0, 0.0); bins],
            vec![crate::complex::Complex::new(1.0, 0.0); bins],
        ];
        let spec = Spectrum::from_parts(rows, 1, 4, Window::Rectangular, true).expect("valid");
        let profile = NoiseProfile::from_magnitudes(alloc::vec![0.5; bins], 44_100.0);
        // The loud frames sit 6 dB above the 0.5 profile, so a +3 dB
        // threshold spares them and gates only the silent frame (whose ratio
        // is −∞, i.e. infinitely far below any threshold).
        let sharp = GateConfig {
            threshold_db: 3.0,
            reduction_db: -12.0,
            smoothing_frames: 0,
        };
        let out = spectral_gate(&spec, &profile, &sharp).expect("valid");
        let mag = |s: &Spectrum, f: usize| s.magnitude(f)[0];
        assert!((mag(&out, 0) - 1.0).abs() < 1e-12, "frame 0 untouched");
        assert!(mag(&out, 1) < 1e-12, "frame 1 gated");
        assert!((mag(&out, 2) - 1.0).abs() < 1e-12, "frame 2 untouched");

        // A 5-frame window (half-width 2) spreads the single gated frame's
        // gain over its neighbours: every output frame whose 5-tap window
        // contains frame 1 averages four unity gains with one `10^(-12/20)`,
        // landing at `(4 + 10^(-12/20))/5`.
        let smooth = GateConfig {
            smoothing_frames: 5,
            ..sharp
        };
        let out = spectral_gate(&spec, &profile, &smooth).expect("valid");
        let expected = (4.0 + dsp_core::math::db_to_linear(-12.0)) / 5.0;
        for f in [0usize, 2, 3] {
            assert!(
                (mag(&out, f) - expected).abs() < 1e-12,
                "frame {f}: {} vs {expected}",
                mag(&out, f)
            );
        }
        // Frame 1 is still all the way down: it was silent to begin with, so
        // no gain can resurrect it.
        assert!(mag(&out, 1) < 1e-12);
        // Frame 4's window only reaches frame 3, so it never sees the gated
        // frame and keeps the full gain.
        assert!((mag(&out, 4) - 1.0).abs() < 1e-12, "{}", mag(&out, 4));
        // Smoothing never increases a bin's magnitude above the untouched value,
        // and never attenuates one by more than the configured reduction
        // (except a bin that was already silent — no gain resurrects zero).
        let floor_gain = dsp_core::math::db_to_linear(-12.0);
        for f in 0..spec.num_frames() {
            let before = spec.magnitude(f)[0];
            let after = mag(&out, f);
            assert!(after <= before + 1e-12, "frame {f} gained energy");
            if before > 1e-12 {
                assert!(
                    after >= before * floor_gain - 1e-12,
                    "frame {f} over-attenuated"
                );
            }
        }
        // The gate is total: no panics on an empty spectrogram.
        let empty = Spectrum::from_parts(Vec::new(), 1, 4, Window::Rectangular, true).expect("ok");
        let _ = spectral_gate(&empty, &profile, &smooth).expect("valid");
    }

    #[test]
    fn gate_error_paths() {
        let cfg = StftConfig::new(64, 32);
        let spec = stft(&[0.5; 256], &cfg).expect("ok");
        let profile = NoiseProfile::from_frames(&spec, &[0]);
        let bad = GateConfig {
            threshold_db: f64::NAN,
            reduction_db: -12.0,
            smoothing_frames: 0,
        };
        assert!(matches!(
            spectral_gate(&spec, &profile, &bad),
            Err(SpectralError::Config(_))
        ));
        let mismatch = NoiseProfile::from_frames(
            &stft(&[0.5; 256], &StftConfig::new(32, 16)).expect("ok"),
            &[0],
        );
        assert!(matches!(
            spectral_gate(&spec, &mismatch, &GateConfig::default()),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            spectral_subtract(&spec, &profile, f64::NAN),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            spectral_subtract(&spec, &profile, -1.0),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            spectral_smooth(&spec, &[]),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            spectral_smooth(&spec, &[f64::NAN]),
            Err(SpectralError::NonFinite)
        ));
    }

    #[test]
    fn subtract_zero_is_identity() {
        let cfg = StftConfig::new(64, 32);
        let x: Vec<f64> = (0..512).map(|i| (0.11 * i as f64).sin()).collect();
        let spec = stft(&x, &cfg).expect("ok");
        let profile = NoiseProfile::from_frames(&spec, &[0]);
        let same = spectral_subtract(&spec, &profile, 0.0).expect("valid");
        assert_eq!(same, spec);
        // Over-subtracting a loud profile floors at zero magnitude.
        let loud = NoiseProfile::from_magnitudes(alloc::vec![1e6; cfg.bin_count()], 0.0);
        let gone = spectral_subtract(&spec, &loud, 1.0).expect("valid");
        assert!(gone.magnitude(0).iter().all(|m| *m == 0.0));
    }

    #[test]
    fn smooth_identity_is_a_no_op() {
        let cfg = StftConfig::new(64, 32);
        let x: Vec<f64> = (0..512).map(|i| (0.21 * i as f64).cos()).collect();
        let spec = stft(&x, &cfg).expect("ok");
        let same = spectral_smooth(&spec, &[1.0]).expect("valid");
        for (a, b) in spec.magnitude(0).iter().zip(same.magnitude(0).iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        // A boxcar over 3 taps smooths a spiky spectrum.
        let boxcar = spectral_smooth(&spec, &[1.0 / 3.0; 3]).expect("valid");
        let peak_spec = spec.magnitude(0).iter().copied().fold(0.0f64, f64::max);
        let peak_box = boxcar.magnitude(0).iter().copied().fold(0.0f64, f64::max);
        assert!(peak_box < peak_spec);
    }

    #[test]
    fn hpss_retains_both_components() {
        let fs = 16_000.0;
        let cfg = StftConfig::new(1024, 256).with_sample_rate(fs);
        let n = 16_384usize;
        let mut x = vec![0.0f64; n];
        // Harmonic: a steady 400 Hz tone, so its spectrum is two sharp bins
        // that stay put across time.
        for (i, v) in x.iter_mut().enumerate() {
            *v = 0.4 * (core::f64::consts::TAU * 400.0 * (i as f64) / fs).sin();
        }
        // Percussive: a three-sample click — broadband, and gone next frame.
        for (offset, gain) in [1.0f64, 0.6, 0.3].iter().enumerate() {
            if let Some(v) = x.get_mut(n / 2 + offset) {
                *v += gain;
            }
        }
        let spec = stft(&x, &cfg).expect("ok");

        let bin_of =
            |hz: f64| ((hz * (cfg.fft_size as f64) / fs) as usize).min(spec.bin_count() - 1);
        let tone_bin = bin_of(400.0);
        let hi_bin = bin_of(2_000.0);
        let nyq = spec.bin_count() - 1;

        // Centred framing pads by fft/2, so the click lands at padded index
        // `n/2 + fft/2`. Pick the frame that carries it at a non-zero window
        // tap (a tap of exactly 0 would discard the click entirely).
        let click_padded = n / 2 + cfg.fft_size / 2;
        let click_frame = (click_padded - 8) / cfg.hop;
        let steady_frame = 10usize;

        let (harm, perc) = harmonic_percussive_split(&spec, 8.0);
        assert_eq!(harm.num_frames(), spec.num_frames());
        assert_eq!(perc.bin_count(), spec.bin_count());

        let power_at = |s: &Spectrum, f: usize, k: usize| {
            s.magnitude(f).get(k).copied().unwrap_or(0.0).powi(2)
        };
        let band_power = |s: &Spectrum, f: usize| s.power(f)[hi_bin..=nyq].iter().sum::<f64>();

        // The harmonic part keeps the tone: at its bin the harmonic estimate
        // is orders of magnitude above the percussive one.
        let (h, p) = (
            power_at(&harm, steady_frame, tone_bin),
            power_at(&perc, steady_frame, tone_bin),
        );
        assert!(h > 1_000.0 * p, "tone bin h={h} p={p}");

        // The percussive part keeps the transient: broadband energy above the
        // tone, where the signal is otherwise silent, appears in the
        // percussive estimate at the click frame and nowhere else.
        let click_hi = band_power(&perc, click_frame);
        let steady_hi = band_power(&perc, steady_frame);
        assert!(
            click_hi > 100.0 * steady_hi,
            "click hi {click_hi} vs {steady_hi}"
        );
        assert!(
            band_power(&harm, click_frame) < 0.01 * click_hi,
            "click is harmonic"
        );

        // The complementary mask splits *magnitude* exactly: |harmonic| +
        // |percussive| == |X| at every bin, at every frame.
        for f in 0..spec.num_frames() {
            let (a, b, c) = (spec.magnitude(f), harm.magnitude(f), perc.magnitude(f));
            for k in 0..spec.bin_count() {
                let lhs = b[k] + c[k];
                let rhs = a[k];
                assert!(
                    (lhs - rhs).abs() <= 1e-9 * rhs.max(1.0),
                    "frame {f} bin {k}: {lhs} vs {rhs}"
                );
            }
        }

        // Degenerate arguments still produce a usable pair.
        let (h0, p0) = harmonic_percussive_split(&spec, 0.0);
        assert_eq!(h0.num_frames(), spec.num_frames());
        assert_eq!(p0.num_frames(), spec.num_frames());
        // A NaN ratio is coerced to the neutral 1.0, not propagated.
        let (h_nan, p_nan) = harmonic_percussive_split(&spec, f64::NAN);
        assert_eq!(h_nan.num_frames(), spec.num_frames());
        assert_eq!(p_nan.num_frames(), spec.num_frames());
        let (h_big, _p_big) = harmonic_percussive_split(&spec, 1e9);
        assert_eq!(h_big.num_frames(), spec.num_frames());
        let empty = Spectrum::from_parts(Vec::new(), 1, 8, Window::Hann, true).expect("valid");
        let (a, b) = harmonic_percussive_split(&empty, 4.0);
        assert_eq!(a.num_frames(), 0);
        assert_eq!(b.num_frames(), 0);
        assert_eq!(HPSS_MEDIAN_HALF_WIDTH, 8);
    }

    #[test]
    fn denoise_improves_snr() {
        let fs = 16_000.0;
        let cfg = StftConfig::new(1024, 256).with_sample_rate(fs);
        let n = 32_768usize;
        let mut rng = lcg(0x5EED);
        let clean: Vec<f64> = (0..n)
            .map(|i| 0.5 * (core::f64::consts::TAU * 500.0 * (i as f64) / fs).sin())
            .collect();
        // Leading noise-only segment for the profile.
        let noise_only: Vec<f64> = (0..n / 2).map(|_| 0.02 * rng()).collect();
        let profile = NoiseProfile::estimate(&noise_only, &cfg).expect("valid");
        // Corrupt the clean signal with noise at the same level.
        let noisy: Vec<f64> = clean
            .iter()
            .zip((0..n).map(|_| 0.02 * rng()))
            .map(|(a, b)| a + b)
            .collect();
        let before = snr_db(&clean, &noisy);
        let gate = GateConfig {
            threshold_db: -3.0,
            reduction_db: -18.0,
            smoothing_frames: 3,
        };
        let out = denoise(&noisy, &cfg, &profile, &gate).expect("valid");
        assert_eq!(out.len(), noisy.len());
        let after = snr_db(&clean, &out);
        assert!(after > before, "denoise did not help: {before} → {after}");
        // Edge cases: identical signals are +∞ (no error at all), a silent
        // reference with any error is −∞, and a finite case is a plain
        // power ratio.
        assert_eq!(snr_db(&clean, &clean), f64::INFINITY);
        assert_eq!(snr_db(&[0.0; 4], &[0.0; 4]), f64::INFINITY);
        assert_eq!(snr_db(&[0.0; 4], &[1.0; 4]), f64::NEG_INFINITY);
        assert!((snr_db(&[1.0; 4], &[3.0; 4]) + 6.020_599_913).abs() < 1e-9);
        assert_eq!(snr_db(&[], &[]), f64::INFINITY);
    }

    #[test]
    fn profile_from_frames_matches_manual_average() {
        let cfg = StftConfig::new(64, 32);
        let x: Vec<f64> = (0..1024).map(|i| (0.07 * i as f64).sin()).collect();
        let spec = stft(&x, &cfg).expect("ok");
        let idx = [1usize, 3, 5];
        let profile = NoiseProfile::from_frames(&spec, &idx);
        let manual: Vec<f64> = (0..spec.bin_count())
            .map(|k| {
                idx.iter()
                    .map(|&f| spec.magnitude(f).get(k).copied().unwrap_or(0.0))
                    .sum::<f64>()
                    / 3.0
            })
            .collect();
        for (a, b) in profile.spectrum().iter().zip(manual.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(profile.frames(), 3);
        assert_eq!(profile.sample_rate(), 0.0);
        // Out-of-range indices are ignored, not fatal.
        let partial = NoiseProfile::from_frames(&spec, &[1, 9_999]);
        assert_eq!(partial.frames(), 1);
        let none = NoiseProfile::from_frames(&spec, &[9_999]);
        assert_eq!(none.frames(), 0);
        assert!(none.spectrum().iter().all(|v| *v == 0.0));
        assert_eq!(none.peak(), 0.0);
        let none = NoiseProfile::from_frames(&spec, &[]);
        assert_eq!(none.frames(), 0);
        assert!((none.with_sample_rate(8_000.0).sample_rate() - 8_000.0).abs() < 1e-9);
    }
}
