//! Short-time Fourier transform and its inverse.
//!
//! [`stft`] frames a signal, windows it, and transforms each frame with the
//! radix-2 FFT owned by [`dsp_core`] (L0), keeping the `fft_size / 2 + 1`
//! useful bins of every frame. [`istft`] reverses it: hermitian-symmetric
//! spectrum expansion, inverse FFT, re-window, and overlap-add divided by the
//! **measured** squared-window weight at each output sample.
//!
//! # Why measured normalisation
//!
//! Classic STFT inversion divides by the constant `sum_k w[k*hop]^2`. That is
//! only correct where the overlap is genuinely constant, and it cannot
//! represent the two places where it is not: the first and last samples of
//! the signal (fewer overlapping frames) and any window whose end taps are
//! zero (Hann, Blackman). Measuring `sum_k w[k*hop]^2` per output sample instead
//! costs one extra accumulator and makes the inverse exact for **any**
//! window and hop — which is what lets the round-trip test assert 1e-15
//! rather than "roughly right".
//!
//! # Centre padding
//!
//! With `center = true` the signal is reflected by `fft_size / 2` on both
//! sides before framing, so frame 0 is centred on sample 0 and the
//! output region is always fully covered by frames with non-negligible taps.
//! [`istft`] then drops those `fft_size / 2` samples again. The net effect is
//! `istft(stft(x), x.len()) == x` for every supported window. With
//! `center = false` the first and last samples are only recoverable when the
//! window's end taps are non-zero (rectangular, Hamming, Blackman–Harris,
//! flat top) — a window that tapers to zero has thrown that energy away.
//!
//! # Frame counts
//!
//! | Layout | Frames |
//! |---|---|
//! | `center = true` | `1 + len / hop` (integer division) |
//! | `center = false` | `(len - fft_size) / hop + 1`, or `0` when `len < fft_size` |
//! |
//! # Example
//!
//! ```rust
//! use dsp_spectral::{istft, stft, StftConfig};
//!
//! let cfg = StftConfig::new(256, 64);
//! let n = 2048;
//! let signal: Vec<f64> = (0..n).map(|i| (0.05 * i as f64).sin()).collect();
//!
//! let spec = stft(&signal, &cfg).expect("valid config and input");
//! assert_eq!(spec.num_frames(), 1 + n / 64);
//! assert_eq!(spec.bin_count(), 129);
//!
//! let back = istft(&spec, signal.len());
//! for (a, b) in signal.iter().zip(back.iter()) {
//!     assert!((a - b).abs() < 1e-12, "round-trip drift");
//! }
//! ```

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use dsp_core::fft::Fft;

use crate::complex::Complex;
use crate::config::StftConfig;
use crate::error::SpectralError;
use crate::window::Window;

/// Below this accumulated squared-window weight an output sample has no
/// usable signal energy (a Hann end tap) and is emitted as an exact zero.
const MIN_WINDOW_WEIGHT: f64 = 1e-12;

/// A short-time Fourier transform: `frames` rows of `bins` complex values.
///
/// Construct one with [`stft`] or [`Spectrum::from_parts`] (the latter for
/// synthetic spectra — feature extraction is defined on any well-formed
/// spectrogram). Every accessor is infallible; out-of-range indices resolve
/// to empty slices or empty vectors rather than panics.
#[derive(Clone, PartialEq)]
pub struct Spectrum {
    frames: Vec<Vec<Complex>>,
    bins: usize,
    hop: usize,
    fft_size: usize,
    window: Window,
    center: bool,
}

impl Default for Spectrum {
    /// The smallest *valid* spectrogram: no frames, 2-point transform, unit
    /// hop, Hann window, centred. Every feature on it returns an empty vector
    /// and [`istft`] returns silence — which is the right answer for
    /// "no analysis was performed".
    fn default() -> Self {
        Self {
            frames: Vec::new(),
            bins: 2,
            hop: 1,
            fft_size: 2,
            window: Window::Hann,
            center: true,
        }
    }
}

impl core::fmt::Debug for Spectrum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Spectrum")
            .field("num_frames", &self.frames.len())
            .field("bins", &self.bins)
            .field("hop", &self.hop)
            .field("fft_size", &self.fft_size)
            .field("window", &self.window)
            .field("center", &self.center)
            .finish()
    }
}

/// Where the signal starts inside the padded buffer, and how many frames fit.
///
/// `center = true` pads by `fft_size / 2` on both sides (`pad`) and emits
/// `1 + len / hop` frames; `center = false` pads nothing and emits
/// `(len - fft_size) / hop + 1` frames.
fn frame_layout(len: usize, cfg: &StftConfig) -> (usize, usize) {
    let n = cfg.fft_size;
    if cfg.center {
        (n / 2, len / cfg.hop + 1)
    } else if len < n {
        (0, 0)
    } else {
        (0, (len - n) / cfg.hop + 1)
    }
}

/// Reflect-pad `samples` by `pad` on both sides (nearest-sample first, the
/// "reflect 101" convention: the edge sample is never repeated, so padding
/// `1, 2, 3, 4, 5` by 2 gives `3, 2, 1, 2, 3, 4, 5, 4, 3`).
///
/// Pads longer than the signal fold back and forth through it, so a signal
/// shorter than `fft_size` still gets a full frame of context.
fn reflect_pad(samples: &[f64], pad: usize) -> Vec<f64> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for k in 0..pad {
        out.push(
            *samples
                .get(mirror_index(n, -(pad as isize) + k as isize))
                .unwrap_or(&0.0),
        );
    }
    out.extend_from_slice(samples);
    for k in 0..pad {
        out.push(
            *samples
                .get(mirror_index(n, n as isize + k as isize))
                .unwrap_or(&0.0),
        );
    }
    out
}

/// Map a virtual index (which may lie outside `0..n`) onto a real sample of a
/// length-`n` buffer by reflecting at both edges without repeating the edge
/// samples ("reflect 101"). The extended sequence `0, 1, …, n−1, n−2, …, 1`
/// has period `2n − 2`, so anything beyond folds back through it.
fn mirror_index(n: usize, v: isize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = (2 * n - 2) as isize;
    let mut m = v.rem_euclid(period);
    if m < 0 {
        m += period;
    }
    let idx = if m < n as isize { m } else { period - m };
    usize::try_from(idx).unwrap_or(0).min(n - 1)
}

impl Spectrum {
    /// Build a spectrogram from raw complex frames.
    ///
    /// Every frame must hold exactly `fft_size / 2 + 1` bins, `hop` must be
    /// positive, and `fft_size` a power of two `>= 2`. Zero frames is legal
    /// (an empty spectrogram), and every feature on it returns an empty vector.
    ///
    /// # Errors
    ///
    /// Returns [`SpectralError::Config`] on a ragged or wrong-width frame
    /// row, a zero hop, or an unsupported transform size.
    pub fn from_parts(
        frames: Vec<Vec<Complex>>,
        hop: usize,
        fft_size: usize,
        window: Window,
        center: bool,
    ) -> Result<Self, SpectralError> {
        if fft_size < 2 || !fft_size.is_power_of_two() {
            return Err(SpectralError::Config(format!(
                "fft_size {fft_size} must be a power of two >= 2"
            )));
        }
        if hop == 0 {
            return Err(SpectralError::Config(String::from("hop must be > 0")));
        }
        let bins = fft_size / 2 + 1;
        if let Some(frame) = frames.iter().find(|f| f.len() != bins) {
            return Err(SpectralError::Config(format!(
                "every frame must hold {bins} bins, found {}",
                frame.len()
            )));
        }
        Ok(Self {
            frames,
            bins,
            hop,
            fft_size,
            window,
            center,
        })
    }

    /// Number of analysis frames.
    #[must_use]
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Useful bins per frame, `fft_size / 2 + 1` (DC through Nyquist).
    #[must_use]
    pub fn bin_count(&self) -> usize {
        self.bins
    }

    /// Samples between successive frames.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Transform size in samples.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The analysis window shape.
    #[must_use]
    pub fn window(&self) -> Window {
        self.window
    }

    /// Whether this spectrogram came from a centre-padded analysis.
    #[must_use]
    pub fn is_centered(&self) -> bool {
        self.center
    }

    /// Frame `i`, or `None` when `i >= num_frames`.
    #[must_use]
    pub fn frame(&self, i: usize) -> Option<&[Complex]> {
        self.frames.get(i).map(Vec::as_slice)
    }

    /// Frame `i`, or [`SpectralError::FrameOutOfRange`] when
    /// `i >= num_frames`.
    ///
    /// # Errors
    ///
    /// Returns [`SpectralError::FrameOutOfRange`] carrying the offending
    /// index.
    pub fn frame_checked(&self, i: usize) -> Result<&[Complex], SpectralError> {
        self.frames
            .get(i)
            .map(Vec::as_slice)
            .ok_or(SpectralError::FrameOutOfRange(i))
    }

    /// All frames, in order.
    #[must_use]
    pub fn frames(&self) -> &[Vec<Complex>] {
        &self.frames
    }

    /// Linear magnitude spectrum `|X[k]|` of frame `frame`; empty when
    /// the index is out of range.
    #[must_use]
    pub fn magnitude(&self, frame: usize) -> Vec<f64> {
        self.frames
            .get(frame)
            .map(|bins| bins.iter().map(|c| c.magnitude()).collect())
            .unwrap_or_default()
    }

    /// Linear power spectrum `|X[k]|²` of frame `frame`; empty when the
    /// index is out of range.
    ///
    /// **Linear, not decibels** — Parseval's identity holds exactly:
    /// `sum_n x[n]^2 = (1/N) * sum_k P[k]`. Use
    /// [`power_db`](Self::power_db) for a decibel view.
    #[must_use]
    pub fn power(&self, frame: usize) -> Vec<f64> {
        self.frames
            .get(frame)
            .map(|bins| bins.iter().map(|c| c.power()).collect())
            .unwrap_or_default()
    }

    /// Magnitude spectrum of frame `frame` in decibels, `20 * log10(|X[k]|)`
    /// with a documented floor: a zero bin reads [`POWER_FLOOR_DB`].
    #[must_use]
    pub fn magnitude_db(&self, frame: usize) -> Vec<f64> {
        linear_to_db_floored(&self.magnitude(frame))
    }

    /// Power spectrum of frame `frame` in decibels, `10 * log10(|X[k]|^2)`,
    /// with the same [`POWER_FLOOR_DB`] floor as
    /// [`magnitude_db`](Self::magnitude_db).
    #[must_use]
    pub fn power_db(&self, frame: usize) -> Vec<f64> {
        self.magnitude_db(frame)
            .into_iter()
            .map(|db| 2.0 * db)
            .collect()
    }

    /// Build a mel filterbank sized for this spectrogram's transform.
    ///
    /// # Panics
    ///
    /// None. Degenerate parameters (`n_mels == 0`, non-positive sample rate,
    /// `f_min >= f_max`) yield a bank with no filters, whose
    /// [energies](MelBank::energies) are empty — callers that need the
    /// argument validated should build the bank with
    /// [`MelBank::new`], which reports those as [`SpectralError::Config`].
    #[must_use]
    pub fn mel_filterbank(
        &self,
        n_mels: usize,
        f_min: f64,
        f_max: f64,
        sample_rate: f64,
    ) -> crate::mel::MelBank {
        crate::mel::MelBank::build(n_mels, f_min, f_max, sample_rate, self.fft_size)
    }

    /// Copy with a different analysis window tag. The frames are unchanged —
    /// this only relabels the window for a subsequent [istft].
    #[must_use]
    pub fn with_window(&self, window: Window) -> Self {
        Self {
            frames: self.frames.clone(),
            bins: self.bins,
            hop: self.hop,
            fft_size: self.fft_size,
            window,
            center: self.center,
        }
    }

    /// How many time-domain samples the frames actually cover, counting from
    /// the start of the analysis buffer.
    ///
    /// With `center = true` this always exceeds `fft_size/2 + len` (the pad is
    /// `fft_size/2`, and the hop's residue cannot spill past the window), so
    /// the whole signal is reconstructible. With `center = false` the frame
    /// count is `(len − fft_size)/hop + 1`, which stops short: the final
    /// `len − covered_len()` samples fall inside no full frame, and
    /// [`istft`](istft) emits them as exact zeros. Use this to find where the
    /// reconstructible region ends.
    #[must_use]
    pub fn covered_len(&self) -> usize {
        self.frames
            .len()
            .saturating_sub(1)
            .saturating_mul(self.hop)
            .saturating_add(self.fft_size)
    }
}

/// Floor used when converting a zero (or subnormal) magnitude to decibels:
/// −200 dBFS, far below anything representable in f64 audio.
pub const POWER_FLOOR_DB: f64 = -200.0;

fn linear_to_db_floored(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&v| {
            if v > 1e-10 {
                20.0 * libm::log10(v)
            } else {
                POWER_FLOOR_DB
            }
        })
        .collect()
}

/// Forward short-time Fourier transform.
///
/// # Errors
///
/// Returns [`SpectralError::Config`] when cfg fails
/// [`StftConfig::validate`], [`SpectralError::EmptyInput`] for an empty
/// samples, and [`SpectralError::NonFinite`] if any sample is NaN or ±∞.
///
/// # Example
///
/// ```rust
/// use dsp_spectral::{Spectrum, Window, stft, StftConfig};
///
/// let cfg = StftConfig::new(16, 8);
/// let spec = stft(&[1.0; 32], &cfg).expect("valid");
/// assert_eq!(spec.num_frames(), 1 + 32 / 8);
/// assert_eq!(spec.bin_count(), 9);
///
/// // A DC signal windowed by a Hann sums to the window's own area, which
/// // lands in bin 0; the rest is the window's own spectral skirt — at this
/// // transform size it has not yet fallen far, which is exactly why a
/// // frequency-domain claim needs a stated transform length.
/// let mag = spec.magnitude(0);
/// let area: f64 = Window::Hann.coefficients(16).iter().sum();
/// assert!((mag[0] - area).abs() < 1e-9, "{} vs {area}", mag[0]);
/// assert!(mag.iter().all(|m| *m <= mag[0] + 1e-12));
///
/// // Out-of-range frame access is empty, or a typed error when asked for
/// // the index back.
/// assert_eq!(spec.frame(999), None);
/// assert!(matches!(
///     spec.frame_checked(999),
///     Err(dsp_spectral::SpectralError::FrameOutOfRange(999))
/// ));
/// let _ = Spectrum::bin_count(&spec);
/// ```
pub fn stft(samples: &[f64], cfg: &StftConfig) -> Result<Spectrum, SpectralError> {
    cfg.validate()?;
    if samples.is_empty() {
        return Err(SpectralError::EmptyInput);
    }
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(SpectralError::NonFinite);
    }

    let n = cfg.fft_size;
    let window = cfg.window.coefficients(n);
    let (pad, n_frames) = frame_layout(samples.len(), cfg);
    let padded = if pad > 0 {
        reflect_pad(samples, pad)
    } else {
        samples.to_vec()
    };
    let bins = n / 2 + 1;
    let fft = Fft::new(n).map_err(|e| SpectralError::Dsp(format!("fft: {e}")))?;
    let mut buf = alloc::vec![0.0f64; n * 2];
    let mut frames = Vec::with_capacity(n_frames);

    for f in 0..n_frames {
        let start = f * cfg.hop;
        for (j, (pair, &tap)) in buf.chunks_exact_mut(2).zip(window.iter()).enumerate() {
            if let [re, im] = pair {
                *re = padded.get(start + j).copied().unwrap_or(0.0) * tap;
                *im = 0.0;
            }
        }
        fft.forward(&mut buf)
            .map_err(|e| SpectralError::Dsp(format!("fft: {e}")))?;
        frames.push(
            buf.chunks_exact(2)
                .take(bins)
                .map(|c| match c {
                    [re, im, ..] => Complex::new(*re, *im),
                    _ => Complex::ZERO,
                })
                .collect(),
        );
    }

    Spectrum::from_parts(frames, cfg.hop, n, cfg.window, cfg.center)
}

/// Inverse short-time Fourier transform by weighted overlap-add.
///
/// Returns exactly `original_len` samples. With `center = true` the leading
/// `fft_size / 2` padded samples are dropped; with `center = false` output
/// sample `i` is input sample `i`.
///
/// Two regions are deliberately **not** reconstructions, and both follow from
/// the analysis rather than from a defect in the inverse:
///
/// - Samples past [`Spectrum::covered_len`] fall inside no frame at all, so
///   they come out as exact `0.0`. Only reachable with `center = false` and a
///   `len` that is not `fft_size + k·hop`.
/// - Samples whose accumulated squared-window weight is below 1e-12 — the very
///   endpoints of a zero-tapered window (Hann, Blackman), where the analysis
///   genuinely carries no information — are also exact `0.0`.
///
/// Everything else reconstructs to f64 round-off. Never fails: an empty
/// spectrogram, a zero `original_len`, or a spectrum with no frames all yield
/// a correspondingly empty or zeroed vector.
#[must_use]
pub fn istft(spec: &Spectrum, original_len: usize) -> Vec<f64> {
    if original_len == 0 || spec.frames.is_empty() {
        return alloc::vec![0.0; original_len];
    }
    let n = spec.fft_size;
    let bins = spec.bins;
    let pad = if spec.center { n / 2 } else { 0 };
    let covered = spec.frames.len().saturating_sub(1) * spec.hop + n;
    let needed = pad + original_len;
    let total = core::cmp::max(covered, needed);

    let window = spec.window.coefficients(n);
    // Unreachable in practice: `fft_size` was validated when the spectrum was
    // built. Handled rather than unwrapped so the function stays total.
    let Ok(fft) = Fft::new(n) else {
        return alloc::vec![0.0; original_len];
    };

    let mut acc = alloc::vec![0.0f64; total];
    let mut weight = alloc::vec![0.0f64; total];
    let mut buf = alloc::vec![0.0f64; n * 2];

    for (f, frame) in spec.frames.iter().enumerate() {
        let start = f * spec.hop;
        // Expand the half-spectrum back to a full hermitian spectrum.
        //
        // Bins 0..bins are stored (DC through Nyquist). Bins n-k for
        // k in 1..bins-1 are the conjugate mirror of bin k, so the
        // upper half is X[n-k] = conj(X[k]). DC and Nyquist are real and
        // map to themselves.
        for (k, chunk) in buf.chunks_exact_mut(2).enumerate() {
            let (re, im) = if k < bins {
                frame.get(k).map_or((0.0, 0.0), |c| (c.re, c.im))
            } else {
                let mirrored = n - k;
                // For k in `bins..n` the mirrored index runs `n/2-1` down to
                // 1: DC (k = n) is unreachable and Nyquist (k = n/2) already
                // took the `k < bins` branch, so every value here is a true
                // interior conjugate mirror and needs only its imaginary part
                // negated.
                match frame.get(mirrored) {
                    Some(c) => (c.re, -c.im),
                    None => (0.0, 0.0),
                }
            };
            if let [a, b] = chunk {
                *a = re;
                *b = im;
            }
        }
        if fft.inverse(&mut buf).is_err() {
            return alloc::vec![0.0; original_len];
        }
        // The inverse transform returns the *windowed* frame (w·x), so
        // re-applying the window here gives w²·x; dividing by the
        // accumulated Σ w² at each output sample then recovers x
        // exactly. Skipping this window multiply is the classic STFT-inverse
        // bug and shows up as a 1/w gain error.
        for (j, chunk) in buf.chunks_exact(2).enumerate() {
            let x = chunk.first().copied().unwrap_or(0.0);
            let tap = window.get(j).copied().unwrap_or(0.0);
            if let Some(slot) = acc.get_mut(start + j) {
                *slot += x * tap;
            }
            if let Some(slot) = weight.get_mut(start + j) {
                *slot += tap * tap;
            }
        }
    }

    (0..original_len)
        .map(|i| {
            let idx = pad + i;
            let w = weight.get(idx).copied().unwrap_or(0.0);
            let a = acc.get(idx).copied().unwrap_or(0.0);
            if w > MIN_WINDOW_WEIGHT {
                a / w
            } else {
                0.0
            }
        })
        .collect()
}

/// Time-scale a spectrogram by re-hopping it: the same frames, laid out with
/// a different stride. The frame *content* is unchanged (same window, same
/// transform, so pitch is untouched) and only the timeline stretches or
/// compresses by factor — the classic overlap-add time-scale primitive,
/// with the usual caveat that frames overlap differently afterwards, so the
/// result is an OLA approximation rather than a resampled waveform.
///
/// A subsequent [istft] on the returned spectrogram yields
/// `≈ (num_frames - 1) * hop' + fft_size - fft_size / 2` samples, i.e.
/// roughly `original_len * factor` when the spectrogram came from a
/// centred STFT.
///
/// # Errors
///
/// Returns [`SpectralError::Config`] when `factor` is not finite and
/// positive, or when the implied hop would round to zero.
pub fn resample_frames(spec: &Spectrum, factor: f64) -> Result<Spectrum, SpectralError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(SpectralError::Config(String::from(
            "resample factor must be finite and > 0",
        )));
    }
    let hop = (spec.hop as f64 * factor + 0.5) as usize;
    if hop == 0 {
        return Err(SpectralError::Config(String::from(
            "resampled hop must be >= 1",
        )));
    }
    Spectrum::from_parts(
        spec.frames.clone(),
        hop,
        spec.fft_size,
        spec.window,
        spec.center,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::{frame_layout, istft, mirror_index, reflect_pad, resample_frames, stft, Spectrum};
    use crate::complex::Complex;
    use crate::config::StftConfig;
    use crate::error::SpectralError;
    use crate::window::Window;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f64::consts::TAU;

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
    fn frame_layout_counts() {
        let cfg = StftConfig::new(1024, 256).with_center(true);
        assert_eq!(frame_layout(1000, &cfg), (512, 1 + 1000 / 256));
        let cfg = StftConfig::new(1024, 256).with_center(false);
        assert_eq!(frame_layout(2048, &cfg), (0, (2048 - 1024) / 256 + 1));
        assert_eq!(frame_layout(1024, &cfg), (0, 1));
        assert_eq!(frame_layout(1023, &cfg), (0, 0));
    }

    #[test]
    fn reflect_padding_layout() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            reflect_pad(&x, 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
        );
        // Long pads fold repeatedly; a single sample repeats.
        // A pad longer than the signal folds back through it: 7 taps of
        // reflection over a 5-sample signal repeats the 8-sample period.
        let long = reflect_pad(&x, 7);
        assert_eq!(long.len(), 5 + 14);
        assert_eq!(
            long,
            vec![
                2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, // left fold
                1.0, 2.0, 3.0, 4.0, 5.0, // signal
                4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, // right fold
            ]
        );
        assert_eq!(reflect_pad(&[9.0], 3), vec![9.0; 7]);
        assert_eq!(reflect_pad(&x, 0), x.to_vec());
        // The virtual index mapping: identity inside, reflect outside.
        assert_eq!(mirror_index(5, 0), 0);
        assert_eq!(mirror_index(5, 4), 4);
        assert_eq!(mirror_index(5, -1), 1);
        assert_eq!(mirror_index(5, -2), 2);
        assert_eq!(mirror_index(5, 5), 3);
        assert_eq!(mirror_index(5, 6), 2);
        assert_eq!(mirror_index(1, 7), 0);
        assert_eq!(mirror_index(0, 3), 0);
        // Every padded sample is one the signal actually contains.
        for &v in &reflect_pad(&x, 12) {
            assert!(x.contains(&v));
        }
    }

    #[test]
    fn empty_and_short_paths() {
        let cfg = StftConfig::new(64, 16);
        assert!(matches!(stft(&[], &cfg), Err(SpectralError::EmptyInput)));
        assert!(matches!(
            stft(&[f64::NAN], &cfg),
            Err(SpectralError::NonFinite)
        ));
        assert!(matches!(
            stft(&[f64::INFINITY; 4], &cfg),
            Err(SpectralError::NonFinite)
        ));
        // centre = false with len < fft_size yields no frames at all.
        let cfg_nc = StftConfig::new(64, 16).with_center(false);
        let spec = stft(&[0.1; 10], &cfg_nc).expect("ok");
        assert_eq!(spec.num_frames(), 0);
        assert_eq!(istft(&spec, 10), vec![0.0; 10]);
        assert!(spectral_calls_on_empty(&spec));
        // Zero-length synthesis request on a populated spectrogram.
        let spec = stft(&[0.1; 200], &cfg).expect("ok");
        assert!(istft(&spec, 0).is_empty());
    }

    fn spectral_calls_on_empty(spec: &Spectrum) -> bool {
        spec.magnitude(0).is_empty()
            && spec.power(0).is_empty()
            && spec.magnitude_db(0).is_empty()
            && spec.power_db(0).is_empty()
            && spec.frame(0).is_none()
            && spec.frame_checked(0).is_err()
    }

    #[test]
    fn impulse_frame_is_flat() {
        // Uncentred, so frame 0 really does start at sample 0 and the
        // impulse sits at tap 8.
        let cfg = StftConfig::new(64, 32).with_center(false);
        let mut x = vec![0.0; 64];
        x[8] = 1.0;
        let spec = stft(&x, &cfg).expect("ok");
        let mag = spec.magnitude(0);
        let tap = spec.window().coefficients(64);
        let want = tap.get(8).copied().unwrap_or(0.0);
        assert!((mag[0] - want).abs() < 1e-12);
        // A windowed impulse has a flat spectrum scaled by that tap (the FFT
        // of w[n]·δ[n−8] is w[n−8]·e^{−i…}, whose magnitude is w[8] flat).
        assert!(mag.iter().all(|m| (*m - mag[0]).abs() < 1e-9));
        assert!((want - 0.150_881_590_956_963_57).abs() < 1e-12);

        // Centred framing pads by fft/2, so the impulse moves to padded index
        // `len/2 + 32`, and every frame covering it is still flat — at
        // whatever tap the impulse lands on. The signal has to be longer than
        // `fft_size` or the frames near the middle also contain the reflected
        // padding, which puts a second copy of the impulse in the frame.
        let len = 512usize;
        let mut x = vec![0.0; len];
        x[len / 2] = 1.0;
        let cfg = StftConfig::new(64, 32);
        let spec = stft(&x, &cfg).expect("ok");
        let tap = spec.window().coefficients(64);
        let padded_index = len / 2 + 32;
        let mut checked = 0;
        for f in 0..spec.num_frames() {
            let start = f * 32;
            let Some(offset) = (0..64).find(|j| start + j == padded_index) else {
                continue;
            };
            let mag = spec.magnitude(f);
            assert!(
                mag.iter().all(|m| (*m - mag[0]).abs() < 1e-9),
                "frame {f} (tap {offset}) not flat: {mag:?}"
            );
            assert!((mag[0] - tap[offset]).abs() < 1e-12, "frame {f} tap");
            checked += 1;
        }
        assert!(checked > 0, "no frame contained the impulse");
    }

    #[test]
    fn from_parts_validates() {
        let bins = vec![Complex::ZERO; 5];
        assert!(Spectrum::from_parts(vec![], 1, 8, Window::Hann, true).is_ok());
        assert!(
            Spectrum::from_parts(vec![], 1, 8, Window::Hann, true)
                .unwrap()
                .num_frames()
                == 0
        );
        assert!(matches!(
            Spectrum::from_parts(vec![], 0, 8, Window::Hann, true),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            Spectrum::from_parts(vec![], 1, 6, Window::Hann, true),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            Spectrum::from_parts(vec![], 1, 0, Window::Hann, true),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            Spectrum::from_parts(vec![vec![Complex::ZERO; 4]], 1, 8, Window::Hann, true),
            Err(SpectralError::Config(_))
        ));
        assert!(Spectrum::from_parts(vec![bins], 1, 8, Window::Hann, true).is_ok());
    }

    #[test]
    fn accessors_and_retag() {
        let cfg = StftConfig::new(8, 4);
        let spec = stft(&[0.5; 32], &cfg).expect("ok");
        assert_eq!(spec.bin_count(), 5);
        assert_eq!(spec.hop(), 4);
        assert_eq!(spec.fft_size(), 8);
        assert_eq!(spec.window(), Window::Hann);
        assert!(spec.is_centered());
        assert_eq!(spec.frames().len(), spec.num_frames());
        let retagged = spec.with_window(Window::Rectangular);
        assert_eq!(retagged.window(), Window::Rectangular);
        assert_eq!(retagged.num_frames(), spec.num_frames());
        assert_ne!(retagged, spec);
        assert!(alloc::format!("{spec:?}").contains("num_frames"));
        let bank = spec.mel_filterbank(0, 0.0, 0.0, 0.0);
        assert!(bank.energies(&[]).is_empty());
    }

    #[test]
    fn resample_changes_hop_and_length() {
        let cfg = StftConfig::new(64, 16);
        let spec = stft(&[0.4; 256], &cfg).expect("ok");
        let faster = resample_frames(&spec, 2.0).expect("ok");
        assert_eq!(faster.hop(), 32);
        assert_eq!(faster.num_frames(), spec.num_frames());
        let slower = resample_frames(&spec, 0.5).expect("ok");
        assert_eq!(slower.hop(), 8);
        assert!(matches!(
            resample_frames(&spec, f64::NAN),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            resample_frames(&spec, 0.0),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            resample_frames(&spec, -1.0),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            resample_frames(&spec, 1e-9),
            Err(SpectralError::Config(_))
        ));
    }

    #[test]
    fn degenerate_istft_paths() {
        let empty = Spectrum::from_parts(Vec::new(), 4, 8, Window::Hann, true).expect("ok");
        assert_eq!(istft(&empty, 32), vec![0.0; 32]);
        // Asking for more samples than the frames cover zero-fills the tail.
        let cfg = StftConfig::new(64, 16);
        let spec = stft(&[0.3; 128], &cfg).expect("ok");
        let long = istft(&spec, 10_000);
        assert_eq!(long.len(), 10_000);
        assert!(long[500..].iter().all(|v| *v == 0.0));
        assert!(long[..100].iter().any(|v| *v != 0.0));
    }

    #[test]
    fn sine_lands_on_its_bin() {
        let n = 1024;
        let bin = 8usize;
        let cfg = StftConfig::new(n, n / 4).with_center(false);
        let x: Vec<f64> = (0..n)
            .map(|i| (TAU * (bin as f64) * (i as f64) / (n as f64)).sin())
            .collect();
        let spec = stft(&x, &cfg).expect("ok");
        let mag = spec.magnitude(0);
        // A windowed sine has amplitude (N/2)·(Σw/N) at its bin: the FFT is
        // unnormalized, so N/2 for a rectangular window, and the window's
        // coherent gain scales that down.
        //
        // A *symmetric* Hann has coherent gain exactly `(N−1)/(2N)`, not 0.5:
        // the cosine sum telescopes to `Σcos(2πn/(N−1)) = 1` (the n = 0 and
        // n = N−1 terms are both 1 and the interior sums to −1), so the
        // window sums to `(N−1)/2`. That 1/(2N) deficit is inaudible (2.4e-4
        // at N = 2048) but is exactly the sort of thing a round-trip
        // reference has to state rather than hand-wave.
        let w = Window::Hann.coefficients(n);
        let coherent: f64 = w.iter().sum::<f64>() / (n as f64);
        let exact = ((n - 1) as f64) / (2.0 * (n as f64));
        assert!(
            (coherent - exact).abs() < 1e-12,
            "Hann coherent gain {coherent}"
        );
        let expect = (n as f64 / 2.0) * coherent;
        // The window's phase response is not perfectly flat across the
        // frame, so the peak sits a few ppm under the ideal.
        assert!(
            (mag[bin] - expect).abs() < 1e-3 * expect,
            "{} vs {expect}",
            mag[bin]
        );
        // Its immediate neighbours carry the window's main lobe, and every
        // bin more than 4 out is below −60 dB of the peak.
        for k in 0..mag.len() {
            if k.abs_diff(bin) > 4 {
                assert!(mag[k] < mag[bin] * 1e-3, "leakage at bin {k}: {}", mag[k]);
            }
        }
        let _ = lcg(1);
    }
}
