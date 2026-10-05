//! Time-varying spectral descriptors.
//!
//! Every function here is a pure, per-frame reduction of a [`Spectrum`] down
//! to one value per frame (plus [`zero_crossing_rate`], which is a time-domain
//! statistic). Definitions are spelled out so a naive reference
//! implementation in a test can reproduce them to 1e-9:
//!
//! | Feature | Definition (m = `\|X[k]\|`, `f_k = k·fs/N`) |
//! |---|---|
//! | [`spectral_centroid`] | `Σ f_k·m_k / Σ m_k` (0 for a silent frame) |
//! | [`spectral_bandwidth`] | `√(Σ m_k·(f_k − centroid)² / Σ m_k)` |
//! | [`spectral_flatness`] | `exp(mean(ln max(m_k, ε))) / mean(m_k)` |
//! | [`spectral_rolloff`] | lowest `f_k` whose cumulative magnitude reaches `threshold·Σ m_k` |
//! | [`spectral_flux`] | `Σ max(0, m_k[t] − m_k[t−1])`, zero for the first frame |
//!
//! The floor `ε = 1e-10` in the flatness geometric mean keeps a frame with
//! any digital-silence bin from producing `−∞`/`0/0`; it is far below any
//! audible level, so a genuinely tonal frame still reads ≈ 0.
//!
//! **Reading flatness:** it is a *contrast* measure, not an absolute
//! loudness one. A uniform magnitude spectrum reads exactly 1. White noise
//! also reads high — ≈ 0.90 at 512 bins — because each bin's magnitude is
//! Rayleigh-distributed and `E[ln R] − ln E[`R`]` is only −0.10; a pure tone
//! reads three orders of magnitude lower. So "flatness ≈ 1" means *flat
//! spectrum*, not *noise present*, and "flatness ≈ 0" means *tonal*.
//!
//! Every function returns a vector with exactly one entry per frame — empty
//! for an empty spectrogram — and never panics on a short or degenerate
//! input.

use alloc::vec::Vec;

use crate::stft::Spectrum;

/// Magnitude floor used inside the geometric mean of
/// [`spectral_flatness`]; 1e-10 is −200 dB.
const FLATNESS_FLOOR: f64 = 1e-10;

/// `fs / fft_size` — Hz per bin. Guarded against a degenerate `fs`.
fn bin_hz(spec: &Spectrum, sample_rate: f64) -> f64 {
    let fft = spec.fft_size() as f64;
    if fft <= 0.0 || !sample_rate.is_finite() {
        0.0
    } else {
        sample_rate / fft
    }
}

/// The magnitude spectra of every frame, as rows.
fn magnitude_rows(spec: &Spectrum) -> Vec<Vec<f64>> {
    (0..spec.num_frames()).map(|f| spec.magnitude(f)).collect()
}

/// Spectral centroid in Hz per frame: the magnitude-weighted mean frequency.
///
/// Returns `0.0` for a silent frame (no magnitude energy) rather than NaN.
///
/// # Example
///
/// ```
/// use dsp_spectral::{stft, spectral_centroid, StftConfig, Window};
///
/// let cfg = StftConfig::new(64, 32).with_window(Window::Rectangular);
/// let spec = stft(&[1.0; 64], &cfg).expect("valid");
/// let centroid = spectral_centroid(&spec, cfg.sample_rate);
/// assert_eq!(centroid.len(), spec.num_frames());
/// // A constant signal under a rectangular window puts all the energy at DC.
/// assert!(centroid.iter().all(|c| c.abs() < 1e-9));
/// ```
#[must_use]
pub fn spectral_centroid(spec: &Spectrum, sample_rate: f64) -> Vec<f64> {
    let hz = bin_hz(spec, sample_rate);
    magnitude_rows(spec)
        .into_iter()
        .map(|mags| {
            let total: f64 = mags.iter().sum();
            if total <= 0.0 {
                return 0.0;
            }
            mags.iter()
                .enumerate()
                .map(|(k, m)| (k as f64) * hz * m)
                .sum::<f64>()
                / total
        })
        .collect()
}

/// Spectral bandwidth in Hz per frame: the magnitude-weighted RMS deviation
/// from the centroid.
///
/// `centroid` is normally the output of [`spectral_centroid`] for the same
/// spectrogram. When it is empty or the wrong length it is ignored and the
/// centroid is recomputed per frame, so the function is total.
#[must_use]
pub fn spectral_bandwidth(spec: &Spectrum, sample_rate: f64, centroid: &[f64]) -> Vec<f64> {
    let hz = bin_hz(spec, sample_rate);
    let use_supplied = centroid.len() == spec.num_frames();
    magnitude_rows(spec)
        .into_iter()
        .enumerate()
        .map(|(t, mags)| {
            let total: f64 = mags.iter().sum();
            if total <= 0.0 {
                return 0.0;
            }
            let c = if use_supplied {
                centroid.get(t).copied().unwrap_or(0.0)
            } else {
                mags.iter()
                    .enumerate()
                    .map(|(k, m)| (k as f64) * hz * m)
                    .sum::<f64>()
                    / total
            };
            let variance = mags
                .iter()
                .enumerate()
                .map(|(k, m)| m * ((k as f64) * hz - c) * ((k as f64) * hz - c))
                .sum::<f64>()
                / total;
            if variance <= 0.0 {
                0.0
            } else {
                libm::sqrt(variance)
            }
        })
        .collect()
}

/// Spectral flatness (Wiener entropy / tonality) per frame: the geometric
/// mean of the magnitude spectrum divided by its arithmetic mean.
///
/// `1.0` means every bin carries identical magnitude (white, untuned); near
/// `0` means the energy sits in a few bins (a tone). Always in `[0, 1]` for
/// finite spectra.
#[must_use]
pub fn spectral_flatness(spec: &Spectrum) -> Vec<f64> {
    magnitude_rows(spec)
        .into_iter()
        .map(|mags| {
            let n = mags.len();
            if n == 0 {
                return 0.0;
            }
            let arithmetic: f64 = mags.iter().sum::<f64>() / (n as f64);
            if arithmetic <= 0.0 {
                return 0.0;
            }
            let log_sum: f64 = mags
                .iter()
                .map(|m| {
                    libm::log(if *m > FLATNESS_FLOOR {
                        *m
                    } else {
                        FLATNESS_FLOOR
                    })
                })
                .sum();
            let geometric = libm::exp(log_sum / (n as f64));
            if geometric > arithmetic {
                1.0
            } else {
                geometric / arithmetic
            }
        })
        .collect()
}

/// Spectral rolloff in Hz per frame: the frequency below which `threshold`
/// (a fraction of the frame's total magnitude, `0 < threshold <= 1`) has
/// accumulated.
///
/// `threshold <= 0` returns `0.0` and `threshold >= 1` returns the highest
/// populated bin, rather than being rejected — the value is clamped, not
/// validated, because this function is a per-frame reduction with no error
/// channel.
#[must_use]
pub fn spectral_rolloff(spec: &Spectrum, threshold: f64, sample_rate: f64) -> Vec<f64> {
    let hz = bin_hz(spec, sample_rate);
    magnitude_rows(spec)
        .into_iter()
        .map(|mags| {
            let total: f64 = mags.iter().sum();
            if total <= 0.0 || !threshold.is_finite() {
                return 0.0;
            }
            let target = total * threshold.clamp(0.0, 1.0);
            let mut cumulative = 0.0;
            let mut found = None;
            for (k, m) in mags.iter().enumerate() {
                cumulative += m;
                if cumulative >= target {
                    found = Some(k);
                    break;
                }
            }
            match found {
                Some(k) => (k as f64) * hz,
                // Unreachable for `0 <= threshold <= 1` on a positive total,
                // but keep the function total: report the highest populated
                // bin rather than DC.
                None => mags
                    .iter()
                    .rposition(|m| *m > 0.0)
                    .map_or(0.0, |last| (last as f64) * hz),
            }
        })
        .collect()
}

/// Half-wave-rectified spectral flux per frame:
/// `Σ_k max(0, |X_k[t]| − |X_k[t−1]|)` — the total positive change in the
/// magnitude spectrum between consecutive frames.
///
/// The first frame is `0.0` by definition (there is no previous frame), so a
/// stationary signal reads ≈ 0 throughout and an abrupt onset produces a
/// single large value at the onset frame.
#[must_use]
pub fn spectral_flux(spec: &Spectrum) -> Vec<f64> {
    let rows = magnitude_rows(spec);
    let mut out = alloc::vec![0.0f64; rows.len()];
    for t in 1..rows.len() {
        let (Some(prev), Some(cur)) = (rows.get(t - 1), rows.get(t)) else {
            continue;
        };
        let flux: f64 = prev
            .iter()
            .zip(cur.iter())
            .map(|(a, b)| if b - a > 0.0 { b - a } else { 0.0 })
            .sum();
        if let Some(slot) = out.get_mut(t) {
            *slot = flux;
        }
    }
    out
}

/// Zero-crossing rate of a time-domain signal: the fraction of adjacent
/// sample pairs whose signs strictly differ.
///
/// Returns `0.0` for an empty or single-sample input. Both conventions are
/// reported in the literature; this one counts a crossing whenever
/// `x[n-1]·x[n] < 0`, so a run of exact zeros is traversed rather than
/// counted as alternating.
#[must_use]
pub fn zero_crossing_rate(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0usize;
    for pair in samples.windows(2) {
        let (a, b) = match pair {
            [a, b, ..] => (*a, *b),
            _ => (0.0, 0.0),
        };
        if a * b < 0.0 {
            crossings += 1;
        }
    }
    crossings as f64 / ((samples.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::{
        spectral_bandwidth, spectral_centroid, spectral_flatness, spectral_flux, spectral_rolloff,
        zero_crossing_rate,
    };
    use crate::complex::Complex;
    use crate::stft::{stft, Spectrum};
    use crate::window::Window;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A spectrogram of `frames` frames whose frame `t` has all its magnitude in
    /// bin `t % bins`, built on an `fft_size`-point transform.
    fn ramp(fft_size: usize, frames: usize) -> Spectrum {
        let row_bins = fft_size / 2 + 1;
        let frames: Vec<Vec<Complex>> = (0..frames)
            .map(|t| {
                (0..row_bins)
                    .map(|k| Complex::new(if k == t % row_bins { 2.0 } else { 0.0 }, 0.0))
                    .collect()
            })
            .collect();
        Spectrum::from_parts(frames, 1, fft_size, Window::Hann, true).expect("valid")
    }

    #[test]
    fn silent_frames_are_zero_not_nan() {
        let spec = Spectrum::from_parts(vec![vec![Complex::ZERO; 5]; 2], 1, 8, Window::Hann, true)
            .expect("valid");
        assert_eq!(spectral_centroid(&spec, 8_000.0), vec![0.0, 0.0]);
        assert_eq!(spectral_bandwidth(&spec, 8_000.0, &[]), vec![0.0, 0.0]);
        assert_eq!(spectral_flatness(&spec), vec![0.0, 0.0]);
        assert_eq!(spectral_rolloff(&spec, 0.95, 8_000.0), vec![0.0, 0.0]);
        assert_eq!(spectral_flux(&spec), vec![0.0, 0.0]);
    }

    #[test]
    fn empty_spectrogram_yields_empty_features() {
        let spec = Spectrum::from_parts(Vec::new(), 1, 8, Window::Hann, true).expect("valid");
        assert!(spectral_centroid(&spec, 8_000.0).is_empty());
        assert!(spectral_bandwidth(&spec, 8_000.0, &[]).is_empty());
        assert!(spectral_flatness(&spec).is_empty());
        assert!(spectral_rolloff(&spec, 0.9, 8_000.0).is_empty());
        assert!(spectral_flux(&spec).is_empty());
        assert_eq!(zero_crossing_rate(&[]), 0.0);
        assert_eq!(zero_crossing_rate(&[0.3]), 0.0);
    }

    #[test]
    fn centroid_of_known_spectra() {
        // fft_size 4 → 3 bins → 1000 Hz per bin at 4 kHz.
        let spec = ramp(4, 3);
        let c = spectral_centroid(&spec, 4_000.0);
        assert!((c[0] - 0.0).abs() < 1e-9);
        assert!((c[1] - 1_000.0).abs() < 1e-9);
        assert!((c[2] - 2_000.0).abs() < 1e-9);
        // Non-finite / zero sample rate degenerates to 0 Hz rather than NaN.
        let spec = ramp(4, 1);
        assert_eq!(spectral_centroid(&spec, 0.0), vec![0.0]);
        assert_eq!(spectral_centroid(&spec, f64::NAN), vec![0.0]);
    }

    #[test]
    fn bandwidth_uses_supplied_or_recomputed_centroid() {
        let spec = ramp(8, 2);
        let c = spectral_centroid(&spec, 4_000.0);
        let supplied = spectral_bandwidth(&spec, 4_000.0, &c);
        let recomputed = spectral_bandwidth(&spec, 4_000.0, &[]);
        let wrong_len = spectral_bandwidth(&spec, 4_000.0, &[0.0]);
        assert!((supplied[1] - recomputed[1]).abs() < 1e-9);
        assert!((supplied[1] - wrong_len[1]).abs() < 1e-9);
        assert!(supplied.iter().all(|b| b.abs() < 1e-9));
        // A two-bin spread: bins 0 and 2 of a 5-bin (8-point) spectrum, both
        // at magnitude 1 → centroid at bin 1, σ = 1 bin = 500 Hz at 4 kHz.
        let frames = vec![vec![
            Complex::new(1.0, 0.0),
            Complex::ZERO,
            Complex::new(1.0, 0.0),
            Complex::ZERO,
            Complex::ZERO,
        ]];
        let spec = Spectrum::from_parts(frames, 1, 8, Window::Hann, true).expect("valid");
        let bw = spectral_bandwidth(&spec, 4_000.0, &[]);
        assert!((bw[0] - 500.0).abs() < 1e-9, "bandwidth {}", bw[0]);
    }

    #[test]
    fn flatness_bounds_and_limits() {
        // Flat spectrum → 1.
        let frames = vec![vec![Complex::new(1.0, 0.0); 5]];
        let spec = Spectrum::from_parts(frames, 1, 8, Window::Hann, true).expect("valid");
        assert!((spectral_flatness(&spec)[0] - 1.0).abs() < 1e-12);
        // Single occupied bin out of five → essentially 0.
        let frames = vec![vec![
            Complex::new(1.0, 0.0),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ]];
        let spec = Spectrum::from_parts(frames, 1, 8, Window::Hann, true).expect("valid");
        assert!(
            spectral_flatness(&spec)[0] < 1e-3,
            "flatness {}",
            spectral_flatness(&spec)[0]
        );
        assert!(spectral_flatness(&spec)[0] >= 0.0);
    }

    #[test]
    fn rolloff_threshold_clamping() {
        let frames = vec![vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 0.0),
        ]];
        let spec = Spectrum::from_parts(frames, 1, 8, Window::Hann, true).expect("valid");
        let hz = 4_000.0 / 8.0;
        // 50 % of total magnitude (2.5) is not reached until bin 3.
        assert!((spectral_rolloff(&spec, 0.5, 4_000.0)[0] - 3.0 * hz).abs() < 1e-9);
        // threshold <= 0 → lowest populated bin.
        assert!(spectral_rolloff(&spec, 0.0, 4_000.0)[0].abs() < 1e-9);
        assert!(spectral_rolloff(&spec, -5.0, 4_000.0)[0].abs() < 1e-9);
        // threshold >= 1 → highest populated bin.
        assert!((spectral_rolloff(&spec, 2.0, 4_000.0)[0] - 3.0 * hz).abs() < 1e-9);
        assert!((spectral_rolloff(&spec, 1.0, 4_000.0)[0] - 3.0 * hz).abs() < 1e-9);
        // Non-finite threshold → 0.
        assert_eq!(spectral_rolloff(&spec, f64::NAN, 4_000.0), vec![0.0]);
    }

    #[test]
    fn flux_is_half_wave_rectified() {
        let frames = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
            vec![Complex::new(3.0, 0.0), Complex::new(1.0, 0.0)],
        ];
        let spec = Spectrum::from_parts(frames, 1, 2, Window::Hann, true).expect("valid");
        let f = spectral_flux(&spec);
        assert_eq!(f[0], 0.0);
        // (3−1) rises, (1−2) falls: only the rise counts.
        assert!((f[1] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn zcr_counts_strict_sign_flips() {
        assert_eq!(zero_crossing_rate(&[1.0, 1.0, 1.0]), 0.0);
        assert_eq!(zero_crossing_rate(&[1.0, -1.0, 1.0, -1.0]), 1.0);
        assert_eq!(zero_crossing_rate(&[1.0, 0.0, -1.0]), 0.0);
        assert_eq!(zero_crossing_rate(&[-1.0, 0.0, 1.0]), 0.0);
        // A Nyquist-frequency square wave crosses at every sample.
        let sq: Vec<f64> = (0..8)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!((zero_crossing_rate(&sq) - 1.0).abs() < 1e-15);
        assert_eq!(zero_crossing_rate(&[0.0; 4]), 0.0);
    }

    #[test]
    fn flux_on_stft_of_a_real_signal() {
        use alloc::vec;

        // A pure tone exactly on a bin centre, long enough that the
        // centre-padded edges are far away: every interior frame has the
        // same magnitude spectrum, so the flux is ~0 there.
        let cfg = crate::config::StftConfig::new(256, 64);
        let tone: Vec<f64> = (0..4096)
            .map(|i| (core::f64::consts::TAU * 8.0 * i as f64 / 256.0).sin())
            .collect();
        let spec = stft(&tone, &cfg).expect("ok");
        let flux = spectral_flux(&spec);
        let interior = &flux[4..flux.len() - 4];
        assert!(!interior.is_empty());
        assert!(
            interior.iter().all(|f| *f < 1e-6),
            "stationary flux {interior:?}"
        );

        // An abrupt onset in the middle: flux spikes on the onset frame only.
        let mut onset = tone.clone();
        for v in onset.iter_mut().skip(2000) {
            *v = 0.0;
        }
        onset[0] = 1.0;
        let spec = stft(&onset, &cfg).expect("ok");
        let flux = spectral_flux(&spec);
        assert!(flux[0] == 0.0);
        assert!(
            flux.iter().skip(1).any(|f| *f > 1.0),
            "onset should produce a flux spike: {flux:?}"
        );
        let _ = vec![0.0; 1];
    }
}
