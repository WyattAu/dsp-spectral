//! Mel-scaled filterbanks, MFCCs, and the real cepstrum.
//!
//! The mel scale is the HTK formulation: `mel = 2595·log10(1 + hz/700)`.
//! It is linear below ~1 kHz and logarithmic above, which matches how
//! auditory critical bandwidth grows, and it inverts in closed form
//! (`mel_to_hz`) to machine precision — hence the round-trip test at 1e-9
//! rather than "about right".
//!
//! [`MelBank`] builds the usual triangular filterbank over the
//! `fft_size/2 + 1` bins, with **unnormalised** (peak-1.0) triangles: because
//! adjacent triangles are exact complements between their centres, the
//! energies of a full-bank transform sum to the frame's total power. That
//! conservation property is what makes the bank verifiable — and it is the
//! reason [`MelBank::energies`] is not Slaney-normalised.
//!
//! [`mfcc`] is the orthonormal DCT-II of the log mel energies (the standard
//! MFCC front end), with an optional sinusoidal cepstral lifter.
//! [`real_cepstrum`] is the inverse transform of the log magnitude spectrum,
//! whose peak lands on the pitch period of a periodic signal.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::f64::consts::PI;
use dsp_core::fft::Fft;

use crate::complex::Complex;
use crate::error::SpectralError;

/// The HTK mel scale constant: `2595·log10(1 + hz/700)`.
const MEL_CONST: f64 = 2595.0;
/// The HTK mel scale's break frequency in Hz.
const MEL_BREAK_HZ: f64 = 700.0;

/// Energy floor inside the MFCC log (`mel_energies` are linear power), 1e-10
/// = −100 dB. Well below speech levels, high enough to keep `ln` finite.
pub const MEL_ENERGY_FLOOR: f64 = 1e-10;

/// Convert Hz to the mel scale (HTK): `2595·log10(1 + hz/700)`.
///
/// `hz_to_mel(mel_to_hz(m)) == m` to ~1e-12 over the audio band, and
/// `hz_to_mel(1000) == 999.9855371396244` (the reference value used by the HTK
/// implementation and by librosa).
///
/// The logarithm's argument vanishes at `hz = −700`, so that exact input
/// returns `−∞`; every frequency above it is finite. Negative frequencies are
/// outside the domain of a mel scale and exist here only for symmetry with
/// `mel_to_hz`.
#[must_use]
pub fn hz_to_mel(hz: f64) -> f64 {
    MEL_CONST * libm::log10(1.0 + hz / MEL_BREAK_HZ)
}

/// Convert the mel scale back to Hz: `700·(10^(mel/2595) − 1)`.
#[must_use]
pub fn mel_to_hz(mel: f64) -> f64 {
    MEL_BREAK_HZ * (libm::pow(10.0, mel / MEL_CONST) - 1.0)
}

/// `n_mels` mel-spaced centre frequencies from `f_min` to `f_max` inclusive.
///
/// Evenly spaced **on the mel scale**, so the spacing widens with frequency
/// (≈ linear Hz below the 700 Hz break, ≈ exponential above). `n_mels == 0`
/// yields an empty vector; the endpoints are exactly `f_min` and `f_max`.
#[must_use]
pub fn mel_frequencies(n_mels: usize, f_min: f64, f_max: f64) -> Vec<f64> {
    if n_mels == 0 {
        return Vec::new();
    }
    let lo = hz_to_mel(f_min);
    let hi = hz_to_mel(f_max);
    if n_mels == 1 {
        return alloc::vec![mel_to_hz((lo + hi) / 2.0)];
    }
    let span = hi - lo;
    let denom = (n_mels - 1) as f64;
    (0..n_mels)
        .map(|i| mel_to_hz(lo + span * (i as f64) / denom))
        .collect()
}

/// A triangular mel filterbank sized for one transform.
///
/// Unnormalised: filter `m` is the triangle rising linearly from
/// `f[m-1]` to `f[m]` and falling back to zero at `f[m+1]`, evaluated at the
/// bin frequencies `k·fs/N` (so the weights are continuous in Hz rather than
/// snapped to integer bins). Adjacent triangles are exact complements, hence
/// for a bin strictly between two centres `Σ_m filter[m][k] == 1` and the
/// energies of a full-bank transform sum to the frame's total power.
#[derive(Debug, Clone, PartialEq)]
pub struct MelBank {
    filters: Vec<Vec<f64>>,
    n_mels: usize,
    sample_rate: f64,
    fft_size: usize,
}

impl MelBank {
    /// Build a validated filterbank.
    ///
    /// # Errors
    ///
    /// Returns [`SpectralError::Config`] when `n_mels == 0`, `fft_size < 2`
    /// or is not a power of two, `sample_rate` is not finite and positive, or
    /// `f_min >= f_max`.
    pub fn new(
        n_mels: usize,
        f_min: f64,
        f_max: f64,
        sample_rate: f64,
        fft_size: usize,
    ) -> Result<Self, SpectralError> {
        if n_mels == 0 {
            return Err(SpectralError::Config(String::from("n_mels must be > 0")));
        }
        if fft_size < 2 || !fft_size.is_power_of_two() {
            return Err(SpectralError::Config(format!(
                "fft_size {fft_size} must be a power of two >= 2"
            )));
        }
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(SpectralError::Config(String::from(
                "sample_rate must be finite and > 0",
            )));
        }
        if !f_min.is_finite() || !f_max.is_finite() || f_min >= f_max {
            return Err(SpectralError::Config(format!(
                "require finite 0 <= f_min ({f_min}) < f_max ({f_max})"
            )));
        }
        if f_max > sample_rate / 2.0 {
            return Err(SpectralError::Config(format!(
                "f_max {f_max} exceeds the Nyquist frequency {}",
                sample_rate / 2.0
            )));
        }
        Ok(Self::build(n_mels, f_min, f_max, sample_rate, fft_size))
    }

    /// Infallible construction for call sites that have already validated
    /// (or deliberately degenerate) their arguments — see
    /// [`Spectrum::mel_filterbank`](crate::Spectrum::mel_filterbank).
    pub(crate) fn build(
        n_mels: usize,
        f_min: f64,
        f_max: f64,
        sample_rate: f64,
        fft_size: usize,
    ) -> Self {
        if n_mels == 0
            || fft_size < 2
            || !fft_size.is_power_of_two()
            || !sample_rate.is_finite()
            || sample_rate <= 0.0
            || !f_min.is_finite()
            || !f_max.is_finite()
            || f_min >= f_max
        {
            return Self {
                filters: Vec::new(),
                n_mels: 0,
                sample_rate: 0.0,
                fft_size: 0,
            };
        }
        let bins = fft_size / 2 + 1;
        let centres = mel_frequencies(n_mels, f_min, f_max);
        let first = centres.first().copied().unwrap_or(f_min);
        let last = centres.last().copied().unwrap_or(f_max);
        // Outer edges: one centre spacing outboard of the first/last centre.
        //
        // Deliberately NOT clamped to `[0, Nyquist]`: when `f_min == 0` the
        // first centre *is* DC, and clamping its rising edge up to 0 would
        // collapse that triangle to nothing and throw DC energy away.
        // Letting the outer edges run off the ends of the represented band
        // costs nothing (no bin lives out there) and makes filter 0 peak at
        // exactly bin 0, so the bank tiles [0, Nyquist] and `energies` sums
        // to the frame's total power.
        let first_gap = centres.get(1).copied().unwrap_or(f_max).max(first) - first;
        let last_gap = last
            - centres
                .get(n_mels.saturating_sub(2))
                .copied()
                .unwrap_or(f_min)
                .min(last);
        let mut edges = Vec::with_capacity(n_mels + 2);
        edges.push(first - first_gap);
        edges.extend(centres.iter().copied());
        edges.push(last + last_gap);

        let bin_hz = sample_rate / (fft_size as f64);
        let filters = centres
            .iter()
            .enumerate()
            .map(|(m, &centre)| {
                let left = edges.get(m).copied().unwrap_or(centre);
                let right = edges.get(m + 2).copied().unwrap_or(centre);
                (0..bins)
                    .map(|k| {
                        let f = (k as f64) * bin_hz;
                        triangle_weight(f, left, centre, right)
                    })
                    .collect()
            })
            .collect();
        Self {
            filters,
            n_mels,
            sample_rate,
            fft_size,
        }
    }

    /// Number of filters.
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Bins per frame this bank was sized for (`fft_size / 2 + 1`).
    #[must_use]
    pub fn bin_count(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Sample rate the frequencies were built for.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Transform size the bank was sized for.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The filter weight rows (`n_mels × bin_count`).
    #[must_use]
    pub fn filters(&self) -> &[Vec<f64>] {
        &self.filters
    }

    /// Filter `m`'s weights, or `None` when `m >= n_mels`.
    #[must_use]
    pub fn filter(&self, m: usize) -> Option<&[f64]> {
        self.filters.get(m).map(Vec::as_slice)
    }

    /// Mel energies of one frame: `E[m] = Σ_k |X[k]|²·filter[m][k]`, linear
    /// power (no logarithm, no decibels).
    ///
    /// A frame shorter than the bank is summed over the bins it does carry;
    /// bins the bank does not cover are ignored. Never panics.
    #[must_use]
    pub fn energies(&self, frame: &[Complex]) -> Vec<f64> {
        self.filters
            .iter()
            .map(|filter| {
                filter
                    .iter()
                    .zip(frame.iter())
                    .map(|(&w, c)| w * c.power())
                    .sum()
            })
            .collect()
    }

    /// Mel energies in decibels (`10·log10`), floored at `floor_db` so a
    /// silent frame yields the floor rather than `−∞`.
    #[must_use]
    pub fn energies_db(&self, frame: &[Complex], floor_db: f64) -> Vec<f64> {
        self.energies(frame)
            .into_iter()
            .map(|e| {
                if e > MEL_ENERGY_FLOOR {
                    10.0 * libm::log10(e)
                } else {
                    floor_db
                }
            })
            .collect()
    }

    /// Mean of [`energies`](Self::energies) over many frames — the mel
    /// spectrum of a whole signal.
    #[must_use]
    pub fn mean_energies(&self, frames: &[&[Complex]]) -> Vec<f64> {
        let mut total = alloc::vec![0.0f64; self.filters.len()];
        let mut count = 0usize;
        for frame in frames {
            for (slot, e) in total.iter_mut().zip(self.energies(frame)) {
                *slot += e;
            }
            count += 1;
        }
        if count > 0 {
            let n = count as f64;
            for slot in total.iter_mut() {
                *slot /= n;
            }
        }
        total
    }
}

/// Triangular magnitude weight of filter `(left, centre, right)` at `f`.
///
/// Rises linearly from 0 at `left` to 1 at `centre`, then falls back to 0 at
/// `right`. A degenerate half-width on either side collapses that side to a
/// step rather than producing a division by zero.
fn triangle_weight(f: f64, left: f64, centre: f64, right: f64) -> f64 {
    if f <= left || f >= right {
        return 0.0;
    }
    let up = centre - left;
    let down = right - centre;
    if f <= centre {
        if up > 0.0 {
            (f - left) / up
        } else {
            1.0
        }
    } else if down > 0.0 {
        (right - f) / down
    } else {
        1.0
    }
}

/// MFCCs: the orthonormal DCT-II of `log(mel_energies)`, first `n_coeffs`
/// terms, no lifter.
///
/// With `K = mel_energies.len()` and `L_k = ln max(E_k, 1e-10)`:
///
/// ```text
/// c[m] = s(m)·√(1/K)·Σ_k L_k·cos(π·m·(k + 1/2)/K),  s(0) = 1, s(m>0) = √2
/// ```
///
/// `n_coeffs` is clamped to `K`; an empty input yields an empty output.
/// Identical energies give `c[0] = Σ L_k/√K` and `c[m>0] = 0`.
#[must_use]
pub fn mfcc(mel_energies: &[f64], n_coeffs: usize) -> Vec<f64> {
    mfcc_with_lifter(mel_energies, n_coeffs, 0)
}

/// [`mfcc`] with an HTK-style sinusoidal cepstral lifter of order `lifter`:
/// coefficient `m` is multiplied by `1 + (L/2)·sin(π·m/L)` (which leaves
/// `m = 0` untouched and suppresses the higher quefrencies). `lifter == 0`
/// disables it, making this identical to [`mfcc`].
#[must_use]
pub fn mfcc_with_lifter(mel_energies: &[f64], n_coeffs: usize, lifter: usize) -> Vec<f64> {
    let k = mel_energies.len();
    let want = core::cmp::min(n_coeffs, k);
    if want == 0 {
        return Vec::new();
    }
    let logs: Vec<f64> = mel_energies
        .iter()
        .map(|&e| {
            let floored = if e > MEL_ENERGY_FLOOR {
                e
            } else {
                MEL_ENERGY_FLOOR
            };
            libm::log(floored)
        })
        .collect();
    let norm = libm::sqrt(1.0 / (k as f64));
    (0..want)
        .map(|m| {
            let sum: f64 = logs
                .iter()
                .enumerate()
                .map(|(i, &l)| {
                    let theta = PI * ((m * (2 * i + 1)) as f64) / ((2 * k) as f64);
                    l * libm::cos(theta)
                })
                .sum();
            let scale = if m == 0 {
                norm
            } else {
                norm * core::f64::consts::SQRT_2
            };
            let mut c = scale * sum;
            if lifter > 0 {
                let q = core::cmp::min(m, lifter);
                let weight =
                    1.0 + 0.5 * (lifter as f64) * libm::sin(PI * (q as f64) / (lifter as f64));
                c *= weight;
            }
            c
        })
        .collect()
}

/// Real cepstrum of a signal: the inverse transform of its log magnitude
/// spectrum, i.e. `c[q] = (1/N)·Σ_k ln|X[k]|·e^{+2πikq/N}` with `X` the
/// `N`-point FFT of `samples` and the log magnitude mirrored hermitianly
/// across DC. The result is real and of length `N`; a periodic signal puts
/// its dominant peak at the pitch period in samples.
///
/// `samples.len()` must be a power of two (zero-padding to a larger power of
/// two would rescale the quefrency axis and move the pitch peak, so it is
/// rejected rather than silently applied).
///
/// # Errors
///
/// Returns [`SpectralError::EmptyInput`] for no samples,
/// [`SpectralError::NonFinite`] for NaN/±∞ samples, and
/// [`SpectralError::Config`] when the length is not a power of two `>= 2`.
///
/// # Example
///
/// ```
/// use dsp_spectral::{real_cepstrum, StftConfig};
///
/// // A *single* sine is one spike in the log spectrum, which is a flat
/// // cepstrum — the pitch peak only appears for a signal with a harmonic
/// // structure. An impulse train is the cleanest comb: its period is 16
/// // samples, so the cepstrum peaks at every multiple of 16.
/// let n = 1024;
/// let x: Vec<f64> = (0..n).map(|i| if i % 16 == 0 { 1.0 } else { 0.0 }).collect();
/// let c = real_cepstrum(&x).expect("power-of-two length");
/// assert_eq!(c.len(), n);
/// // Search the first period to pick the fundamental rather than one of its
/// // (equally tall) multiples.
/// let peak = (1..=16)
///     .max_by(|a, b| c[*a].partial_cmp(&c[*b]).expect("finite"))
///     .expect("non-empty");
/// assert_eq!(peak, 16);
/// let _ = StftConfig::hann_fft2048();
/// ```
pub fn real_cepstrum(samples: &[f64]) -> Result<Vec<f64>, SpectralError> {
    if samples.is_empty() {
        return Err(SpectralError::EmptyInput);
    }
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(SpectralError::NonFinite);
    }
    let n = samples.len();
    if n < 2 || !n.is_power_of_two() {
        return Err(SpectralError::Config(format!(
            "real_cepstrum needs a power-of-two length >= 2, got {n}"
        )));
    }
    let fft = Fft::new(n).map_err(|e| SpectralError::Dsp(format!("fft: {e}")))?;
    let mut buf: Vec<f64> = samples.iter().flat_map(|&v| [v, 0.0]).collect();
    fft.forward(&mut buf)
        .map_err(|e| SpectralError::Dsp(format!("fft: {e}")))?;

    let half = n / 2;
    let logs: Vec<f64> = buf
        .chunks_exact(2)
        .take(half + 1)
        .map(|c| {
            let re = c.first().copied().unwrap_or(0.0);
            let im = c.get(1).copied().unwrap_or(0.0);
            let mag = libm::sqrt(re * re + im * im);
            if mag > MEL_ENERGY_FLOOR {
                libm::log(mag)
            } else {
                libm::log(MEL_ENERGY_FLOOR)
            }
        })
        .collect();

    // Hermitian log-magnitude spectrum: C[k] = C[N-k], so the inverse
    // transform comes out real. Build it symmetrically about DC.
    let mut out: Vec<f64> = alloc::vec![0.0; n];
    for k in 0..=half {
        let v = logs.get(k).copied().unwrap_or(0.0);
        if let Some(slot) = out.get_mut(k) {
            *slot = v;
        }
        if k > 0 {
            if let Some(slot) = out.get_mut(n - k) {
                *slot = v;
            }
        }
    }
    let mut spectrum: Vec<f64> = out.iter().flat_map(|&v| [v, 0.0]).collect();
    fft.inverse(&mut spectrum)
        .map_err(|e| SpectralError::Dsp(format!("fft: {e}")))?;
    Ok(spectrum
        .chunks_exact(2)
        .map(|c| c.first().copied().unwrap_or(0.0))
        .collect())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::{
        hz_to_mel, mel_frequencies, mel_to_hz, mfcc, mfcc_with_lifter, real_cepstrum, MelBank,
        MEL_ENERGY_FLOOR,
    };
    use crate::complex::Complex;
    use crate::error::SpectralError;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn mel_scale_reference_values() {
        // HTK reference: 1000 Hz → 999.9855 mel.
        assert!((hz_to_mel(1_000.0) - 999.985_537_6).abs() < 1e-6);
        assert!((hz_to_mel(0.0)).abs() < 1e-15);
        assert!((mel_to_hz(0.0)).abs() < 1e-12);
        assert!((mel_to_hz(999.985_537_6) - 1_000.0).abs() < 1e-5);
        // Round trip across the audio band.
        let mut worst = 0.0f64;
        for i in 0..=2_000 {
            let hz = (i as f64) * 20_000.0 / 2_000.0;
            let back = mel_to_hz(hz_to_mel(hz));
            worst = worst.max((back - hz).abs());
        }
        assert!(worst < 1e-9, "mel round-trip error {worst}");
    }

    #[test]
    fn mel_frequencies_are_monotone_and_anchored() {
        let f = mel_frequencies(40, 0.0, 8_000.0);
        assert_eq!(f.len(), 40);
        assert!(f[0].abs() < 1e-9);
        assert!((f[39] - 8_000.0).abs() < 1e-9);
        for w in f.windows(2) {
            assert!(w[1] > w[0], "not increasing: {w:?}");
        }
        assert!(mel_frequencies(0, 0.0, 100.0).is_empty());
        // A single filter lands on the mel-scale midpoint, which in Hz is
        // *above* the arithmetic midpoint because the mel scale is convex.
        let one = mel_frequencies(1, 100.0, 1_000.0);
        assert_eq!(one.len(), 1);
        assert!((one[0] - 466.19).abs() < 0.1, "single centre {}", one[0]);
        assert!((mel_frequencies(1, 0.0, 3_000.0)[0] - 909.35).abs() < 0.1);
        // Degenerate spans still return the right count.
        assert_eq!(mel_frequencies(4, 500.0, 500.0).len(), 4);
    }

    #[test]
    fn mel_bank_validation() {
        assert!(MelBank::new(0, 0.0, 1_000.0, 8_000.0, 512).is_err());
        assert!(MelBank::new(4, 0.0, 1_000.0, 8_000.0, 6).is_err());
        assert!(MelBank::new(4, 0.0, 1_000.0, 8_000.0, 0).is_err());
        assert!(MelBank::new(4, 0.0, 1_000.0, 0.0, 512).is_err());
        assert!(MelBank::new(4, 0.0, 1_000.0, f64::NAN, 512).is_err());
        assert!(MelBank::new(4, 1_000.0, 1_000.0, 8_000.0, 512).is_err());
        assert!(MelBank::new(4, 2_000.0, 1_000.0, 8_000.0, 512).is_err());
        assert!(MelBank::new(4, 0.0, f64::NAN, 8_000.0, 512).is_err());
        // Above Nyquist is rejected.
        assert!(MelBank::new(4, 0.0, 5_000.0, 8_000.0, 512).is_err());
        let bank = MelBank::new(8, 0.0, 3_000.0, 8_000.0, 512).expect("valid");
        assert_eq!(bank.n_mels(), 8);
        assert_eq!(bank.bin_count(), 257);
        assert!((bank.sample_rate() - 8_000.0).abs() < 1e-9);
        assert_eq!(bank.fft_size(), 512);
        assert_eq!(bank.filters().len(), 8);
        assert_eq!(bank.filter(0).map(<[f64]>::len), Some(257));
        assert!(bank.filter(8).is_none());
        // Degenerate build path (used by Spectrum::mel_filterbank).
        let empty = MelBank::build(0, 0.0, 0.0, 0.0, 0);
        assert_eq!(empty.n_mels(), 0);
        assert!(empty.energies(&[]).is_empty());
        assert!(empty.energies_db(&[], -100.0).is_empty());
        assert_eq!(empty.mean_energies(&[]), Vec::<f64>::new());
        let degenerate = MelBank::build(4, 100.0, 0.0, 8_000.0, 512);
        assert_eq!(degenerate.filters().len(), 0);
    }

    #[test]
    fn mel_energies_are_power_weighted() {
        // Span the whole band (0 → Nyquist) so the bank tiles it: the row
        // sums below show every bin covered by exactly one unit of weight.
        let bank = MelBank::new(4, 0.0, 4_000.0, 8_000.0, 8).expect("valid");
        // bins = 5, fft = 8, fs = 8000 → 1000 Hz per bin.
        let frame = vec![
            Complex::new(1.0, 0.0), // |X|² = 1 at DC
            Complex::new(0.0, 2.0), // 4 at 1 kHz
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ];
        let e = bank.energies(&frame);
        assert_eq!(e.len(), 4);
        // The bank tiles [0, Nyquist], so the energies sum to the frame's
        // total power exactly.
        assert!(
            (e.iter().sum::<f64>() - 5.0).abs() < 1e-9,
            "energy not conserved: {e:?}"
        );
        // Each filter's own contribution is that bin's power times its
        // weight at that bin.
        let weight = |m: usize, k: usize| bank.filter(m).expect("filter")[k];
        for m in 0..4 {
            let want: f64 = (0..5).map(|k| weight(m, k) * frame[k].power()).sum();
            assert!((e[m] - want).abs() < 1e-12, "mel {m}: {e:?}");
        }
        // Every row is a sampled triangle: non-negative, ≤ 1, and — the
        // property that matters — the rows *tile* the band, summing to
        // exactly 1 at every bin. (A row's best tap is only 1.0 when its
        // centre lands on a bin frequency; between bins the sampled peak
        // falls short, which is why the tiling identity, not the peak, is
        // what is asserted.)
        for m in 0..4 {
            let row = bank.filter(m).expect("filter");
            assert!(
                row.iter().all(|w| (0.0..=1.0).contains(w)),
                "mel {m} out of range: {row:?}"
            );
            assert!(row.iter().any(|w| *w > 0.0), "mel {m} is empty");
        }
        for k in 0..5 {
            let rowsum: f64 = (0..4).map(|m| weight(m, k)).sum();
            assert!((rowsum - 1.0).abs() < 1e-12, "bin {k} rowsum {rowsum}");
        }
        // Filter 0 peaks at DC (weight exactly 1) and falls from there.
        assert!((weight(0, 0) - 1.0).abs() < 1e-12);
        assert!(weight(0, 1) < 0.5);

        // A silent mel band reads exactly the floor, not −∞.
        let db = bank.energies_db(&frame, -120.0);
        assert_eq!(db.len(), 4);
        assert!(db.iter().all(|v| *v >= -120.0 && v.is_finite()));
        assert_eq!(db[3], -120.0);
        assert!((db[1] - 10.0 * libm::log10(e[1])).abs() < 1e-12);
        let mean = bank.mean_energies(&[&frame]);
        assert!((mean[0] - e[0]).abs() < 1e-15);
        assert!(bank.mean_energies(&[]).iter().all(|v| *v == 0.0));
        // A short frame sums only over the bins it carries, so dropping a
        // bin with real power strictly reduces the total.
        let mut partial = frame.clone();
        partial[0] = Complex::ZERO;
        let short = bank.energies(&partial);
        assert!(short.iter().sum::<f64>() < e.iter().sum::<f64>());
        // An empty frame is zero energy in every band.
        assert!(bank.energies(&[]).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn mfcc_shape_and_reference() {
        let energies = vec![1.0, 1.0, 1.0, 1.0];
        let c = mfcc(&energies, 3);
        assert_eq!(c.len(), 3);
        // Equal log-energies: only c[0] survives, = ln(1)·Σ1/√4 = 0 here.
        assert!(c[1].abs() < 1e-15 && c[2].abs() < 1e-15);
        // Constant non-unit energies: c[0] = K·ln(e)/√K = √K·ln(e).
        let energies = vec![2.0, 2.0, 2.0, 2.0];
        let c = mfcc(&energies, 4);
        assert!((c[0] - 2.0 * core::f64::consts::LN_2).abs() < 1e-12);
        assert!(c[1..].iter().all(|v| v.abs() < 1e-12));
        // n_coeffs is clamped to the input length; empty in → empty out.
        assert_eq!(mfcc(&energies, 99).len(), 4);
        assert_eq!(mfcc(&energies, 0).len(), 0);
        assert!(mfcc(&[], 13).is_empty());
        // Floored energies: a zero bin becomes ln(1e-10), never −∞.
        let floored = mfcc(&[0.0, 0.0], 2);
        assert!(floored.iter().all(|v| v.is_finite()));

        // Naive O(K·M) DCT-II reference, agreeing to 1e-12. A unit impulse at
        // mel bin 0 over K = 4 leaves logs (0, ln ε, ln ε, ln ε), which is
        // what makes every coefficient non-zero — the floored tail dominates.
        let e = [1.0f64, 0.0, 0.0, 0.0];
        let logs: Vec<f64> = e
            .iter()
            .map(|&x| {
                if x > MEL_ENERGY_FLOOR {
                    libm::log(x)
                } else {
                    libm::log(MEL_ENERGY_FLOOR)
                }
            })
            .collect();
        let k = e.len();
        let got = mfcc(&e, 4);
        for m in 0..k {
            let sum: f64 = logs
                .iter()
                .enumerate()
                .map(|(i, &l)| {
                    l * libm::cos(
                        core::f64::consts::PI * ((m * (2 * i + 1)) as f64) / ((2 * k) as f64),
                    )
                })
                .sum();
            let scale = if m == 0 {
                1.0 / libm::sqrt(k as f64)
            } else {
                core::f64::consts::SQRT_2 / libm::sqrt(k as f64)
            };
            let want = scale * sum;
            assert!((got[m] - want).abs() < 1e-12, "c{m}: {got:?} vs {want}");
        }
        // Orthonormality: the DCT-II basis rows are orthonormal, so a
        // single occupied mel bin with the rest floored reconstructs
        // 1/√K on c[0] with the floored tail on the rest.
        assert!((got[0] - (-34.538_776_394_910_684)).abs() < 1e-9);

        // Linearity in the log energies: squaring every energy doubles every
        // log, hence doubles every DCT coefficient.
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let squared: Vec<f64> = base.iter().map(|v| v * v).collect();
        let a = mfcc(&base, 4);
        let b = mfcc(&squared, 4);
        for m in 0..4 {
            assert!((b[m] - 2.0 * a[m]).abs() < 1e-12, "c{m}: {b:?} vs {a:?}");
        }
        // Orthonormality check. For the orthonormal DCT-II, a log-energy
        // vector proportional to the m-th basis row cos(π·m·(k + ½)/K) puts
        // all of its coefficient energy in slot m — *except* for m = 0, whose
        // row is the constant 1 and lands entirely in c[0] too. So for every
        // m, slot m must dominate the rest.
        let k = 8usize;
        for m in 0..k {
            // Pick log E_k = C·cos(...) with C large enough to clear the
            // energy floor everywhere, then exponentiate.
            let c_amp = 4.0f64;
            let row: Vec<f64> = (0..k)
                .map(|i| {
                    c_amp
                        * libm::cos(
                            core::f64::consts::PI * ((m * (2 * i + 1)) as f64) / ((2 * k) as f64),
                        )
                })
                .map(libm::exp)
                .collect();
            let c = mfcc(&row, k);
            assert_eq!(c.len(), k);
            let total: f64 = c.iter().map(|v| v * v).sum();
            let in_m = c[m] * c[m];
            assert!(
                in_m > 0.99 * total,
                "m={m}: c={c:?}, slot m holds {in_m} of {total}"
            );
        }
    }

    #[test]
    fn mfcc_lifter_scales_higher_quefrencies() {
        let energies: Vec<f64> = (0..8).map(|i| (i as f64 + 1.0) * 0.01).collect();
        let plain = mfcc(&energies, 5);
        let lifted = mfcc_with_lifter(&energies, 5, 20);
        assert_eq!(plain.len(), lifted.len());
        assert!((plain[0] - lifted[0]).abs() < 1e-12);
        assert!(lifted[4].abs() > plain[4].abs());
        assert_eq!(mfcc_with_lifter(&energies, 5, 0), plain);
        // L = 1 clamps the quefrency index to L, so every coefficient but
        // c[0] takes the same small weight.
        let one = mfcc_with_lifter(&energies, 4, 1);
        let w = 1.0 + 0.5 * libm::sin(core::f64::consts::PI);
        assert!((one[1] - plain[1] * w).abs() < 1e-12);
    }

    #[test]
    fn real_cepstrum_peaks_at_the_pitch_lag() {
        // A *pure tone* is a single spike in the log spectrum, so its cepstrum
        // is flat (a DC spike in the cepstrum) — the pitch peak only appears
        // for a signal with harmonic structure. A periodic impulse train has
        // exactly that: a comb of harmonics, whose cepstrum peaks at the
        // period.
        let n = 1_024usize;
        let period = 16usize;
        let x: Vec<f64> = (0..n)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect();
        let c = real_cepstrum(&x).expect("valid");
        assert_eq!(c.len(), n);
        // A comb of harmonics has its cepstral energy at *every* multiple of
        // the period (each harmonic pair contributes its own peak), so the
        // check is that the global maximum lands on a multiple of the period
        // — and that multiples dominate everything in between.
        let peak = (1..n)
            .max_by(|a, b| {
                c[*a]
                    .partial_cmp(&c[*b])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        assert_eq!(peak % period, 0, "peak {peak} is not a pitch multiple");
        for q in [1usize, 2, 3, 4] {
            let on = c.get(q * period).copied().unwrap_or(0.0);
            let off = c.get(q * period + 1).copied().unwrap_or(0.0);
            assert!(
                on > 10.0 * off.abs() + 0.05,
                "q*period={} : {on} vs {off}",
                q * period
            );
        }
        assert!(c.iter().all(|v| v.is_finite()));
        // The peak is well clear of the floor elsewhere (excluding the DC
        // region, which carries the overall log gain).
        let peak_val = c[period];
        let median_rest: f64 = {
            let mut others: Vec<f64> = (period + 1..n - period).map(|q| c[q]).collect();
            others.sort_by(f64::total_cmp);
            others.get(others.len() / 2).copied().unwrap_or(0.0)
        };
        assert!(
            peak_val > 10.0 * median_rest.abs() + 0.05,
            "peak {peak_val} vs {median_rest}"
        );
        // Reversal symmetry: the cepstrum is real, so c[q] == c[N-q].
        for q in 1..n / 2 {
            assert!((c[q] - c[n - q]).abs() < 1e-12, "q={q}");
        }
    }

    #[test]
    fn real_cepstrum_validation() {
        assert!(matches!(real_cepstrum(&[]), Err(SpectralError::EmptyInput)));
        assert!(matches!(
            real_cepstrum(&[f64::NAN; 8]),
            Err(SpectralError::NonFinite)
        ));
        assert!(matches!(
            real_cepstrum(&[1.0; 100]),
            Err(SpectralError::Config(_))
        ));
        assert!(matches!(
            real_cepstrum(&[1.0; 1]),
            Err(SpectralError::Config(_))
        ));
        assert!(real_cepstrum(&[1.0; 64]).is_ok());
        // A zero signal floors every log, so the cepstrum is flat at DC only.
        let c = real_cepstrum(&vec![0.0; 64]).expect("valid");
        assert!(c[1..].iter().all(|v| v.abs() < 1e-15));
        // Every bin floored to the same log value, so the inverse transform
        // is a pure DC spike at that value.
        assert!((c[0] - libm::log(MEL_ENERGY_FLOOR)).abs() < 1e-12);
    }
}
