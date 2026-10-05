//! Analysis windows — symmetric cosine-sum shapes for the STFT.
//!
//! [`Window::coefficients`] generates the **symmetric** convention
//! (`θ = 2πn/(N−1)`, the STFT/NDFT convention), so the first and last taps
//! are exact endpoints of the shape. That makes each variant's edge
//! behaviour a documented, testable property:
//!
//! | Variant | `w[0] = w[N−1]` | Peak |
//! |---|---|---|
//! | [`Window::Hann`] | `0` | `1.0` |
//! | [`Window::Hamming`] | `0.08` | `1.0` |
//! | [`Window::Blackman`] | `0` | `1.0` |
//! | [`Window::BlackmanHarris`] | `6.0e-5` | `1.0` |
//! | [`Window::FlatTop`] | `−4.21e-4` | `1.0` |
//! | [`Window::Rectangular`] | `1` | `1` |
//!
//! Hann and Blackman are exactly zero at both ends, which is the whole point
//! of the window (no discontinuity when the analysis frame is stitched back
//! together) and also why [`istft`](crate::istft) normalises by the
//! *measured* per-sample overlap instead of a constant: a zero end-tap
//! contributes nothing to the denominator and must not divide by it.
//!
//! `dsp-core` ships the same cosine-sum shapes with a periodic/symmetric
//! switch; this crate keeps a symmetric-only enum because the STFT is the
//! only consumer, and adds [`Window::FlatTop`] (the amplitude-calibration
//! window) which `dsp-core` does not carry.

use alloc::vec::Vec;
use core::f64::consts::TAU;
use libm::cos;

/// Window shapes available to [`StftConfig`](crate::StftConfig).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Window {
    /// Hann (raised cosine): `0.5 − 0.5·cos θ`. Endpoints `0`, main lobe
    /// ≈ 4 bins — the default analysis window.
    #[default]
    Hann,
    /// Hamming: `0.54 − 0.46·cos θ`. Endpoints `0.08`, ≈ −42 dB sidelobes.
    Hamming,
    /// Exact Blackman: `0.42 − 0.5·cos θ + 0.08·cos 2θ`. Endpoints `0`,
    /// ≈ −58 dB sidelobes.
    Blackman,
    /// 4-term Blackman–Harris: `0.35875 − 0.48829·cos θ + 0.14128·cos 2θ −
    /// 0.01168·cos 3θ`. Endpoints `6.0e-5`, ≈ −92 dB sidelobes.
    BlackmanHarris,
    /// Flat top: `0.21557895 − 0.41663158·cos θ + 0.277263158·cos 2θ −
    /// 0.083578947·cos 3θ + 0.006947368·cos 4θ`. Endpoints `−4.21e-4`,
    /// ≈ −93 dB sidelobes — the flat passband makes it the right window for
    /// amplitude calibration, at the cost of a very wide main lobe.
    FlatTop,
    /// Rectangular (Dirichlet): `w[n] = 1`. No taper: maximal main-lobe
    /// width, ≈ −13 dB sidelobes, and only valid for overlap-add
    /// reconstruction when `hop == fft_size`.
    Rectangular,
}

impl Window {
    /// Cosine-sum coefficients `(a0, a1, …)` with
    /// `w[n] = a0 + Σ a_k·cos(kθ)`, `θ = 2πn/(N−1)`.
    fn coefficients_of(self) -> &'static [f64] {
        match self {
            Window::Rectangular => &[1.0],
            Window::Hann => &[0.5, -0.5],
            Window::Hamming => &[0.54, -0.46],
            Window::Blackman => &[0.42, -0.5, 0.08],
            Window::BlackmanHarris => &[0.35875, -0.48829, 0.14128, -0.01168],
            Window::FlatTop => &[
                0.21557895,
                -0.41663158,
                0.277_263_158,
                -0.083_578_947,
                0.006_947_368,
            ],
        }
    }

    /// Generate `n` taps of this window, symmetric (`θ = 2πn/(N−1)`).
    ///
    /// `n = 0` yields an empty vector and `n = 1` yields `[1.0]` (a one-tap
    /// window has no sweep; this matches `dsp_core::window::Window`).
    #[must_use]
    pub fn coefficients(&self, n: usize) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return alloc::vec![1.0];
        }
        let coeffs = self.coefficients_of();
        let denom = (n - 1) as f64;
        (0..n)
            .map(|i| {
                let theta = TAU * (i as f64) / denom;
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(k, &a)| if k == 0 { a } else { a * cos((k as f64) * theta) })
                    .sum()
            })
            .collect()
    }

    /// The nominal squared-window overlap constant
    /// `Σ_{k : k·hop < N} w[k·hop]²` — the divisor a fixed-COLA overlap-add
    /// implementation would use. Reported (never relied upon) so a host can
    /// check its window/hop pair against
    /// [`StftConfig::is_cola`](crate::StftConfig::is_cola).
    ///
    /// The sum only counts taps that land on multiples of `hop`, so it is
    /// phase-dependent unless the window/hop pair is COLA (a rectangular
    /// window at `hop == N`, or Hann at 75 % overlap, both are). The value is
    /// reported regardless — it is the divisor at tap 0, nothing more.
    ///
    /// # Panics
    ///
    /// None: `hop == 0` yields `0.0`.
    #[must_use]
    pub fn overlap_sum(&self, n: usize, hop: usize) -> f64 {
        if n == 0 || hop == 0 {
            return 0.0;
        }
        let w = self.coefficients(n);
        let mut acc = 0.0;
        let mut idx = 0usize;
        while idx < n {
            if let Some(&tap) = w.get(idx) {
                acc += tap * tap;
            }
            idx += hop;
        }
        acc
    }

    /// Human-readable name, for diagnostics and CLI output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Window::Hann => "hann",
            Window::Hamming => "hamming",
            Window::Blackman => "blackman",
            Window::BlackmanHarris => "blackman-harris",
            Window::FlatTop => "flat-top",
            Window::Rectangular => "rectangular",
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::Window;
    use alloc::vec;

    #[test]
    fn length_and_endpoint_documents() {
        for w in [
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::BlackmanHarris,
            Window::FlatTop,
            Window::Rectangular,
        ] {
            let c = w.coefficients(65);
            assert_eq!(c.len(), 65);
            // Symmetric by construction.
            for i in 0..65 {
                assert!((c[i] - c[64 - i]).abs() < 1e-15, "{} tap {i}", w.name());
            }
            // Documented endpoint values.
            let first = c[0];
            let expected = match w {
                Window::Hann | Window::Blackman => 0.0,
                Window::Hamming => 0.08,
                Window::BlackmanHarris => 0.000_06,
                Window::FlatTop => -0.000_421_051,
                Window::Rectangular => 1.0,
            };
            assert!(
                (first - expected).abs() < 1e-6,
                "{} endpoint {first} vs {expected}",
                w.name()
            );
        }
    }

    #[test]
    fn peak_and_tapered_peak_are_one() {
        for w in [
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::BlackmanHarris,
            Window::FlatTop,
        ] {
            let c = w.coefficients(64);
            // Symmetric cosine-sum windows peak *between* the two middle
            // taps (their sweep is 2π/(N−1), not 2π/N), so an even-length
            // window's best tap is cos²(π/N) short of 1.
            let peak = c.iter().copied().fold(0.0f64, f64::max);
            assert!((peak - 1.0).abs() < 4e-3, "{} peak {peak}", w.name());
        }
    }

    #[test]
    fn degenerate_lengths() {
        assert!(Window::Hann.coefficients(0).is_empty());
        assert_eq!(Window::Hann.coefficients(1), vec![1.0]);
        assert_eq!(Window::Hann, Window::default());
        assert_eq!(Window::Rectangular.overlap_sum(0, 4), 0.0);
        assert_eq!(Window::Rectangular.overlap_sum(8, 0), 0.0);
    }

    #[test]
    fn overlap_sum_values() {
        // Rectangular at hop == n: a single frame per output sample.
        assert!((Window::Rectangular.overlap_sum(8, 8) - 1.0).abs() < 1e-15);
        // Rectangular at 25 % overlap: 4 full frames.
        assert!((Window::Rectangular.overlap_sum(8, 2) - 4.0).abs() < 1e-15);
        // Symmetric Hann at N = 8 has taps [0, .188, .611, .950, .950, .611,
        // .188, 0] (it peaks between the two middle taps). At 50 % overlap
        // that is w[0]² + w[4]² = 0.9034; at 25 % it is
        // w[0]² + w[2]² + w[4]² + w[6]² = 1.3125.
        assert!((Window::Hann.overlap_sum(8, 4) - 0.903_420_659_183_551).abs() < 1e-12);
        assert!((Window::Hann.overlap_sum(8, 2) - 1.3125).abs() < 1e-12);
        // Hamming at 50 % overlap: 0.08² + 0.9239².
        assert!((Window::Hamming.overlap_sum(8, 4) - 0.917_366_554_610_575_8).abs() < 1e-12);
        // Blackman at 50 % overlap: w[0]² + w[4]² = 0 + 0.920364².
        assert!((Window::Blackman.overlap_sum(8, 4) - 0.847_069_189_521_953_9).abs() < 1e-12);
    }

    #[test]
    fn names_are_stable() {
        assert_eq!(Window::BlackmanHarris.name(), "blackman-harris");
        assert_eq!(Window::FlatTop.name(), "flat-top");
        assert_eq!(Window::Rectangular.name(), "rectangular");
        assert_eq!(Window::Hamming.name(), "hamming");
        assert_eq!(Window::Blackman.name(), "blackman");
        assert_eq!(Window::Hann.name(), "hann");
    }
}