//! The DCT-II, standalone.
//!
//! Extracted from the MFCC path, which always carried the orthonormal
//! DCT-II inline (`mfcc` is exactly `dct2` of the log-mel energies). A
//! standalone transform deserves a standalone function: image
//! placeholders (ThumbHash-adjacent work), perceptual hashes, and any
//! spectral compaction need the same cosine basis without a mel bank in
//! front of it.

extern crate alloc;

use alloc::vec::Vec;
use core::f64::consts::PI;

/// The orthonormal DCT-II of `input`, same length.
///
/// With `N = input.len()`:
///
/// ```text
/// c[m] = s(m)·√(1/N)·Σ_k x[k]·cos(π·m·(2k + 1)/(2N)),  s(0) = 1, s(m>0) = √2
/// ```
///
/// Orthonormal: the basis rows are orthonormal, so the transform preserves
/// L2 energy and its inverse is the transpose. A constant input maps to
/// DC only; a unit impulse spreads energy flat across every coefficient.
/// An empty input yields an empty output.
#[must_use]
pub fn dct2(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    (0..n)
        .map(|m| {
            let sum: f64 = input
                .iter()
                .enumerate()
                .map(|(k, &x)| {
                    let theta = PI * ((m * (2 * k + 1)) as f64) / ((2 * n) as f64);
                    x * libm::cos(theta)
                })
                .sum();
            let scale = libm::sqrt(1.0 / (n as f64));
            if m == 0 {
                scale * sum
            } else {
                scale * core::f64::consts::SQRT_2 * sum
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::{vec, vec::Vec};
    use core::f64::consts::PI;

    use super::dct2;

    /// Tolerance for floating-point agreement with the reference sums.
    const EPS: f64 = 1e-9;

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    fn median(xs: &[f64]) -> f64 {
        let mut sorted = xs.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted.get(sorted.len() / 2).copied().unwrap_or_default()
    }

    #[allow(clippy::expect_used)]
    #[test]
    fn constant_input_maps_to_dc_only() {
        let n = 32;
        let input = vec![3.0; n];
        let out = dct2(&input);
        assert_eq!(out.len(), n);
        // c[0] = mean × √N; every other coefficient cancels exactly in
        // exact arithmetic (residual is float noise).
        let dc = out.first().copied().unwrap_or_default();
        assert!((dc - 3.0 * (n as f64).sqrt()).abs() < EPS);
        for coefficient in out.iter().skip(1) {
            assert!(coefficient.abs() < EPS, "non-DC {coefficient}");
        }
    }

    #[allow(clippy::expect_used)]
    #[test]
    fn unit_impulse_preserves_energy() {
        let n = 16;
        let mut input = vec![0.0; n];
        if let Some(first) = input.first_mut() {
            *first = 1.0;
        }
        let out = dct2(&input);
        // c[m>0] = √(2/N)·cos(π·m/(2N)) — bounded by √(2/N).
        for coefficient in out.iter().skip(1) {
            assert!(
                coefficient.abs() <= (2.0 / n as f64).sqrt() + EPS,
                "{coefficient}"
            );
        }
        // Energy is preserved: the orthonormal transform is an isometry.
        let energy_in: f64 = input.iter().map(|x| x * x).sum();
        let energy_out: f64 = out.iter().map(|x| x * x).sum();
        assert!((energy_in - energy_out).abs() < EPS);
    }

    #[allow(clippy::expect_used)]
    #[test]
    fn cosine_rows_map_to_single_coefficients() {
        // dct2 applied to the raw cosine row m yields a single nonzero
        // coefficient at index m — the orthogonality of the cosine sums:
        // Σ_k cos(θ_jk)·cos(θ_mk) = (N/2)·δ_jm (N when j = m = 0).
        let n = 8;
        for m in 0..n {
            let mut row = vec![0.0; n];
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = (PI * ((m * (2 * k + 1)) as f64) / ((2 * n) as f64)).cos();
            }
            let out = dct2(&row);
            for (j, coefficient) in out.iter().enumerate() {
                let expected = if j == m && j == 0 {
                    (n as f64).sqrt()
                } else if j == m {
                    (n as f64 / 2.0).sqrt()
                } else {
                    0.0
                };
                assert!(
                    (coefficient - expected).abs() < 1e-9,
                    "row {m}, coefficient {j}: {coefficient} vs {expected}"
                );
            }
        }
    }

    #[allow(clippy::expect_used)]
    #[test]
    fn phash_style_pipeline_is_deterministic_and_separates() {
        // The thumbnail-perceptual-hash shape: 64 low-frequency
        // coefficients, median threshold, 64-bit hash. Deterministic, and
        // visually-similar inputs hash close while distinct inputs do not.
        let gradient: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let noisy_gradient: Vec<f64> = gradient.iter().map(|x| x + 0.5).collect();
        let checker: Vec<f64> = (0..64)
            .map(|i| if i % 2 == 0 { 0.0 } else { 255.0 })
            .collect();

        let hash = |input: &[f64]| -> u64 {
            let coefficients = dct2(input);
            let low: Vec<f64> = coefficients.iter().take(8).copied().collect();
            let med = median(&low);
            low.iter()
                .map(|&c| if c > med { 1u64 } else { 0 })
                .enumerate()
                .fold(0u64, |acc, (i, bit)| acc | (bit << i))
        };

        assert_eq!(
            hash(&gradient),
            hash(&noisy_gradient),
            "perturbation survives"
        );
        assert_ne!(
            hash(&gradient),
            hash(&checker),
            "distinct structure differs"
        );
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(dct2(&[]).is_empty());
    }

    #[allow(clippy::expect_used)]
    #[test]
    fn agrees_with_mean_reference() {
        // c[0] is √N × the mean; a hand-checked small case.
        let input = [1.0, 2.0, 3.0, 4.0];
        let out = dct2(&input);
        let expected_dc = 10.0 / 2.0; // sum / √N
        let dc = out.first().copied().unwrap_or_default();
        assert!((dc - expected_dc).abs() < EPS, "{dc}");
        // Reference c[1]: √(2/4)·Σ x[k]·cos(π·1·(2k+1)/8).
        let reference: f64 = input
            .iter()
            .enumerate()
            .map(|(k, &x)| x * (PI * (2 * k + 1) as f64 / 8.0).cos())
            .sum::<f64>()
            * 0.5_f64.sqrt();
        let first = out.get(1).copied().unwrap_or_default();
        assert!((first - reference).abs() < EPS);
    }

    #[test]
    fn mean_helper_agrees_for_the_threshold_tests() {
        assert!((mean(&[2.0, 4.0]) - 3.0).abs() < EPS);
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < EPS);
    }
}
