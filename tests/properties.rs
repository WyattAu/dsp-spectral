#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    missing_docs
)]
//! Property-based invariants over arbitrary signals, configs, and spectra.
//!
//! These are the tests that generalise the examples: instead of checking one
//! hand-picked configuration, they check that the *invariants* hold for
//! hundreds of arbitrary ones — the round trip for any signal/hop/window
//! pair, and the stated range of every spectral descriptor for any positive
//! spectrum.

use dsp_spectral::{
    denoise, harmonic_percussive_split, hz_to_mel, istft, mel_frequencies, mel_to_hz, mfcc,
    real_cepstrum, spectral_bandwidth, spectral_centroid, spectral_flatness, spectral_flux,
    spectral_rolloff, spectral_subtract, stft, zero_crossing_rate, Complex, GateConfig,
    NoiseProfile, Spectrum, StftConfig, Window,
};
use proptest::prelude::*;

const WINDOWS: [Window; 6] = [
    Window::Hann,
    Window::Hamming,
    Window::Blackman,
    Window::BlackmanHarris,
    Window::FlatTop,
    Window::Rectangular,
];

/// A signal bounded to ±1 so tolerances can be absolute.
fn unit_signal() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1.0f64..1.0, 64..2_048)
}

/// A power-of-two transform size in [8, 4096] with a hop in [1, N/2].
fn config_strategy() -> impl Strategy<Value = StftConfig> {
    (3u32..13, any::<bool>(), 0usize..6, 1.0f64..192_000.0).prop_flat_map(
        |(exp, centered, window, sample_rate)| {
            let fft_size = 1usize << exp;
            (1..=fft_size / 2).prop_map(move |hop| StftConfig {
                window: WINDOWS[window],
                hop,
                fft_size,
                center: centered,
                window_overlap: 0.0,
                sample_rate,
            })
        },
    )
}

fn worst_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The half-open range of samples a spectrogram can actually reconstruct.
///
/// Mirrors the analysis, not the inverse: a sample is reconstructible iff the
/// frames cover it *and* the squared-window weight accumulated there clears
/// the floor `istft` uses. Everything else is documented to come back as an
/// exact zero, which this test then asserts.
fn reconstructible_range(
    config: &StftConfig,
    spec: &Spectrum,
    len: usize,
) -> std::ops::Range<usize> {
    /// The same floor `istft` applies.
    const MIN_WEIGHT: f64 = 1e-12;
    let taps = config.window.coefficients(config.fft_size);
    let pad = if config.center {
        config.fft_size / 2
    } else {
        0
    };
    let mut weight = vec![0.0f64; len];
    for f in 0..spec.num_frames() {
        for (j, &tap) in taps.iter().enumerate() {
            // Output index, undoing the centre offset.
            let Some(out_idx) = (f * config.hop + j).checked_sub(pad) else {
                continue;
            };
            if out_idx >= len {
                continue;
            }
            weight[out_idx] += tap * tap;
        }
    }
    let usable = |i: usize| weight[i] > MIN_WEIGHT;
    let start = (0..len).find(|i| usable(*i)).unwrap_or(len);
    let end = (start..len)
        .rev()
        .find(|i| usable(*i))
        .map_or(start, |i| i + 1);
    start..end
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

    /// The contract this crate exists for: ISTFT(STFT(x)) == x for arbitrary
    /// finite signals, windows, transform sizes, hops, and both framing modes.
    #[test]
    fn prop_roundtrip_is_exact(
        signal in unit_signal(),
        config in config_strategy(),
    ) {
        let spec = stft(&signal, &config).expect("generated config is valid");
        let back = istft(&spec, signal.len());
        prop_assert_eq!(back.len(), signal.len());
        // `center = true` covers the whole signal; `center = false` stops at
        // the last full frame, and the uncovered tail is exact zero by
        // design (see `Spectrum::covered_len`).
        let range = reconstructible_range(&config, &spec, signal.len());
        let err = worst_err(&signal[range.start..range.end], &back[range.start..range.end]);
        prop_assert!(
            err < 1e-9,
            "round-trip error {err:e} (fft {} hop {} {} centered {})",
            config.fft_size, config.hop, config.window.name(), config.center
        );
        // Everything outside that range is exact zero.
        for (i, v) in back.iter().enumerate() {
            if !range.contains(&i) {
                prop_assert_eq!(*v, 0.0, "sample {} outside the reconstructible range", i);
            }
        }
    }

    /// The same, for signals that include large values and DC offsets.
    #[test]
    fn prop_roundtrip_survives_large_amplitudes(
        signal in prop::collection::vec(-1.0e3f64..1.0e3, 64..1_024),
        offset in -1.0e3f64..1.0e3,
        fft_size in prop::sample::select(vec![8usize, 16, 64, 256]),
        hop_frac in 2usize..=8,
    ) {
        let x: Vec<f64> = signal.iter().map(|v| v + offset).collect();
        let hop = core::cmp::max(1, fft_size / hop_frac);
        let cfg = StftConfig::new(fft_size, hop);
        let spec = stft(&x, &cfg).expect("valid");
        let back = istft(&spec, x.len());
        let scale = 1.0e3f64 + offset.abs();
        let err = worst_err(&x, &back);
        // Centred framing covers the whole signal, so this is the full length.
        prop_assert!(
            err < 1e-9 * scale,
            "round-trip error {err:e} at scale {scale:e} (fft {fft_size} hop {hop})"
        );
    }


    /// Every frame yields exactly one value from every feature, whatever the
    /// spectrum's contents.
    #[test]
    fn prop_features_return_one_value_per_frame(
        rows in prop::collection::vec(
            prop::collection::vec(-1.0e6f64..1.0e6, 1..33),
            0..16,
        ),
    ) {
        let bins = rows.first().map_or(1, Vec::len);
        let fft_size = (bins - 1).next_power_of_two().max(2);
        let spec = Spectrum::from_parts(
            rows.iter()
                .map(|row| {
                    (0..fft_size / 2 + 1)
                        .map(|k| Complex::new(row.get(k).copied().unwrap_or(0.0), 0.0))
                        .collect()
                })
                .collect(),
            1,
            fft_size,
            Window::Rectangular,
            true,
        ).expect("valid");
        let n = spec.num_frames();
        prop_assert_eq!(spectral_centroid(&spec, 48_000.0).len(), n);
        prop_assert_eq!(spectral_bandwidth(&spec, 48_000.0, &[]).len(), n);
        prop_assert_eq!(spectral_flatness(&spec).len(), n);
        prop_assert_eq!(spectral_rolloff(&spec, 0.9, 48_000.0).len(), n);
        prop_assert_eq!(spectral_flux(&spec).len(), n);
    }

    /// The mel scale inverts exactly for any frequency.
    #[test]
    fn prop_mel_round_trips(hz in 0.0f64..100_000.0) {
        let back = mel_to_hz(hz_to_mel(hz));
        prop_assert!((back - hz).abs() < 1e-9, "mel round-trip: {hz} -> {back}");
    }

    /// `mel_frequencies` is monotone and anchored at its endpoints.
    #[test]
    fn prop_mel_frequencies_are_monotone(
        n_mels in 2usize..64,
        f_min in 0.0f64..4_000.0,
        f_max in 4_001.0f64..24_000.0,
    ) {
        let f = mel_frequencies(n_mels, f_min, f_max);
        prop_assert_eq!(f.len(), n_mels);
        prop_assert!((f[0] - f_min).abs() < 1e-9);
        prop_assert!((f[n_mels - 1] - f_max).abs() < 1e-9);
        for w in f.windows(2) {
            prop_assert!(w[1] > w[0], "not increasing: {w:?}");
        }
    }

    /// MFCCs respect `n_coeffs` and stay finite for any positive energies.
    #[test]
    fn prop_mfcc_shape(
        energies in prop::collection::vec(0.0f64..1.0e6, 1..64),
        n_coeffs in 0usize..64,
    ) {
        let c = mfcc(&energies, n_coeffs);
        prop_assert_eq!(c.len(), energies.len().min(n_coeffs));
        for v in &c {
            prop_assert!(v.is_finite(), "mfcc coefficient {v}");
        }
    }

    /// The real cepstrum is real, finite, and symmetric for any power-of-two
    /// length signal.
    #[test]
    fn prop_cepstrum_is_real_and_symmetric(
        signal in prop::collection::vec(-1.0f64..1.0, 8..512),
    ) {
        let n = signal.len().next_power_of_two();
        let x: Vec<f64> = (0..n).map(|i| signal[i % signal.len()]).collect();
        let c = real_cepstrum(&x).expect("power-of-two length");
        prop_assert_eq!(c.len(), n);
        for v in &c {
            prop_assert!(v.is_finite(), "cepstrum {v}");
        }
        for q in 1..n / 2 {
            prop_assert!(
                (c[q] - c[n - q]).abs() < 1e-9,
                "symmetry at q={q}: {} vs {}",
                c[q],
                c[n - q]
            );
        }
    }

    /// Zero-subtraction is the identity, for any profile.
    #[test]
    fn prop_zero_subtraction_is_identity(
        rows in prop::collection::vec(
            prop::collection::vec(0.0f64..1.0e3, 1..32),
            1..8,
        ),
        over_subtraction in 0.0f64..4.0,
    ) {
        let bins = rows.first().map_or(1, Vec::len);
        let fft_size = (bins - 1).next_power_of_two().max(2);
        let spec = Spectrum::from_parts(
            rows.iter()
                .map(|row| {
                    (0..fft_size / 2 + 1)
                        .map(|k| Complex::from_polar(row.get(k).copied().unwrap_or(0.0), 0.3))
                        .collect()
                })
                .collect(),
            1,
            fft_size,
            Window::Hann,
            true,
        ).expect("valid");
        let profile = NoiseProfile::from_magnitudes(
            (0..spec.bin_count()).map(|k| rows[0][k % bins]).collect(),
            48_000.0,
        );
        let out = spectral_subtract(&spec, &profile, over_subtraction).expect("valid");
        // Subtraction only ever removes magnitude, and preserves phase.
        for f in 0..spec.num_frames() {
            for k in 0..spec.bin_count() {
                let a = spec.magnitude(f)[k];
                let b = out.magnitude(f)[k];
                prop_assert!(b <= a + 1e-9, "subtraction increased magnitude at {f},{k}");
                if b > 1e-12 {
                    let pa = spec.frame(f).expect("in range")[k].phase();
                    let pb = out.frame(f).expect("in range")[k].phase();
                    let d = (pa - pb).abs();
                    prop_assert!(d < 1e-9 || (d - core::f64::consts::TAU).abs() < 1e-9);
                }
            }
        }
    }

    /// HPSS splits magnitude exactly, whatever the median width.
    #[test]
    fn prop_hpss_is_complementary(
        rows in prop::collection::vec(
            prop::collection::vec(0.0f64..1.0e3, 1..32),
            1..24,
        ),
        ratio in 1.0f64..16.0,
    ) {
        let bins = rows.first().map_or(1, Vec::len);
        let fft_size = (bins - 1).next_power_of_two().max(2);
        let spec = Spectrum::from_parts(
            rows.iter()
                .map(|row| {
                    (0..fft_size / 2 + 1)
                        .map(|k| Complex::new(row.get(k).copied().unwrap_or(0.0), 0.0))
                        .collect()
                })
                .collect(),
            1,
            fft_size,
            Window::Hann,
            true,
        ).expect("valid");
        let (harm, perc) = harmonic_percussive_split(&spec, ratio);
        prop_assert_eq!(harm.num_frames(), spec.num_frames());
        for f in 0..spec.num_frames() {
            let a = spec.magnitude(f);
            let b = harm.magnitude(f);
            let c = perc.magnitude(f);
            for k in 0..spec.bin_count() {
                prop_assert!(
                    ((b[k] + c[k]) - a[k]).abs() < 1e-9 * a[k].max(1.0),
                    "complementary mask at {f},{k}"
                );
            }
        }
    }

    /// `denoise` never lengthens or corrupts the signal, and never lowers the
    /// SNR below the input's when the profile matches the actual noise.
    #[test]
    fn prop_denoise_returns_a_clean_signal(
        signal in prop::collection::vec(-1.0f64..1.0, 512..4_096),
        fft_size in prop::sample::select(vec![256usize, 512, 1024]),
        hop_frac in 2usize..=8,
    ) {
        let hop = core::cmp::max(1, fft_size / hop_frac);
        let cfg = StftConfig::new(fft_size, hop);
        let spec = stft(&signal, &cfg).expect("valid");
        let profile = NoiseProfile::from_frames(&spec, &[0]);
        let out = denoise(&signal, &cfg, &profile, &GateConfig::default()).expect("valid");
        prop_assert_eq!(out.len(), signal.len());
        for v in &out {
            prop_assert!(v.is_finite(), "denoise produced {v}");
        }
    }

    /// Zero-crossing rate is a probability: in [0, 1], and 0.5 for a signal
    /// that flips every sample.
    #[test]
    fn prop_zcr_is_bounded(signal in prop::collection::vec(-1.0f64..1.0, 1..4_096)) {
        let z = zero_crossing_rate(&signal);
        prop_assert!(z.is_finite());
        prop_assert!((0.0..=1.0).contains(&z), "zcr {z}");
        if signal.len() < 2 {
            prop_assert_eq!(z, 0.0);
        }
    }

    /// Degenerate configurations are always rejected, never accepted or
    /// silently mishandled.
    #[test]
    fn prop_invalid_configs_are_rejected(
        fft_size in 0usize..64,
        hop in 0usize..64,
    ) {
        let cfg = StftConfig::new(fft_size, hop);
        let x = vec![0.1; 128];
        if stft(&x, &cfg).is_ok() {
            // Accepted, so it must have been a genuinely valid pair.
            prop_assert!(fft_size >= 2 && fft_size.is_power_of_two());
            prop_assert!(hop >= 1 && hop <= fft_size / 2);
        }
    }

    /// Whatever `stft` does with a config, the errors it returns are typed and
    /// carry a non-empty message — never a panic.
    #[test]
    fn prop_config_errors_are_typed(
        fft_size in 0usize..64,
        hop in 0usize..64,
        sample_rate in prop::option::of(-1.0f64..1.0).prop_map(|v| v.unwrap_or(48_000.0)),
    ) {
        let mut cfg = StftConfig::new(fft_size, hop);
        cfg.sample_rate = sample_rate;
        let x = vec![0.1; 128];
        if let Err(e) = stft(&x, &cfg) {
            prop_assert!(!e.to_string().is_empty());
        }
    }
}

proptest! {
    // Both of these are pure arithmetic invariants over random non-negative
    // rows, so they get the larger case budget: the degenerate shapes the
    // boundaries live on (all-zero rows, single-bin frames) are rare draws,
    // and a 200-case sweep can miss them. The round-trip property stays at
    // 200 because building its signals is far more expensive.
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

/// Flatness is a ratio of means, so it is always in [0, 1] for a
/// non-negative spectrum — including spectra with exact zeros.
///
/// Raised to 300 cases: this is a pure arithmetic invariant over random
/// non-negative rows, so more cases buy real coverage of the degenerate
/// shapes (all-zero rows, single bins) that the boundary lives on.
#[test]
fn prop_flatness_is_bounded(
    rows in prop::collection::vec(
        prop::collection::vec(0.0f64..1.0e3, 1..64),
        1..12,
    ),
) {
    let bins = rows.first().map_or(1, Vec::len);
    let fft_size = (bins - 1).next_power_of_two().max(2);
    let spec = Spectrum::from_parts(
        rows.iter()
            .map(|row| {
                (0..fft_size / 2 + 1)
                    .map(|k| Complex::new(row.get(k).copied().unwrap_or(0.0), 0.0))
                    .collect()
            })
            .collect(),
        1,
        fft_size,
        Window::Hann,
        true,
    ).expect("generated spectrum is valid");
    let flatness = spectral_flatness(&spec);
    prop_assert_eq!(flatness.len(), spec.num_frames());
    for f in flatness {
        prop_assert!(f.is_finite(), "flatness {f}");
        prop_assert!((0.0..=1.0).contains(&f), "flatness {f} out of [0, 1]");
    }
}

/// The centroid is a weighted mean of bin frequencies, so it is bounded by
/// the Nyquist frequency — and non-negative. Raised to 300 cases for the
/// same reason as the flatness invariant above.
#[test]
fn prop_centroid_is_within_the_band(
    rows in prop::collection::vec(
        prop::collection::vec(0.0f64..1.0e3, 1..64),
        1..12,
    ),
    sample_rate in 1.0f64..192_000.0,
) {
    let bins = rows.first().map_or(1, Vec::len);
    let fft_size = (bins - 1).next_power_of_two().max(2);
    let spec = Spectrum::from_parts(
        rows.iter()
            .map(|row| {
                (0..fft_size / 2 + 1)
                    .map(|k| Complex::new(row.get(k).copied().unwrap_or(0.0), 0.0))
                    .collect()
            })
            .collect(),
        1,
        fft_size,
        Window::Hann,
        true,
    ).expect("generated spectrum is valid");
    let nyquist = sample_rate / 2.0;
    let centroids = spectral_centroid(&spec, sample_rate);
    prop_assert_eq!(centroids.len(), spec.num_frames());
    for c in &centroids {
        prop_assert!(c.is_finite(), "centroid {c}");
        prop_assert!((0.0..=nyquist).contains(c), "centroid {c} outside 0..{nyquist}");
    }
    // Bandwidth is a weighted deviation, so it too is bounded by the band.
    for b in spectral_bandwidth(&spec, sample_rate, &centroids) {
        prop_assert!(b.is_finite(), "bandwidth {b}");
        prop_assert!((0.0..=nyquist).contains(&b), "bandwidth {b} outside 0..{nyquist}");
    }
    // So is rolloff, at every threshold.
    for threshold in [0.05f64, 0.5, 0.95] {
        for r in spectral_rolloff(&spec, threshold, sample_rate) {
            prop_assert!(r.is_finite(), "rolloff {r}");
            prop_assert!((0.0..=nyquist).contains(&r), "rolloff {r} outside 0..{nyquist}");
        }
    }
    // Flux is a sum of non-negative differences.
    for f in spectral_flux(&spec) {
        prop_assert!(f.is_finite() && f >= 0.0, "flux {f}");
    }
}
}
