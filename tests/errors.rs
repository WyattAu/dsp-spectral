#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    missing_docs
)]
//! Error contract: every variant reachable, every `Display` stable, every
//! public entry point total on empty/short/non-finite input.

use core::f64::consts::TAU;
use core::fmt::Write as _;

use dsp_spectral::{
    denoise, harmonic_percussive_split, hz_to_mel, istft, mel_frequencies, mel_to_hz, mfcc,
    real_cepstrum, resample_frames, spectral_bandwidth, spectral_centroid, spectral_flatness,
    spectral_flux, spectral_gate, spectral_rolloff, spectral_smooth, spectral_subtract, stft,
    zero_crossing_rate, Complex, GateConfig, MelBank, NoiseProfile, SpectralError, Spectrum,
    StftConfig, Window,
};

/// Every `SpectralError` variant renders exactly this text.
#[test]
fn display_covers_every_variant() {
    let cases: Vec<(SpectralError, &str)> = vec![
        (
            SpectralError::Config("hop must be > 0".into()),
            "invalid spectral configuration: hop must be > 0",
        ),
        (SpectralError::EmptyInput, "input is empty"),
        (
            SpectralError::FrameOutOfRange(7),
            "frame index 7 is out of range",
        ),
        (
            SpectralError::NonFinite,
            "input contains a non-finite value (NaN or infinity)",
        ),
        (
            SpectralError::Dsp("size 100 is not a power of two".into()),
            "dsp-core kernel failed: size 100 is not a power of two",
        ),
    ];
    for (err, want) in cases {
        assert_eq!(err.to_string(), want);
        // And through the fmt machinery directly.
        let mut s = String::new();
        write!(s, "{err}").unwrap();
        assert_eq!(s, want);
        // Implements std::error::Error.
        fn assert_error<E: core::error::Error>(_: &E) {}
        assert_error(&err);
    }
    // Debug is derived and non-empty.
    assert!(!format!("{:?}", SpectralError::EmptyInput).is_empty());
}

#[test]
fn errors_are_comparable_and_cloneable() {
    assert_eq!(SpectralError::EmptyInput, SpectralError::EmptyInput);
    assert_ne!(SpectralError::EmptyInput, SpectralError::NonFinite);
    assert_ne!(
        SpectralError::FrameOutOfRange(1),
        SpectralError::FrameOutOfRange(2)
    );
    let cloned = SpectralError::Config("x".into()).clone();
    assert_eq!(cloned, SpectralError::Config("x".into()));
}

#[test]
fn stft_rejects_invalid_configurations() {
    let x = vec![0.1; 1_024];

    // fft_size not a power of two.
    let mut cfg = StftConfig::new(1_000, 256);
    assert!(matches!(
        stft(&x, &cfg),
        Err(SpectralError::Config(m)) if m.contains("power of two")
    ));

    // fft_size below 2.
    cfg = StftConfig::new(1, 0);
    assert!(stft(&x, &cfg).is_err());

    // hop == 0.
    cfg = StftConfig::new(256, 0);
    assert!(matches!(
        stft(&x, &cfg),
        Err(SpectralError::Config(m)) if m.contains("hop")
    ));

    // hop > fft_size, and hop in the half-open band above fft_size/2.
    for hop in [513usize, 1_024, 4_096] {
        cfg = StftConfig::new(512, hop);
        assert!(
            matches!(stft(&x, &cfg), Err(SpectralError::Config(_))),
            "hop {hop} should be rejected"
        );
    }
    // The boundary itself is accepted.
    cfg = StftConfig::new(512, 256);
    assert!(stft(&x, &cfg).is_ok());

    // Non-finite / non-positive sample rate.
    for fs in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
        cfg = StftConfig::new(256, 64).with_sample_rate(fs);
        assert!(matches!(
            stft(&x, &cfg),
            Err(SpectralError::Config(m)) if m.contains("sample_rate")
        ));
    }

    // Negative / non-finite window_overlap.
    let mut bad = StftConfig::new(256, 64);
    bad.window_overlap = -1.0;
    assert!(matches!(stft(&x, &bad), Err(SpectralError::Config(_))));
    bad.window_overlap = f32::NAN;
    assert!(matches!(stft(&x, &bad), Err(SpectralError::Config(_))));
}

#[test]
fn stft_rejects_empty_and_non_finite_input() {
    let cfg = StftConfig::new(64, 16);
    assert_eq!(stft(&[], &cfg).unwrap_err(), SpectralError::EmptyInput);

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut x = vec![0.0; 128];
        x[7] = bad;
        assert_eq!(stft(&x, &cfg).unwrap_err(), SpectralError::NonFinite);
    }

    // NoiseProfile::estimate shares the same contract.
    assert_eq!(
        NoiseProfile::estimate(&[], &cfg).unwrap_err(),
        SpectralError::EmptyInput
    );
    assert_eq!(
        NoiseProfile::estimate(&[f64::NAN; 128], &cfg).unwrap_err(),
        SpectralError::NonFinite
    );
    // And real_cepstrum.
    assert_eq!(real_cepstrum(&[]).unwrap_err(), SpectralError::EmptyInput);
    assert_eq!(
        real_cepstrum(&[f64::INFINITY; 8]).unwrap_err(),
        SpectralError::NonFinite
    );
}

#[test]
fn frame_index_errors_are_reported_precisely() {
    let spec = stft(&[0.1; 512], &StftConfig::new(64, 16)).expect("valid");
    assert_eq!(spec.num_frames(), 1 + 512 / 16);
    assert!(spec.frame(spec.num_frames()).is_none());
    assert!(spec.frame(usize::MAX).is_none());
    assert_eq!(
        spec.frame_checked(spec.num_frames()).unwrap_err(),
        SpectralError::FrameOutOfRange(spec.num_frames())
    );
    assert_eq!(
        spec.frame_checked(usize::MAX).unwrap_err(),
        SpectralError::FrameOutOfRange(usize::MAX)
    );
    // In-range indices work.
    for f in 0..spec.num_frames() {
        assert_eq!(
            spec.frame_checked(f).expect("in range").len(),
            spec.bin_count()
        );
    }
    // Out-of-range magnitude/power accessors yield empty vectors.
    assert!(spec.magnitude(spec.num_frames()).is_empty());
    assert!(spec.power(usize::MAX).is_empty());
    assert!(spec.magnitude_db(usize::MAX).is_empty());
    assert!(spec.power_db(usize::MAX).is_empty());
}

#[test]
fn spectrum_construction_validates_its_shape() {
    let bins = vec![Complex::ZERO; 5];
    // Valid, including zero frames.
    assert!(Spectrum::from_parts(vec![], 1, 8, Window::Hann, true).is_ok());
    assert!(Spectrum::from_parts(vec![bins.clone()], 1, 8, Window::Hann, true).is_ok());
    // Wrong bin count.
    assert!(matches!(
        Spectrum::from_parts(vec![vec![Complex::ZERO; 4]], 1, 8, Window::Hann, true),
        Err(SpectralError::Config(m)) if m.contains("bins")
    ));
    // Zero hop.
    assert!(matches!(
        Spectrum::from_parts(vec![], 0, 8, Window::Hann, true),
        Err(SpectralError::Config(m)) if m.contains("hop")
    ));
    // Bad transform size.
    for fft in [0usize, 1, 6, 100] {
        assert!(
            matches!(
                Spectrum::from_parts(vec![], 1, fft, Window::Hann, true),
                Err(SpectralError::Config(_))
            ),
            "fft {fft}"
        );
    }
    // `Default` is the smallest valid spectrogram.
    let d = Spectrum::default();
    assert_eq!(d.num_frames(), 0);
    assert_eq!(d.bin_count(), 2);
    assert_eq!(d.hop(), 1);
    assert_eq!(d.fft_size(), 2);
    assert_eq!(d.window(), Window::Hann);
    assert!(d.is_centered());
    assert_eq!(istft(&d, 16), vec![0.0; 16]);
}

#[test]
fn noise_profile_validation_errors() {
    let cfg = StftConfig::new(64, 16);
    let spec = stft(&[0.2; 256], &cfg).expect("valid");
    let good = NoiseProfile::from_frames(&spec, &[0]);

    // Bin-count mismatch.
    let other = stft(&[0.2; 256], &StftConfig::new(32, 8)).expect("valid");
    let mismatched = NoiseProfile::from_frames(&other, &[0]);
    assert!(matches!(
        mismatched.validate_for(&spec),
        Err(SpectralError::Config(m)) if m.contains("bins")
    ));
    // Every operation that takes a profile validates it.
    for r in [
        spectral_gate(&spec, &mismatched, &GateConfig::default()).err(),
        spectral_subtract(&spec, &mismatched, 1.0).err(),
        denoise(&[0.2; 256], &cfg, &mismatched, &GateConfig::default()).err(),
    ] {
        assert!(matches!(r, Some(SpectralError::Config(_))));
    }
    // A valid profile passes.
    assert!(good.validate_for(&spec).is_ok());
    // Non-finite profiles are rejected: build one through the public escape
    // hatch.
    let broken = NoiseProfile::from_magnitudes(vec![f64::NAN; spec.bin_count()], 8_000.0);
    assert_eq!(
        broken.validate_for(&spec).unwrap_err(),
        SpectralError::NonFinite
    );
    assert!(spectral_gate(&spec, &broken, &GateConfig::default()).is_err());
}

#[test]
fn gate_config_validation_errors() {
    let cfg = StftConfig::new(64, 16);
    let spec = stft(&[0.2; 256], &cfg).expect("valid");
    let profile = NoiseProfile::from_frames(&spec, &[0]);
    for gate in [
        GateConfig {
            threshold_db: f64::NAN,
            reduction_db: 0.0,
            smoothing_frames: 0,
        },
        GateConfig {
            threshold_db: f64::INFINITY,
            reduction_db: 0.0,
            smoothing_frames: 0,
        },
        GateConfig {
            threshold_db: 0.0,
            reduction_db: f64::NAN,
            smoothing_frames: 0,
        },
        GateConfig {
            threshold_db: 0.0,
            reduction_db: f64::NEG_INFINITY,
            smoothing_frames: 0,
        },
    ] {
        assert!(
            matches!(gate.validate(), Err(SpectralError::Config(_))),
            "{gate:?}"
        );
        assert!(spectral_gate(&spec, &profile, &gate).is_err());
    }
    // A valid gate passes.
    assert!(GateConfig::default().validate().is_ok());
    // And smoothing_frames is unbounded (any width is accepted).
    for n in [0usize, 1, 2, 1_000] {
        let g = GateConfig {
            smoothing_frames: n,
            ..GateConfig::default()
        };
        assert!(g.validate().is_ok());
        assert!(spectral_gate(&spec, &profile, &g).is_ok());
    }
}

#[test]
fn subtraction_and_smoothing_argument_errors() {
    let cfg = StftConfig::new(64, 16);
    let spec = stft(&[0.2; 256], &cfg).expect("valid");
    let profile = NoiseProfile::from_frames(&spec, &[0]);
    for alpha in [f64::NAN, f64::INFINITY, -1.0, -0.001] {
        assert!(
            matches!(
                spectral_subtract(&spec, &profile, alpha),
                Err(SpectralError::Config(_))
            ),
            "alpha {alpha}"
        );
    }
    assert!(spectral_subtract(&spec, &profile, 0.0).is_ok());
    assert!(matches!(
        spectral_smooth(&spec, &[]),
        Err(SpectralError::Config(m)) if m.contains("kernel")
    ));
    assert!(matches!(
        spectral_smooth(&spec, &[1.0, f64::NAN]),
        Err(SpectralError::NonFinite)
    ));
    assert!(matches!(
        spectral_smooth(&spec, &[f64::INFINITY]),
        Err(SpectralError::NonFinite)
    ));
    assert!(spectral_smooth(&spec, &[0.0]).is_ok());
}

#[test]
fn mel_bank_validation_errors() {
    // n_mels must be positive.
    assert!(matches!(
        MelBank::new(0, 0.0, 1_000.0, 8_000.0, 512),
        Err(SpectralError::Config(m)) if m.contains("n_mels")
    ));
    // fft_size must be a power of two >= 2.
    for fft in [0usize, 1, 6, 100] {
        assert!(
            matches!(
                MelBank::new(4, 0.0, 1_000.0, 8_000.0, fft),
                Err(SpectralError::Config(_))
            ),
            "fft {fft}"
        );
    }
    // sample_rate must be finite and positive.
    for fs in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            MelBank::new(4, 0.0, 1_000.0, fs, 512),
            Err(SpectralError::Config(m)) if m.contains("sample_rate")
        ));
    }
    // f_min < f_max, both finite.
    for (lo, hi) in [(1_000.0, 1_000.0), (2_000.0, 1_000.0)] {
        assert!(matches!(
            MelBank::new(4, lo, hi, 8_000.0, 512),
            Err(SpectralError::Config(m)) if m.contains("f_min")
        ));
    }
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(MelBank::new(4, 0.0, bad, 8_000.0, 512).is_err());
        assert!(MelBank::new(4, bad, 1_000.0, 8_000.0, 512).is_err());
    }
    // f_max must not exceed Nyquist.
    assert!(matches!(
        MelBank::new(4, 0.0, 5_000.0, 8_000.0, 512),
        Err(SpectralError::Config(m)) if m.contains("Nyquist")
    ));
    // The boundary is accepted.
    assert!(MelBank::new(4, 0.0, 4_000.0, 8_000.0, 512).is_ok());
}

#[test]
fn resample_frames_validates_its_factor() {
    let spec = stft(&[0.2; 512], &StftConfig::new(64, 16)).expect("valid");
    for factor in [0.0f64, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            matches!(
                resample_frames(&spec, factor),
                Err(SpectralError::Config(_))
            ),
            "factor {factor}"
        );
    }
    // A factor so small the hop rounds to zero is rejected.
    assert!(matches!(
        resample_frames(&spec, 1e-9),
        Err(SpectralError::Config(m)) if m.contains("hop")
    ));
    // Valid factors preserve everything but the hop.
    for factor in [0.25f64, 0.5, 1.0, 2.0, 7.5] {
        let r = resample_frames(&spec, factor).expect("valid");
        assert_eq!(r.num_frames(), spec.num_frames());
        assert_eq!(r.fft_size(), spec.fft_size());
        assert_eq!(r.window(), spec.window());
        assert_eq!(r.is_centered(), spec.is_centered());
        assert_eq!(r.hop(), ((spec.hop() as f64 * factor) as usize).max(1));
        assert_eq!(r.frames(), spec.frames());
    }
}

#[test]
fn cepstrum_length_validation() {
    // Zero samples is its own error, distinct from a bad length.
    assert_eq!(real_cepstrum(&[]).unwrap_err(), SpectralError::EmptyInput);
    for n in [1usize, 3, 5, 6, 7, 100, 1_000] {
        assert!(
            matches!(
                real_cepstrum(&vec![0.1; n]),
                Err(SpectralError::Config(m)) if m.contains("power-of-two")
            ),
            "n={n} must be rejected"
        );
    }
    for n in [2usize, 4, 8, 16, 64, 1_024] {
        let c = real_cepstrum(&vec![0.1; n]).expect("power of two");
        assert_eq!(c.len(), n);
    }
}

#[test]
fn every_feature_is_total_on_an_empty_spectrogram() {
    let empty = Spectrum::default();
    assert!(spectral_centroid(&empty, 48_000.0).is_empty());
    assert!(spectral_bandwidth(&empty, 48_000.0, &[]).is_empty());
    assert!(spectral_bandwidth(&empty, 48_000.0, &[1.0, 2.0]).is_empty());
    assert!(spectral_flatness(&empty).is_empty());
    assert!(spectral_rolloff(&empty, 0.9, 48_000.0).is_empty());
    assert!(spectral_flux(&empty).is_empty());

    // And on a silent spectrogram of a realistic size: zeros, not NaNs.
    let cfg = StftConfig::new(256, 64);
    let silent = stft(&[0.0; 1_024], &cfg).expect("valid");
    assert!(silent.num_frames() > 1);
    for v in spectral_centroid(&silent, cfg.sample_rate) {
        assert_eq!(v, 0.0);
    }
    for v in spectral_bandwidth(&silent, cfg.sample_rate, &[]) {
        assert_eq!(v, 0.0);
    }
    for v in spectral_flatness(&silent) {
        assert_eq!(v, 0.0);
    }
    for v in spectral_rolloff(&silent, 0.9, cfg.sample_rate) {
        assert_eq!(v, 0.0);
    }
    for v in spectral_flux(&silent) {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn every_feature_is_total_on_a_single_sample() {
    // A one-sample signal is shorter than any useful window; the centred STFT
    // still produces frames (the reflected padding supplies context).
    for len in [1usize, 2, 3, 5] {
        let cfg = StftConfig::new(64, 16);
        let x: Vec<f64> = (0..len).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let spec = stft(&x, &cfg).expect("valid");
        assert!(spec.num_frames() >= 1);
        assert_eq!(
            spectral_centroid(&spec, cfg.sample_rate).len(),
            spec.num_frames()
        );
        assert_eq!(spectral_flatness(&spec).len(), spec.num_frames());
        assert_eq!(spectral_flux(&spec).len(), spec.num_frames());
        assert_eq!(
            spectral_rolloff(&spec, 0.9, cfg.sample_rate).len(),
            spec.num_frames()
        );
        // Round-trip still reconstructs (a lone sample at the centre is
        // surrounded by its own reflection).
        let y = istft(&spec, len);
        assert_eq!(y.len(), len);
        assert!((y[0] - x[0]).abs() < 1e-10, "len {len}: {}", y[0]);
    }
    // And the frequency helpers tolerate zero-length input.
    assert_eq!(zero_crossing_rate(&[]), 0.0);
    assert_eq!(zero_crossing_rate(&[1.0]), 0.0);
    assert_eq!(hz_to_mel(0.0), 0.0);
    assert_eq!(mel_to_hz(0.0), 0.0);
    assert!(mel_frequencies(0, 0.0, 1.0).is_empty());
    assert!(mfcc(&[], 4).is_empty());
    assert!(mfcc(&[1.0], 0).is_empty());
    assert!(mfcc(&[], 0).is_empty());
}

#[test]
fn features_tolerate_a_degenerate_sample_rate() {
    let cfg = StftConfig::new(64, 16);
    let spec = stft(&[0.3; 256], &cfg).expect("valid");
    for fs in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
        // The configuration would be rejected, but the feature functions take
        // a bare rate and must still return finite values.
        for v in spectral_centroid(&spec, fs) {
            assert!(v.is_finite(), "centroid {v} at fs={fs}");
        }
        for v in spectral_bandwidth(&spec, fs, &[]) {
            assert!(v.is_finite(), "bandwidth {v} at fs={fs}");
        }
        for v in spectral_rolloff(&spec, 0.9, fs) {
            assert!(v.is_finite(), "rolloff {v} at fs={fs}");
        }
    }
    // A non-finite rolloff threshold degenerates to 0 rather than NaN.
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for v in spectral_rolloff(&spec, threshold, 48_000.0) {
            assert_eq!(v, 0.0);
        }
    }
}

#[test]
fn window_coefficients_handle_every_length() {
    for window in [
        Window::Hann,
        Window::Hamming,
        Window::Blackman,
        Window::BlackmanHarris,
        Window::FlatTop,
        Window::Rectangular,
    ] {
        assert!(window.coefficients(0).is_empty());
        assert_eq!(window.coefficients(1), vec![1.0]);
        for n in [2usize, 3, 4, 5, 64, 65, 1_024] {
            let c = window.coefficients(n);
            assert_eq!(c.len(), n);
            assert!(c.iter().all(|v| v.is_finite()), "{} n={n}", window.name());
            assert!(!window.name().is_empty());
            // overlap_sum is total for every hop, including zero.
            for hop in [0usize, 1, 2, n / 2, n, 2 * n] {
                let v = window.overlap_sum(n, hop);
                assert!(
                    v.is_finite() && v >= 0.0,
                    "{} n={n} hop={hop}",
                    window.name()
                );
            }
        }
    }
    // The default window is Hann.
    assert_eq!(Window::default(), Window::Hann);
    assert!(matches!(Window::default(), Window::Hann));
    // Window is Copy + Eq so it can be used as a map key.
    let set = [Window::Hann, Window::FlatTop];
    assert_eq!(set[0], Window::Hann);
    assert_ne!(set[0], set[1]);
}

#[test]
fn mel_scale_is_total_over_its_input_domain() {
    for hz in [0.0f64, 1e-9, 1.0, 700.0, 1_000.0, 20_000.0, 1e9] {
        let m = hz_to_mel(hz);
        assert!(m.is_finite() && m >= 0.0, "mel({hz}) = {m}");
        assert!(mel_to_hz(m).is_finite());
    }
    // Frequencies above the −700 Hz log singularity stay finite and negative;
    // at exactly −700 the argument hits zero, so the result is −∞ (documented).
    assert!(hz_to_mel(-699.0).is_finite());
    assert!(hz_to_mel(-699.0) < 0.0);
    assert_eq!(hz_to_mel(-700.0), f64::NEG_INFINITY);
    assert!(mel_to_hz(-100.0).is_finite());
    // Degenerate count/spans.
    assert!(mel_frequencies(0, 100.0, 200.0).is_empty());
    assert_eq!(mel_frequencies(1, 100.0, 200.0).len(), 1);
    assert_eq!(mel_frequencies(3, 500.0, 500.0).len(), 3);
    assert_eq!(mel_frequencies(3, -100.0, 100.0).len(), 3);
}

#[test]
fn hpss_is_total_on_every_input() {
    let empty = Spectrum::default();
    for ratio in [0.0f64, 1.0, 4.0, 8.0, 1e6, f64::NAN, f64::INFINITY, -1.0] {
        let (h, p) = harmonic_percussive_split(&empty, ratio);
        assert_eq!(h.num_frames(), 0);
        assert_eq!(p.num_frames(), 0);
    }
    let cfg = StftConfig::new(64, 16);
    let spec = stft(&[0.3; 256], &cfg).expect("valid");
    for ratio in [0.0f64, 1.0, 8.0, 1e6, f64::NAN, -5.0] {
        let (h, p) = harmonic_percussive_split(&spec, ratio);
        assert_eq!(h.num_frames(), spec.num_frames());
        assert_eq!(p.num_frames(), spec.num_frames());
        assert_eq!(h.bin_count(), spec.bin_count());
        assert!(h.magnitude(0).iter().all(|v| v.is_finite()));
        assert!(p.magnitude(0).iter().all(|v| v.is_finite()));
    }
    // An uncentred spectrogram too.
    let unc = stft(&[0.3; 256], &StftConfig::new(64, 16).with_center(false)).expect("valid");
    let (h, _) = harmonic_percussive_split(&unc, 4.0);
    assert!(!h.is_centered());
    assert!(!h.magnitude(0).iter().any(|v| !v.is_finite()));
}

#[test]
fn denoise_is_total_on_short_and_silent_input() {
    let cfg = StftConfig::new(64, 16);
    for len in [1usize, 2, 7, 63, 64, 65, 1_024] {
        let x: Vec<f64> = (0..len)
            .map(|i| 0.2 * (TAU * 0.01 * i as f64).sin())
            .collect();
        let spec = stft(&x, &cfg).expect("valid");
        let profile = NoiseProfile::from_frames(&spec, &[0]);
        let out = denoise(&x, &cfg, &profile, &GateConfig::default()).expect("valid");
        assert_eq!(out.len(), len, "len {len}");
        assert!(out.iter().all(|v| v.is_finite()), "len {len}");
    }
    // A silent signal against a silent profile is a no-op, not a crash.
    let x = vec![0.0; 256];
    let profile = NoiseProfile::estimate(&x, &cfg).expect("valid");
    let out = denoise(&x, &cfg, &profile, &GateConfig::default()).expect("valid");
    assert!(out.iter().all(|v| *v == 0.0));
}

#[test]
fn window_shape_endpoints_are_as_documented() {
    // The table in the module docs, asserted.
    let cases: Vec<(Window, f64)> = vec![
        (Window::Hann, 0.0),
        (Window::Hamming, 0.08),
        (Window::Blackman, 0.0),
        (Window::BlackmanHarris, 6.0e-5),
        (Window::FlatTop, -4.210_510e-4),
        (Window::Rectangular, 1.0),
    ];
    for (window, endpoint) in cases {
        for n in [9usize, 65, 257] {
            let c = window.coefficients(n);
            let first = c[0];
            let last = c[n - 1];
            assert!(
                (first - endpoint).abs() < 1e-6,
                "{} n={n}: first {first} vs {endpoint}",
                window.name()
            );
            assert!(
                (last - endpoint).abs() < 1e-6,
                "{} n={n}: last {last} vs {endpoint}",
                window.name()
            );
            // Symmetric, by construction.
            for i in 0..n {
                assert!(
                    (c[i] - c[n - 1 - i]).abs() < 1e-15,
                    "{} i={i}",
                    window.name()
                );
            }
        }
    }
}
