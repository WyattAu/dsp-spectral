#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    missing_docs
)]
//! Analysis–synthesis: the round-trip is the contract this crate rests on.
//!
//! These are the tests that would catch the class of bug that is invisible
//! in a spectrogram plot — a synthesis window applied once instead of twice,
//! a missing `1/N`, a hop-phase error in the overlap-add — because they
//! measure the *waveform*, not its picture.

use core::f64::consts::TAU;

use dsp_spectral::{istft, stft, StftConfig, Window};

/// Deterministic LCG so every failure is reproducible.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_845_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn white(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = lcg(seed);
    (0..n).map(|_| rng()).collect()
}

/// Largest absolute sample error over a pair of buffers.
fn worst_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

const WINDOWS: [Window; 6] = [
    Window::Hann,
    Window::Hamming,
    Window::Blackman,
    Window::BlackmanHarris,
    Window::FlatTop,
    Window::Rectangular,
];

/// The critical test: ISTFT(STFT(x)) reconstructs white noise to 1e-10, for
/// every window and several transform/hop combinations.
#[test]
fn roundtrip_white_noise_is_exact() {
    for window in WINDOWS {
        for (fft_size, hop) in [
            (2048usize, 512usize),
            (1024, 256),
            (256, 64),
            (64, 16),
            (64, 1),
            (8, 4),
        ] {
            let cfg = StftConfig::new(fft_size, hop).with_window(window);
            let x = white(8_192, 0xA11CE);
            let spec = stft(&x, &cfg).expect("valid config");
            let y = istft(&spec, x.len());
            let err = worst_err(&x, &y);
            assert!(
                err < 1e-10,
                "{} n={fft_size} hop={hop}: round-trip error {err:e}",
                window.name()
            );
        }
    }
}

#[test]
fn roundtrip_holds_for_hann_fft2048_defaults() {
    let cfg = StftConfig::hann_fft2048();
    assert_eq!(cfg.fft_size, 2048);
    assert_eq!(cfg.hop, 512);
    let x = white(44_100, 7);
    let y = istft(&stft(&x, &cfg).expect("valid"), x.len());
    let err = worst_err(&x, &y);
    assert!(err < 1e-12, "error {err:e}");
}

#[test]
fn roundtrip_holds_for_musical_content() {
    // A mix of a chord, a bass line, and a click train — content with real
    // spectral structure, so any phase or windowing error shows up.
    let fs = 48_000.0;
    let n = 16_384usize;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let chord =
                (TAU * 220.0 * t).sin() + (TAU * 277.18 * t).sin() + (TAU * 329.63 * t).sin();
            let bass = 0.6 * (TAU * 55.0 * t).sin();
            let click = if i % 1_024 == 0 { 0.8 } else { 0.0 };
            0.2 * chord + bass + click
        })
        .collect();
    for window in WINDOWS {
        let cfg = StftConfig::new(2048, 512).with_window(window);
        let y = istft(&stft(&x, &cfg).expect("valid"), x.len());
        let err = worst_err(&x, &y);
        assert!(err < 1e-10, "{}: error {err:e}", window.name());
    }
}

#[test]
fn roundtrip_holds_at_every_signal_length() {
    // Frame counts change discontinuously at multiples of the hop, and the
    // centre-padding math is easy to get wrong for the very short cases where
    // the reflected padding wraps more than once.
    let cfg = StftConfig::new(256, 64);
    for len in [
        1usize, 2, 3, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000,
    ] {
        let x = white(len, len as u64 + 1);
        let spec = stft(&x, &cfg).expect("valid");
        let y = istft(&spec, len);
        assert_eq!(y.len(), len, "len {len}");
        let err = worst_err(&x, &y);
        assert!(err < 1e-10, "len {len}: error {err:e}");
    }
}

#[test]
fn roundtrip_holds_for_large_amplitude_and_dc_offset() {
    // Linear paths must not lose precision to scaling: a full-scale signal
    // with a large DC offset is the worst case for the `1/N` normalisation.
    let cfg = StftConfig::new(512, 128);
    for scale in [1e-8f64, 1.0, 1e3, 1e6] {
        let x: Vec<f64> = (0..4096)
            .map(|i| scale * (0.9 * (TAU * 3.0 * i as f64 / 4096.0).sin() + 0.1))
            .collect();
        let y = istft(&stft(&x, &cfg).expect("valid"), x.len());
        let err = worst_err(&x, &y);
        // Tolerance scales with the signal: what matters is that the relative
        // error stays at the f64 noise floor.
        assert!(err < 1e-12 * scale, "scale {scale:e}: error {err:e}");
    }
}

#[test]
fn roundtrip_is_exact_for_silence() {
    let cfg = StftConfig::new(1024, 256);
    let x = vec![0.0; 4096];
    let y = istft(&stft(&x, &cfg).expect("valid"), x.len());
    assert_eq!(y.len(), 4096);
    assert!(
        y.iter().all(|v| *v == 0.0),
        "silence should stay exactly silent"
    );
}

#[test]
fn parseval_holds_for_every_frame() {
    // Parseval on the *windowed* frame: `Σ_n (x·w)² = (1/N)·Σ_k |X[k]|²`,
    // which is the identity that pins the FFT's unnormalised forward / 1/N
    // inverse convention. Getting it backwards by a factor of N is the
    // classic way to be off by 20·log10(N) dB on every spectral magnitude.
    let n = 512usize;
    let hop = n / 4;
    let cfg = StftConfig::new(n, hop).with_center(false);
    let x = white(n * 4, 99);
    let spec = stft(&x, &cfg).expect("valid");
    let w = Window::Hann.coefficients(n);
    for f in 0..spec.num_frames() {
        let start = f * hop;
        let time_energy: f64 = (0..n)
            .map(|j| {
                let v = x.get(start + j).copied().unwrap_or(0.0) * w[j];
                v * v
            })
            .sum();
        // The spectrum is stored as a half-spectrum (DC..Nyquist), so the
        // full Parseval sum is `2·Σ_half − DC − Nyquist` before dividing by N.
        let freq_energy = full_spectrum_power(&spec, f) / (n as f64);
        assert!(
            (time_energy - freq_energy).abs() < 1e-9 * time_energy.max(1.0),
            "frame {f}: {time_energy} vs {freq_energy}"
        );
    }
}

#[test]
fn parseval_holds_without_a_window_too() {
    // With a rectangular window the frame's spectrum is exactly the DFT of the
    // raw samples, so the identity is textbook: `Σ x² = (1/N)·Σ|X|²`.
    let n = 256usize;
    let cfg = StftConfig::new(n, n / 4)
        .with_center(false)
        .with_window(Window::Rectangular);
    let x = white(n * 4, 11);
    let spec = stft(&x, &cfg).expect("valid");
    for f in 0..spec.num_frames() {
        let time_energy: f64 = (0..n).map(|j| x[f * (n / 4) + j].powi(2)).sum();
        let freq_energy = full_spectrum_power(&spec, f) / (n as f64);
        assert!(
            (time_energy - freq_energy).abs() < 1e-10 * time_energy.max(1.0),
            "frame {f}: {time_energy} vs {freq_energy}"
        );
    }
}

/// `Σ |X[k]|²` over the *full* spectrum, reconstructed from the stored
/// half-spectrum (`DC..Nyquist`) by mirroring — DC and Nyquist count once.
fn full_spectrum_power(spec: &dsp_spectral::Spectrum, frame: usize) -> f64 {
    let half = spec.power(frame);
    let inner = half
        .iter()
        .skip(1)
        .take(half.len().saturating_sub(2))
        .sum::<f64>();
    2.0 * inner + half.first().copied().unwrap_or(0.0) + half.last().copied().unwrap_or(0.0)
}

#[test]
fn roundtrip_with_uncentred_analysis_recovers_the_interior() {
    // Without centre padding the first and last samples of a zero-tapered
    // window are genuinely unrecoverable — but the interior must be exact,
    // and the recoverable endpoints of a non-zero-tapered window must be too.
    let n = 256usize;
    let x = white(n * 4, 5);
    for (window, exact_edges) in [
        (Window::Rectangular, true),
        (Window::Hamming, true),
        (Window::BlackmanHarris, true),
        (Window::FlatTop, true),
        (Window::Hann, false),
        (Window::Blackman, false),
    ] {
        let cfg = StftConfig::new(n, n / 4)
            .with_window(window)
            .with_center(false);
        let y = istft(&stft(&x, &cfg).expect("valid"), x.len());
        let interior = worst_err(&x[n..x.len() - n], &y[n..x.len() - n]);
        assert!(interior < 1e-10, "{} interior: {interior:e}", window.name());
        let edges =
            worst_err(&x[..n], &y[..n]).max(worst_err(&x[x.len() - n..], &y[x.len() - n..]));
        if exact_edges {
            assert!(edges < 1e-10, "{} edges: {edges:e}", window.name());
        } else {
            // A zero end-tap means the analysis genuinely carries no
            // information there; the output must be exact zero, not garbage.
            assert_eq!(y[0], 0.0, "{} first sample", window.name());
            assert_eq!(y[x.len() - 1], 0.0, "{} last sample", window.name());
        }
    }
}

#[test]
fn roundtrip_survives_identity_transformations() {
    // The framing-preserving transformations must compose with the inverse:
    // anything that leaves the bins alone (zero over-subtraction, an identity
    // smoothing kernel, a gate that never fires) has to come back to the
    // input, and the HPSS pair has to recombine through its own mask.
    use dsp_spectral::{harmonic_percussive_split, snr_db, spectral_smooth, spectral_subtract};
    let cfg = StftConfig::new(1024, 256);
    let x = white(8_192, 3);
    let spec = stft(&x, &cfg).expect("valid");
    let profile = dsp_spectral::NoiseProfile::from_frames(&spec, &[0, 1, 2]);

    let subbed = spectral_subtract(&spec, &profile, 0.0).expect("valid");
    assert_eq!(subbed, spec, "zero over-subtraction must be the identity");
    let smoothed = spectral_smooth(&spec, &[1.0]).expect("valid");

    // A gate whose threshold sits below every bin's headroom also never fires.
    let noop = dsp_spectral::spectral_gate(
        &spec,
        &profile,
        &dsp_spectral::GateConfig {
            threshold_db: -200.0,
            reduction_db: -12.0,
            smoothing_frames: 0,
        },
    )
    .expect("valid");
    assert_eq!(noop, spec);

    // HPSS splits rather than preserves, so check the complementary
    // recombination: |harmonic| + |percussive| == |input| at every bin.
    let (harm, perc) = harmonic_percussive_split(&spec, 4.0);
    for f in 0..spec.num_frames() {
        let a = spec.magnitude(f);
        let b = harm.magnitude(f);
        let c = perc.magnitude(f);
        for k in 0..spec.bin_count() {
            assert!(
                ((b[k] + c[k]) - a[k]).abs() < 1e-9 * a[k].max(1.0),
                "frame {f} bin {k}"
            );
        }
    }

    for (label, s) in [
        ("subtract", &subbed),
        ("smooth", &smoothed),
        ("noop gate", &noop),
    ] {
        let y = istft(s, x.len());
        let err = worst_err(&x, &y);
        assert!(err < 1e-10, "{label}: error {err:e}");
        assert!(snr_db(&x, &y) > 200.0, "{label}: snr {}", snr_db(&x, &y));
    }
}
