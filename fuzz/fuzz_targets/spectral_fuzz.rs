//! Fuzz target — `spectral_fuzz`: arbitrary bytes become samples, a config, and
//! a noise profile, then the whole pipeline runs.
//!
//! The contract under fuzz is *totality*: every public entry point resolves to
//! a value or a typed [`SpectralError`] for any input at all — never a panic,
//! never an allocation that grows with the input, never a silent wrong
//! answer. Alongside that, three invariants are checked on whatever the fuzzer
//! produces:
//!
//! 1. Frames are internally consistent: every row has `fft_size/2 + 1` bins,
//!    and `num_frames × bin_count` describes the same data as `frames()`.
//! 2. Any reconstruction the library claims to support is length-exact:
//!    `istft` returns exactly the requested length.
//! 3. Successful analyses round-trip to f64 accuracy over the reconstructible
//!    region, so a mutation cannot make the inverse quietly wrong.
//!
//! Inputs are bounded (`MAX_SAMPLES`, the transform size) so a fuzzer run
//! cannot be turned into an OOM or an hours-long case by a single input.

#![no_main]

use dsp_spectral::{
    Complex, GateConfig, NoiseProfile, Spectrum, StftConfig, Window, denoise,
    harmonic_percussive_split, mel_frequencies, mfcc, real_cepstrum, resample_frames, spectral_bandwidth,
    spectral_centroid, spectral_flatness, spectral_flux, spectral_gate, spectral_rolloff,
    spectral_smooth, spectral_subtract, stft, zero_crossing_rate,
};
use libfuzzer_sys::fuzz_target;

/// Upper bound on the sample count decoded from an input.
const MAX_SAMPLES: usize = 4_096;
/// Upper bound on the transform size (as a power-of-two exponent shift): big
/// enough to be interesting, small enough that a case runs in milliseconds.
const MAX_FFT_SHIFT: u32 = 20;
const _: () = assert!((1usize << MAX_FFT_SHIFT) <= (1 << 20));
/// Below this, `istft` divides by a window weight that is numerically zero.
const MIN_WEIGHT: f64 = 1e-12;

const WINDOWS: [Window; 6] = [
    Window::Hann,
    Window::Hamming,
    Window::Blackman,
    Window::BlackmanHarris,
    Window::FlatTop,
    Window::Rectangular,
];

/// Decode arbitrary bytes into samples in ±1: any bit pattern is a legal
/// `f64`, but NaN and ±∞ are filtered out so the interesting paths are the
/// arithmetic ones, not the non-finite guards.
fn samples_from(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .take(MAX_SAMPLES)
        .map(|c| {
            let bits = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            let v = f64::from_bits(bits);
            if v.is_finite() {
                (v % 2.0).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// A power-of-two transform size in `[2, MAX_FFT]`.
fn fft_from(data: &[u8]) -> usize {
    let exp = u32::from(data.first().copied().unwrap_or(1) & 0x0f).clamp(1, MAX_FFT_SHIFT);
    1usize << exp
}

/// An arbitrary (usually invalid) config: the point is that `validate` catches
/// it and `stft` returns `Err` rather than dividing by a zero hop.
fn config_from(data: &[u8]) -> StftConfig {
    let fft_size = fft_from(data);
    let window = WINDOWS[data.get(1).copied().unwrap_or(0) as usize % WINDOWS.len()];
    StftConfig {
        window,
        // Half the bytes drive a raw hop, which is often too large.
        hop: usize::from(data.get(2).copied().unwrap_or(1)),
        fft_size,
        center: data.get(3).copied().unwrap_or(0) & 1 == 0,
        window_overlap: f32::from_bits(u32::from_le_bytes([
            data.get(4).copied().unwrap_or(0),
            data.get(5).copied().unwrap_or(0),
            data.get(6).copied().unwrap_or(0),
            data.get(7).copied().unwrap_or(0),
        ])),
        sample_rate: f64::from(u32::from_le_bytes([
            data.get(8).copied().unwrap_or(1),
            data.get(9).copied().unwrap_or(0),
            data.get(10).copied().unwrap_or(0),
            data.get(11).copied().unwrap_or(0),
        ])) as f64,
    }
}

/// Force a config into its valid region, for the paths that need one.
fn valid_config(data: &[u8]) -> StftConfig {
    let fft_size = fft_from(data);
    let hop_seed = data.get(1).copied().unwrap_or(1);
    StftConfig {
        window: WINDOWS[data.get(2).copied().unwrap_or(0) as usize % WINDOWS.len()],
        hop: 1 + usize::from(hop_seed) % (fft_size / 2).max(1),
        fft_size,
        center: true,
        window_overlap: 0.0,
        sample_rate: 48_000.0,
    }
}

/// Everything the library reports must be finite (or explicitly the documented
/// sentinels) — a NaN leaking into a feature is a wrong answer, not a panic.
fn assert_finite(label: &str, values: &[f64], lo: f64, hi: f64) {
    for v in values {
        assert!(v.is_finite(), "{label} produced {v}");
        assert!((lo..=hi).contains(v), "{label} produced {v}, outside {lo}..={hi}");
    }
}

/// Structural invariants that must hold for any successfully built spectrum.
fn check_spectrum(label: &str, spec: &Spectrum) {
    assert_eq!(spec.frames().len(), spec.num_frames(), "{label}: frames() length");
    let bins = spec.bin_count();
    assert_eq!(bins, spec.fft_size() / 2 + 1, "{label}: bin count");
    assert!(spec.hop() > 0, "{label}: hop");
    for f in 0..spec.num_frames() {
        let frame = spec.frame(f).expect("index from num_frames");
        assert_eq!(frame.len(), bins, "{label}: frame {f} width");
        assert!(frame.iter().all(|c| c.is_finite()), "{label}: frame {f} non-finite");
        assert_eq!(spec.magnitude(f).len(), bins, "{label}: magnitude {f}");
        assert_eq!(spec.power(f).len(), bins, "{label}: power {f}");
        // Out-of-range access is empty, never a panic.
        assert!(spec.magnitude(spec.num_frames()).is_empty(), "{label}: oob magnitude");
    }
}

/// The samples a spectrogram can actually reconstruct, derived from the
/// analysis rather than from `istft`.
fn reconstructible(spec: &Spectrum, config: &StftConfig, len: usize) -> std::ops::Range<usize> {
    let taps = config.window.coefficients(config.fft_size);
    let pad = if spec.is_centered() {
        config.fft_size / 2
    } else {
        0
    };
    let mut weight = vec![0.0f64; len];
    for f in 0..spec.num_frames() {
        for j in 0..config.fft_size {
            if let Some(out_idx) = (f * spec.hop() + j).checked_sub(pad) {
                if out_idx < len {
                    weight[out_idx] += taps[j] * taps[j];
                }
            }
        }
    }
    let usable = |i: usize| weight[i] > MIN_WEIGHT;
    let start = (0..len).find(|i| usable(*i)).unwrap_or(len);
    let end = (start..len).rev().find(|i| usable(*i)).map_or(start, |i| i + 1);
    start..end
}

fuzz_target!(|data: &[u8]| {
    // ---- Arbitrary config: must be a typed error, never a panic. ----
    let arbitrary = config_from(data);
    let samples = samples_from(data);
    let _ = stft(&samples, &arbitrary);
    let _ = arbitrary.validate();

    if samples.is_empty() {
        return;
    }

    // ---- Valid config: run the whole pipeline. ----
    let built = valid_config(data);
    let cfg = built;
    if cfg.validate().is_err() {
        return;
    }
    let Ok(spec) = stft(&samples, &cfg) else {
        return;
    };
    check_spectrum("stft", &spec);

    // Features: bounded and finite.
    let fs = cfg.sample_rate;
    let nyquist = fs / 2.0;
    let centroids = spectral_centroid(&spec, fs);
    assert_eq!(centroids.len(), spec.num_frames(), "centroid count");
    assert_finite("centroid", &centroids, 0.0, nyquist);
    let bandwidths = spectral_bandwidth(&spec, fs, &centroids);
    assert_finite("bandwidth", &bandwidths, 0.0, nyquist.max(1.0));
    let flatness = spectral_flatness(&spec);
    assert_finite("flatness", &flatness, 0.0, 1.0);
    for threshold in [0.1f64, 0.5, 0.9, 1.0] {
        let rolloff = spectral_rolloff(&spec, threshold, fs);
        assert_finite("rolloff", &rolloff, 0.0, nyquist);
    }
    let flux = spectral_flux(&spec);
    assert_finite("flux", &flux, 0.0, f64::MAX);
    assert_eq!(flux.first().copied(), Some(0.0), "flux[0] is zero");
    let zcr = zero_crossing_rate(&samples);
    assert!((0.0..=1.0).contains(&zcr), "zcr {zcr}");

    // Reconstruct where the analysis supports it.
    let back = dsp_spectral::istft(&spec, samples.len());
    assert_eq!(back.len(), samples.len(), "istft length");
    assert!(back.iter().all(|v| v.is_finite()), "istft non-finite");
    let range = reconstructible(&spec, &cfg, samples.len());
    if !range.is_empty() {
        let err = samples[range.clone()]
            .iter()
            .zip(back[range.clone()].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            err < 1e-9 * 2.0,
            "round-trip error {err:e} for fft {} hop {} {}",
            cfg.fft_size,
            cfg.hop,
            cfg.window.name()
        );
    }

    // Restoration: every entry point is total, and each result is a
    // well-formed spectrum.
    let profile = NoiseProfile::from_frames(&spec, &[0]);
    if let Ok(profiled) = NoiseProfile::estimate(&samples, &cfg) {
        assert_eq!(profiled.spectrum().len(), spec.bin_count(), "profile bins");
        assert!(profiled.spectrum().iter().all(|v| v.is_finite()), "profile non-finite");
    }
    for gate in [
        GateConfig::default(),
        GateConfig {
            threshold_db: -12.0,
            reduction_db: -6.0,
            smoothing_frames: 0,
        },
        GateConfig {
            threshold_db: 12.0,
            reduction_db: -24.0,
            smoothing_frames: 3,
        },
    ] {
        if let Ok(gated) = spectral_gate(&spec, &profile, &gate) {
            check_spectrum("gate", &gated);
        }
        let _ = denoise(&samples, &cfg, &profile, &gate);
    }
    for alpha in [0.0f64, 1.0, 2.0] {
        if let Ok(subtracted) = spectral_subtract(&spec, &profile, alpha) {
            check_spectrum("subtract", &subtracted);
            // Subtraction can only reduce magnitude.
            for f in 0..spec.num_frames() {
                for (k, (&a, &b)) in spec
                    .magnitude(f)
                    .iter()
                    .zip(subtracted.magnitude(f).iter())
                    .enumerate()
                {
                    assert!(b <= a + 1e-9, "subtraction raised bin {k} of frame {f}");
                }
            }
        }
    }
    for kernel in [vec![1.0], vec![0.25, 0.5, 0.25], vec![1.0; 5]] {
        if let Ok(smoothed) = spectral_smooth(&spec, &kernel) {
            check_spectrum("smooth", &smoothed);
        }
    }
    for ratio in [1.0f64, 4.0, 8.0] {
        let (harm, perc) = harmonic_percussive_split(&spec, ratio);
        check_spectrum("hpss-harmonic", &harm);
        check_spectrum("hpss-percussive", &perc);
        // The complementary mask means the magnitudes sum back exactly.
        for f in 0..spec.num_frames() {
            let a = spec.magnitude(f);
            let b = harm.magnitude(f);
            let c = perc.magnitude(f);
            for k in 0..spec.bin_count() {
                assert!(
                    ((b[k] + c[k]) - a[k]).abs() < 1e-6 * a[k].max(1.0),
                    "hpss mask at {f},{k}: {} + {} vs {}",
                    b[k],
                    c[k],
                    a[k]
                );
            }
        }
    }
    if let Ok(rescaled) = resample_frames(&spec, 1.5) {
        check_spectrum("resample", &rescaled);
        assert_eq!(rescaled.num_frames(), spec.num_frames(), "resample frames");
    }

    // Mel / cepstral front end.
    let n_mels = 1 + usize::from(data.get(12).copied().unwrap_or(0) % 40);
    let bank = spec.mel_filterbank(n_mels, 0.0, fs / 2.0, fs);
    assert_eq!(bank.n_mels(), n_mels, "mel bands");
    assert_eq!(bank.bin_count(), spec.bin_count(), "mel bin count");
    let centres = mel_frequencies(n_mels, 0.0, fs / 2.0);
    assert_eq!(centres.len(), n_mels, "mel centres");
    for frame in spec.frames().iter().take(8) {
        let energies = bank.energies(frame);
        assert_eq!(energies.len(), n_mels, "mel energies");
        assert!(energies.iter().all(|v| v.is_finite() && *v >= 0.0), "mel energy");
        // The bank tiles the band, so its energies sum to the frame's power.
        let total: f64 = energies.iter().sum();
        let power: f64 = frame.iter().map(|c| c.power()).sum();
        assert!(
            (total - power).abs() < 1e-6 * power.max(1.0),
            "mel energy conservation: {total} vs {power}"
        );
        for coeffs in [mfcc(&energies, 13), mfcc(&energies, 1)] {
            assert!(coeffs.iter().all(|v| v.is_finite()), "mfcc non-finite");
        }
        assert!(bank.energies_db(frame, -200.0).iter().all(|v| v.is_finite()));
    }

    // The cepstrum only accepts power-of-two lengths >= 2, and must be total
    // for those. Pad/truncate to the next power of two.
    let n = samples.len().next_power_of_two().max(2);
    let padded: Vec<f64> = (0..n).map(|i| samples[i % samples.len()]).collect();
    match real_cepstrum(&padded) {
        Ok(c) => {
            assert_eq!(c.len(), n, "cepstrum length");
            assert!(c.iter().all(|v| v.is_finite()), "cepstrum non-finite");
        }
        Err(e) => panic!("cepstrum rejected a power-of-two length: {e}"),
    }

    // Hand-built spectra must validate their shape rather than trusting it.
    let bogus = Spectrum::from_parts(
        vec![vec![Complex::new(1.0, 2.0); spec.bin_count() + 1]],
        1,
        cfg.fft_size,
        cfg.window,
        cfg.center,
    );
    assert!(bogus.is_err(), "from_parts accepted a ragged frame");
});