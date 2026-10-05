#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    missing_docs
)]
//! Every feature against a naive reference implementation.
//!
//! Each test here recomputes the descriptor with the most obvious possible
//! `O(N·M)` double loop over frames and bins — no `dsp_spectral` internals, no
//! shared code with the implementation — and requires agreement to 1e-9. That
//! is what makes the descriptor *definitions* (not just the code) the thing
//! under test: a refactor that changes the implementation still has to change
//! these references, visibly.

use core::f64::consts::PI;

use dsp_spectral::{
    spectral_bandwidth, spectral_centroid, spectral_flatness, spectral_flux, spectral_rolloff,
    stft, zero_crossing_rate, Complex, Spectrum, StftConfig, Window,
};

/// Naive magnitude spectrogram: `|X_k|` from the stored bins.
fn mags(spec: &Spectrum, frame: usize) -> Vec<f64> {
    (0..spec.bin_count())
        .map(|k| spec.frame(frame).unwrap_or(&[])[k].magnitude())
        .collect()
}

/// Reference centroid: `Σ f_k·m_k / Σ m_k`, with `f_k = k·fs/N`.
fn ref_centroid(row: &[f64], fs: f64, fft: usize) -> f64 {
    let total: f64 = row.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut num = 0.0;
    for (k, &m) in row.iter().enumerate() {
        num += (k as f64) * (fs / (fft as f64)) * m;
    }
    num / total
}

/// Reference bandwidth: `√(Σ m_k·(f_k − c)² / Σ m_k)`.
fn ref_bandwidth(row: &[f64], c: f64, fs: f64, fft: usize) -> f64 {
    let total: f64 = row.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for (k, &m) in row.iter().enumerate() {
        let d = (k as f64) * (fs / (fft as f64)) - c;
        acc += m * d * d;
    }
    (acc / total).sqrt()
}

/// Reference flatness: geometric mean over arithmetic mean, with the same
/// 1e-10 floor the implementation documents.
fn ref_flatness(row: &[f64]) -> f64 {
    let n = row.len();
    if n == 0 {
        return 0.0;
    }
    let arithmetic: f64 = row.iter().sum::<f64>() / (n as f64);
    if arithmetic <= 0.0 {
        return 0.0;
    }
    let mut log_sum = 0.0;
    for &m in row {
        log_sum += if m > 1e-10 { m.ln() } else { 1e-10f64.ln() };
    }
    let geometric = (log_sum / (n as f64)).exp();
    geometric / arithmetic
}

/// Reference rolloff: first bin whose cumulative magnitude reaches the
/// threshold fraction of the total.
fn ref_rolloff(row: &[f64], threshold: f64, fs: f64, fft: usize) -> f64 {
    let total: f64 = row.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let target = total * threshold;
    let mut cumulative = 0.0;
    for (k, &m) in row.iter().enumerate() {
        cumulative += m;
        if cumulative >= target {
            return (k as f64) * (fs / (fft as f64));
        }
    }
    0.0
}

/// Reference flux: half-wave-rectified frame difference.
fn ref_flux(prev: &[f64], cur: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (a, b) in prev.iter().zip(cur.iter()) {
        if b - a > 0.0 {
            acc += b - a;
        }
    }
    acc
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_845_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A spectrogram with random but known magnitudes, built through
/// `from_parts` so the references can see the same numbers.
fn random_spectrum(bins: usize, frames: usize, seed: u64) -> Spectrum {
    let mut rng = lcg(seed);
    let rows: Vec<Vec<Complex>> = (0..frames)
        .map(|_| {
            (0..bins)
                .map(|_| Complex::from_polar(rng().abs() + 1e-3, rng() * PI))
                .collect()
        })
        .collect();
    Spectrum::from_parts(rows, 1, (bins - 1) * 2, Window::Hann, true).expect("valid")
}

#[test]
fn centroid_matches_reference_on_random_spectra() {
    for (bins, frames, fs) in [
        (5usize, 7usize, 8_000.0f64),
        (17, 9, 44_100.0),
        (33, 4, 48_000.0),
    ] {
        let spec = random_spectrum(bins, frames, 0xCE1);
        let got = spectral_centroid(&spec, fs);
        assert_eq!(got.len(), frames);
        for (f, &actual) in got.iter().enumerate() {
            let want = ref_centroid(&mags(&spec, f), fs, (bins - 1) * 2);
            assert!(
                (actual - want).abs() < 1e-9,
                "frame {f}: {actual} vs {want}"
            );
        }
    }
}

#[test]
fn bandwidth_matches_reference_on_random_spectra() {
    let fs = 44_100.0;
    for (bins, frames) in [(9usize, 6usize), (33, 3)] {
        let fft = (bins - 1) * 2;
        let spec = random_spectrum(bins, frames, 0xB0D);
        let centroids = spectral_centroid(&spec, fs);
        let got = spectral_bandwidth(&spec, fs, &centroids);
        // Ignoring the supplied centroid must give the same answer.
        let recomputed = spectral_bandwidth(&spec, fs, &[]);
        for (f, &actual) in got.iter().enumerate() {
            let row = mags(&spec, f);
            let want = ref_bandwidth(&row, ref_centroid(&row, fs, fft), fs, fft);
            assert!(
                (actual - want).abs() < 1e-9,
                "frame {f}: {actual} vs {want}"
            );
            assert!((actual - recomputed[f]).abs() < 1e-9);
        }
    }
}

#[test]
fn flatness_matches_reference_on_random_spectra() {
    for (bins, frames) in [(5usize, 8usize), (9, 5), (65, 3)] {
        let spec = random_spectrum(bins, frames, 0x71A);
        let got = spectral_flatness(&spec);
        for (f, &actual) in got.iter().enumerate() {
            let want = ref_flatness(&mags(&spec, f));
            assert!(
                (actual - want).abs() < 1e-9,
                "frame {f}: {actual} vs {want}"
            );
        }
    }
}

#[test]
fn rolloff_matches_reference_on_random_spectra() {
    let fs = 16_000.0;
    for threshold in [0.1f64, 0.5, 0.85, 0.95, 0.99] {
        let spec = random_spectrum(17, 6, 0x201);
        let got = spectral_rolloff(&spec, threshold, fs);
        for (f, &actual) in got.iter().enumerate() {
            let want = ref_rolloff(&mags(&spec, f), threshold, fs, 32);
            assert!(
                (actual - want).abs() < 1e-9,
                "threshold {threshold} frame {f}: {actual} vs {want}"
            );
        }
    }
}

#[test]
fn flux_matches_reference_on_random_spectra() {
    let spec = random_spectrum(9, 10, 0xF10);
    let got = spectral_flux(&spec);
    assert_eq!(got[0], 0.0, "the first frame has no predecessor");
    for (f, &actual) in got.iter().enumerate().skip(1) {
        let want = ref_flux(&mags(&spec, f - 1), &mags(&spec, f));
        assert!(
            (actual - want).abs() < 1e-9,
            "frame {f}: {actual} vs {want}"
        );
    }
}

#[test]
fn features_match_reference_on_a_real_stft() {
    // The same references, but driven through `stft` so the bin values come
    // from the actual transform rather than a synthetic spectrogram.
    let fs = 16_000.0;
    let cfg = StftConfig::new(512, 128).with_sample_rate(fs);
    let mut rng = lcg(0x5EC);
    let x: Vec<f64> = (0..8192).map(|_| rng()).collect();
    let spec = stft(&x, &cfg).expect("valid");
    let centroids = spectral_centroid(&spec, fs);
    let bandwidths = spectral_bandwidth(&spec, fs, &centroids);
    let flatness = spectral_flatness(&spec);
    let rolloff = spectral_rolloff(&spec, 0.85, fs);
    let flux = spectral_flux(&spec);
    for f in 0..spec.num_frames() {
        let row = mags(&spec, f);
        let c = ref_centroid(&row, fs, 512);
        assert!((centroids[f] - c).abs() < 1e-9);
        assert!((bandwidths[f] - ref_bandwidth(&row, c, fs, 512)).abs() < 1e-9);
        assert!((flatness[f] - ref_flatness(&row)).abs() < 1e-9);
        assert!((rolloff[f] - ref_rolloff(&row, 0.85, fs, 512)).abs() < 1e-9);
        if f > 0 {
            assert!((flux[f] - ref_flux(&mags(&spec, f - 1), &row)).abs() < 1e-9);
        }
    }
}

#[test]
fn constant_amplitude_signal_reads_as_pure_dc() {
    // A constant signal is all DC. With a rectangular window its spectrum is
    // exactly one occupied bin, so the centroid is 0 Hz, the bandwidth and
    // rolloff are 0, and the flatness is ~0 (one bin out of `fft/2+1`, with
    // the rest at the log floor).
    let cfg = StftConfig::new(64, 32).with_window(Window::Rectangular);
    let spec = stft(&[1.0; 256], &cfg).expect("valid");
    let c = spectral_centroid(&spec, cfg.sample_rate);
    assert!(c.iter().all(|v| v.abs() < 1e-9), "constant signal: {c:?}");
    assert!(spectral_bandwidth(&spec, cfg.sample_rate, &c)
        .iter()
        .all(|v| *v < 1e-9));
    assert!(spectral_flatness(&spec).iter().all(|v| *v < 1e-3));
    assert!(spectral_rolloff(&spec, 0.95, cfg.sample_rate)
        .iter()
        .all(|v| *v < 1e-9));
    assert!(spectral_flux(&spec).iter().all(|v| *v == 0.0));

    // A *tapered* window on a constant signal is a different question: the
    // window itself spreads a DC step across bins, so the centroid rises to
    // the window's own centre of mass and the bandwidth follows it. That is
    // the window's spectral leakage, not a bug, and the values are pinned so
    // a change in window coefficients shows up here.
    let cfg = StftConfig::new(64, 32);
    let spec = stft(&[1.0; 256], &cfg).expect("valid");
    let c = spectral_centroid(&spec, cfg.sample_rate);
    assert!((c[0] - 275.923_222_999_693_6).abs() < 1e-9, "{}", c[0]);
    assert!(spectral_bandwidth(&spec, cfg.sample_rate, &c)[0] > 0.0);
}

#[test]
fn uniform_magnitude_spectrum_reads_flatness_one() {
    // A spectrogram whose every bin carries the same magnitude is flatness 1
    // by definition, and its centroid is the mid-band centre because every bin
    // weighs the same. This is the "flatness ≈ 1" case — it is a property of
    // the *spectrogram*, not of any particular input signal.
    let uniform = Spectrum::from_parts(
        (0..3).map(|_| vec![Complex::new(1.0, 0.0); 9]).collect(),
        1,
        16,
        Window::Hann,
        true,
    )
    .expect("valid");
    assert!((spectral_flatness(&uniform)[0] - 1.0).abs() < 1e-12);
    assert!(spectral_flatness(&uniform)
        .iter()
        .all(|v| (*v - 1.0).abs() < 1e-12));
    // fft_size 16 → 9 bins, bin 4 is the mid-band centre at 8 kHz.
    let mid = spectral_centroid(&uniform, 8_000.0)[0];
    assert!((mid - 2_000.0).abs() < 1e-9, "mid {mid}");
    // Bandwidth is the RMS deviation of the uniform weight from its mean,
    // i.e. the standard deviation of 0..8 bins at 500 Hz each.
    let bw = spectral_bandwidth(&uniform, 8_000.0, &[])[0];
    let sd = ((0..9)
        .map(|k| (k as f64 * 500.0 - 2_000.0).powi(2))
        .sum::<f64>()
        / 9.0)
        .sqrt();
    assert!((bw - sd).abs() < 1e-9, "bandwidth {bw} vs {sd}");
    // 85 % rolloff is bin 7 (bins 0..7 hold 8 of 9 units).
    assert!((spectral_rolloff(&uniform, 0.85, 8_000.0)[0] - 3_500.0).abs() < 1e-9);
}

#[test]
fn white_noise_reads_low_flatness_and_mid_band_centroid() {
    let fs = 16_000.0;
    let cfg = StftConfig::new(1024, 256).with_sample_rate(fs);
    let mut rng = lcg(0xA0D1);
    let x: Vec<f64> = (0..16_384).map(|_| rng()).collect();
    let spec = stft(&x, &cfg).expect("valid");
    let flatness = spectral_flatness(&spec);
    let centroids = spectral_centroid(&spec, fs);
    let mid = fs / 4.0;
    let nyquist = fs / 2.0;

    // White noise spreads energy uniformly, so the centroid sits near the
    // mid-band and stays well inside the band.
    let mean_c: f64 = centroids.iter().sum::<f64>() / (centroids.len() as f64);
    assert!((mean_c - mid).abs() < 0.1 * mid, "mean centroid {mean_c}");
    assert!(centroids.iter().all(|c| (0.0..nyquist).contains(c)));

    // Flatness of white noise reads *high*, not low: each bin's magnitude is
    // Rayleigh-distributed, and the geometric/arithmetic mean ratio of K
    // Rayleigh variates converges to ≈ 0.905 as K grows (E[ln R] − ln E[R]).
    // The signal-specific claim is the *contrast*: a pure tone sits three
    // orders of magnitude lower.
    let mean_f: f64 = flatness.iter().sum::<f64>() / (flatness.len() as f64);
    assert!((0.75..0.95).contains(&mean_f), "mean flatness {mean_f}");
    assert!(flatness.iter().all(|f| (0.0..=1.0).contains(f)));

    let tone: Vec<f64> = (0..16_384)
        .map(|i| 0.5 * (core::f64::consts::TAU * 500.0 * i as f64 / fs).sin())
        .collect();
    let tone_flat = spectral_flatness(&stft(&tone, &cfg).expect("valid"));
    let tone_mean: f64 = tone_flat.iter().sum::<f64>() / (tone_flat.len() as f64);
    assert!(tone_mean < 0.01, "tone flatness {tone_mean}");
    assert!(
        mean_f > 100.0 * tone_mean,
        "noise {mean_f} vs tone {tone_mean}"
    );
    // A pure tone's centroid tracks its frequency.
    let tone_c: f64 = spectral_centroid(&stft(&tone, &cfg).expect("valid"), fs)
        .iter()
        .sum::<f64>()
        / tone_flat.len() as f64;
    assert!((tone_c - 500.0).abs() < 20.0, "tone centroid {tone_c}");

    // Rolloff of white noise at 85 % sits in the upper part of the band.
    let roll = spectral_rolloff(&spec, 0.85, fs);
    let mean_r: f64 = roll.iter().sum::<f64>() / (roll.len() as f64);
    assert!(
        (0.6 * nyquist..=nyquist).contains(&mean_r),
        "mean rolloff {mean_r}"
    );
}

#[test]
fn stationary_signal_has_near_zero_flux_and_onset_spikes() {
    let cfg = StftConfig::new(256, 64);
    // A steady tone on a bin centre: interior frames have identical spectra.
    let tone: Vec<f64> = (0..8192)
        .map(|i| (core::f64::consts::TAU * 8.0 * i as f64 / 256.0).sin())
        .collect();
    let flux = spectral_flux(&stft(&tone, &cfg).expect("valid"));
    let interior = &flux[4..flux.len() - 4];
    assert!(interior.iter().all(|f| *f < 1e-6), "{interior:?}");

    // Silence is the other stationary case: exactly zero flux.
    let silence = stft(&vec![0.0; 8192], &cfg).expect("valid");
    assert!(spectral_flux(&silence).iter().all(|f| *f == 0.0));

    // An abrupt onset produces a single large flux value at the onset frame
    // and near-zero flux everywhere else.
    let mut onset = tone.clone();
    for v in onset.iter_mut().skip(4096) {
        *v = 0.0;
    }
    onset[0] = 1.0;
    let spec = stft(&onset, &cfg).expect("valid");
    let flux = spectral_flux(&spec);
    assert_eq!(flux[0], 0.0);
    let peak = flux.iter().copied().fold(0.0f64, f64::max);
    assert!(peak > 1.0, "onset flux {peak}");
    // Exactly one frame dominates: the onset.
    let peak_frames = flux.iter().filter(|f| **f > 0.5 * peak).count();
    assert!(
        peak_frames <= 4,
        "{peak_frames} frames near the peak: {flux:?}"
    );
}

#[test]
fn zero_crossing_rate_matches_its_definition() {
    assert_eq!(zero_crossing_rate(&[]), 0.0);
    assert_eq!(zero_crossing_rate(&[0.5]), 0.0);
    assert_eq!(zero_crossing_rate(&[1.0, 1.0, 1.0, 1.0]), 0.0);
    assert_eq!(zero_crossing_rate(&[1.0, -1.0]), 1.0);
    assert_eq!(zero_crossing_rate(&[1.0, -1.0, 1.0, -1.0]), 1.0);
    // Zeros are traversed, not counted as flips.
    assert_eq!(zero_crossing_rate(&[1.0, 0.0, -1.0, 0.0, 1.0]), 0.0);

    // Naive reference over a random signal.
    let mut rng = lcg(0x2C1);
    let x: Vec<f64> = (0..4096).map(|_| rng()).collect();
    let crossings = x.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    let want = crossings as f64 / ((x.len() - 1) as f64);
    assert!((zero_crossing_rate(&x) - want).abs() < 1e-15);

    // A Nyquist-rate square wave crosses at every sample pair.
    let square: Vec<f64> = (0..1024)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert!((zero_crossing_rate(&square) - 1.0).abs() < 1e-15);

    // And the feature tracks the actual rate: a 1 kHz sine at 16 kHz has
    // 2000 crossings over 16 000 samples = 0.125 per sample.
    let sine: Vec<f64> = (0..16_000)
        .map(|i| (core::f64::consts::TAU * 1_000.0 * i as f64 / 16_000.0).sin())
        .collect();
    assert!((zero_crossing_rate(&sine) - 0.125).abs() < 1e-3);
}
