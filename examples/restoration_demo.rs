#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs
)]
// The demo is a linear script over a signal it just synthesised: every
// `expect` guards a call whose inputs this file controls (a valid config, a
// non-empty buffer), and an index is a literal or a checked range. The
// crate-level denies are lifted here only, and only because a demo that
// aborts with a message beats one that threads Results through 200 lines of
// printing. The library itself holds the denies.

//! End-to-end demonstration of the `dsp-spectral` restoration chain.
//!
//! Synthesises a test signal — a musical tone, broadband room noise, and a
//! pair of clicks — profiles the noise from a leading *noise-only* segment,
//! then runs the restoration primitives in turn and reports what each one did:
//!
//! 1. noise profiling + spectral gating (the headline SNR number),
//! 2. spectral subtraction (how far can we push it?),
//! 3. harmonic/percussive separation (what survives in each part?),
//! 4. the spectral descriptors of the result.
//!
//! Run with `cargo run --example restoration_demo`.

use core::f64::consts::TAU;

use dsp_spectral::{
    denoise, harmonic_percussive_split, mel_frequencies, mfcc, snr_db, spectral_bandwidth,
    spectral_centroid, spectral_flatness, spectral_flux, spectral_rolloff, spectral_subtract, stft,
    zero_crossing_rate, GateConfig, NoiseProfile, StftConfig, Window,
};

const FS: f64 = 16_000.0;
const FFT: usize = 1024;
const HOP: usize = 256;

fn main() {
    let mut rng = Lcg::new(0x5EED_1234_ABCD_0001);

    // ---- The test signal -------------------------------------------------
    // 0.00–0.50 s: noise only (this is what gets profiled).
    // 0.50–1.00 s: the "program" — a two-note chord.
    // 1.00–1.50 s: the program with two clicks on top.
    let total = (FS * 1.5) as usize;
    let noise_only_end = (FS * 0.5) as usize;
    let clicks = [(FS * 1.10) as usize, (FS * 1.35) as usize];

    let mut clean = vec![0.0f64; total];
    for (i, v) in clean.iter_mut().enumerate() {
        let t = i as f64 / FS;
        if i >= noise_only_end {
            // A 440 Hz tone with a 660 Hz fifth, amplitude-shaped so the onset
            // at 0.5 s is not a click of its own.
            let since = (i - noise_only_end) as f64 / FS;
            let env = 1.0 - (-since * 6.0).exp();
            *v = 0.25 * env * ((TAU * 440.0 * t).sin() + 0.5 * (TAU * 660.0 * t).sin());
        }
    }
    for &c in &clicks {
        if let Some(v) = clean.get_mut(c) {
            *v += 0.8;
        }
    }

    // Zero-mean uniform noise at ±0.02, so the profile is genuinely broadband
    // (a DC-offset "noise" would profile as a single spike at bin 0 and tell
    // the gate nothing useful).
    let mut noisy = clean.clone();
    for v in noisy.iter_mut() {
        *v += 0.02 * (rng.next_f64() * 2.0 - 1.0);
    }

    println!("dsp-spectral restoration demo");
    println!(
        "  signal      {} samples @ {FS:.0} Hz ({} ms)",
        noisy.len(),
        noisy.len() * 1000 / FS as usize
    );
    println!(
        "  stft        {FFT}-point {} window, hop {HOP} (75 % overlap)",
        Window::Hann.name()
    );
    println!("  noise floor 0.02 (uniform), tone amplitude 0.25, clicks 0.8");
    println!();

    let input_snr = snr_db(&clean, &noisy);
    println!("input SNR: {:.2} dB", input_snr);
    println!(
        "input ZCR: {:.4} per sample ({:.0} crossings/s)",
        zero_crossing_rate(&noisy),
        zero_crossing_rate(&noisy) * FS
    );
    println!();

    // ---- 1. Profile the noise from the leading noise-only segment --------
    let cfg = StftConfig::new(FFT, HOP).with_sample_rate(FS);
    let noise_segment = &noisy[..noise_only_end];
    let profile = NoiseProfile::estimate(noise_segment, &cfg).expect("valid config");
    let noise_spec = stft(noise_segment, &cfg).expect("valid");
    println!("--- 1. noise profile ---");
    println!(
        "  estimated from {} frames of the leading {} ms",
        profile.frames(),
        noise_segment.len() * 1000 / FS as usize
    );
    println!("  peak noise magnitude: {:.4}", profile.peak());
    println!(
        "  noise ZCR: {:.4} per sample",
        zero_crossing_rate(noise_segment)
    );
    // The profile's shape is worth printing: a flat floor looks different from
    // a tonal one at a glance.
    let p = profile.spectrum();
    let bar = |v: f64, width: usize| {
        "#".repeat((v / profile.peak() * width as f64).clamp(0.0, width as f64) as usize)
    };
    for (label, bin) in [
        ("      0 Hz", 0usize),
        ("    500 Hz", 32),
        ("   1000 Hz", 64),
        ("   2000 Hz", 128),
        ("   4000 Hz", 256),
        ("   7000 Hz", 448),
    ] {
        let v = p.get(bin).copied().unwrap_or(0.0);
        println!("  {label} |{:<30}| {:.4}", bar(v, 30), v);
    }
    println!();

    // ---- 2. Spectral gating ----------------------------------------------
    println!("--- 2. spectral gate ---");
    for gate_cfg in [
        GateConfig {
            threshold_db: 3.0,
            reduction_db: -12.0,
            smoothing_frames: 0,
        },
        GateConfig {
            threshold_db: 6.0,
            reduction_db: -18.0,
            smoothing_frames: 3,
        },
        GateConfig {
            threshold_db: 9.0,
            reduction_db: -24.0,
            smoothing_frames: 5,
        },
    ] {
        let restored = denoise(&noisy, &cfg, &profile, &gate_cfg).expect("valid");
        let s = snr_db(&clean, &restored);
        println!(
            "  thr {:>5.1} dB  red {:>6.1} dB  smooth {} frames  →  SNR {:>6.2} dB  ({:+.2} dB)",
            gate_cfg.threshold_db,
            gate_cfg.reduction_db,
            gate_cfg.smoothing_frames,
            s,
            s - input_snr
        );
    }
    let best_gate = GateConfig {
        threshold_db: 6.0,
        reduction_db: -18.0,
        smoothing_frames: 3,
    };
    let gated = denoise(&noisy, &cfg, &profile, &best_gate).expect("valid");
    println!();

    // ---- 3. Spectral subtraction -----------------------------------------
    println!("--- 3. spectral subtraction (over the gated signal) ---");
    let gated_spec = stft(&gated, &cfg).expect("valid");
    let gated_profile = NoiseProfile::from_frames(&gated_spec, &[0]);
    for alpha in [0.5f64, 1.0, 2.0, 4.0] {
        let subtracted = spectral_subtract(&gated_spec, &gated_profile, alpha).expect("valid");
        let out = dsp_spectral::istft(&subtracted, gated.len());
        println!(
            "  α = {alpha:.1}  →  SNR {:>6.2} dB  ({:+.2} dB)",
            snr_db(&clean, &out),
            snr_db(&clean, &out) - input_snr
        );
    }
    println!();

    // ---- 4. Harmonic / percussive separation -----------------------------
    println!("--- 4. harmonic / percussive separation (HPSS) ---");
    let (harm, perc) = harmonic_percussive_split(&gated_spec, 8.0);
    let harm_sig = dsp_spectral::istft(&harm, gated.len());
    let perc_sig = dsp_spectral::istft(&perc, gated.len());
    let mean_energy =
        |s: &[f64]| -> f64 { s.iter().map(|v| v * v).sum::<f64>() / (s.len() as f64) };
    println!(
        "  harmonic  RMS {:.5}  (the sustained tone lives here)",
        mean_energy(&harm_sig).sqrt()
    );
    println!(
        "  percussive RMS {:.5}  (the clicks live here)",
        mean_energy(&perc_sig).sqrt()
    );
    // The clicks are where the two parts differ most. Centre framing pads by
    // `fft_size / 2`, so sample `c` lands in frame `(c + fft_size/2) / hop`.
    let click_frames: Vec<usize> = clicks.iter().map(|&c| (c + FFT / 2) / HOP).collect();
    println!(
        "  click frames {:?} (t = {:.2} s and {:.2} s)",
        click_frames,
        click_frames.first().copied().unwrap_or(0) as f64 * HOP as f64 / FS,
        click_frames.get(1).copied().unwrap_or(0) as f64 * HOP as f64 / FS,
    );
    // Energy above 4 kHz: the tone has none, so anything there is click.
    let hi_bin = (4_000.0 * FFT as f64 / FS) as usize;
    let band = |s: &dsp_spectral::Spectrum, f: usize| -> f64 {
        s.power(f).iter().skip(hi_bin).sum::<f64>()
    };
    for (label, s) in [("harmonic", &harm), ("percussive", &perc)] {
        let at_click: f64 = click_frames.iter().map(|f| band(s, *f)).fold(0.0, f64::max);
        let elsewhere: f64 = (2..s.num_frames() - 2)
            .filter(|f| !click_frames.contains(f))
            .map(|f| band(s, f))
            .sum::<f64>()
            / ((s.num_frames() - 4) as f64);
        println!(
            "  {label:<11} >4 kHz energy: {at_click:>10.2} at a click vs {elsewhere:>8.2} mean elsewhere ({:.0}×)",
            at_click / elsewhere.max(1e-12)
        );
    }
    println!();

    // ---- 5. Descriptors of the restored signal ---------------------------
    println!("--- 5. spectral features of the restored signal ---");
    let spec = stft(&gated, &cfg).expect("valid");
    let centroids = spectral_centroid(&spec, FS);
    let bandwidths = spectral_bandwidth(&spec, FS, &centroids);
    let flatness = spectral_flatness(&spec);
    let rolloff = spectral_rolloff(&spec, 0.85, FS);
    let flux = spectral_flux(&spec);

    let interior = &flux[2..flux.len() - 2];
    let peak_flux = interior.iter().copied().fold(0.0f64, f64::max);
    let at = interior
        .iter()
        .position(|f| (f - peak_flux).abs() < 1e-12)
        .map(|i| i + 2)
        .unwrap_or(0);
    let stats = |label: &str, unit: &str, v: &[f64], scale: f64| {
        let mean = v.iter().sum::<f64>() / (v.len().max(1) as f64);
        println!(
            "  {label:<22} mean {:>8.2} {unit:<7} min {:>8.2}  max {:>8.2}",
            mean * scale,
            v.iter().copied().fold(f64::INFINITY, f64::min) * scale,
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max) * scale,
        );
    };
    stats("centroid", "Hz", &centroids, 1.0);
    stats("bandwidth", "Hz", &bandwidths, 1.0);
    stats("flatness", "", &flatness, 1.0);
    stats("rolloff (85 %)", "Hz", &rolloff, 1.0);
    println!(
        "  spectral flux          peak {:.2} at frame {at} (t = {:.3} s), mean {:.2}",
        peak_flux,
        at as f64 * HOP as f64 / FS,
        interior.iter().sum::<f64>() / (interior.len().max(1) as f64)
    );
    println!(
        "  zero-crossing rate     {:.4} per sample",
        zero_crossing_rate(&gated)
    );

    // The tone's centroid should sit near 440 Hz — the mean over the whole
    // file is dragged higher by the noise-only first half.
    let tone_region = &centroids[noise_only_end / HOP + 2..noise_only_end / HOP + 40];
    let tone_centroid = tone_region.iter().sum::<f64>() / (tone_region.len().max(1) as f64);
    println!(
        "  centroid over the tone region: {tone_centroid:.1} Hz (the tone is 440 Hz + 660 Hz)"
    );
    println!();

    // ---- 6. Mel / MFCC front end -----------------------------------------
    println!("--- 6. mel / MFCC front end ---");
    let n_mels = 26usize;
    // Span the whole band (0 → Nyquist) so the bank tiles it and the energies
    // sum to the frame's power exactly.
    let bank = spec.mel_filterbank(n_mels, 0.0, FS / 2.0, FS);
    let centres = mel_frequencies(n_mels, 0.0, FS / 2.0);
    let frame = spec.frame(tone_region.len() / 2 + 2).expect("in range");
    let energies = bank.energies(frame);
    let coeffs = mfcc(&energies, 13);
    println!(
        "  {n_mels} mel bands spanning 0 Hz – 8 kHz, centres {:.1} Hz … {:.1} Hz",
        centres.first().copied().unwrap_or(0.0),
        centres.last().copied().unwrap_or(0.0)
    );
    let total_energy: f64 = energies.iter().sum();
    let frame_power: f64 = frame.iter().map(|c| c.power()).sum();
    println!(
        "  mel energies sum to {total_energy:.4} vs frame power {frame_power:.4} (the bank tiles the band)"
    );
    for (m, &c) in coeffs.iter().enumerate() {
        let bar_len = ((c.abs() / coeffs.iter().map(|v| v.abs()).fold(0.0f64, f64::max) * 40.0)
            .clamp(0.0, 40.0)) as usize;
        println!("  MFCC[{m:>2}] {:>9.4} |{:<40}|", c, "#".repeat(bar_len));
    }
    println!();

    // ---- Summary ---------------------------------------------------------
    println!("--- summary ---");
    let final_snr = snr_db(&clean, &gated);
    println!("  input SNR        {input_snr:>7.2} dB");
    println!(
        "  gated SNR        {final_snr:>7.2} dB  ({:+.2} dB)",
        final_snr - input_snr
    );
    println!(
        "  analysis–synthesis round-trip: {:.1} dB SNR ({} reconstruction error)",
        snr_db(&noisy, &dsp_spectral::istft(&noise_spec, noise_only_end)),
        {
            let rt = dsp_spectral::istft(&noise_spec, noise_only_end);
            noisy[..noise_only_end]
                .iter()
                .zip(rt.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
        }
    );
}

/// A small deterministic PRNG, so the demo is reproducible run to run without
/// pulling in a dependency.
struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// The next value in `[0, 1)`, from the 53 high bits of a 64-bit LCG
    /// (Numerical Recipes' constants).
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_845_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}
