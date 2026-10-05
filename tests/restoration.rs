#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    missing_docs
)]
//! Restoration primitives: noise profiling, gating, subtraction, HPSS,
//! smoothing, and the end-to-end `denoise` SNR claim.

use core::f64::consts::TAU;

use dsp_spectral::{
    denoise, harmonic_percussive_split, snr_db, spectral_gate, spectral_smooth, spectral_subtract,
    stft, Complex, GateConfig, NoiseProfile, Spectrum, StftConfig, Window,
};

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_845_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

const FS: f64 = 16_000.0;

#[test]
fn noise_profile_differs_between_noise_and_a_tone() {
    // The whole premise of `estimate`: a profile of room tone is broad and
    // low, a profile of a tone is spiky and high. If these came out the same
    // the gate would have nothing to compare against.
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let mut rng = lcg(0xA5);
    let noise: Vec<f64> = (0..8_192).map(|_| 0.05 * rng()).collect();
    let tone: Vec<f64> = (0..8_192)
        .map(|i| 0.5 * (TAU * 500.0 * i as f64 / FS).sin())
        .collect();

    let p_noise = NoiseProfile::estimate(&noise, &cfg).expect("valid");
    let p_tone = NoiseProfile::estimate(&tone, &cfg).expect("valid");
    assert_eq!(p_noise.spectrum().len(), cfg.bin_count());
    assert_eq!(p_tone.spectrum().len(), cfg.bin_count());
    assert!(p_noise.frames() > 0);
    assert!((p_noise.sample_rate() - FS).abs() < 1e-9);

    // Count bins carrying more than 1 % of the profile's own peak, so the
    // comparison is relative and independent of the signal's absolute level.
    let active = |p: &NoiseProfile| -> usize {
        let floor = 0.01 * p.peak();
        p.spectrum().iter().filter(|v| **v > floor).count()
    };
    // Room tone is broad: most of the band carries real energy.
    let noise_active = active(&p_noise);
    assert!(
        noise_active > cfg.bin_count() * 3 / 4,
        "only {noise_active}/{} active bins",
        cfg.bin_count()
    );
    // A pure tone concentrates it in a couple of bins (the rest is leakage).
    let tone_active = active(&p_tone);
    assert!(tone_active < 20, "{tone_active} active bins for a tone");

    // And the peak magnitudes differ by orders of magnitude.
    assert!(
        p_tone.peak() > 20.0 * p_noise.peak(),
        "{} vs {}",
        p_tone.peak(),
        p_noise.peak()
    );
    assert!(!p_noise
        .spectrum()
        .iter()
        .all(|a| a == &p_tone.spectrum()[0]));
}

#[test]
fn from_frames_equals_the_manual_average() {
    // The identity a caller relies on when they profile a selected subset of
    // frames: `from_frames` is a plain arithmetic mean of those frames'
    // magnitude spectra.
    let cfg = StftConfig::new(512, 128).with_sample_rate(FS);
    let mut rng = lcg(0x3B);
    let x: Vec<f64> = (0..4_096).map(|_| rng()).collect();
    let spec = stft(&x, &cfg).expect("valid");
    for indices in [
        vec![0usize],
        vec![1, 2, 3],
        vec![0, 5, 10, 15],
        (0..spec.num_frames()).collect(),
    ] {
        let profile = NoiseProfile::from_frames(&spec, &indices);
        assert_eq!(profile.frames(), indices.len());
        for k in 0..spec.bin_count() {
            let mut sum = 0.0;
            for &i in &indices {
                sum += spec.magnitude(i)[k];
            }
            let want = sum / (indices.len() as f64);
            let got = profile.magnitude_at(k);
            assert!((got - want).abs() < 1e-12, "bin {k}: {got} vs {want}");
        }
    }
    // Out-of-range indices are ignored, not fatal.
    let partial = NoiseProfile::from_frames(&spec, &[0, 99_999]);
    assert_eq!(partial.frames(), 1);
    // An empty selection is an all-zero profile of the right width.
    let none = NoiseProfile::from_frames(&spec, &[]);
    assert_eq!(none.frames(), 0);
    assert_eq!(none.spectrum().len(), spec.bin_count());
    assert!(none.spectrum().iter().all(|v| *v == 0.0));
    assert_eq!(none.peak(), 0.0);
    // `estimate` over a whole buffer agrees with `from_frames` over all of it.
    let estimated = NoiseProfile::estimate(&x, &cfg).expect("valid");
    let manual = NoiseProfile::from_frames(&spec, &(0..spec.num_frames()).collect::<Vec<_>>());
    for (a, b) in estimated.spectrum().iter().zip(manual.spectrum().iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    assert_eq!(estimated.frames(), manual.frames());
}

#[test]
fn gate_applies_exactly_the_configured_reduction() {
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let mut rng = lcg(0x77);
    let x: Vec<f64> = (0..8_192).map(|_| 0.1 * rng()).collect();
    let spec = stft(&x, &cfg).expect("valid");
    let profile = NoiseProfile::from_frames(&spec, &(0..spec.num_frames()).collect::<Vec<_>>());

    for reduction_db in [-3.0f64, -12.0, -30.0, -60.0] {
        let gate = GateConfig {
            threshold_db: 6.0,
            reduction_db,
            smoothing_frames: 0,
        };
        let gated = spectral_gate(&spec, &profile, &gate).expect("valid");
        let want = dsp_core::math::db_to_linear(reduction_db);
        let mut gated_bins = 0usize;
        for f in 0..spec.num_frames() {
            let before = spec.magnitude(f);
            let after = gated.magnitude(f);
            for k in 0..spec.bin_count() {
                let b = before[k];
                let a = after[k];
                if b < 1e-12 {
                    assert!(a < 1e-12);
                    continue;
                }
                let gain = a / b;
                assert!(
                    (gain - 1.0).abs() < 1e-9 || (gain - want).abs() < 1e-9,
                    "reduction {reduction_db} frame {f} bin {k}: gain {gain}"
                );
                if (gain - want).abs() < 1e-9 {
                    gated_bins += 1;
                    let measured = dsp_core::math::linear_to_db(gain);
                    assert!(
                        (measured - reduction_db).abs() < 1e-6,
                        "measured {measured} dB, asked for {reduction_db}"
                    );
                }
            }
        }
        assert!(gated_bins > 0, "reduction {reduction_db} gated nothing");
    }
}

#[test]
fn gate_leaves_above_threshold_bins_within_tolerance() {
    // A signal far above its own noise profile: with a low threshold the gate
    // must be a pass-through, bin for bin.
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let mut rng = lcg(0x19);
    let noise: Vec<f64> = (0..4_096).map(|_| 0.001 * rng()).collect();
    let tone: Vec<f64> = (0..4_096)
        .map(|i| 0.5 * (TAU * 500.0 * i as f64 / FS).sin())
        .collect();
    let profile = NoiseProfile::estimate(&noise, &cfg).expect("valid");
    let spec = stft(&tone, &cfg).expect("valid");
    let gate = GateConfig {
        threshold_db: 20.0,
        reduction_db: -60.0,
        smoothing_frames: 0,
    };
    let gated = spectral_gate(&spec, &profile, &gate).expect("valid");
    let mut checked = 0usize;
    for f in 2..spec.num_frames() - 2 {
        let before = spec.magnitude(f);
        let after = gated.magnitude(f);
        for k in 0..spec.bin_count() {
            // A sine's DC and very-low bins sit *below* the noise floor, so
            // the gate legitimately fires there; the claim is only about bins
            // that clear the profile by more than the threshold.
            if before[k] < 10.0 * profile.magnitude_at(k) {
                continue;
            }
            assert!(
                (after[k] - before[k]).abs() < 1e-9 * before[k].max(1.0),
                "frame {f} bin {k}: {} → {}",
                before[k],
                after[k]
            );
            checked += 1;
        }
    }
    // The tone is spectrally narrow (a windowed sine lives in a few bins), so
    // the number of checked bins is a handful per frame, not the whole band.
    assert!(
        checked >= (spec.num_frames() - 4) * 3,
        "only {checked} bins checked"
    );
}

#[test]
fn gate_threshold_decides_the_split() {
    // Sweeping the threshold from "gate everything" to "gate nothing" must
    // move the gated-bin count monotonically.
    let cfg = StftConfig::new(512, 128).with_sample_rate(FS);
    let mut rng = lcg(0x2C);
    let mut x: Vec<f64> = (0..4_096).map(|_| 0.1 * rng()).collect();
    for v in x.iter_mut() {
        *v += 0.3 * (TAU * 300.0 * 1.0).cos();
    }
    let spec = stft(&x, &cfg).expect("valid");
    let profile = NoiseProfile::from_frames(&spec, &(0..spec.num_frames()).collect::<Vec<_>>());
    let total = spec.num_frames() * spec.bin_count();

    let count_gated = |threshold: f64| -> usize {
        let gate = GateConfig {
            threshold_db: threshold,
            reduction_db: -12.0,
            smoothing_frames: 0,
        };
        let gated = spectral_gate(&spec, &profile, &gate).expect("valid");
        let mut n = 0;
        for f in 0..spec.num_frames() {
            let b = spec.magnitude(f);
            let a = gated.magnitude(f);
            for k in 0..spec.bin_count() {
                if b[k] > 1e-12 && a[k] < b[k] * 0.5 {
                    n += 1;
                }
            }
        }
        n
    };

    // The threshold is the headroom a bin needs *above* the profile, so a
    // large positive threshold gates everything (even the signal) and a large
    // negative one gates nothing.
    let aggressive = count_gated(200.0);
    let moderate = count_gated(0.0);
    let gentle = count_gated(-200.0);
    assert_eq!(aggressive, total, "+200 dB must gate everything");
    assert_eq!(gentle, 0, "−200 dB must gate nothing");
    assert!(
        moderate > gentle && moderate < aggressive,
        "{moderate} between {gentle} and {aggressive}"
    );
}

#[test]
fn gate_smoothing_averages_the_gain_along_time() {
    // A hand-built spectrogram with a known gain pattern, so the smoothed
    // output is exactly the moving average.
    let bins = 2usize;
    let rows = |mag: f64| vec![Complex::new(mag, 0.0); bins];
    let frames: Vec<Vec<Complex>> = vec![rows(1.0), rows(0.0), rows(1.0), rows(1.0), rows(1.0)];
    let spec = Spectrum::from_parts(frames, 1, 2, Window::Rectangular, true).expect("valid");
    // A profile above every bin's magnitude makes the threshold irrelevant:
    // all bins are gated, all gains identical.
    let profile = NoiseProfile::from_magnitudes(vec![0.5; bins], FS);
    // A threshold above the loud frames' +6.02 dB of headroom gates
    // everything, so the gain pattern is uniform and smoothing cannot change
    // anything. Check that first.
    let all_gated = GateConfig {
        threshold_db: 100.0,
        reduction_db: -12.0,
        smoothing_frames: 0,
    };
    let uniform = spectral_gate(&spec, &profile, &all_gated).expect("valid");
    for f in 0..spec.num_frames() {
        let want = if spec.magnitude(f)[0] == 0.0 {
            0.0
        } else {
            dsp_core::math::db_to_linear(-12.0)
        };
        assert!((uniform.magnitude(f)[0] - want).abs() < 1e-12);
    }
    // Now a genuinely varying gain: a frame whose magnitude sits below the
    // profile is gated, one above it is not.
    let profile2 = NoiseProfile::from_magnitudes(vec![0.5; bins], FS);
    let gate = GateConfig {
        threshold_db: 6.0,
        reduction_db: -12.0,
        smoothing_frames: 0,
    };
    let sharp = spectral_gate(&spec, &profile2, &gate).expect("valid");
    // Frame 1 is silent → its ratio is −∞ → gated. Frames 0/2/3/4 sit at
    // 20·log10(1/0.5) = +6.02 dB, just above the 6 dB threshold → untouched.
    assert!(sharp.magnitude(1)[0] < 1e-12);
    assert!((sharp.magnitude(0)[0] - 1.0).abs() < 1e-12);
    assert!((sharp.magnitude(4)[0] - 1.0).abs() < 1e-12);

    // The smoothed version bleeds the gated frame's gain into its neighbours.
    let smooth = spectral_gate(
        &spec,
        &profile2,
        &GateConfig {
            smoothing_frames: 5,
            ..gate
        },
    )
    .expect("valid");
    let expected = (4.0 + dsp_core::math::db_to_linear(-12.0)) / 5.0;
    assert!(
        (smooth.magnitude(0)[0] - expected).abs() < 1e-12,
        "{}",
        smooth.magnitude(0)[0]
    );
    assert!((smooth.magnitude(2)[0] - expected).abs() < 1e-12);
    assert!(
        smooth.magnitude(1)[0] < 1e-12,
        "a silent frame stays silent"
    );
    // Frame 4's window never reaches frame 1.
    assert!(
        (smooth.magnitude(4)[0] - 1.0).abs() < 1e-12,
        "{}",
        smooth.magnitude(4)[0]
    );
}

#[test]
fn spectral_subtraction_with_zero_is_the_identity() {
    let cfg = StftConfig::new(512, 128).with_sample_rate(FS);
    let mut rng = lcg(0x5E);
    let x: Vec<f64> = (0..4_096).map(|_| rng()).collect();
    let spec = stft(&x, &cfg).expect("valid");
    let profile = NoiseProfile::from_frames(&spec, &[0, 1]);
    for alpha in [0.0f64, 1e-12] {
        let same = spectral_subtract(&spec, &profile, alpha).expect("valid");
        if alpha == 0.0 {
            assert_eq!(same, spec, "over-subtraction 0 must be the exact identity");
        } else {
            for f in 0..spec.num_frames() {
                let a = spec.magnitude(f);
                let b = same.magnitude(f);
                for k in 0..spec.bin_count() {
                    assert!(
                        (a[k] - b[k]).abs() < 1e-9 * a[k].max(1.0),
                        "frame {f} bin {k}"
                    );
                }
            }
        }
    }
    // Every bin's phase is preserved at any over-subtraction, which is the
    // defining property of magnitude-domain subtraction.
    let subbed = spectral_subtract(&spec, &profile, 2.0).expect("valid");
    for f in 0..spec.num_frames() {
        let a = spec.frame(f).expect("in range");
        let b = subbed.frame(f).expect("in range");
        for k in 0..spec.bin_count() {
            if a[k].magnitude() < 1e-9 || b[k].magnitude() < 1e-9 {
                continue;
            }
            let pa = a[k].phase();
            let pb = b[k].phase();
            let d = (pa - pb).abs();
            assert!(d < 1e-9 || (d - TAU).abs() < 1e-9, "frame {f} bin {k}: {d}");
        }
    }
    // Over-subtraction floors at zero, never negative.
    let loud = NoiseProfile::from_magnitudes(vec![1e9; cfg.bin_count()], FS);
    let gone = spectral_subtract(&spec, &loud, 1.0).expect("valid");
    assert!(gone.magnitude(0).iter().all(|m| *m == 0.0));
}

#[test]
fn spectral_subtraction_reduces_only_the_subtracted_part() {
    // α = 1 on a synthetic spectrogram: the output magnitude is exactly
    // `max(0, |X| − |N|)` and the phase is untouched.
    // An 8-point transform has 5 bins, so the five magnitudes tile it exactly.
    let bins = 5usize;
    let mags = [1.0f64, 0.5, 0.2, 0.05, 0.9];
    let rows: Vec<Vec<Complex>> = (0..3)
        .map(|_| {
            mags.iter()
                .enumerate()
                .map(|(k, m)| Complex::from_polar(*m, 0.3 * (k as f64) + 0.1))
                .collect()
        })
        .collect();
    let spec = Spectrum::from_parts(rows, 1, 8, Window::Rectangular, true).expect("valid");
    let noise = [0.4f64, 0.4, 0.4, 0.4, 0.4];
    let profile = NoiseProfile::from_magnitudes(noise.to_vec(), FS);
    let out = spectral_subtract(&spec, &profile, 1.0).expect("valid");
    for f in 0..spec.num_frames() {
        let a = spec.frame(f).expect("in range");
        let b = out.frame(f).expect("in range");
        for k in 0..bins {
            let want = (mags[k] - noise[k]).max(0.0);
            assert!((b[k].magnitude() - want).abs() < 1e-12, "frame {f} bin {k}");
            if want > 0.0 {
                // Phase is preserved wherever the bin survives; a bin floored
                // to zero has no phase left to preserve (atan2(0, 0) = 0).
                assert!(
                    (b[k].phase() - a[k].phase()).abs() < 1e-12,
                    "frame {f} bin {k} phase moved"
                );
            } else {
                assert_eq!(b[k], Complex::ZERO, "frame {f} bin {k} should be zeroed");
            }
        }
        // At least one bin survives and one is floored, so both branches of
        // the check above are actually exercised.
        assert!(mags.iter().zip(noise.iter()).any(|(m, nz)| m - nz > 0.0));
        assert!(mags.iter().zip(noise.iter()).any(|(m, nz)| m - nz <= 0.0));
    }
}

#[test]
fn spectral_smooth_is_a_no_op_with_an_identity_kernel() {
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let mut rng = lcg(0x4D);
    let x: Vec<f64> = (0..8_192).map(|_| rng()).collect();
    let spec = stft(&x, &cfg).expect("valid");
    // The kernel is applied as-is with offsets `j − width/2`, so the identity
    // is a centre tap: `[1]`, or `[0, 1, 0]` at width 3.
    for kernel in [
        vec![1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ] {
        let out = spectral_smooth(&spec, &kernel).expect("valid");
        for f in 0..spec.num_frames() {
            let a = spec.frame(f).expect("in range");
            let b = out.frame(f).expect("in range");
            for k in 0..spec.bin_count() {
                assert!(
                    (a[k].magnitude() - b[k].magnitude()).abs() < 1e-12,
                    "kernel {kernel:?} frame {f} bin {k}"
                );
            }
        }
    }
}

#[test]
fn spectral_smooth_averages_the_magnitude_spectrum() {
    // A known triangular spectrum smoothed by a boxcar: the interior bins
    // become the mean of their three taps, the edges get edge-replicated
    // contributions, and the phase is preserved.
    let mags = [1.0f64, 4.0, 9.0, 16.0, 25.0];
    let rows: Vec<Vec<Complex>> = vec![mags
        .iter()
        .enumerate()
        .map(|(k, m)| Complex::from_polar(*m, 0.2 * (k as f64)))
        .collect()];
    let spec = Spectrum::from_parts(rows, 1, 8, Window::Rectangular, true).expect("valid");
    let kernel = [1.0 / 3.0; 3];
    let out = spectral_smooth(&spec, &kernel).expect("valid");
    let before = spec.frame(0).expect("in range");
    let after = out.frame(0).expect("in range");
    for k in 0usize..5 {
        // Three taps at offsets −1, 0, +1, each clamped into range at the
        // edges — so k = 0 reads bins 0, 0, 1.
        let want: f64 = (0..3)
            .map(|j| {
                let kk = (k as isize + (j as isize - 1)).clamp(0, 4) as usize;
                mags[kk] / 3.0
            })
            .sum();
        assert!(
            (after[k].magnitude() - want).abs() < 1e-12,
            "bin {k}: {} vs {want}",
            after[k].magnitude()
        );
        assert!(
            (after[k].phase() - before[k].phase()).abs() < 1e-12,
            "bin {k} phase moved"
        );
    }
    // A boxcar genuinely smooths: the peak drops.
    assert!(after[4].magnitude() < before[4].magnitude());
}

#[test]
fn hpss_separates_a_harmonic_and_a_percussive_mixture() {
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let n = 16_384usize;
    // Harmonic: a steady 400 Hz tone (sharply peaked, stationary in time).
    let mut x: Vec<f64> = (0..n)
        .map(|i| 0.4 * (TAU * 400.0 * i as f64 / FS).sin())
        .collect();
    // Percussive: a three-sample click (broadband, transient in time).
    for (offset, gain) in [1.0f64, 0.6, 0.3].iter().enumerate() {
        if let Some(v) = x.get_mut(n / 2 + offset) {
            *v += gain;
        }
    }
    let spec = stft(&x, &cfg).expect("valid");
    let bin_of = |hz: f64| ((hz * (cfg.fft_size as f64) / FS) as usize).min(spec.bin_count() - 1);
    let tone_bin = bin_of(400.0);
    let hi_bin = bin_of(2_000.0);
    let nyq = spec.bin_count() - 1;
    let click_frame = (n / 2 + cfg.fft_size / 2 - 8) / cfg.hop;
    let steady_frame = 10usize;

    let (harm, perc) = harmonic_percussive_split(&spec, 8.0);
    assert_eq!(harm.num_frames(), spec.num_frames());
    assert_eq!(perc.bin_count(), spec.bin_count());
    assert_eq!(harm.hop(), spec.hop());
    assert_eq!(harm.fft_size(), spec.fft_size());
    assert!(harm.is_centered());

    let p2 = |s: &Spectrum, f: usize, k: usize| s.magnitude(f)[k].powi(2);
    let band = |s: &Spectrum, f: usize| s.power(f)[hi_bin..=nyq].iter().sum::<f64>();

    // The harmonic part keeps the tone at its bin.
    assert!(
        p2(&harm, steady_frame, tone_bin) > 1_000.0 * p2(&perc, steady_frame, tone_bin),
        "tone bin"
    );
    // The percussive part keeps the transient's broadband energy.
    assert!(
        band(&perc, click_frame) > 100.0 * band(&perc, steady_frame),
        "click high band"
    );
    assert!(
        band(&harm, click_frame) < 0.01 * band(&perc, click_frame),
        "click is not harmonic"
    );

    // The mask is complementary, so magnitude splits exactly.
    for f in 0..spec.num_frames() {
        let a = spec.magnitude(f);
        let b = harm.magnitude(f);
        let c = perc.magnitude(f);
        for k in 0..spec.bin_count() {
            assert!(
                ((b[k] + c[k]) - a[k]).abs() < 1e-9 * a[k].max(1.0),
                "frame {f} bin {k}: {} + {} vs {}",
                b[k],
                c[k],
                a[k]
            );
        }
    }
}

#[test]
fn hpss_on_a_pure_tone_puts_everything_in_the_harmonic_part() {
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let n = 16_384usize;
    let x: Vec<f64> = (0..n)
        .map(|i| 0.5 * (TAU * 440.0 * i as f64 / FS).sin())
        .collect();
    let spec = stft(&x, &cfg).expect("valid");
    let (harm, perc) = harmonic_percussive_split(&spec, 8.0);
    let frame_energy = |s: &Spectrum| -> f64 {
        (2..s.num_frames() - 2)
            .map(|f| s.power(f).iter().sum::<f64>())
            .sum::<f64>()
            / ((s.num_frames() - 4) as f64)
    };
    let eh = frame_energy(&harm);
    let ep = frame_energy(&perc);
    assert!(eh > 100.0 * ep, "harmonic {eh} vs percussive {ep}");
}

#[test]
fn hpss_on_a_pure_transient_puts_everything_in_the_percussive_part() {
    // A click train with a long gap: nothing is stationary, so the time-axis
    // median finds no harmonic support.
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let n = 16_384usize;
    let x: Vec<f64> = (0..n)
        .map(|i| if i % 4_096 < 3 { 1.0 } else { 0.0 })
        .collect();
    let spec = stft(&x, &cfg).expect("valid");
    let (harm, perc) = harmonic_percussive_split(&spec, 8.0);
    let frame_energy = |s: &Spectrum| -> f64 {
        (2..s.num_frames() - 2)
            .map(|f| s.power(f).iter().sum::<f64>())
            .sum::<f64>()
            / ((s.num_frames() - 4) as f64)
    };
    let eh = frame_energy(&harm);
    let ep = frame_energy(&perc);
    assert!(ep > 100.0 * eh, "harmonic {eh} vs percussive {ep}");
}

#[test]
fn denoise_improves_snr_with_a_real_margin() {
    // The headline claim, asserted with a margin rather than "it went up".
    let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
    let n = 32_768usize;
    let mut rng = lcg(0x5EED);

    // A noise-only segment to profile...
    let room: Vec<f64> = (0..n / 2).map(|_| 0.02 * rng()).collect();
    let profile = NoiseProfile::estimate(&room, &cfg).expect("valid");

    // ...and a signal at the same noise level.
    let clean: Vec<f64> = (0..n)
        .map(|i| {
            0.5 * (TAU * 500.0 * i as f64 / FS).sin() + 0.1 * (TAU * 1_500.0 * i as f64 / FS).sin()
        })
        .collect();
    let noisy: Vec<f64> = clean
        .iter()
        .zip((0..n).map(|_| 0.02 * rng()))
        .map(|(a, b)| a + b)
        .collect();
    let before = snr_db(&clean, &noisy);
    assert!(before > 20.0, "input SNR should be realistic: {before}");

    let gate = GateConfig {
        threshold_db: 6.0,
        reduction_db: -18.0,
        smoothing_frames: 3,
    };
    let out = denoise(&noisy, &cfg, &profile, &gate).expect("valid");
    assert_eq!(out.len(), noisy.len());
    let after = snr_db(&clean, &out);

    // A real margin: gating 18 dB of a noise floor measured from the same
    // distribution must recover a meaningful fraction of it.
    assert!(after > before + 5.0, "SNR {before} → {after} dB");

    // And the gate is not simply muting everything: the signal survives.
    let retained = snr_db(&clean, &out) - snr_db(&clean, &vec![0.0; n]);
    assert!(retained > 5.0, "signal was destroyed: {retained}");
    assert!(out.iter().any(|v| v.abs() > 0.1), "output is silent");
}

#[test]
fn denoise_end_to_end_over_a_parametric_sweep() {
    // The SNR improvement must hold across the reasonable configuration space,
    // not just at one lucky point.
    let n = 16_384usize;
    let mut gains = Vec::new();
    for (threshold, reduction, smoothing) in
        [(3.0f64, -12.0f64, 0usize), (6.0, -18.0, 3), (9.0, -24.0, 5)]
    {
        let cfg = StftConfig::new(1024, 256).with_sample_rate(FS);
        let mut rng = lcg(0xF00D + threshold as u64);
        let room: Vec<f64> = (0..n / 2).map(|_| 0.02 * rng()).collect();
        let profile = NoiseProfile::estimate(&room, &cfg).expect("valid");
        let clean: Vec<f64> = (0..n)
            .map(|i| 0.5 * (TAU * 700.0 * i as f64 / FS).sin())
            .collect();
        let noisy: Vec<f64> = clean
            .iter()
            .zip((0..n).map(|_| 0.02 * rng()))
            .map(|(a, b)| a + b)
            .collect();
        let before = snr_db(&clean, &noisy);
        let out = denoise(
            &noisy,
            &cfg,
            &profile,
            &GateConfig {
                threshold_db: threshold,
                reduction_db: reduction,
                smoothing_frames: smoothing,
            },
        )
        .expect("valid");
        let after = snr_db(&clean, &out);
        gains.push(after - before);
        assert!(
            after > before + 2.0,
            "thr={threshold} red={reduction} sm={smoothing}: {before} → {after}"
        );
    }
    // Stronger reduction should not do worse than the weakest.
    assert!(gains[2] > gains[0], "gains {gains:?}");
}

#[test]
fn denoise_propagates_errors_rather_than_panicking() {
    let cfg = StftConfig::new(256, 64);
    let x = vec![0.1; 1_024];
    let profile = NoiseProfile::estimate(&x, &cfg).expect("valid");
    // Empty input.
    assert!(denoise(&[], &cfg, &profile, &GateConfig::default()).is_err());
    // Non-finite input.
    let mut bad = x.clone();
    bad[10] = f64::NAN;
    assert!(denoise(&bad, &cfg, &profile, &GateConfig::default()).is_err());
    // Invalid configuration.
    let mut bad_cfg = cfg.clone();
    bad_cfg.hop = 0;
    assert!(denoise(&x, &bad_cfg, &profile, &GateConfig::default()).is_err());
    // Mismatched profile.
    let other = NoiseProfile::estimate(&x, &StftConfig::new(64, 16)).expect("valid");
    assert!(denoise(&x, &cfg, &other, &GateConfig::default()).is_err());
    // Invalid gate.
    assert!(denoise(
        &x,
        &cfg,
        &profile,
        &GateConfig {
            threshold_db: f64::NAN,
            reduction_db: 0.0,
            smoothing_frames: 0
        }
    )
    .is_err());
}

#[test]
fn snr_db_edge_cases() {
    assert_eq!(snr_db(&[], &[]), f64::INFINITY);
    assert_eq!(snr_db(&[1.0, 2.0], &[1.0, 2.0]), f64::INFINITY);
    assert_eq!(snr_db(&[0.0, 0.0], &[0.0, 0.0]), f64::INFINITY);
    assert_eq!(snr_db(&[0.0, 0.0], &[1.0, 1.0]), f64::NEG_INFINITY);
    // A 4× amplitude error over 4 samples: 10·log10(4/16) = −6.02 dB.
    assert!((snr_db(&[1.0; 4], &[3.0; 4]) + 6.020_599_913_279_624).abs() < 1e-9);
    // Mismatched lengths zip to the shorter, so only the first sample counts:
    // a unit reference against a zero first sample has signal 1 and error 1,
    // i.e. 0 dB.
    assert!((snr_db(&[1.0], &[0.0; 8]) - 0.0).abs() < 1e-12);
    // Doubling the sample doubles both signal and error, so the ratio — and
    // therefore the SNR — is unchanged.
    assert!((snr_db(&[1.0], &[2.0; 8]) - snr_db(&[1.0], &[0.0; 8])).abs() < 1e-12);
    // An empty processed buffer zips to nothing, so there is no error at all.
    assert_eq!(snr_db(&[1.0, 2.0], &[]), f64::INFINITY);
}
