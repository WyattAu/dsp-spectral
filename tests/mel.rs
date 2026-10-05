#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, missing_docs)]
//! Mel scale, filterbank, MFCC, and cepstrum.

use core::f64::consts::{PI, TAU};

use dsp_spectral::{
    Complex, MelBank, Spectrum, Window, hz_to_mel, mel_frequencies, mel_to_hz, mfcc,
    mfcc_with_lifter, real_cepstrum, stft,
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

/// HTK reference: 1000 Hz → 999.9855… mel.
const HZ_1000_IN_MEL: f64 = 999.985_537_139_6;

#[test]
fn mel_scale_matches_htk_reference_points() {
    assert!((hz_to_mel(0.0) - 0.0).abs() < 1e-15);
    assert!((hz_to_mel(100.0) - 150.489_102_407_097_1).abs() < 1e-9, "{}", hz_to_mel(100.0));
    assert!((hz_to_mel(1_000.0) - HZ_1000_IN_MEL).abs() < 1e-9);
    assert!((hz_to_mel(4_000.0) - 2_146.064_527_506_190_2).abs() < 1e-9);
    assert!((hz_to_mel(8_000.0) - 2_840.023_046_708_318_8).abs() < 1e-9);
    assert!((hz_to_mel(20_000.0) - 3_816.913_632_623_705).abs() < 1e-9);
    assert!((hz_to_mel(44_100.0) - 4_687.037_032_488_187).abs() < 1e-9);

    // Inverses.
    for &mel in &[0.0f64, 1.0, 500.0, HZ_1000_IN_MEL, 2_000.0, 3_284.52] {
        let hz = mel_to_hz(mel);
        assert!((hz_to_mel(hz) - mel).abs() < 1e-9, "mel {mel} → {hz}");
    }
}

#[test]
fn mel_scale_round_trips_within_1e9_across_the_audio_band() {
    let mut worst = 0.0f64;
    let mut worst_at = 0.0f64;
    for i in 0..=44_100 {
        let hz = i as f64;
        let back = mel_to_hz(hz_to_mel(hz));
        let err = (back - hz).abs();
        if err > worst {
            worst = err;
            worst_at = hz;
        }
    }
    assert!(worst < 1e-9, "worst {worst:e} at {worst_at} Hz");

    // And the inverse direction.
    let mut worst = 0.0f64;
    for i in 0..=4_000 {
        let mel = i as f64 / 10.0;
        worst = worst.max((hz_to_mel(mel_to_hz(mel)) - mel).abs());
    }
    assert!(worst < 1e-9, "inverse worst {worst:e}");
}

#[test]
fn mel_scale_is_strictly_monotone() {
    let mut prev = hz_to_mel(0.0);
    for i in 1..=100_000 {
        let m = hz_to_mel(i as f64 * 0.2);
        assert!(m > prev, "not increasing at {} Hz", i as f64 * 0.2);
        prev = m;
    }
    assert!(mel_to_hz(hz_to_mel(1.0)) > mel_to_hz(hz_to_mel(0.0)));
}

#[test]
fn mel_frequencies_are_evenly_spaced_on_the_mel_scale() {
    let n = 32usize;
    let f = mel_frequencies(n, 20.0, 8_000.0);
    assert_eq!(f.len(), n);
    assert!((f[0] - 20.0).abs() < 1e-9);
    assert!((f[n - 1] - 8_000.0).abs() < 1e-9);
    for w in f.windows(2) {
        assert!(w[1] > w[0]);
    }
    // Uniform in mel, not in Hz: the gaps must grow with frequency.
    let gaps: Vec<f64> = f.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gaps.windows(2).all(|w| w[1] > w[0]), "gaps must widen: {gaps:?}");
    assert_eq!(gaps.len(), n - 1);
    // ... and the mel gaps are equal.
    // Each centre sits an exact whole number of steps above f_min in mel.
    let mels: Vec<f64> = f.iter().map(|&hz| hz_to_mel(hz)).collect();
    let lo = mels[0];
    let step = (mels[n - 1] - lo) / ((n - 1) as f64);
    for (i, m) in mels.iter().enumerate() {
        assert!(
            ((m - lo) - (step * (i as f64))).abs() < 1e-9,
            "mel {i}: {} is not {i} steps above {lo}",
            m
        );
    }

    // Degenerate counts.
    assert!(mel_frequencies(0, 20.0, 8_000.0).is_empty());
    let one = mel_frequencies(1, 20.0, 8_000.0);
    assert_eq!(one.len(), 1);
    assert!(one[0] > 20.0 && one[0] < 8_000.0);
    // A degenerate span still returns the requested count.
    assert_eq!(mel_frequencies(4, 1_000.0, 1_000.0).len(), 4);
}

#[test]
fn mel_filterbank_tiles_the_band_and_conserves_energy() {
    // The bank is built so adjacent triangles are exact complements: the
    // filter weights sum to 1 at every bin, hence `energies` sums to the
    // frame's total power. Verified against a naive weighted sum.
    for (fs, fft_size, n_mels) in [
        (8_000.0f64, 8usize, 4usize),
        (16_000.0f64, 64usize, 8usize),
        (44_100.0f64, 512usize, 26usize),
        (48_000.0f64, 1024usize, 40usize),
        (8_000.0f64, 32usize, 2usize),
        (8_000.0f64, 32usize, 3usize),
    ] {
        let bank = MelBank::new(n_mels, 0.0, fs / 2.0, fs, fft_size).expect("valid");
        assert_eq!(bank.n_mels(), n_mels);
        assert_eq!(bank.bin_count(), fft_size / 2 + 1);
        let bins = fft_size / 2 + 1;

        // Every row is a proper triangle: non-negative, ≤ 1, and with a
        // genuine interior maximum. (The peak only reaches 1.0 when the
        // filter's centre lands exactly on a bin frequency — with 5 bins and
        // 4 filters it does not, so the height check is left to the
        // finely-resolved bank below.)
        for m in 0..n_mels {
            let row = bank.filter(m).expect("filter");
            assert_eq!(row.len(), bins);
            assert!(row.iter().all(|w| (0.0..=1.0).contains(w)), "mel {m}: {row:?}");
            let peak = row.iter().copied().fold(0.0f64, f64::max);
            assert!(peak > 0.0, "mel {m} is empty: {row:?}");
            assert!(row.contains(&peak), "mel {m} peak is not a sample");
        }
        // The rows tile the band.
        for k in 0..bins {
            let rowsum: f64 = (0..n_mels).map(|m| bank.filter(m).expect("filter")[k]).sum();
            assert!((rowsum - 1.0).abs() < 1e-12, "bin {k} rowsum {rowsum}");
        }

        // And energy is conserved for an arbitrary frame.
        let mut rng = lcg(0xE1E);
        let frame: Vec<Complex> = (0..bins).map(|_| Complex::from_polar(rng().abs() + 0.1, rng())).collect();
        let e = bank.energies(&frame);
        let naive: f64 = (0..n_mels)
            .map(|m| {
                (0..bins)
                    .map(|k| bank.filter(m).expect("filter")[k] * frame[k].power())
                    .sum::<f64>()
            })
            .sum();
        let total_power: f64 = frame.iter().map(|c| c.power()).sum();
        assert!((e.iter().sum::<f64>() - naive).abs() < 1e-9);
        assert!(
            (naive - total_power).abs() < 1e-9 * total_power,
            "fs={fs} n={fft_size} m={n_mels}: {naive} vs {total_power}"
        );
    }
}

#[test]
fn finely_resolved_mel_filters_peak_at_one() {
    // With enough bins per filter the sampled triangle reaches its full
    // height of 1 at the centre. This is the assertion that would catch a
    // triangle normalised to something other than unit peak (Slaney's
    // area-normalised variant, say).
    for (fs, fft_size, n_mels) in [
        (8_000.0f64, 2048usize, 20usize),
        (16_000.0f64, 4096usize, 40usize),
    ] {
        let bank = MelBank::new(n_mels, 0.0, fs / 2.0, fs, fft_size).expect("valid");
        let bin_hz = fs / (fft_size as f64);
        for (m, &centre) in mel_frequencies(n_mels, 0.0, fs / 2.0).iter().enumerate() {
            let row = bank.filter(m).expect("filter");
            // The row's peak is the bin nearest the centre, and it is high.
            let peak_bin = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
                .map(|(i, _)| i)
                .expect("non-empty");
            assert!(
                peak_bin.abs_diff((centre / bin_hz).round() as usize) <= 1,
                "fs={fs} mel {m}: peak at bin {peak_bin}, centre {centre}"
            );
            let peak = row.get(peak_bin).copied().unwrap_or(0.0);
            // How close the sampled peak gets to 1 depends on how many bins
            // span a filter, which is exactly the resolution argument: the
            // sampled triangle reaches its apex only when a bin lands on the
            // centre, and converges to it as the transform grows.
            let bins_per_filter = (row.len() as f64) / (n_mels as f64);
            let floor = 1.0 - (2.0 / bins_per_filter).min(1.0);
            assert!(
                peak > floor,
                "fs={fs} mel {m}: peak {peak}, floor {floor} ({bins_per_filter} bins/filter)"
            );
        }
    }
}

#[test]
fn mel_filterbank_is_narrow_at_low_frequencies_and_wide_at_high() {
    // The whole point of the mel scale: constant-Q in the low band, constant
    // bandwidth in the high band.
    let fs = 16_000.0;
    let fft_size = 4096;
    let bank = MelBank::new(40, 20.0, 7_800.0, fs, fft_size).expect("valid");
    let centres = mel_frequencies(40, 20.0, 7_800.0);
    let bin_hz = fs / (fft_size as f64);
    // Width in bins of each filter.
    let widths: Vec<usize> = bank
        .filters()
        .iter()
        .map(|row| row.iter().filter(|w| **w > 0.0).count())
        .collect();
    assert!(widths[0] < widths[39], "{widths:?}");
    // The low-frequency filters are narrow in Hz; the high ones are wide.
    let low_hz = widths[0] as f64 * bin_hz;
    let high_hz = widths[39] as f64 * bin_hz;
    assert!(low_hz < 200.0, "low filter {low_hz} Hz");
    assert!(high_hz > 600.0, "high filter {high_hz} Hz");
    // And the mel centres are what the filters peak on (within one bin).
    for (m, &centre) in centres.iter().enumerate() {
        let row = bank.filter(m).expect("filter");
        let peak_bin = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .map(|(i, _)| i)
            .expect("non-empty");
        let peak_hz = (peak_bin as f64) * bin_hz;
        assert!(
            (peak_hz - centre).abs() <= 2.0 * bin_hz,
            "mel {m}: peak {peak_hz} vs centre {centre}"
        );
    }
}

#[test]
fn mel_energies_track_where_the_energy_is() {
    let fs = 16_000.0;
    let fft_size = 1024;
    let bank = MelBank::new(24, 20.0, 7_800.0, fs, fft_size).expect("valid");
    let bin_hz = fs / (fft_size as f64);
    let bins = fft_size / 2 + 1;
    let centres = mel_frequencies(24, 20.0, 7_800.0);
    let nearest_band = |hz: f64| -> usize {
        centres
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1 - hz)
                    .abs()
                    .partial_cmp(&(b.1 - hz).abs())
                    .expect("finite")
            })
            .map(|(i, _)| i)
            .expect("non-empty")
    };
    for tone_hz in [100.0f64, 1_000.0, 6_000.0] {
        let expected_band = nearest_band(tone_hz);
        let bin = ((tone_hz / bin_hz).round() as usize).min(bins - 1);
        let frame: Vec<Complex> = (0..bins)
            .map(|k| Complex::new(if k == bin { 1.0 } else { 0.0 }, 0.0))
            .collect();
        let e = bank.energies(&frame);
        let (peak_m, &peak) = e
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .expect("non-empty");
        // The bin sits between filter centres, so the winning band's weight
        // is below 1 — but it is still the clear maximum.
        assert!(peak > 0.5, "{tone_hz} Hz: peak band {peak_m} at {peak}");
        assert!(
            expected_band.abs_diff(peak_m) <= 1,
            "{tone_hz} Hz (centre {} Hz) landed in band {peak_m}, expected ~{expected_band}",
            centres[expected_band]
        );
        // The total is exactly the (unit) input power.
        assert!((e.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn mfcc_matches_a_naive_dct_ii_reference() {
    // Naive O(K·M) orthonormal DCT-II of the log energies, computed with plain
    // loops, against the implementation — to 1e-12.
    let mut rng = lcg(0xD07);
    for (k, n_coeffs) in [
        (4usize, 4usize),
        (13usize, 13usize),
        (26usize, 13usize),
        (40usize, 20usize),
        (26usize, 99usize),
    ] {
        let energies: Vec<f64> = (0..k).map(|_| rng().abs() * 1e3 + 1e-6).collect();
        let got = mfcc(&energies, n_coeffs);
        assert_eq!(got.len(), k.min(n_coeffs));
        let logs: Vec<f64> = energies
            .iter()
            .map(|&e| if e > 1e-10 { e.ln() } else { 1e-10f64.ln() })
            .collect();
        for (m, &actual) in got.iter().enumerate() {
            let mut sum = 0.0;
            for (i, &l) in logs.iter().enumerate() {
                sum += l * (PI * ((m * (2 * i + 1)) as f64) / ((2 * k) as f64)).cos();
            }
            let scale = if m == 0 {
                1.0 / (k as f64).sqrt()
            } else {
                2.0f64.sqrt() / (k as f64).sqrt()
            };
            let want = scale * sum;
            assert!((actual - want).abs() < 1e-12, "K={k} m={m}: {actual} vs {want}");
        }
    }
}

#[test]
fn mfcc_of_a_constant_frame_has_only_c0() {
    // Every mel band equal → the log vector is constant → the DCT-II puts
    // everything in slot 0 and nothing in the rest. `c[0] = √K · ln(E)`.
    for k in [4usize, 13, 26] {
        for e in [0.5f64, 1.0, 2.0, 1e4] {
            let c = mfcc(&vec![e; k], k);
            assert_eq!(c.len(), k);
            let want = (k as f64).sqrt() * e.ln();
            assert!((c[0] - want).abs() < 1e-12, "K={k} e={e}: {} vs {want}", c[0]);
            assert!(c[1..].iter().all(|v| v.abs() < 1e-12), "K={k}: {c:?}");
        }
    }
}

#[test]
fn mfcc_respects_n_coeffs_and_degenerate_inputs() {
    let e: Vec<f64> = (0..26).map(|i| (i as f64 + 1.0) * 0.01).collect();
    for n in 0..=30 {
        assert_eq!(mfcc(&e, n).len(), n.min(26));
    }
    assert!(mfcc(&[], 13).is_empty());
    assert!(mfcc(&[], 0).is_empty());
    // A silent frame floors every log, so the coefficients stay finite and
    // c[0] takes the floor's value rather than −∞.
    let silent = mfcc(&vec![0.0; 13], 13);
    assert!(silent.iter().all(|v| v.is_finite()), "{silent:?}");
    let floor = 13f64.sqrt() * 1e-10f64.ln();
    assert!((silent[0] - floor).abs() < 1e-12);
}

#[test]
fn mfcc_lifter_leaves_c0_and_lifts_high_quefrencies() {
    let e: Vec<f64> = (0..26).map(|i| (i as f64 + 1.0) * 0.001).collect();
    let plain = mfcc(&e, 13);
    // HTK lifter: `1 + (L/2)·sin(π·m/L)`, which is 1 at m = 0 and rises.
    for l in [2usize, 13, 22] {
        let lifted = mfcc_with_lifter(&e, 13, l);
        assert_eq!(lifted.len(), 13);
        assert!((lifted[0] - plain[0]).abs() < 1e-15, "lifter {l} moved c[0]");
        for m in 1..13 {
            let q = m.min(l);
            let weight = 1.0 + 0.5 * (l as f64) * (PI * (q as f64) / (l as f64)).sin();
            assert!(
                (lifted[m] - plain[m] * weight).abs() < 1e-12,
                "lifter {l} m={m}: {} vs {}",
                lifted[m],
                plain[m] * weight
            );
            // Beyond the lifter order the quefrency index is clamped to L, so
            // the weight is fixed at `1 + (L/2)·sin(π) = 1` — i.e. no change.
            if m > l {
                assert!(
                    (lifted[m] - plain[m]).abs() < 1e-12,
                    "lifter {l} m={m}: {} vs {}",
                    lifted[m],
                    plain[m]
                );
            }
        }
    }
    // lifter = 0 is the identity.
    assert_eq!(mfcc_with_lifter(&e, 13, 0), plain);
}

#[test]
fn mfcc_from_a_real_stft_is_stable_across_frames() {
    // The end-to-end front end: stft → mel → log → DCT, on a tone. The MFCCs
    // must be finite, the right length, and dominated by c[0] (overall energy)
    // for a stationary tone.
    let fs = 16_000.0;
    let cfg = dsp_spectral::StftConfig::new(1024, 256).with_sample_rate(fs);
    let x: Vec<f64> = (0..8192)
        .map(|i| 0.5 * (TAU * 500.0 * i as f64 / fs).sin())
        .collect();
    let spec = stft(&x, &cfg).expect("valid");
    let bank = MelBank::new(26, 20.0, 7_800.0, fs, cfg.fft_size).expect("valid");
    // Stationary tone ⇒ the MFCC vector is stationary too. Skip the first and
    // last frame: centre padding reflects the signal there, so the edge frames
    // legitimately see a different spectrum.
    let mut reference: Option<Vec<f64>> = None;
    for f in 2..spec.num_frames() - 2 {
        let energies = bank.energies(spec.frame(f).expect("in range"));
        let coeffs = mfcc(&energies, 13);
        assert_eq!(coeffs.len(), 13);
        assert!(coeffs.iter().all(|v| v.is_finite()), "frame {f}: {coeffs:?}");
        match &reference {
            None => reference = Some(coeffs),
            Some(prev) => {
                for m in 0..13 {
                    assert!(
                        (coeffs[m] - prev[m]).abs() < 1e-9,
                        "frame {f} coefficient {m} moved: {} vs {}",
                        coeffs[m],
                        prev[m]
                    );
                }
            }
        }
    }
    assert!(reference.is_some());
    // The log energies fall off across the band for a pure tone (all the
    // energy is in the low mel filters), so c[0] is *not* necessarily the
    // largest coefficient — the spectrum's log-slope shows up in the low
    // orders. What must be bounded is the overall scale.
    let r = reference.expect("set");
    assert!(r.iter().all(|v| v.abs() < 100.0), "{r:?}");
}

#[test]
fn real_cepstrum_peaks_at_the_pitch_period() {
    // A periodic impulse train has a harmonic comb in its log spectrum, so its
    // cepstrum peaks at every multiple of the period. The first multiple is
    // the pitch.
    for (n, period) in [(1_024usize, 16usize), (2_048usize, 40usize), (4_096usize, 100usize)] {
        let x: Vec<f64> = (0..n)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect();
        let c = real_cepstrum(&x).expect("valid");
        assert_eq!(c.len(), n);
        assert!(c.iter().all(|v| v.is_finite()));
        // Cepstral energy is *periodic* in the period (the harmonic comb
        // repeats), so the global maximum lands on any multiple of it. Search
        // only the first full period to identify the pitch lag itself.
        let peak = (1..=period)
            .max_by(|a, b| c[*a].partial_cmp(&c[*b]).expect("finite"))
            .expect("non-empty");
        assert_eq!(peak, period, "n={n} period={period}");
        // The first few multiples are clear peaks. Beyond those the ripple
        // from the finite harmonic count makes later multiples unreliable in
        // sign, so the strong claim is kept to where the comb dominates.
        for q in 1..4.min(n / period) {
            let at = q * period;
            let on = c[at];
            let neighbours: Vec<f64> = (at.saturating_sub(4)..at + 5)
                .filter(|j| *j != at)
                .map(|j| c[j])
                .collect();
            let near = neighbours.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(on > near, "n={n} period={period} q={q}: {on} vs {near}");
        }
        // Cepstral energy sits on the multiples, not between them, for the
        // first few harmonics (where the comb is strong).
        for q in 1..4.min(n / period) {
            let on = c[q * period];
            let off = c[q * period + 1];
            assert!(
                on > 100.0 * off.abs(),
                "n={n} period={period} q={q}: {on} vs {off}"
            );
        }
    }
}

#[test]
fn real_cepstrum_tracks_the_fundamental_of_a_harmonic_tone() {
    // Two partials at f0 and 2f0 give a cepstral peak at the f0 period.
    let fs = 16_000.0f64;
    let f0 = 200.0f64;
    let n = 4_096usize;
    let period = (fs / f0).round() as usize;
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (TAU * f0 * t).sin() + 0.5 * (TAU * 2.0 * f0 * t).sin()
        })
        .collect();
    let c = real_cepstrum(&x).expect("valid");
    // Search only the plausible pitch range, since a two-partial tone also has
    // a peak at the half-period (the 2f0 component alone would).
    let lo = period / 2;
    let hi = period * 3 / 2;
    let peak = (lo..hi)
        .max_by(|a, b| c[*a].partial_cmp(&c[*b]).expect("finite"))
        .expect("non-empty range");
    // The integer-period 200 Hz tone at 16 kHz has period exactly 80 samples,
    // but the 400 Hz partial's own lag is 40 — i.e. the half-period. Searching
    // [period/2, 3·period/2) excludes it, and the peak lands within a sample
    // of the true lag.
    assert!(
        peak.abs_diff(period) <= 1,
        "cepstral peak {peak} vs period {period}"
    );
}

#[test]
fn real_cepstrum_is_real_and_symmetric() {
    let n = 1_024usize;
    let mut rng = lcg(0xCE);
    let x: Vec<f64> = (0..n).map(|_| rng()).collect();
    let c = real_cepstrum(&x).expect("valid");
    // The log magnitude spectrum was mirrored hermitianly, so the inverse is
    // real and symmetric: c[q] == c[N-q].
    for q in 1..n / 2 {
        assert!((c[q] - c[n - q]).abs() < 1e-12, "q={q}: {} vs {}", c[q], c[n - q]);
    }
    // c[0] carries the mean log magnitude. Compare it against that mean,
    // computed naively from the FFT magnitudes.
    let fft = dsp_core::fft::Fft::new(n).expect("power of two");
    let mut buf: Vec<f64> = x.iter().flat_map(|&v| [v, 0.0]).collect();
    fft.forward(&mut buf).expect("length matches");
    let log_sum: f64 = buf
        .chunks_exact(2)
        .map(|c| {
            let re = c[0];
            let im = c[1];
            let m = (re * re + im * im).sqrt();
            if m > 1e-10 { m.ln() } else { 1e-10f64.ln() }
        })
        .sum();
    let want_c0 = log_sum / (n as f64);
    assert!((c[0] - want_c0).abs() < 1e-9, "c[0] {} vs {want_c0}", c[0]);
}

#[test]
fn real_cepstrum_and_mel_reject_invalid_inputs_without_panicking() {
    assert!(real_cepstrum(&[]).is_err());
    assert!(real_cepstrum(&[1.0; 3]).is_err()); // not a power of two
    assert!(real_cepstrum(&[1.0; 1]).is_err());
    assert!(real_cepstrum(&[f64::NAN; 8]).is_err());
    assert!(real_cepstrum(&[f64::INFINITY; 8]).is_err());
    // Power-of-two lengths from 2 upward are all accepted.
    for n in [2usize, 4, 8, 16, 64] {
        assert!(real_cepstrum(&vec![0.5; n]).is_ok(), "n={n}");
    }
    // A zero signal floors every log to ln(1e-10), giving a flat cepstrum.
    let c = real_cepstrum(&vec![0.0; 64]).expect("valid");
    assert!((c[0] - 1e-10f64.ln()).abs() < 1e-12);
    assert!(c[1..].iter().all(|v| v.abs() < 1e-15));
}

#[test]
fn mel_filterbank_handles_short_and_empty_frames() {
    let bank = MelBank::new(8, 20.0, 7_800.0, 16_000.0, 512).expect("valid");
    // A frame shorter than the bank: summed over the bins it carries.
    let short: Vec<Complex> = (0..10).map(|_| Complex::new(1.0, 0.0)).collect();
    let e = bank.energies(&short);
    assert_eq!(e.len(), 8);
    assert!(e.iter().all(|v| v.is_finite()));
    let naive: f64 = (0..8)
        .map(|m| (0..10).map(|k| bank.filter(m).expect("filter")[k]).sum::<f64>())
        .sum();
    assert!((e.iter().sum::<f64>() - naive * 1.0).abs() < 1e-9);
    // An empty frame is zero energy everywhere.
    assert!(bank.energies(&[]).iter().all(|v| *v == 0.0));
    // The floor makes a silent frame's dB view exactly the floor.
    assert!(bank.energies_db(&[], -200.0).iter().all(|v| *v == -200.0));
}

#[test]
fn mel_bank_from_a_spectrum_matches_a_directly_built_one() {
    // `Spectrum::mel_filterbank` is the convenience path and must produce
    // exactly what `MelBank::new` does for the same parameters.
    let cfg = dsp_spectral::StftConfig::new(1024, 256).with_sample_rate(16_000.0);
    let mut rng = lcg(0x1F);
    let x: Vec<f64> = (0..4096).map(|_| rng()).collect();
    let spec = stft(&x, &cfg).expect("valid");
    let from_spec = spec.mel_filterbank(26, 20.0, 7_800.0, cfg.sample_rate);
    let direct = MelBank::new(26, 20.0, 7_800.0, cfg.sample_rate, cfg.fft_size).expect("valid");
    assert_eq!(from_spec, direct);
    // Degenerate arguments degrade to an empty bank rather than panicking.
    let empty = spec.mel_filterbank(0, 0.0, 0.0, 0.0);
    assert_eq!(empty.n_mels(), 0);
    assert!(empty.energies(spec.frame(0).expect("in range")).is_empty());
    let _ = Window::Hann;
    let _ = Spectrum::default();
}