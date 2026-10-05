//! Spectral audio analysis and restoration — STFT/ISTFT, mel/MFCC features,
//! spectral gating, HPSS source separation.
//!
//! `dsp-spectral` is the estate's **L1** audio layer: the analysis and
//! restoration vocabulary that noise-reduction tools, voice activity
//! detectors, speech front ends, and audio-restoration pipelines are built
//! from. The numerical kernels (radix-2 FFT, cosine-sum windows, dB
//! conversion) are owned by [`dsp_core`] (L0); this crate owns the framing,
//! the spectral algebra on top, and the restoration primitives.
//!
//! # Design
//!
//! - **Analysis–synthesis is exact.** `istft(stft(x), x.len()) == x` to
//!   better than 1e-10, for every window and hop, because the inverse
//!   divides by the *measured* per-sample squared-window overlap instead of a
//!   constant COLA divisor. See [`stft`].
//! - **f64 throughout**, matching `dsp-core`; f32 is a consumer concern.
//! - **Typed errors, never panics.** Every fallible entry point returns
//!   [`SpectralError`]; the crate denies `clippy::unwrap_used`, `expect_used`,
//!   `panic`, and `indexing_slicing`, so an untrusted buffer (a decoded
//!   media file, a live capture) can only produce a typed error.
//! - **Feature definitions are pinned.** Each descriptor is spelled out in
//!   its own module docs and checked against a naive `O(N·M)` reference
//!   implementation in the test suite to 1e-9.
//! - **Composable restoration.** [`spectral_gate`], [`spectral_subtract`],
//!   [`harmonic_percussive_split`], and [`spectral_smooth`] all preserve the
//!   framing, so they chain freely.
//! - **Verified.** proptest round-trips over 200 arbitrary
//!   signal/config pairs and bounds-checks the feature invariants over 300
//!   cases each; the `spectral_fuzz` cargo-fuzz target feeds arbitrary bytes
//!   as samples and arbitrary configs through the whole pipeline.
//!
//! # Example
//!
//! Denoise a noisy recording, then read out its spectral character:
//!
//! ```
//! use dsp_spectral::{
//!     GateConfig, NoiseProfile, denoise, mel_frequencies, mfcc, snr_db, spectral_centroid,
//!     spectral_flatness, stft, StftConfig,
//! };
//!
//! let fs = 16_000.0;
//! let cfg = StftConfig::new(512, 128).with_sample_rate(fs);
//! let n = 4_096;
//!
//! // A 500 Hz tone, and a deterministic uniform-noise generator.
//! let clean: Vec<f64> = (0..n)
//!     .map(|i| 0.5 * (core::f64::consts::TAU * 500.0 * i as f64 / fs).sin())
//!     .collect();
//! let mut next_noise = |seed: u64| {
//!     let mut s = seed;
//!     move || {
//!         s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
//!         0.02 * (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0)
//!     }
//! };
//!
//! // Profile the noise from a noise-only segment, then gate. The profile
//! // must come from noise at the same level as the noise being removed, or
//! // the gate has nothing to compare against.
//! let mut r = next_noise(0xA11CE);
//! let noise_only: Vec<f64> = (0..n).map(|_| r()).collect();
//! let profile = NoiseProfile::estimate(&noise_only, &cfg).expect("valid config");
//! let mut r = next_noise(0xB0B);
//! let noisy: Vec<f64> = clean.iter().zip((0..n).map(|_| r())).map(|(a, b)| a + b).collect();
//!
//! let gate = GateConfig {
//!     threshold_db: -3.0,
//!     reduction_db: -18.0,
//!     smoothing_frames: 3,
//! };
//! let restored = denoise(&noisy, &cfg, &profile, &gate).expect("valid");
//! assert_eq!(restored.len(), noisy.len());
//!
//! // The point of the whole crate: the restoration measurably helps.
//! let before = snr_db(&clean, &noisy);
//! let after = snr_db(&clean, &restored);
//! assert!(after > before, "SNR {before} dB -> {after} dB");
//!
//! // A lone tone centres on its own frequency and is far from flat.
//! let tone = stft(&clean, &cfg).expect("valid");
//! let centroid = spectral_centroid(&tone, fs);
//! // The first and last frames are half zero-padded (`center: true`), which
//! // smears the tone's sidelobes, so the tight claim is made on the interior.
//! let interior = &centroid[1..centroid.len() - 1];
//! assert!(
//!     interior.iter().all(|c| (c - 500.0).abs() < 160.0),
//!     "centroid {centroid:?}"
//! );
//! // Broadband noise pulls the centroid up and the flatness toward 1, so the
//! // restored signal must sit between the clean tone and the noisy input.
//! let restored_centroid = spectral_centroid(&stft(&restored, &cfg).expect("valid"), fs);
//! let noisy_centroid = spectral_centroid(&stft(&noisy, &cfg).expect("valid"), fs);
//! let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
//! assert!(
//!     mean(&restored_centroid) < mean(&noisy_centroid),
//!     "restoration should pull the centroid back toward the tone"
//! );
//! assert!(
//!     spectral_flatness(&tone).iter().all(|f| (0.0..0.5).contains(f)),
//!     "a pure tone is never flat"
//! );
//!
//! // Mel/MFCC front end.
//! assert_eq!(mel_frequencies(26, 20.0, 7_800.0).len(), 26);
//! let spec = stft(&restored, &cfg).expect("valid");
//! let bank = spec.mel_filterbank(26, 20.0, 7_800.0, fs);
//! let coeffs = mfcc(&bank.energies(spec.frame(0).expect("in range")), 13);
//! assert_eq!(coeffs.len(), 13);
//! ```
//!
//! # Modules
//!
//! | Module | Contents |
//! |---|---|
//! | [`stft`] | [`stft`] / [`istft`] / [`Spectrum`], frame layout, time-scale helper |
//! | [`config`] | [`StftConfig`] — window, hop, transform size, centring, overlap |
//! | [`window`] | [`Window`] — Hann, Hamming, Blackman, Blackman–Harris, flat top, rectangular |
//! | [`features`] | centroid, bandwidth, flatness, rolloff, flux, zero-crossing rate |
//! | [`mel`] | [`MelBank`], mel scale, MFCC (DCT-II), real cepstrum |
//! | [`restore`] | [`NoiseProfile`], [`spectral_gate`], [`spectral_subtract`], [`harmonic_percussive_split`], [`spectral_smooth`], [`denoise`] |
//! | [`complex`] | [`Complex`] — the minimal complex value the spectral bins carry |
//! | [`error`] | [`SpectralError`] |

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Pedantic is surfaced (not denied) by the shared CI job; these four groups
// are allowed here with the reason they are inherent to the domain, and
// nothing else is suppressed.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "spectral algebra is index math: bin numbers and hop offsets are               usize by construction and become f64 bin frequencies, and f64               magnitudes become back into linear gains. Every conversion is               bounded by an explicit range check (bins < fft_size/2 + 1, taps               indexed through .get()) before it is used."
)]

extern crate alloc;

pub mod complex;
pub mod config;
pub mod error;
pub mod features;
pub mod mel;
pub mod restore;
pub mod stft;
pub mod window;

pub use complex::Complex;
pub use config::StftConfig;
pub use error::SpectralError;
pub use features::{
    spectral_bandwidth, spectral_centroid, spectral_flatness, spectral_flux, spectral_rolloff,
    zero_crossing_rate,
};
pub use mel::{
    hz_to_mel, mel_frequencies, mel_to_hz, mfcc, mfcc_with_lifter, real_cepstrum, MelBank,
};
pub use restore::{
    denoise, harmonic_percussive_split, snr_db, spectral_gate, spectral_smooth, spectral_subtract,
    GateConfig, NoiseProfile,
};
pub use stft::{istft, resample_frames, stft, Spectrum};
pub use window::Window;
