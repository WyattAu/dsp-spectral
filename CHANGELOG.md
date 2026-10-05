# Changelog

All notable changes to this project are documented here. Format: [Keep a
Changelog](https://keepachangelog.com/) — versions follow [semver](https://semver.org).

## [Unreleased]

## [0.1.0] - 2026-10-06

### Added

- **`stft`** — short-time Fourier transform and its inverse. `stft` frames,
  windows, and transforms with the radix-2 FFT owned by `dsp-core`, keeping the
  `fft_size/2 + 1` useful bins of every frame; `istft` reverses it by hermitian
  expansion, inverse transform, re-window, and overlap-add **divided by the
  measured per-sample squared-window weight**. That last choice is what makes
  the round trip exact (better than 1e-15 for white noise and musical content,
  every window, every valid hop) instead of approximately right for the handful
  of window/hop pairs that happen to be COLA. `Spectrum` is the spectrogram
  itself, with infallible accessors (out-of-range indices yield empty slices,
  never panics) and a fallible `frame_checked` for callers that want the index
  back. `resample_frames` re-hops a spectrogram for pitch-preserving
  time-scaling.
- **`config`** — `StftConfig` (window, hop, transform size, centre padding,
  nominal overlap constant, sample rate) with `validate` returning a typed
  error for every invalid field, plus `bin_frequency`, `nominal_overlap`, and a
  `is_cola` predicate. `StftConfig::hann_fft2048` is the de-facto default.
- **`window`** — symmetric Hann, Hamming, Blackman, Blackman–Harris, flat top,
  and rectangular, with the endpoint values documented and tested.
- **`features`** — `spectral_centroid`, `spectral_bandwidth`,
  `spectral_flatness`, `spectral_rolloff`, `spectral_flux`, and
  `zero_crossing_rate`. Each definition is spelled out in the module docs and
  checked against a naive `O(N·M)` reference to 1e-9.
- **`mel`** — the HTK mel scale (`hz_to_mel` / `mel_to_hz`, exact inverse to
  1e-9 across the audio band), `mel_frequencies`, `MelBank` (unnormalised
  triangular filters that tile `[0, Nyquist]`, so `energies` sums to the
  frame's total power exactly), `mfcc` / `mfcc_with_lifter` (orthonormal
  DCT-II with an optional HTK sinusoidal lifter), and `real_cepstrum`.
- **`restore`** — `NoiseProfile` (from a whole buffer or from named frames),
  `GateConfig` + `spectral_gate` (per-bin hard threshold against the profile
  with optional time-axis gain smoothing), `spectral_subtract` (magnitude-domain
  Boll subtraction, phase preserved), `harmonic_percussive_split` (Fitzgerald
  HPSS via median filtering with a complementary Wiener mask),
  `spectral_smooth` (frequency-axis magnitude convolution), `denoise`
  (`stft → gate → istft` in one call), and `snr_db` for measuring the result.
- **`complex`** — `Complex`, the minimal `Copy` complex value the bins carry
  (`dsp-core` exposes its FFT over interleaved `f64` buffers, so there was no
  existing type to reuse).
- **`examples/restoration_demo`** — tone + noise + clicks, profiled from a
  leading noise-only segment, through gate → subtract → HPSS, printing
  before/after SNR and the extracted features.

### Design

- `#![no_std]` + `alloc`; the only dependencies are `dsp-core` (L0, the FFT,
  cosine-sum windows, dB conversion) and `libm` (no_std transcendentals, so no
  new supply-chain surface). Builds for `thumbv7em-none-eabihf` and
  `wasm32-unknown-unknown`.
- f64 throughout, matching `dsp-core`.
- Typed errors, never panics: the crate denies `clippy::unwrap_used`,
  `expect_used`, `panic`, and `indexing_slicing`, and forbids unsafe.
- Verified with 130+ tests: naive-reference agreement for every feature to
  1e-9, proptest round-trips over 200 arbitrary signal/config pairs and
  bounds-invariants over hundreds of cases each, doctests on every public entry
  point, and a `spectral_fuzz` cargo-fuzz target. Line coverage ≥ 90 % under
  `cargo llvm-cov`.
- Estate layer: **L1** (substrate over `dsp-core`'s L0 kernels).

[0.1.0]: https://github.com/WyattAu/dsp-spectral/releases/tag/v0.1.0
