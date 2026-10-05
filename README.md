# dsp-spectral

Spectral audio analysis and restoration — STFT/ISTFT, mel/MFCC features,
spectral gating, HPSS source separation.

**Layer:** L1 — substrate · **Estate deps:** `dsp-core` (L0) · **Runtime
deps:** `dsp-core`, `libm` · `#![no_std]` + `alloc` (CI exercises
`thumbv7em-none-eabihf` and `wasm32-unknown-unknown`).

## What's here

| Module | Contents |
|---|---|
| [`stft`](https://docs.rs/dsp-spectral/latest/dsp_spectral/stft/index.html) | [`stft`] / [`istft`] / [`Spectrum`], frame layout, centre padding, [`resample_frames`] |
| [`config`](https://docs.rs/dsp-spectral/latest/dsp_spectral/config/index.html) | [`StftConfig`] — window, hop, transform size, centring, nominal overlap, sample rate |
| [`window`](https://docs.rs/dsp-spectral/latest/dsp_spectral/window/index.html) | [`Window`] — Hann, Hamming, Blackman, Blackman–Harris, flat top, rectangular (symmetric convention) |
| [`features`](https://docs.rs/dsp-spectral/latest/dsp_spectral/features/index.html) | [`spectral_centroid`], [`spectral_bandwidth`], [`spectral_flatness`], [`spectral_rolloff`], [`spectral_flux`], [`zero_crossing_rate`] |
| [`mel`](https://docs.rs/dsp-spectral/latest/dsp_spectral/mel/index.html) | [`MelBank`], [`hz_to_mel`] / [`mel_to_hz`], [`mel_frequencies`], [`mfcc`] (orthonormal DCT-II + lifter), [`real_cepstrum`] |
| [`restore`](https://docs.rs/dsp-spectral/latest/dsp_spectral/restore/index.html) | [`NoiseProfile`], [`spectral_gate`], [`spectral_subtract`], [`harmonic_percussive_split`], [`spectral_smooth`], [`denoise`], [`snr_db`] |
| [`complex`](https://docs.rs/dsp-spectral/latest/dsp_spectral/complex/index.html) | [`Complex`] — the minimal complex value the spectral bins carry |
| [`error`](https://docs.rs/dsp-spectral/latest/dsp_spectral/error/index.html) | [`SpectralError`] — the exhaustive error taxonomy |

## Guarantees

- **Analysis–synthesis is exact.** `istft(stft(x), x.len()) == x` to better
  than **1e-15** for white noise and musical content, for every window and
  every valid hop. The inverse divides by the *measured* per-sample
  squared-window overlap rather than a constant COLA divisor, so it is exact
  for any window/hop pair — including the ones that are not COLA, which is all
  the tapered windows here.
- **Two documented non-reconstructions**, both properties of the analysis and
  not of the inverse: samples past [`Spectrum::covered_len`] (only reachable
  with `center = false` and a `len` that is not `fft_size + k·hop`), and the
  endpoints of a zero-tapered window when uncentred. Both come back as exact
  `0.0`; everything else reconstructs.
- **Every feature is pinned by a naive reference.** Each descriptor's
  definition is spelled out in its module docs and checked against an
  `O(N·M)` double-loop reference implementation in the test suite to **1e-9**,
  on synthetic spectrograms and on real STFTs.
- **Totality under untrusted input.** Every fallible entry point returns a
  typed [`SpectralError`]; the crate denies `clippy::unwrap_used`, `expect_used`,
  `panic`, and `indexing_slicing`, and `#![forbid(unsafe_code)]`. The
  `spectral_fuzz` cargo-fuzz target drives arbitrary bytes through the whole
  pipeline.
- **Verified.** 130+ tests: proptest round-trips over 200 arbitrary
  signal/config pairs and bounds-invariants over hundreds of cases each, plus
  doctests on every public entry point. `cargo llvm-cov` line coverage ≥ 90 %.
- **Restoration measurably works.** The end-to-end `denoise` test asserts a
  real SNR margin, and the demo reports **+7.2 dB** on the built-in test
  signal at the default gate settings.

## Example

```rust
use dsp_spectral::{
    GateConfig, NoiseProfile, denoise, mfcc, spectral_centroid, stft, StftConfig,
};

let cfg = StftConfig::new(1024, 256).with_sample_rate(16_000.0);

// Profile the noise floor from a noise-only segment, then gate the noisy
// recording against it.
let noise_only: Vec<f64> = vec![/* room tone */];
let profile = NoiseProfile::estimate(&noise_only, &cfg).expect("valid config");
let restored: Vec<f64> = denoise(&noisy, &cfg, &profile, &GateConfig::default())?;

// Read the spectrum back out.
let spec = stft(&restored, &cfg).expect("valid");
let centroid = spectral_centroid(&spec, cfg.sample_rate);
let bank = spec.mel_filterbank(26, 0.0, cfg.sample_rate / 2.0, cfg.sample_rate);
let mfccs = mfcc(&bank.energies(spec.frame(0).expect("in range")), 13);
# Ok::<(), dsp_spectral::SpectralError>(())
```

## Conventions worth knowing before you use it

- **f64 samples throughout**, matching `dsp-core`; f32 is a consumer concern.
- **The FFT is unnormalized.** `Σ_n (x·w)² = (1/N)·Σ_k |X[k]|²` (Parseval,
  tested), so a full-scale sine on a bin centre reads `N/2 · coherent gain` —
  about `N/4` for a Hann window, whose *symmetric* coherent gain is exactly
  `(N−1)/(2N)` and not 0.5.
- **`GateConfig::threshold_db` is positive to gate noise.** A noise-only bin
  measures *at* the profile (0 dB of headroom), so `threshold_db = +6` means
  "attenuate everything within 6 dB of the floor".
- **`power` is linear**, not decibels, and Parseval holds on it directly;
  `power_db` / `magnitude_db` (floored at −200 dBFS) are there for display.
- **`MelBank` filters are unnormalised**, so the bank tiles `[0, Nyquist]` and
  `energies` sums to the frame's total power exactly (tested to 1e-9).
- **`hop <= fft_size / 2`**, which `validate` enforces: a larger hop leaves
  samples covered by no frame at all, which no normalisation can repair.

## Demo

```sh
cargo run --release --example restoration_demo
```

Synthesises tone + noise + clicks, estimates the noise profile from a leading
noise-only segment, then runs gate → subtract → HPSS and prints the before/after
SNR plus the extracted features. Output is deterministic (a fixed-seed LCG), so
the numbers below are reproducible:

```text
input SNR          21.66 dB
gated SNR          28.85 dB  (+7.18 dB)
analysis–synthesis round-trip: 308.6 dB SNR (1.9e-17 reconstruction error)
```

## Development

```sh
cargo build --locked --all-features
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all --check
cargo deny check advisories licenses bans
cargo llvm-cov --all-features --fail-under-lines 90
cargo +nightly fuzz run spectral_fuzz -- -max_total_time=60
```

[MIT OR Apache-2.0](LICENSE-MIT) · [GitHub](https://github.com/WyattAu/dsp-spectral) · [crates.io](https://crates.io/crates/dsp-spectral)