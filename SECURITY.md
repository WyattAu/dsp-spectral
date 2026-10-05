# Security Policy — dsp-spectral

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Report privately via [GitHub security advisories] for this repository, or
email **wyatt_au@protonmail.com**. Do **not** open a public issue for
security reports.

You will receive an acknowledgement within **72 hours**. Coordinated
disclosure: we ask for up to 90 days before public disclosure while a
patch ships.

## Scope notes

`dsp-spectral` is a numeric DSP library. Security considerations for
integrators:

- **No network, filesystem, or process state** — the crate is pure
  computation over caller-owned buffers and has no attack surface beyond
  its numeric inputs.
- **`process` paths are panic-free by construction** (crate denies
  `unwrap_used`/`expect_used`/`panic`/`indexing_slicing`; no unsafe) and
  allocation-free, so malicious or corrupted inputs cannot stall a
  real-time thread via panics, allocator contention, or denormal
  microcode (state is flushed to zero below 1e-20).
- **NaN propagation**: `stft` and `NoiseProfile::estimate` *reject* non-finite
  input with `SpectralError::NonFinite` rather than propagating it. The
  per-frame feature reducers, by contrast, are total by design and will report
  a finite `0.0` for a silent frame — a caller who feeds NaN into a
  hand-built `Spectrum` gets NaN out of the feature functions, which is IEEE
  behaviour and not masked.
- **Unbounded input**: `stft`, `NoiseProfile::estimate`, and `spectral_gate`
  allocate proportionally to their input (one `Vec` per frame). Callers
  bounding an untrusted buffer should bound its length before analysis; the
  fuzz target caps samples at 4096 and the transform at 2²⁰ for the same
  reason.
- `#![forbid(unsafe_code)]` — no unsafe blocks exist in this crate.

[GitHub security advisories]:
    https://github.com/WyattAu/dsp-spectral/security/advisories/new
