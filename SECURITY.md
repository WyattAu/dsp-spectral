# Security Policy — dsp-core

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

`dsp-core` is a numeric DSP library. Security considerations for
integrators:

- **No network, filesystem, or process state** — the crate is pure
  computation over caller-owned buffers and has no attack surface beyond
  its numeric inputs.
- **`process` paths are panic-free by construction** (crate denies
  `unwrap_used`/`expect_used`/`panic`/`indexing_slicing`; no unsafe) and
  allocation-free, so malicious or corrupted inputs cannot stall a
  real-time thread via panics, allocator contention, or denormal
  microcode (state is flushed to zero below 1e-20).
- **NaN propagation**: NaN inputs propagate through the math as IEEE NaN.
  Callers gating untrusted input should validate before feeding process
  paths; the crate never panics on NaN but downstream stages will see it.
- `#![forbid(unsafe_code)]` — no unsafe blocks exist in this crate.

[GitHub security advisories]:
    https://github.com/WyattAu/dsp-core/security/advisories/new
