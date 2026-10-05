//! The crate-wide error type.
//!
//! Every fallible entry point in `dsp-spectral` — [`stft`](crate::stft),
//! [`real_cepstrum`](crate::real_cepstrum), the restoration primitives —
//! validates its inputs up front and returns [`SpectralError`] instead of
//! panicking. The crate denies `clippy::unwrap_used`, `expect_used`,
//! `panic`, and `indexing_slicing`, so an untrusted sample buffer can only
//! ever produce a typed error.

use alloc::string::String;
use core::fmt;

/// Errors returned by the spectral layer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralError {
    /// A configuration value was outside its valid range: a non-power-of-two
    /// or too-small `fft_size`, a zero or oversized `hop`, an empty or
    /// empty-kernel convolution filter, an out-of-range mel count, a
    /// non-positive sample rate, or a noise profile whose length does not
    /// match the spectrum it is applied to.
    Config(String),
    /// An input slice had no samples at all.
    EmptyInput,
    /// A frame index was outside `0..num_frames`.
    FrameOutOfRange(usize),
    /// An input sample, or a derived quantity, was NaN or ±∞.
    NonFinite,
    /// A numerical kernel from `dsp-core` failed (its own typed error,
    /// rendered here).
    Dsp(String),
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpectralError::Config(msg) => write!(f, "invalid spectral configuration: {msg}"),
            SpectralError::EmptyInput => write!(f, "input is empty"),
            SpectralError::FrameOutOfRange(i) => write!(f, "frame index {i} is out of range"),
            SpectralError::NonFinite => {
                write!(f, "input contains a non-finite value (NaN or infinity)")
            }
            SpectralError::Dsp(msg) => write!(f, "dsp-core kernel failed: {msg}"),
        }
    }
}

impl core::error::Error for SpectralError {}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Unit tests over the error contract; assertions instead of unwraps.
    use super::SpectralError;
    use alloc::string::{String, ToString};

    #[test]
    fn display_is_stable() {
        assert_eq!(
            SpectralError::Config(String::from("hop must be > 0")).to_string(),
            "invalid spectral configuration: hop must be > 0"
        );
        assert_eq!(SpectralError::EmptyInput.to_string(), "input is empty");
        assert_eq!(
            SpectralError::FrameOutOfRange(7).to_string(),
            "frame index 7 is out of range"
        );
        assert_eq!(
            SpectralError::NonFinite.to_string(),
            "input contains a non-finite value (NaN or infinity)"
        );
        assert_eq!(
            SpectralError::Dsp(String::from("size 100 is not a power of two")).to_string(),
            "dsp-core kernel failed: size 100 is not a power of two"
        );
    }

    #[test]
    fn is_std_error_and_equatable() {
        fn assert_error<E: core::error::Error>(_: &E) {}
        assert_error(&SpectralError::NonFinite);
        assert_eq!(SpectralError::EmptyInput, SpectralError::EmptyInput);
        assert_ne!(SpectralError::NonFinite, SpectralError::EmptyInput);
    }
}
