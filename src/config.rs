//! STFT framing configuration.
//!
//! [`StftConfig`] is the single knob-bundle every entry point takes: the
//! window shape, the hop (stride) between frames, the transform size, whether
//! the signal is centre-padded, the declared overlap constant, and the
//! sample rate that pins the frequency axis. [`StftConfig::validate`] is
//! called by every fallible entry point, so an invalid configuration is a
//! typed [`SpectralError::Config`] rather than a panic or a wrong answer.

use alloc::format;
use alloc::string::String;

use crate::error::SpectralError;
use crate::window::Window;

/// Default transform size for [`StftConfig::new`] — 2048.
pub const DEFAULT_FFT_SIZE: usize = 2048;
/// Default hop for [`StftConfig::new`] — 512 (75 % overlap).
pub const DEFAULT_HOP: usize = 512;
/// Default sample rate for [`StftConfig::new`] — 48 kHz.
pub const DEFAULT_SAMPLE_RATE: f64 = 48_000.0;

/// Framing parameters for [`stft`](crate::stft) / [`istft`](crate::istft).
#[derive(Debug, Clone, PartialEq)]
pub struct StftConfig {
    /// Analysis window shape (symmetric convention).
    pub window: Window,
    /// Samples between successive frame starts. Must satisfy
    /// `1 <= hop <= fft_size / 2`.
    pub hop: usize,
    /// Transform size in samples. Must be a power of two `>= 2`; the useful
    /// bin count is `fft_size / 2 + 1`.
    pub fft_size: usize,
    /// Centre-pad the signal by `fft_size / 2` on both sides (reflected) so
    /// frame `0` is centred on sample `0`. This is what makes
    /// `istft(stft(x), x.len()) == x` exact at the signal edges; with
    /// `center = false` the window's zero-valued end taps cannot contribute
    /// signal energy and the first/last samples are unrecoverable.
    pub center: bool,
    /// The nominal squared-window overlap constant
    /// `Σ_{k : k·hop < fft_size} w[k·hop]²`, filled in by
    /// [`new`](StftConfig::new) for the configured window/hop pair.
    ///
    /// Informational only: [`istft`](crate::istft) measures the actual
    /// per-sample overlap and divides by that, which is exact for any hop and
    /// any window. Mutating `window` or `hop` in place leaves this stale;
    /// [`StftConfig::with_window`] / `with_hop` refresh it, and
    /// [`is_cola`](StftConfig::is_cola) reports whether the pair satisfies
    /// constant overlap-add at all.
    pub window_overlap: f32,
    /// Sample rate in Hz — pins the frequency axis (`bin k → k·fs/N`) and
    /// is carried into [`NoiseProfile`](crate::NoiseProfile). Must be finite
    /// and `> 0`. The transform itself never uses it.
    pub sample_rate: f64,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self::new(DEFAULT_FFT_SIZE, DEFAULT_HOP)
    }
}

impl StftConfig {
    /// A Hann-windowed configuration with `center = true`,
    /// `sample_rate = 48 kHz`, and `window_overlap` measured for the pair.
    #[must_use]
    pub fn new(fft_size: usize, hop: usize) -> Self {
        let window = Window::Hann;
        Self {
            window_overlap: window.overlap_sum(fft_size, hop) as f32,
            window,
            hop,
            fft_size,
            center: true,
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }

    /// The de-facto default: 2048-point Hann STFT, 512-sample hop
    /// (43.07 Hz bins at 44.1 kHz, 21 ms frames, 75 % overlap).
    #[must_use]
    pub fn hann_fft2048() -> Self {
        Self::new(2048, 512)
    }

    /// Validate every field.
    ///
    /// # Errors
    ///
    /// Returns [`SpectralError::Config`] when `hop == 0`, `fft_size < 2`,
    /// `fft_size` is not a power of two, `hop > fft_size / 2`,
    /// `sample_rate` is not finite and positive, or `window_overlap` is not
    /// finite and non-negative.
    pub fn validate(&self) -> Result<(), SpectralError> {
        if self.fft_size < 2 {
            return Err(SpectralError::Config(String::from(
                "fft_size must be >= 2",
            )));
        }
        if !self.fft_size.is_power_of_two() {
            return Err(SpectralError::Config(format!(
                "fft_size {} is not a power of two",
                self.fft_size
            )));
        }
        if self.hop == 0 {
            return Err(SpectralError::Config(String::from("hop must be > 0")));
        }
        // `hop > fft_size / 2` would leave samples near the end of the
        // signal covered by no frame at all (or by a frame's zero-valued end
        // tap), which no amount of window normalisation can repair.
        if self.hop > self.fft_size / 2 {
            return Err(SpectralError::Config(format!(
                "hop {} must not exceed fft_size / 2 ({})",
                self.hop,
                self.fft_size / 2
            )));
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(SpectralError::Config(String::from(
                "sample_rate must be finite and > 0",
            )));
        }
        if !self.window_overlap.is_finite() || self.window_overlap < 0.0 {
            return Err(SpectralError::Config(String::from(
                "window_overlap must be finite and >= 0",
            )));
        }
        Ok(())
    }

    /// Frequency of bin `k` in Hz (`k·fs/N`); `None` when `k` is out of range.
    ///
    /// Valid `k` are `0..=fft_size/2` — bin `fft_size/2` is Nyquist.
    #[must_use]
    pub fn bin_frequency(&self, k: usize) -> Option<f64> {
        if k >= self.bin_count() {
            None
        } else {
            Some(self.sample_rate * (k as f64) / (self.fft_size as f64))
        }
    }

    /// Useful bin count, `fft_size / 2 + 1` (DC through Nyquist).
    #[must_use]
    pub fn bin_count(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Frame duration in seconds.
    #[must_use]
    pub fn frame_duration(&self) -> f64 {
        (self.fft_size as f64) / self.sample_rate
    }

    /// Hop duration in seconds — the time between successive frames.
    #[must_use]
    pub fn hop_duration(&self) -> f64 {
        (self.hop as f64) / self.sample_rate
    }

    /// Nyquist frequency in Hz.
    #[must_use]
    pub fn nyquist(&self) -> f64 {
        self.sample_rate / 2.0
    }

    /// The measured squared-window overlap for the current window/hop pair.
    #[must_use]
    pub fn nominal_overlap(&self) -> f64 {
        self.window.overlap_sum(self.fft_size, self.hop)
    }

    /// Whether the window/hop pair satisfies constant overlap-add: every
    /// output position accumulates the same squared-window weight (within
    /// `tolerance`, relative to [`nominal_overlap`](Self::nominal_overlap)).
    ///
    /// The probe is deliberately the *interior* only: the first and last
    /// `fft_size` samples of any signal are always covered by fewer frames,
    /// which no window/hop pair can fix — and which is exactly why
    /// [`istft`](crate::istft) normalises by the measured per-sample weight
    /// instead of a constant. This predicate answers the narrower question
    /// "is the steady state flat?", which is what a host wants when it needs
    /// a constant synthesis gain.
    ///
    /// A rectangular window is COLA at any hop that divides `fft_size` (every
    /// output position then sees the same number of full frames). The tapered
    /// windows are COLA only *approximately*: symmetric Hann at
    /// `hop == fft_size/4` gives 1.4939 at phase 0 and 1.4853 half a hop
    /// along — a 0.6 % ripple that a symmetric window's taper-to-zero
    /// endpoints make unavoidable. That ripple is precisely why [`istft`]
    /// normalises by the measured per-sample overlap instead of a constant.
    #[must_use]
    pub fn is_cola(&self, tolerance: f64) -> bool {
        let n = self.fft_size;
        if n == 0 || self.hop == 0 {
            return false;
        }
        let w = self.window.coefficients(n);
        let overlap_at = |residue: usize| {
            let mut acc = 0.0;
            let mut idx = residue;
            while idx < n {
                if let Some(&tap) = w.get(idx) {
                    acc += tap * tap;
                }
                idx += self.hop;
            }
            acc
        };
        let nominal = self.nominal_overlap();
        if nominal <= 0.0 {
            return false;
        }
        let probes = self.hop.min(n);
        (0..probes).all(|r| (overlap_at(r) - nominal).abs() <= tolerance * nominal)
    }

    /// Return a copy with `window` replaced and `window_overlap` re-measured.
    #[must_use]
    pub fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self.window_overlap = self.nominal_overlap() as f32;
        self
    }

    /// Return a copy with `hop` replaced and `window_overlap` re-measured.
    #[must_use]
    pub fn with_hop(mut self, hop: usize) -> Self {
        self.hop = hop;
        self.window_overlap = self.nominal_overlap() as f32;
        self
    }

    /// Return a copy with `fft_size` replaced and `window_overlap`
    /// re-measured.
    #[must_use]
    pub fn with_fft_size(mut self, fft_size: usize) -> Self {
        self.fft_size = fft_size;
        self.window_overlap = self.nominal_overlap() as f32;
        self
    }

    /// Return a copy with `sample_rate` replaced.
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Return a copy with `center` replaced.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::{DEFAULT_HOP, DEFAULT_SAMPLE_RATE, StftConfig};
    use crate::error::SpectralError;
    use crate::window::Window;
    use alloc::string::ToString;

    #[test]
    fn hann_fft2048_defaults() {
        let cfg = StftConfig::hann_fft2048();
        assert_eq!(cfg.fft_size, 2048);
        assert_eq!(cfg.hop, 512);
        assert_eq!(cfg.window, Window::Hann);
        assert!(cfg.center);
        assert_eq!(cfg.sample_rate, DEFAULT_SAMPLE_RATE);
        assert_eq!(cfg.bin_count(), 1025);
        // Hann at 75 % overlap: w[0]² + w[n/4]² + w[n/2]² + w[3n/4]² = 2.
        // Symmetric Hann at 75 % overlap: w[0]² + w[n/4]² + w[n/2]² + w[3n/4]²,
        // where n/2 is one tap past the window's true peak (symmetric windows
        // peak at (n−1)/2), giving ≈ 1.5 rather than the periodic window's
        // clean 2.0.
        assert!((cfg.nominal_overlap() - 1.5).abs() < 1e-2, "overlap {}", cfg.nominal_overlap());
        assert!((f64::from(cfg.window_overlap) - 1.5).abs() < 1e-2);
        assert_eq!(StftConfig::default(), StftConfig::new(2048, DEFAULT_HOP));
        assert!((cfg.frame_duration() - 2048.0 / 48_000.0).abs() < 1e-15);
        assert!((cfg.hop_duration() - 512.0 / 48_000.0).abs() < 1e-15);
        assert!((cfg.nyquist() - 24_000.0).abs() < 1e-9);
    }

    #[test]
    fn frequency_axis() {
        let cfg = StftConfig::new(1024, 256).with_sample_rate(8_000.0);
        assert!((cfg.bin_frequency(0).unwrap() - 0.0).abs() < 1e-15);
        assert!((cfg.bin_frequency(512).unwrap() - 4_000.0).abs() < 1e-9);
        assert!((cfg.bin_frequency(256).unwrap() - 2_000.0).abs() < 1e-9);
        // Bin 512 is Nyquist; 513 is past the end of the half-spectrum.
        assert_eq!(cfg.bin_frequency(513), None);
    }

    #[test]
    fn validate_rejects_bad_configs() {
        let base = StftConfig::new(1024, 256);
        assert!(base.validate().is_ok());

        let mut cfg = base.clone();
        cfg.fft_size = 100;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("not a power of two"));

        let mut cfg = base.clone();
        cfg.fft_size = 0;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
        cfg.fft_size = 1;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));

        let mut cfg = base.clone();
        cfg.hop = 0;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));

        // hop > fft_size and hop in (fft_size/2, fft_size] both rejected.
        let mut cfg = base.clone();
        cfg.hop = 2048;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
        let mut cfg = base.clone();
        cfg.hop = 513;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
        // hop == fft_size / 2 is the boundary and is accepted.
        let mut cfg = base.clone();
        cfg.hop = 512;
        assert!(cfg.validate().is_ok());

        let mut cfg = base.clone();
        cfg.sample_rate = 0.0;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
        cfg.sample_rate = f64::NAN;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
        cfg.sample_rate = -1.0;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));

        let mut cfg = base.clone();
        cfg.window_overlap = -1.0;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
        cfg.window_overlap = f32::NAN;
        assert!(matches!(cfg.validate(), Err(SpectralError::Config(_))));
    }

    #[test]
    fn cola_detection() {
        // Rectangular at hop == n: every position sees exactly one frame.
        let cfg = StftConfig::new(64, 64)
            .with_window(Window::Rectangular);
        assert!(cfg.is_cola(1e-9));
        // Rectangular is COLA whenever the hop divides the window (then every
        // position sees the same number of full frames) — and not at hop 7,
        // where the residue classes cover 9 vs 10 frames.
        for hop in [1usize, 2, 4, 8, 16, 32, 64] {
            let cfg = StftConfig::new(64, hop).with_window(Window::Rectangular);
            assert!(cfg.is_cola(1e-9), "rect hop {hop}");
        }
        let cfg = StftConfig::new(64, 7).with_window(Window::Rectangular);
        assert!(!cfg.is_cola(1e-9));
        // Rectangular at 50 % overlap is COLA too.
        let cfg = StftConfig::new(256, 128).with_window(Window::Rectangular);
        assert!(cfg.is_cola(1e-9));

        // Tapered windows only *approach* COLA. Symmetric Hann at 25 %
        // overlap has ~0.6 % ripple, so the predicate is tolerance-sensitive:
        // strict 1e-9 says no, a realistic 1e-2 says yes.
        let cfg = StftConfig::new(256, 64);
        assert!(!cfg.is_cola(1e-9));
        assert!(cfg.is_cola(2e-2), "hann 75 % overlap should be near-COLA");
        // Hann at 50 % overlap: phase 0 sums to 1.0, half a hop along to
        // 0.5 — 100 % ripple, so no tolerance below 1 makes it pass.
        let cfg = StftConfig::new(256, 128);
        assert!(!cfg.is_cola(0.5));
        // Degenerate pairs are never COLA.
        let mut cfg = StftConfig::new(256, 64);
        cfg.hop = 0;
        assert!(!cfg.is_cola(1e-9));
        let cfg = StftConfig::new(0, 1);
        assert!(!cfg.is_cola(1e-9));
    }

    #[test]
    fn builders_refresh_overlap() {
        let cfg = StftConfig::new(256, 64).with_window(Window::Rectangular);
        assert!((cfg.nominal_overlap() - 4.0).abs() < 1e-12);
        assert!((f64::from(cfg.window_overlap) - 4.0).abs() < 1e-12);
        let cfg = cfg.with_hop(256);
        assert!((cfg.nominal_overlap() - 1.0).abs() < 1e-12);
        let cfg = cfg.with_fft_size(512);
        assert!((cfg.nominal_overlap() - 2.0).abs() < 1e-12);
        let _ = StftConfig::hann_fft2048().with_hop(256).nominal_overlap();
        let cfg = cfg.with_center(false).with_sample_rate(44_100.0);
        assert!(!cfg.center);
        assert!((cfg.sample_rate - 44_100.0).abs() < 1e-9);
    }
}