//! Complex arithmetic for spectral bins.
//!
//! `dsp-core` deliberately exposes the FFT over *interleaved `f64` buffers*
//! (`dsp_core::fft::Fft`) rather than a complex element type, so there is no
//! complex type to reuse and this crate owns a minimal one: a
//! `Copy` Cartesian pair with the handful of operations the spectral layer
//! actually performs (magnitude, phase, scaling, conjugation, element-wise
//! product). It is 16 bytes, `no_std`, and allocates nothing.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex number in Cartesian form.
///
/// The FFT is unnormalized in both directions, so a bin's
/// [`magnitude`](Complex::magnitude) is a raw amplitude (a full-scale sine
/// sitting exactly on a bin centre reads `N/2`) and its
/// [`power`](Complex::power) is the unnormalized power-spectrum value used by
/// Parseval's identity `Σ x² = (1/N)·Σ|X|²`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// The additive identity, `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Construct from Cartesian parts.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Construct from polar form: `magnitude · e^{i·phase}`.
    #[must_use]
    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            re: magnitude * libm::cos(phase),
            im: magnitude * libm::sin(phase),
        }
    }

    /// Magnitude `|z| = √(re² + im²)`.
    #[must_use]
    pub fn magnitude(self) -> f64 {
        libm::sqrt(self.power())
    }

    /// Squared magnitude `|z|²` — the linear power-spectrum value.
    #[must_use]
    pub fn power(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase `arg(z)` in `(−π, π]`.
    #[must_use]
    pub fn phase(self) -> f64 {
        libm::atan2(self.im, self.re)
    }

    /// Conjugate.
    #[must_use]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// `true` when both parts are finite (neither NaN nor ±∞).
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    /// Scalar (real) scaling — the per-bin gain a spectral gate applies.
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl MulAssign<f64> for Complex {
    fn mul_assign(&mut self, rhs: f64) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        rhs * self
    }
}

impl Div<f64> for Complex {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::Complex;
    use core::f64::consts::{FRAC_PI_4, PI};

    #[test]
    fn magnitude_power_phase_round_trip() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.magnitude() - 5.0).abs() < 1e-15);
        assert!((z.power() - 25.0).abs() < 1e-13);
        // arg(3 + 4i) = atan(4/3) = 0.9272952180016122 rad.
        assert!((z.phase() - 0.9272952180016122).abs() < 1e-12);
        let polar = Complex::from_polar(z.magnitude(), z.phase());
        assert!((polar.re - 3.0).abs() < 1e-14 && (polar.im - 4.0).abs() < 1e-14);
        // Negative real axis: phase is ±π, magnitude positive.
        assert!((Complex::new(-2.0, 0.0).phase().abs() - PI).abs() < 1e-15);
        assert!((Complex::new(-2.0, 0.0).magnitude() - 2.0).abs() < 1e-15);
        assert_eq!(Complex::ZERO.magnitude(), 0.0);
    }

    #[test]
    fn arithmetic_operators() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        assert_eq!(a + b, Complex::new(4.0, 1.0));
        assert_eq!(a - b, Complex::new(-2.0, 3.0));
        // (1+2i)(3−i) = 3 − i + 6i − 2i² = 5 + 5i
        assert_eq!(a * b, Complex::new(5.0, 5.0));
        assert_eq!(a * 2.0, Complex::new(2.0, 4.0));
        assert_eq!(2.0 * a, Complex::new(2.0, 4.0));
        assert_eq!(a / 2.0, Complex::new(0.5, 1.0));
        assert_eq!(-a, Complex::new(-1.0, -2.0));
        assert_eq!(a.conj(), Complex::new(1.0, -2.0));
        assert_eq!(a + b, b + a);
        // i·i = −1
        let i = Complex::new(0.0, 1.0);
        assert_eq!(i * i, Complex::new(-1.0, 0.0));

        let mut acc = a;
        acc += b;
        assert_eq!(acc, Complex::new(4.0, 1.0));
        acc -= b;
        assert_eq!(acc, a);
        acc *= b;
        assert_eq!(acc, a * b);
        acc *= 0.5;
        assert!((acc.re - 2.5).abs() < 1e-15);
    }

    #[test]
    fn quarter_turn_rotations() {
        let z = Complex::from_polar(2.0, FRAC_PI_4);
        assert!((z.re - core::f64::consts::SQRT_2).abs() < 1e-15);
        assert!((z.im - core::f64::consts::SQRT_2).abs() < 1e-15);
        assert!(z.is_finite());
        assert!(!Complex::new(f64::NAN, 0.0).is_finite());
        assert!(!Complex::new(0.0, f64::INFINITY).is_finite());
        assert_eq!(Complex::default(), Complex::ZERO);
    }
}
