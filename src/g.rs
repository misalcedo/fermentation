/// A positive monotone non-decreasing function g, used to calculate the decayed weight of an item.
/// Implementors are responsible for ensuring the range of the function adheres to these requirements.
pub trait Function {
    fn invoke(&self, age: f64) -> f64;
}

impl Function for () {
    fn invoke(&self, _: f64) -> f64 {
        1.0
    }
}

/// • Exponential decay: g(n) = exp(α * n) for parameter α > 0.
#[derive(Copy, Clone)]
pub struct Exponential(f64);

impl Exponential {
    /// ## Panic
    /// Panics when alpha is not greater than 0.
    pub fn new(alpha: f64) -> Self {
        if !(alpha > 0.0) {
            panic!("alpha must be greater than 0, given {alpha}");
        }

        Self(alpha)
    }
}

impl Function for Exponential {
    fn invoke(&self, age: f64) -> f64 {
        (self.0 * age).exp()
    }
}

/// Polynomial decay: g(n) = n ^ β for some parameter β > 0.
#[derive(Copy, Clone)]
pub struct Polynomial(i32);

impl Polynomial {
    /// ## Panic
    /// Panics when beta is not greater than 0.
    pub fn new(beta: i32) -> Self {
        if !(beta > 0) {
            panic!("beta must be greater than 0, given {beta}");
        }

        Self(beta)
    }
}

impl Function for Polynomial {
    fn invoke(&self, age: f64) -> f64 {
        age.powi(self.0)
    }
}

/// Landmark Window: g(n) = 1 for n > 0, and 0 otherwise.
#[derive(Copy, Clone)]
pub struct LandmarkWindow;

impl Function for LandmarkWindow {
    fn invoke(&self, age: f64) -> f64 {
        if age > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Copy, Clone)]
pub struct Custom<F>(F);

impl<F> From<F> for Custom<F> where F: Fn(f64) -> f64 {
    fn from(f: F) -> Self {
        Self(f)
    }
}

impl<F> Custom<F> where F: Fn(f64) -> f64 {
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<F> Function for Custom<F> where F: Fn(f64) -> f64 {
    fn invoke(&self, age: f64) -> f64 {
        self.0(age)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_decay() {
        assert_eq!(().invoke(1.0), 1.0);
        assert_eq!(().invoke(0.0), 1.0);
        assert_eq!(().invoke(-1.0), 1.0);
    }

    #[test]
    fn exponential() {
        assert_eq!(Exponential::new(1.0).invoke(1.0), 1.0_f64.exp());
    }

    #[test]
    #[should_panic]
    fn negative_exponential() {
        Exponential::new(-1.0);
    }

    #[test]
    #[should_panic]
    fn zero_exponential() {
        Exponential::new(0.0);
    }

    #[test]
    fn polynomial() {
        assert_eq!(Polynomial::new(3).invoke(2.0), 8.0);
    }

    #[test]
    #[should_panic]
    fn negative_polynomial() {
        Polynomial::new(-3);
    }

    #[test]
    #[should_panic]
    fn zero_polynomial() {
        Polynomial::new(0);
    }

    #[test]
    fn landmark() {
        assert_eq!(LandmarkWindow.invoke(1.0), 1.0);
        assert_eq!(LandmarkWindow.invoke(0.0), 0.0);
        assert_eq!(LandmarkWindow.invoke(-1.0), 0.0);
    }

    #[test]
    fn custom() {
        assert_eq!(Custom::from(|n| n * 0.2).invoke(1.0), 0.2);
        assert_eq!(Custom::from(|n| n * 0.2).invoke(0.0), 0.0);
        assert_eq!(Custom::from(|n| n * 0.2).invoke(-1.0), -0.2);
    }
}