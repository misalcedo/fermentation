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

pub struct Exponential(f64);

impl Exponential {
    pub fn new(alpha: f64) -> Self {
        Self(alpha)
    }
}

impl Function for Exponential {
    fn invoke(&self, age: f64) -> f64 {
        (self.0 * age).exp()
    }
}

pub struct Polynomial(i32);

impl Polynomial {
    pub fn new(beta: i32) -> Self {
        Self(beta)
    }
}

impl Function for Polynomial {
    fn invoke(&self, age: f64) -> f64 {
        age.powi(self.0)
    }
}

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
    fn polynomial() {
        assert_eq!(Polynomial::new(3).invoke(2.0), 8.0);
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