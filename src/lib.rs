//! An implementation of Forward Decay to enable various aggregations over stream of items.
//! See [the research paper](http://dimacs.rutgers.edu/~graham/pubs/papers/fwddecay.pdf) for more details on forward decay.

use std::time::{Duration, Instant};

mod aggregate;

/// An item in a stream of inputs.
pub trait Item {
    /// The arrival timestamp for this item.
    fn timestamp(&self) -> Instant;

    /// The age in seconds (including fractional time) for this item.
    fn age(&self, landmark: Instant) -> f64;

    /// The value associated with this item.
    fn value(&self) -> f64;
}

impl Item for Instant {
    fn timestamp(&self) -> Instant {
        *self
    }

    fn age(&self, landmark: Instant) -> f64 {
        self.checked_duration_since(landmark)
            .as_ref()
            .map(Duration::as_secs_f64)
            .unwrap_or_else(|| -1.0 * landmark.duration_since(*self).as_secs_f64())
    }

    fn value(&self) -> f64 {
        f64::NAN
    }
}


impl Item for (Instant, f64) {
    fn timestamp(&self) -> Instant {
        self.0
    }

    fn age(&self, landmark: Instant) -> f64 {
        self.0.age(landmark)
    }

    fn value(&self) -> f64 {
        self.1
    }
}

impl<I> Item for &I where I: Item {
    fn timestamp(&self) -> Instant {
        (*self).timestamp()
    }

    fn age(&self, landmark: Instant) -> f64 {
        (*self).age(landmark)
    }

    fn value(&self) -> f64 {
        (*self).value()
    }
}

/// A decay function takes some information about the ith item, and returns a weight for this item.
/// It can depend on a variety of properties of the item such as ti, vi as well as the current time t,
/// but for brevity we will write it simply as w(i, t), or just w(i) when t is implicit.
/// We define a function w(i, t) to be a decay function if it satisfies the following properties:
/// 1. w(i, t) = 1 when ti = t and 0 ≤ w(i, t) ≤ 1 for all t ≥ ti.
/// 2. w is monotone non-increasing as time increases: t' ≥ t ⇒ w(i, t') ≤ w(i, t).
///
/// The forward decay is computed on the amount of time between the arrival of an item and a fixed point L,
/// known as the landmark. By convention, this landmark is some time earlier than all other items;
/// we discuss how this landmark can be chosen below.
/// Thus, we are looking forward in time from the landmark to see the item,
/// instead of looking backward from the current time.
///
/// ## Examples
///
/// ### No decay
/// g(n) = 1 for all n.
///
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::ForwardDecay;
///
/// let landmark = Instant::now();
/// let fd = ForwardDecay::new(Instant::now(), |_| 1.0);
///
/// let weight = fd.weight(landmark + Duration::from_secs(5), landmark + Duration::from_secs(10));
///
/// assert_eq!(weight, 1.0);
/// ```
///
/// ### Polynomial decay
/// g(n) = n ^ β for some parameter β > 0
///
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::ForwardDecay;
///
/// let beta = 2;
/// let landmark = Instant::now();
/// let fd = ForwardDecay::new(Instant::now(), |n| n.powi(beta));
///
/// let weight = fd.weight(landmark + Duration::from_secs(5), landmark + Duration::from_secs(10));
/// let result = format!("{weight:.8}");
///
/// assert!(vec!["0.24999999", "0.25000000"].contains(&result.as_str()));
/// ```
///
/// ### Exponential decay
/// g(n) = exp(αn) for parameter α>0.
///
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::ForwardDecay;
///
/// let alpha = 0.2;
/// let landmark = Instant::now();
/// let fd = ForwardDecay::new(landmark, |n| (alpha * n).exp());
///
/// let weight = fd.weight(landmark + Duration::from_secs(5), landmark + Duration::from_secs(10));
///
/// assert_eq!(weight, 0.3678794411714423);
/// ```
///
/// ### Landmark Window
/// g(n) = 1 for n > 0, and 0 otherwise.
///
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::ForwardDecay;
///
/// let beta = 2;
/// let landmark = Instant::now();
/// let fd = ForwardDecay::new(landmark, |n| if n > 0.0 { 1.0 } else { 0.0 });
///
/// let weight = fd.weight(landmark + Duration::from_secs(5), landmark + Duration::from_secs(10));
///
/// assert_eq!(weight, 1.0);
/// ```
pub struct ForwardDecay<G> {
    landmark: Instant,
    g: G,
}

impl<G> ForwardDecay<G>
where
    G: Fn(f64) -> f64,
{
    /// Create a new instance with a positive monotone non-decreasing function and a landmark time.
    pub fn new(landmark: Instant, g: G) -> Self {
        Self {
            landmark,
            g,
        }
    }

    /// Update the landmark to the given timestamp.
    pub fn set_landmark(&mut self, landmark: Instant) {
        self.landmark = landmark;
    }

    /// Given a positive monotone non-decreasing function g, and a landmark time L,
    /// the decayed weight of an item with arrival time ti > L measured at time t ≥ ti
    /// is given by w(i, t) = g(ti − L) / g(t − L).
    pub fn weight<I>(&self, item: I, timestamp: Instant) -> f64
    where
        I: Item,
    {
        (self.g)(item.age(self.landmark)) / (self.g)(timestamp.age(self.landmark))
    }

    /// The weight of an item without the normalizing factor of 1 / g(t - L).
    /// Has the property of remaining constant for a given item when the landmark remains constant.
    pub fn raw_weight<I>(&self, item: I) -> f64
    where
        I: Item,
    {
        (self.g)(item.age(self.landmark))
    }

    /// The weighted value of the item without the normalizing factor of 1 / g(t - L).
    /// Has the property of remaining constant for a given item when the landmark remains constant.
    pub fn raw_weighted_value<I>(&self, item: I) -> f64
    where
        I: Item,
    {
        (self.g)(item.age(self.landmark)) * item.value()
    }

    /// In order to normalize values given that the function value increases with time,
    /// we typically need to include a normalizing factor in terms of g(t),
    /// the function of the current time.
    pub fn normalizing_factor(&self, timestamp: Instant) -> f64
    {
        (self.g)(timestamp.age(self.landmark))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example() {
        let landmark = Instant::now();
        let stream = vec![5, 7, 3, 8, 4];
        let fd = ForwardDecay::new(landmark, |n: f64| n.powi(2));
        let now = landmark + Duration::from_secs(10);

        let result: Vec<f64> = stream.into_iter()
            .map(|i| landmark + Duration::from_secs(i))
            .map(|i| fd.weight(i, now))
            .collect();
        let weights = vec![0.25, 0.49, 0.09, 0.64, 0.16];

        assert_eq!(result, weights);
    }

    #[test]
    fn age() {
        let landmark = Instant::now();

        assert_eq!((landmark - Duration::from_secs(1)).age(landmark), -1.0);
        assert_eq!(landmark.age(landmark), 0.0);
        assert_eq!((landmark + Duration::from_secs(5)).age(landmark), 5.0);
        assert_eq!((landmark + Duration::from_secs(10)).age(landmark), 10.0);
    }
}

