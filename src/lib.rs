//! An implementation of Forward Decay to enable various aggregations over stream of items.
//! See [the research paper](http://dimacs.rutgers.edu/~graham/pubs/papers/fwddecay.pdf) for more details on forward decay.

use std::time::{Duration, Instant};

mod aggregate;
pub mod g;
mod item;

pub use aggregate::ArithmeticAggregation;
pub use item::Item;
use crate::g::Function;

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
/// ## Numerical Issues
/// A common feature of the above techniques—indeed,
/// the key technique that allows us to track the decayed weights
/// efficiently—is that they maintain counts and other quantities
/// based on g(ti − L), and only scale by g(t − L) at query time.
/// But while g(ti−L)/g(t−L) is guaranteed to lie between zero
/// and one, the intermediate values of g(ti − L) could become
/// very large.
///
/// For polynomial functions, these values should
/// not grow too large, and should be effectively represented in
/// practice by floating point values without loss of precision. For
/// exponential functions, these values could grow quite large as
/// new values of (ti − L) become large, and potentially exceed
/// the capacity of common floating point types. However, since
/// the values stored by the algorithms are linear combinations
/// of g values (scaled sums), they can be rescaled relative to a
/// new landmark. That is, by the analysis of exponential decay,
/// the choice of L does not affect the final result.
/// We can therefore multiply each value based on L by a factor
/// of exp(−α(L' − L)), and obtain the correct value as if we
/// had instead computed relative to a new landmark L'
/// (and then use this new L' at query time).
/// This can be done with a linear
/// pass over whatever data structure is being used.
pub struct ForwardDecay<G> {
    landmark: Instant,
    g: G,
}

impl<G> ForwardDecay<G>
where
    G: Function,
{
    /// Create a new instance with a positive monotone non-decreasing function and a landmark time.
    pub fn new(landmark: Instant, g: G) -> Self {
        Self {
            landmark,
            g,
        }
    }

    /// The function g for this decay model.
    pub fn g(&self) -> &G {
        &self.g
    }

    /// The landmark for this decay model.
    pub fn landmark(&self) -> Instant {
        self.landmark
    }

    /// Update the landmark to the given timestamp.
    /// Returns the age of the new landmark relative to the previous landmark.
    pub fn set_landmark(&mut self, landmark: Instant) -> f64 {
        let age = landmark.age(self.landmark);
        self.landmark = landmark;
        age
    }

    /// Given a positive monotone non-decreasing function g, and a landmark time L,
    /// the decayed weight of an item with arrival time ti > L measured at time t ≥ ti
    /// is given by w(i, t) = g(ti − L) / g(t − L).
    pub fn weight<I>(&self, item: I, timestamp: Instant) -> f64
    where
        I: Item,
    {
        self.g.invoke(item.age(self.landmark)) / self.g.invoke(timestamp.age(self.landmark))
    }

    /// The weight of an item without the normalizing factor of 1 / g(t - L).
    /// Has the property of remaining constant for a given item when the landmark remains constant.
    pub fn static_weight<I>(&self, item: I) -> f64
    where
        I: Item,
    {
        self.g.invoke(item.age(self.landmark))
    }

    /// The weighted value of the item without the normalizing factor of 1 / g(t - L).
    /// Has the property of remaining constant for a given item when the landmark remains constant.
    pub fn static_weighted_value<I>(&self, item: I) -> f64
    where
        I: Item,
    {
        self.g.invoke(item.age(self.landmark)) * item.value()
    }

    /// In order to normalize values given that the function value increases with time,
    /// we typically need to include a normalizing factor in terms of g(t),
    /// the function of the current time.
    pub fn normalizing_factor(&self, timestamp: Instant) -> f64
    {
        self.g.invoke(timestamp.age(self.landmark))
    }

    pub fn aggregate<I>(self) -> ArithmeticAggregation<G, I>
    where
        I: Item,
    {
        ArithmeticAggregation::<G, I>::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example() {
        let landmark = Instant::now();
        let stream = vec![5, 7, 3, 8, 4];
        let fd = ForwardDecay::new(landmark, g::Polynomial::new(2));
        let now = landmark + Duration::from_secs(10);

        let result: Vec<f64> = stream.into_iter()
            .map(|i| landmark + Duration::from_secs(i))
            .map(|i| fd.weight(i, now))
            .collect();
        let weights = vec![0.25, 0.49, 0.09, 0.64, 0.16];

        assert_eq!(result, weights);
    }

    #[test]
    fn scaled_exponential() {
        let landmark = Instant::now();
        let tick = Duration::from_secs(1);
        let new_landmark = landmark + tick;
        let stream = vec![5, 7, 3, 8, 4];
        let alpha = 1.0;

        let mut fd = ForwardDecay::new(landmark, g::Exponential::new(alpha));

        let previous_weights: Vec<f64> = stream.iter()
            .map(|i| landmark + Duration::from_secs(*i))
            .map(|i| fd.static_weight(i))
            .collect();
        let age = fd.set_landmark(new_landmark);
        let factor = fd.g().invoke(-age);
        let new_weights: Vec<f64> = stream.iter()
            .map(|i| landmark + Duration::from_secs(*i))
            .map(|i| fd.static_weight(i))
            .collect();

        let factors: Vec<f64> = new_weights.iter().zip(previous_weights).map(|(a, b)| ((a / b) - factor).abs()).collect();
        let epsilon = 0.001;

        assert_eq!(age, tick.as_secs_f64());
        assert!(factors.iter().all(|d| *d < epsilon));
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

