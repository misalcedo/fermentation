//! An implementation of Forward Decay to enable various aggregations over stream of items.
//! See [the research paper](http://dimacs.rutgers.edu/~graham/pubs/papers/fwddecay.pdf) for more details on forward decay.

use std::time::{Duration, Instant};

mod aggregate;

pub use aggregate::AggregateComputation;

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

impl<I> Item for &I
where
    I: Item,
{
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
/// let expected = 0.25;
/// let epsilon = 0.00001;
///
/// assert!(weight >= (expected - epsilon) && weight <= (expected + epsilon));
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
/// let  item = landmark + Duration::from_secs(5);
/// let  now = landmark + Duration::from_secs(10);
///
/// let mut fd = ForwardDecay::new(landmark, |n| if n > 0.0 { 1.0 } else { 0.0 });
///
/// assert_eq!(fd.weight(item, now), 1.0);
///
/// fd.set_landmark(item + Duration::from_secs(1));
///
/// assert_ne!(fd.landmark(), landmark);
/// assert_eq!(fd.weight(item, now), 0.0);
/// ```
///
/// ### Aggregate Computations
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::{AggregateComputation, ForwardDecay};
///
/// let landmark = Instant::now();
/// let stream = vec![(5, 4.0), (7, 8.0), (3, 3.0), (8, 6.0), (4, 4.0)];
/// let fd = ForwardDecay::new(landmark, |n: f64| n.powi(2));
/// let now = landmark + Duration::from_secs(10);
///
/// let mut sum = fd.sum();
/// let mut count = fd.count();
/// let mut average = fd.average();
/// let mut min = fd.min();
/// let mut max = fd.max();
///
/// stream.into_iter()
///     .map(|(offset, value)| (landmark + Duration::from_secs(offset), value))
///     .for_each(|item| {
///         sum.update(&item);
///         count.update(&item);
///         average.update(&item);
///         min.update(&item);
///         max.update(&item);
///     });
///
/// let epsilon = 0.01;
///
/// assert_eq!(sum.query(now), 9.67);
/// assert_eq!(count.query(now), 1.63);
/// assert!(average.query(now) >= (5.93 - epsilon) && average.query(now) <= (5.93 + epsilon));
/// assert_eq!(min.query(now), Some(3.0 * 0.09));
/// assert_eq!(max.query(now), Some(8.0 * 0.49));
/// ```
///
/// ### Changing Landmark with Exponential Decay
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::{AggregateComputation, ForwardDecay, Item};
///
/// let landmark = Instant::now();
/// let new_landmark = landmark + Duration::from_secs(1);
/// let age = new_landmark.age(landmark);
/// let stream = vec![(5, 4.0), (7, 8.0), (3, 3.0), (8, 6.0), (4, 4.0)];
/// let now = landmark + Duration::from_secs(10);
/// let alpha = 0.1;
///
/// let mut fd = ForwardDecay::new(landmark, |n: f64| (alpha * n).exp());
///
/// let (scaled_sum, scaled_count, scaled_average) = {
///     let mut sum = fd.sum();
///     let mut count = fd.count();
///     let mut average = fd.average();
///
///     stream.iter()
///         .map(|(offset, value)| (landmark + Duration::from_secs(*offset), *value))
///         .for_each(|item| {
///             sum.update(&item);
///             count.update(&item);
///             average.update(&item);
///         });
///
///     sum.scale(fd.g()(-age));
///     count.scale(fd.g()(-age));
///     average.scale(fd.g()(-age));
///
///     (sum.query(now), count.query(now), average.query(now))
/// };
///
/// let actual_age = fd.set_landmark(new_landmark);
///
/// assert_eq!(age, actual_age);
///
/// let mut sum = fd.sum();
/// let mut count = fd.count();
/// let mut average = fd.average();
///
/// stream.into_iter()
///     .map(|(offset, value)| (landmark + Duration::from_secs(offset), value))
///     .for_each(|item| {
///         sum.update(&item);
///         count.update(&item);
///         average.update(&item);
///     });
///
/// assert_eq!(scaled_sum, sum.query(now));
/// assert_eq!(scaled_count, count.query(now));
/// assert_eq!(scaled_average, average.query(now));
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
        (self.g)(item.age(self.landmark)) / (self.g)(timestamp.age(self.landmark))
    }

    /// The weight of an item without the normalizing factor of 1 / g(t - L).
    /// Has the property of remaining constant for a given item when the landmark remains constant.
    pub fn static_weight<I>(&self, item: I) -> f64
    where
        I: Item,
    {
        (self.g)(item.age(self.landmark))
    }

    /// The weighted value of the item without the normalizing factor of 1 / g(t - L).
    /// Has the property of remaining constant for a given item when the landmark remains constant.
    pub fn static_weighted_value<I>(&self, item: I) -> f64
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

    pub fn sum<I>(&self) -> aggregate::Sum<'_, G, I>
    where
        I: Item,
    {
        aggregate::Sum::<'_, G, I>::new(self)
    }

    pub fn count<I>(&self) -> aggregate::Count<'_, G, I>
    where
        I: Item,
    {
        aggregate::Count::<'_, G, I>::new(self)
    }

    pub fn average<I>(&self) -> aggregate::Average<'_, G, I>
    where
        I: Item,
    {
        aggregate::Average::<'_, G, I>::new(self)
    }

    pub fn min<I>(&self) -> aggregate::Min<'_, G, I>
    where
        I: Item + Clone,
    {
        aggregate::Min::<'_, G, I>::new(self)
    }

    pub fn max<I>(&self) -> aggregate::Max<'_, G, I>
    where
        I: Item + Clone,
    {
        aggregate::Max::<'_, G, I>::new(self)
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
    fn scaled_exponential() {
        let landmark = Instant::now();
        let tick = Duration::from_secs(1);
        let new_landmark = landmark + tick;
        let stream = vec![5, 7, 3, 8, 4];
        let now = landmark + Duration::from_secs(10);
        let alpha = 1.0;

        let mut fd = ForwardDecay::new(landmark, |n: f64| (alpha * n).exp());

        let previous_weights: Vec<f64> = stream.iter()
            .map(|i| landmark + Duration::from_secs(*i))
            .map(|i| fd.static_weight(i))
            .collect();
        let age = fd.set_landmark(new_landmark);
        let factor = fd.g()(-age);
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

