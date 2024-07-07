use std::marker::PhantomData;
use std::time::Instant;
use crate::{Aggregator, ForwardDecay, Item};
use crate::g::{Exponential, Function};

/// Decayed aggregate sum, count and average over a stream of items.
///
/// ## Examples
/// ### Basic Aggregation
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::{aggregate::BasicAggregator, Aggregator, ForwardDecay, g};
///
/// let decay = ForwardDecay::new(Instant::now(), g::Polynomial::new(2));
/// let landmark = decay.landmark();
/// let now = landmark + Duration::from_secs(10);
/// let stream = vec![
///     (landmark + Duration::from_secs(5), 4.0),
///     (landmark + Duration::from_secs(7), 8.0),
///     (landmark + Duration::from_secs(3), 3.0),
///     (landmark + Duration::from_secs(8), 6.0),
///     (landmark + Duration::from_secs(4), 4.0),
/// ];
///
/// let mut aggregator = BasicAggregator::new(decay);
///
/// for item in stream {
///     aggregator.update(item);
/// }
///
/// let epsilon = 0.01;
///
/// assert_eq!(aggregator.sum(now), 9.67);
/// assert_eq!(aggregator.count(now), 1.63);
/// assert!(aggregator.average() >= (5.93 - epsilon) && aggregator.average() <= (5.93 + epsilon));
/// ```
///
/// ### Update Landmark
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::{aggregate::BasicAggregator, Aggregator, ForwardDecay, g};
///
/// let decay = ForwardDecay::new(Instant::now(), g::Exponential::new(0.2));
/// let landmark = decay.landmark();
/// let new_landmark = landmark + Duration::from_secs(1);
/// let now = landmark + Duration::from_secs(10);
/// let stream = vec![
///     (landmark + Duration::from_secs(5), 4.0),
///     (landmark + Duration::from_secs(7), 8.0),
///     (landmark + Duration::from_secs(3), 3.0),
///     (landmark + Duration::from_secs(8), 6.0),
///     (landmark + Duration::from_secs(4), 4.0),
/// ];
///
/// let mut aggregator = BasicAggregator::new(decay);
/// let mut clone = aggregator.clone();
///
/// clone.reset(new_landmark);
///
/// for item in stream {
///     aggregator.update(item);
///     clone.update(item);
/// }
///
/// aggregator.update_landmark(new_landmark);
///
/// let epsilon = 0.0001;
///
/// assert!((aggregator.sum(now) - clone.sum(now)).abs() < epsilon);
/// assert!((aggregator.count(now) - clone.count(now)).abs() < epsilon);
/// assert!((aggregator.average() - clone.average()).abs() < epsilon);
/// ```
#[derive(Copy, Clone)]
pub struct BasicAggregator<G, I> {
    decay: ForwardDecay<G>,
    sum: f64,
    count: f64,
    _phantom_data: PhantomData<I>
}

impl<G, I> Aggregator for BasicAggregator<G, I> where G: Function, I: Item {
    type Item = I;

    fn update(&mut self, item: I) {
        let static_weight = self.decay.static_weight(&item);

        self.sum += static_weight * item.value();
        self.count += static_weight;
    }

    fn reset(&mut self, landmark: Instant) {
        self.decay.set_landmark(landmark);
        self.sum = 0.0;
        self.count = 0.0;
    }
}

impl<I> BasicAggregator<Exponential, I>
where
    I: Item,
{
    pub fn update_landmark(&mut self, landmark: Instant) {
        let age = self.decay.set_landmark(landmark);
        let factor = self.decay.g().invoke(age);

        self.sum /= factor;
        self.count /= factor;
    }
}

impl<G, I> BasicAggregator<G, I>
where
    G: Function,
    I: Item,
{
    pub fn new(decay: ForwardDecay<G>) -> Self {
        Self {
            decay,
            sum: 0.0,
            count: 0.0,
            _phantom_data: Default::default()
        }
    }

    pub fn sum(&self, timestamp: Instant) -> f64 {
        self.sum / self.decay.normalizing_factor(timestamp)
    }

    pub fn static_sum(&self) -> f64 {
        self.sum
    }

    pub fn count(&self, timestamp: Instant) -> f64 {
        self.count / self.decay.normalizing_factor(timestamp)
    }

    pub fn static_count(&self) -> f64 {
        self.count
    }

    pub fn average(&self) -> f64 {
        self.sum / self.count
    }

    pub fn decay(&mut self) -> &ForwardDecay<G> {
        &self.decay
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Add;
    use std::time::{Duration, Instant};
    use crate::g;
    use super::*;

    #[test]
    fn example() {
        let landmark = Instant::now();
        let now = landmark + Duration::from_secs(10);
        let stream = vec![
            (landmark.add(Duration::from_secs(5)), 4.0),
            (landmark.add(Duration::from_secs(7)), 8.0),
            (landmark.add(Duration::from_secs(3)), 3.0),
            (landmark.add(Duration::from_secs(8)), 6.0),
            (landmark.add(Duration::from_secs(4)), 4.0),
        ];

        let fd = ForwardDecay::new(landmark, g::Polynomial::new(2));
        let mut aggregator = BasicAggregator::new(fd);

        for item in stream {
            aggregator.update(item);
        }

        let epsilon = 0.01;

        assert_eq!(aggregator.sum(now), 9.67);
        assert_eq!(aggregator.static_sum(), 967.0);
        assert_eq!(aggregator.count(now), 1.63);
        assert_eq!(aggregator.static_count(), 163.0);
        assert!(aggregator.average() >= (5.93 - epsilon) && aggregator.average() <= (5.93 + epsilon));
    }
}