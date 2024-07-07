use std::time::Instant;
use crate::{ForwardDecay, Item};
use crate::aggregate::{Aggregator, BasicAggregator};
use crate::g::{Exponential, Function};

/// A composite aggregator that uses a separate [BasicAggregator] for positive and negative values.
///
/// ## Examples
/// ### Decayed Error Percentage
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::{ForwardDecay, g};
/// use fermentation::aggregate::{SignAggregator, Aggregator};
///
/// let decay = ForwardDecay::new(Instant::now(), g::Polynomial::new(2));
/// let landmark = decay.landmark();
/// let now = landmark + Duration::from_secs(10);
///
/// // A negative sign denotes an error.
/// let stream = vec![
///     (landmark + Duration::from_secs(5), -4.0),
///     (landmark + Duration::from_secs(7), 8.0),
///     (landmark + Duration::from_secs(3), 3.0),
///     (landmark + Duration::from_secs(8), -6.0),
///     (landmark + Duration::from_secs(4), 4.0),
/// ];
///
/// let mut aggregator = SignAggregator::from(decay);
///
/// for item in stream {
///     aggregator.update(item);
/// }
///
/// let epsilon = 0.01;
///
/// assert_eq!(aggregator.positive().sum(now), 4.83);
/// assert_eq!(aggregator.positive().static_sum(), 483.0);
/// assert_eq!(aggregator.negative().sum(now), -4.84);
/// assert_eq!(aggregator.negative().static_sum(), -484.0);
/// assert_eq!(aggregator.positive().count(now), 0.74);
/// assert_eq!(aggregator.positive().static_count(), 74.0);
/// assert_eq!(aggregator.negative().count(now), 0.89);
/// assert_eq!(aggregator.negative().static_count(), 89.0);
/// assert!(aggregator.positive().average() >= (6.53 - epsilon) && aggregator.positive().average() <= (6.53 + epsilon));
/// assert!(aggregator.negative().average() >= (-5.44 - epsilon) && aggregator.negative().average() <= (-5.44 + epsilon));
///
/// let errors = aggregator.negative().static_sum().abs();
/// let successes = aggregator.positive().static_sum();
/// let percent = 100.0 * errors / (errors + successes);
///
/// assert!(percent >= (50.05 - epsilon) && percent <= (50.05 + epsilon));
/// ```
///
/// ### Update Landmark
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::{ForwardDecay, g};
/// use fermentation::aggregate::{SignAggregator, Aggregator};
///
/// let decay = ForwardDecay::new(Instant::now(), g::Exponential::new(0.2));
/// let landmark = decay.landmark();
/// let new_landmark = landmark + Duration::from_secs(1);
/// let now = landmark + Duration::from_secs(10);
/// let stream = vec![
///     (landmark + Duration::from_secs(5), -4.0),
///     (landmark + Duration::from_secs(7), 8.0),
///     (landmark + Duration::from_secs(3), 3.0),
///     (landmark + Duration::from_secs(8), -6.0),
///     (landmark + Duration::from_secs(4), 4.0),
/// ];
///
/// let mut aggregator = SignAggregator::new(decay, decay);
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
/// assert!((aggregator.positive().sum(now) - clone.positive().sum(now)).abs() < epsilon);
/// assert!((aggregator.negative().sum(now) - clone.negative().sum(now)).abs() < epsilon);
/// assert!((aggregator.positive().count(now) - clone.positive().count(now)).abs() < epsilon);
/// assert!((aggregator.negative().count(now) - clone.negative().count(now)).abs() < epsilon);
/// assert!((aggregator.positive().average() - clone.positive().average()).abs() < epsilon);
/// assert!((aggregator.negative().average() - clone.negative().average()).abs() < epsilon);
/// ```
#[derive(Copy, Clone)]
pub struct SignAggregator<G, I> {
    positive: BasicAggregator<G, I>,
    negative: BasicAggregator<G, I>,
}

impl<G, I> Aggregator for SignAggregator<G, I> where G: Function, I: Item {
    type Item = I;

    fn update(&mut self, item: I) {
        if item.value().is_sign_positive() {
            self.positive.update(item);
        } else {
            self.negative.update(item);
        }
    }

    fn reset(&mut self, landmark: Instant) {
        self.positive.reset(landmark);
        self.negative.reset(landmark);
    }
}

impl<I> SignAggregator<Exponential, I>
where
    I: Item,
{
    pub fn update_landmark(&mut self, landmark: Instant) {
        self.positive.update_landmark(landmark);
        self.negative.update_landmark(landmark);
    }
}

impl<G, I> From<ForwardDecay<G>> for SignAggregator<G, I>
where
    G: Function + Clone,
    I: Item,
{
    fn from(value: ForwardDecay<G>) -> Self {
        Self::new(value.clone(), value)
    }
}
impl<G, I> SignAggregator<G, I>
where
    G: Function,
    I: Item,
{
    pub fn new(positive: ForwardDecay<G>, negative: ForwardDecay<G>) -> Self {
        Self {
            positive: BasicAggregator::new(positive),
            negative: BasicAggregator::new(negative),
        }
    }

    pub fn positive(&self) -> &BasicAggregator<G, I> {
        &self.positive
    }

    pub fn negative(&self) -> &BasicAggregator<G, I> {
        &self.negative
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
            (landmark.add(Duration::from_secs(5)), -4.0),
            (landmark.add(Duration::from_secs(7)), 8.0),
            (landmark.add(Duration::from_secs(3)), 3.0),
            (landmark.add(Duration::from_secs(8)), -6.0),
            (landmark.add(Duration::from_secs(4)), 4.0),
        ];

        let fd = ForwardDecay::new(landmark, g::Polynomial::new(2));
        let mut aggregator = SignAggregator::from(fd);

        for item in stream {
            aggregator.update(item);
        }

        let epsilon = 0.01;

        assert_eq!(aggregator.positive().sum(now), 4.83);
        assert_eq!(aggregator.negative().sum(now), -4.84);
        assert_eq!(aggregator.positive().count(now), 0.74);
        assert_eq!(aggregator.negative().count(now), 0.89);
        assert!(aggregator.positive().average() >= (6.53 - epsilon) && aggregator.positive().average() <= (6.53 + epsilon));
        assert!(aggregator.negative().average() >= (-5.44 - epsilon) && aggregator.negative().average() <= (-5.44 + epsilon));
    }
}