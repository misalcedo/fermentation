use std::mem;
use std::time::Instant;

use crate::{ForwardDecay, Item};
use crate::g::{Exponential, Function};

#[derive(Default)]
enum MinMax<I> {
    #[default]
    Neither,
    Same(I),
    Both(I, I)
}

impl<I> MinMax<I> {
    fn min(&self) -> Option<&I> {
        match self {
            MinMax::Neither => None,
            MinMax::Same(min_max) => Some(min_max),
            MinMax::Both(min, _) => Some(min)
        }
    }

    fn max(&self) -> Option<&I> {
        match self {
            MinMax::Neither => None,
            MinMax::Same(min_max) => Some(min_max),
            MinMax::Both(_, max) => Some(max)
        }
    }
}

/// A decay function in either the forward or backward setting assigns a weight to each item in the
/// input (and the value of this weight can vary over time).
/// Aggregate computations over such data must now use these weights to scale the contribution
/// of each item. In most cases, this leads to a natural weighted generalization of the aggregate.
pub struct ArithmeticAggregation<G, I> {
    decay: ForwardDecay<G>,
    sum: f64,
    count: f64,
    min_max: MinMax<I>,
}

impl<I> ArithmeticAggregation<Exponential, I>
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

impl<G, I> ArithmeticAggregation<G, I>
where
    G: Function,
    I: Item,
{
    pub fn new(decay: ForwardDecay<G>) -> Self {
        Self {
            decay,
            sum: 0.0,
            count: 0.0,
            min_max: MinMax::Neither,
        }
    }

    pub fn update(&mut self, item: I) {
        let static_weight = self.decay.static_weight(&item);

        self.sum += static_weight * item.value();
        self.count += static_weight;

        self.min_max = match mem::take(&mut self.min_max) {
            MinMax::Neither => MinMax::Same(item),
            MinMax::Same(min_max) => {
                let min_max_static_weight = self.decay.static_weighted_value(&min_max);
                let item_static_weight = self.decay.static_weighted_value(&item);

                if min_max_static_weight <= item_static_weight {
                    MinMax::Both(min_max, item)
                } else {
                    MinMax::Both(item, min_max)
                }
            }
            MinMax::Both(min, max) => {
                let min_static_weight = self.decay.static_weighted_value(&min);
                let max_static_weight = self.decay.static_weighted_value(&max);
                let item_static_weight = self.decay.static_weighted_value(&item);

                if item_static_weight < min_static_weight {
                    MinMax::Both(item, max)
                } else if item_static_weight > max_static_weight {
                    MinMax::Both(min, item)
                } else {
                    MinMax::Both(min, max)
                }
            }
        }
    }

    pub fn sum(&self, timestamp: Instant) -> f64 {
        self.sum / self.decay.normalizing_factor(timestamp)
    }

    pub fn count(&self, timestamp: Instant) -> f64 {
        self.count / self.decay.normalizing_factor(timestamp)
    }

    pub fn average(&self) -> f64 {
        self.sum / self.count
    }

    pub fn min(&self) -> Option<&I> {
        self.min_max.min()
    }

    pub fn max(&self) -> Option<&I> {
        self.min_max.max()
    }

    pub fn decay(&mut self) -> &ForwardDecay<G> {
        &self.decay
    }

    pub fn reset(&mut self, landmark: Instant) {
        self.decay.set_landmark(landmark);
        self.sum = 0.0;
        self.count = 0.0;
        self.min_max = MinMax::Neither;
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};
    use crate::g;
    use super::*;

    #[test]
    fn example() {
        let landmark = Instant::now();
        let now = landmark + Duration::from_secs(10);
        let stream = vec![
            item(landmark, 5, 4.0),
            item(landmark, 7, 8.0),
            item(landmark, 3, 3.0),
            item(landmark, 8, 6.0),
            item(landmark, 4, 4.0),
        ];

        let fd = ForwardDecay::new(landmark, g::Polynomial::new(2));
        let mut aggregates = ArithmeticAggregation::new(fd);

        for item in stream {
            aggregates.update(item);
        }

        assert_eq!(aggregates.sum(now), 9.67);
        assert_eq!(aggregates.count(now), 1.63);
        assert_almost_eq(aggregates.average(), 5.93, 0.01);
        assert_eq!(aggregates.min(), Some(&(landmark + Duration::from_secs(3), 3.0)));
        assert_eq!(aggregates.max(), Some(&(landmark + Duration::from_secs(7), 8.0)));
    }

    fn item(landmark: Instant, offset_seconds: u64, value: f64) -> (Instant, f64) {
        (landmark + Duration::from_secs(offset_seconds), value)
    }

    fn assert_almost_eq(left: f64, right: f64, epsilon: f64) {
        assert!(left >= (right - epsilon) && left <= (right + epsilon), "assertion 'left approximately equals right' failed");
    }
}