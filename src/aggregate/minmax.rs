use std::mem;
use std::time::Instant;

use crate::{Aggregator, ForwardDecay, Item};
use crate::g::Function;

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

/// An aggregation computation over a stream of items to determine the decayed min and max.
///
/// ## Example
/// ```rust
/// use std::time::{Duration, Instant};
/// use fermentation::{aggregate::MinMaxAggregator, Aggregator, ForwardDecay, g};
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
/// let mut aggregator = MinMaxAggregator::new(decay);
///
/// for item in stream {
///     aggregator.update(item);
/// }
///
/// assert_eq!(aggregator.min(), Some(&(landmark + Duration::from_secs(3), 3.0)));
/// assert_eq!(aggregator.max(), Some(&(landmark + Duration::from_secs(7), 8.0)));
///
/// aggregator.reset(landmark);
///
/// assert_eq!(aggregator.min(), None);
/// assert_eq!(aggregator.max(), None);
/// ```
pub struct MinMaxAggregator<G, I> {
    decay: ForwardDecay<G>,
    min_max: MinMax<I>,
}

impl<G, I> Aggregator for MinMaxAggregator<G, I> where G: Function, I: Item {
    type Item = I;

    fn update(&mut self, item: I) {
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

    fn reset(&mut self, landmark: Instant) {
        self.decay.set_landmark(landmark);
        self.min_max = MinMax::Neither;
    }

}

impl<G, I> MinMaxAggregator<G, I>
where
    G: Function,
    I: Item,
{
    pub fn new(decay: ForwardDecay<G>) -> Self {
        Self {
            decay,
            min_max: MinMax::Neither,
        }
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
        let stream = vec![
            (landmark.add(Duration::from_secs(5)), 4.0),
            (landmark.add(Duration::from_secs(7)), 8.0),
            (landmark.add(Duration::from_secs(3)), 3.0),
            (landmark.add(Duration::from_secs(8)), 6.0),
            (landmark.add(Duration::from_secs(4)), 4.0),
        ];

        let fd = ForwardDecay::new(landmark, g::Polynomial::new(2));
        let mut aggregator = MinMaxAggregator::new(fd);

        for item in stream {
            aggregator.update(item);
        }

        assert_eq!(aggregator.min(), Some(&(landmark + Duration::from_secs(3), 3.0)));
        assert_eq!(aggregator.max(), Some(&(landmark + Duration::from_secs(7), 8.0)));
    }
}