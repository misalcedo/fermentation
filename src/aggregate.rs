use std::marker::PhantomData;
use std::time::Instant;

use crate::{ForwardDecay, Item};

/// A decay function in either the forward or backward setting assigns a weight to each item in the
/// input (and the value of this weight can vary over time).
/// Aggregate computations over such data must now use these weights to scale the contribution
/// of each item. In most cases, this leads to a natural weighted generalization of the aggregate.
pub trait AggregateComputation {
    type Item: Item;
    type Output;

    fn update(&mut self, item: &Self::Item);

    fn query(&self, timestamp: Instant) -> Self::Output;
}

pub struct Sum<'a, G, I> {
    decay: &'a ForwardDecay<G>,
    total: f64,
    _phantom: PhantomData<I>,
}

impl<'a, G, I> Sum<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item,
{
    pub fn new(decay: &'a ForwardDecay<G>) -> Self {
        Self {
            decay,
            total: 0.0,
            _phantom: Default::default(),
        }
    }
}

impl<'a, G, I> AggregateComputation for Sum<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item,
{
    type Item = I;
    type Output = f64;

    fn update(&mut self, item: &Self::Item) {
        self.total += self.decay.raw_weight(&item) * item.value();
    }

    fn query(&self, timestamp: Instant) -> Self::Output {
        self.total / self.decay.normalizing_factor(timestamp)
    }
}

pub struct Count<'a, G, I> {
    decay: &'a ForwardDecay<G>,
    total: f64,
    _phantom: PhantomData<I>,
}

impl<'a, G, I> Count<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item,
{
    pub fn new(decay: &'a ForwardDecay<G>) -> Self {
        Self {
            decay,
            total: 0.0,
            _phantom: Default::default(),
        }
    }
}

impl<'a, G, I> AggregateComputation for Count<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item,
{
    type Item = I;
    type Output = f64;

    fn update(&mut self, item: &Self::Item) {
        self.total += self.decay.raw_weight(item);
    }

    fn query(&self, timestamp: Instant) -> Self::Output {
        self.total / self.decay.normalizing_factor(timestamp)
    }
}

pub struct Average<'a, G, I> {
    decay: &'a ForwardDecay<G>,
    sum: f64,
    count: f64,
    _phantom: PhantomData<I>,
}

impl<'a, G, I> Average<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item,
{
    pub fn new(decay: &'a ForwardDecay<G>) -> Self {
        Self {
            decay,
            sum: 0.0,
            count: 0.0,
            _phantom: Default::default(),
        }
    }
}

impl<'a, G, I> AggregateComputation for Average<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item,
{
    type Item = I;
    type Output = f64;

    fn update(&mut self, item: &Self::Item) {
        let raw_weight = self.decay.raw_weight(&item);

        self.sum += raw_weight * item.value();
        self.count += raw_weight;
    }

    fn query(&self, _: Instant) -> Self::Output {
        self.sum / self.count
    }
}

pub struct Min<'a, G, I> {
    decay: &'a ForwardDecay<G>,
    item: Option<I>,
}

impl<'a, G, I> Min<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item,
{
    pub fn new(decay: &'a ForwardDecay<G>) -> Self {
        Self {
            decay,
            item: None,
        }
    }
}

impl<'a, G, I> AggregateComputation for Min<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item + Clone,
{
    type Item = I;
    type Output = Option<f64>;

    fn update(&mut self, other: &Self::Item) {
        self.item = match self.item.take() {
            None => Some(other.clone()),
            Some(item) if self.decay.raw_weighted_value(&item) > self.decay.raw_weighted_value(other) => Some(other.clone()),
            item => item
        }
    }

    fn query(&self, timestamp: Instant) -> Self::Output {
        let item = self.item.as_ref()?;
        Some(self.decay.weight(item, timestamp) * item.value())
    }
}

pub struct Max<'a, G, I> {
    decay: &'a ForwardDecay<G>,
    item: Option<I>,
}

impl<'a, G, I> Max<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item + Clone,
{
    pub fn new(decay: &'a ForwardDecay<G>) -> Self {
        Self {
            decay,
            item: None,
        }
    }
}

impl<'a, G, I> AggregateComputation for Max<'a, G, I>
where
    G: Fn(f64) -> f64,
    I: Item + Clone,
{
    type Item = I;
    type Output = Option<f64>;

    fn update(&mut self, other: &Self::Item) {
        self.item = match self.item.take() {
            None => Some(other.clone()),
            Some(item) if self.decay.raw_weighted_value(&item) < self.decay.raw_weighted_value(other) => Some(other.clone()),
            item => item
        }
    }

    fn query(&self, timestamp: Instant) -> Self::Output {
        let item = self.item.as_ref()?;
        Some(self.decay.weight(item, timestamp) * item.value())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn example() {
        let landmark = Instant::now();
        let stream = vec![
            item(landmark, 5, 4.0),
            item(landmark, 7, 8.0),
            item(landmark, 3, 3.0),
            item(landmark, 8, 6.0),
            item(landmark, 4, 4.0),
        ];
        let fd = ForwardDecay::new(landmark, |n: f64| n.powi(2));

        let mut sum = Sum::new(&fd);
        let mut count = Count::new(&fd);
        let mut average = Average::new(&fd);
        let mut min = Min::new(&fd);
        let mut max = Max::new(&fd);

        let now = landmark + Duration::from_secs(10);

        for item in stream {
            sum.update(&item);
            count.update(&item);
            average.update(&item);
            min.update(&item);
            max.update(&item);
        }

        assert_eq!(sum.query(now), 9.67);
        assert_eq!(count.query(now), 1.63);
        assert_almost_eq(average.query(now), 5.93, 0.01);
        assert_eq!(min.query(now), Some(3.0 * 0.09));
        assert_eq!(max.query(now), Some(8.0 * 0.49));
    }

    fn item(landmark: Instant, offset_seconds: u64, value: f64) -> (Instant, f64) {
        (landmark + Duration::from_secs(offset_seconds), value)
    }

    fn assert_almost_eq(left: f64, right: f64, epsilon: f64) {
        assert!(left >= (right - epsilon) && left <= (right + epsilon), "assertion 'left approximately equals right' failed");
    }
}