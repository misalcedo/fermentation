use std::time::Instant;

pub use basic::BasicAggregator;
pub use minmax::MinMaxAggregator;

mod basic;
mod minmax;

pub trait Aggregator {
    type Item;

    fn update(&mut self, item: Self::Item);

    fn reset(&mut self, landmark: Instant);
}