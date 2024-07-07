use std::time::Instant;

pub use basic::BasicAggregator;
pub use minmax::MinMaxAggregator;
pub use sign::SignAggregator;

mod basic;
mod minmax;
mod sign;

pub trait Aggregator {
    type Item;

    fn update(&mut self, item: Self::Item);

    fn reset(&mut self, landmark: Instant);
}