//! Aggregate computations on streams of items using a forward decay model.

use std::time::Instant;

pub use basic::BasicAggregator;
pub use minmax::MinMaxAggregator;
pub use sign::SignAggregator;

mod basic;
mod minmax;
mod sign;

/// Aggregates information about items in an unordered stream.
pub trait Aggregator {
    type Item;

    /// Update the aggregation with the given item.
    fn update(&mut self, item: Self::Item);

    /// Reset the aggregation to the initial state.
    /// This is equivalent to creating a new aggregator with the same decay model and the given landmark.
    fn reset(&mut self, landmark: Instant);
}