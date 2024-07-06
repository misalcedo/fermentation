use std::time::{Duration, Instant};

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