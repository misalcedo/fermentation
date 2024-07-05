use std::time::{Duration, Instant};

pub trait Item {
    fn timestamp(&self) -> Instant;
}

impl Item for Instant {
    fn timestamp(&self) -> Instant {
        *self
    }
}

/// A decay function takes some information about the ith item, and returns a weight for this item.
/// It can depend on a variety of properties of the item such as ti, vi as well as the current time t,
/// but for brevity we will write it simply as w(i, t), or just w(i) when t is implicit.
/// We define a function w(i, t) to be a decay function if it satisfies the following properties:
/// 1. w(i, t) = 1 when ti = t and 0 ≤ w(i, t) ≤ 1 for all t ≥ ti.
/// 2. w is monotone non-increasing as time increases: t' ≥ t ⇒ w(i, t') ≤ w(i, t).
pub trait DecayFunction {
    fn weight<I>(&self, item: I, timestamp: Instant) -> f64
    where
        I: Item;
}

/// Given a positive monotone non-decreasing function g, and a landmark time L,
/// the decayed weight of an item with arrival time ti > L measured at time t ≥ ti
/// is given by w(i, t) = g(ti − L) / g(t − L).
pub trait ForwardDecayFunction {
    fn invoke(&self, age: Duration) -> f64;
}

impl<F> ForwardDecayFunction for F
where
    F: Fn(Duration) -> f64,
{
    fn invoke(&self, age: Duration) -> f64 {
        self(age)
    }
}

/// The forward decay is computed on the amount of time between the arrival of an item and a fixed point L,
/// known as the landmark. By convention, this landmark is some time earlier than all other items;
/// we discuss how this landmark can be chosen below.
/// Thus, we are looking forward in time from the landmark to see the item,
/// instead of looking backward from the current time.
pub struct ForwardDecay<G> {
    landmark: Instant,
    g: G,
}

impl<G> ForwardDecay<G>
where
    G: ForwardDecayFunction,
{
    pub fn new(landmark: Instant, g: G) -> Self {
        Self {
            landmark,
            g,
        }
    }

    pub fn set_landmark(&mut self, landmark: Instant) {
        self.landmark = landmark;
    }
}

impl<G> DecayFunction for ForwardDecay<G>
where
    G: ForwardDecayFunction,
{
    fn weight<I>(&self, item: I, timestamp: Instant) -> f64
    where
        I: Item,
    {
        // Return early with a weight of 0 for any items before or at the landmark.
        if self.landmark >= timestamp {
            return 0.0;
        }

        let item_duration = item.timestamp().duration_since(self.landmark);
        let timestamp_duration = timestamp.duration_since(self.landmark);

        self.g.invoke(item_duration) / self.g.invoke(timestamp_duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example() {
        let landmark = Instant::now();
        let stream = vec![5, 7, 3, 8, 4];
        let fd = ForwardDecay::new(landmark, |n: Duration| {
            let seconds = n.as_secs_f64();
            seconds * seconds
        });
        let now = landmark + Duration::from_secs(10);

        let result: Vec<f64> = stream.into_iter()
            .map(|i| landmark + Duration::from_secs(i))
            .map(|i| fd.weight(i, now))
            .collect();
        let weights = vec![0.25, 0.49, 0.09, 0.64, 0.16];

        assert_eq!(result, weights);
    }
}