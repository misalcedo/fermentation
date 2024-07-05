use std::collections::VecDeque;

pub trait Counter {
    type Key;

    fn update(&mut self, key: Self::Key, value: f64);
    fn value(&self) -> f64;
}
pub struct DecayingCounter<K> {
    key: K,
    value: f64,
    buckets: VecDeque<f64>
}

impl<K> DecayingCounter<K> {
    pub fn new(key: K) -> Self {
        Self {
            key,
            value: 0.0,
            buckets: Default::default(),
        }
    }

    fn decay(&mut self) {
    }
}

impl<K> Counter for DecayingCounter<K> {
    type Key = K;

    fn update(&mut self, key: Self::Key, value: f64) {
        todo!()
    }

    fn value(&self) -> f64 {
        todo!()
    }
}