//! An implementation of the [SpaceSaving](https://www.cs.ucsb.edu/sites/default/files/documents/2005-23.pdf) algorithm.
//! The algorithm is adjusted according to support the [forward decay model](http://dimacs.rutgers.edu/~graham/pubs/papers/expdecay.pdf).

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;
use std::time::Instant;

use crate::ForwardDecay;
use crate::g::{Exponential, Function};

/// An aggregation computation that implements the [SpaceSaving[(http://dimacs.rutgers.edu/~graham/pubs/papers/expdecay.pdf) algorithm.
/// Instead of a StreamSummary, this implementation uses a [BTreeSet] to maintain an ordered list of counters.
/// The use of a [BTreeSet] avoids having to implement a [LinkedList](https://rust-unofficial.github.io/too-many-lists/) that allows shareable cursors.
#[derive(Debug)]
pub struct BTreeSpaceSaving<E, G> {
    capacity: usize,
    decay: ForwardDecay<G>,
    hits: f64,
    elements: HashMap<E, Count>,
    counts: BTreeSet<Counter<E>>,
}

impl<E> BTreeSpaceSaving<E, Exponential>
where
    E: Clone + Hash + Eq + Ord,
{
    pub fn update_landmark(&mut self, landmark: Instant) {
        let age = self.decay.set_landmark(landmark);
        let factor = self.decay.g().invoke(age);

        self.hits /= factor;

        let counts = std::mem::take(&mut self.counts);
        for mut counter in counts {
            counter.count /= factor;
            counter.error /= factor;

            self.counts.insert(counter);
        }
    }
}

impl<E, G> BTreeSpaceSaving<E, G>
where
    E: Clone + Hash + Eq + Ord,
    G: Function
{
    /// Initializes a new aggregator with the given capacity and decay model.
    /// The error bound for the results are 1/capacity.
    pub fn new(capacity: usize, decay: ForwardDecay<G>) -> Self {
        Self {
            capacity,
            decay,
            hits: 0.0,
            elements: Default::default(),
            counts: Default::default(),
        }
    }

    /// Increments the given element's counter by a single hit.
    pub fn hit(&mut self, element: E) -> Count {
        let now = Instant::now();
        let weight = self.decay.static_weight(now);

        self.hits += weight;

        let count = self.elements.get(&element).copied();
        let mut counter = Counter::new(element, count.unwrap_or_default());

        match count {
            None => {
                if self.counts.len() >= self.capacity {
                    if let Some(min) = self.counts.pop_first() {
                        self.elements.remove(&min.element);
                        counter.count = min.count;
                        counter.error = min.count;
                    }
                }
            }
            Some(_) => {
                self.counts.remove(&counter);
            }
        }

        counter.count += weight;

        let key = counter.key();

        if let Some(value) = self.elements.get_mut(&counter.element) {
            *value = key;
        } else {
            self.elements.insert(counter.element.clone(), key);
        }

        self.counts.insert(counter);

        key
    }

    pub fn top(&self, k: usize) -> Result<Vec<&E>, Vec<&E>> {
        let mut top_k = Vec::with_capacity(k);
        let mut order = true;
        let mut guarantee = false;
        let mut min = f64::INFINITY;

        let mut iterator = self.counts.iter().rev().peekable();
        while let Some(counter) = iterator.next() {
            if top_k.len() >= k {
                break;
            }

            let guaranteed_count = counter.guaranteed_count();

            min = min.min(guaranteed_count);

            if let Some(next) = iterator.peek() {
                order &= guaranteed_count >= next.count;
            }

            top_k.push(&counter.element);
        }

        if let Some(next) = iterator.next() {
            guarantee = next.count <= min;
        }

        if order && guarantee {
            Ok(top_k)
        } else {
            Err(top_k)
        }
    }

    pub fn frequent(&self, phi: f64) -> Result<Vec<&E>, Vec<&E>> {
        let threshold = (phi * self.hits).ceil();
        let mut hitters = Vec::new();
        let mut guaranteed = true;

        for counter in self.counts.iter().rev() {
            if counter.count <= threshold {
                break;
            }

            guaranteed &= counter.guaranteed_count() >= threshold;

            hitters.push(&counter.element);
        }

        if guaranteed {
            Ok(hitters)
        } else {
            Err(hitters)
        }
    }

    pub fn get(&self, element: &E, timestamp: Instant) -> Option<Count> {
        let mut count = self.elements.get(element).copied()?;
        count.count /= self.decay.normalizing_factor(timestamp);
        count.error /= self.decay.normalizing_factor(timestamp);
        Some(count)
    }

    pub fn hits(&self, timestamp: Instant) -> f64 {
        self.hits / self.decay.normalizing_factor(timestamp)
    }

    pub fn decay(&self) -> &ForwardDecay<G> {
        &self.decay
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Counter<E> {
    count: f64,
    error: f64,
    element: E,
}

impl<E> Counter<E> {
    fn new(element: E, count: Count) -> Self {
        Self { count: count.count, error: count.error, element }
    }

    fn key(&self) -> Count {
        Count { count: self.count, error: self.error }
    }

    fn guaranteed_count(&self) -> f64 {
        self.count - self.error
    }
}

impl<E> Ord for Counter<E> where E: Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("unable to compare counters")
    }
}

impl<E> PartialOrd for Counter<E> where E: PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self.count, self.error).partial_cmp(&(other.count, other.error))
    }
}


// Counters will not contain NaN, so we can safely compare them for equality.
impl<E> Eq for Counter<E> where E: Eq {
}

#[derive(Debug, Default, Copy, Clone, PartialOrd, PartialEq)]
pub struct Count {
    count: f64,
    error: f64,
}
