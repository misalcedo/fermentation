use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::hash::Hash;

mod decay;

/// http://dimacs.rutgers.edu/~graham/pubs/papers/fwddecay.pdf
/// https://pdfs.semanticscholar.org/bb07/cb2360590fd788f5ee3739567f36080d350b.pdf
/// http://datagenetics.com/blog/november22019/index.html
trait Decay {
    fn exponential_weight(&self) -> f64;
}

impl Decay for Duration {
    fn exponential_weight(&self) -> f64 {
        self.as_secs_f64().exponential_weight()
    }
}

impl Decay for f64 {
    fn exponential_weight(&self) -> f64 {
        1.0 / 2.0f64.powf(*self)
    }
}

struct Record<K> {
    key: K,
    value: f64,
    weight: f64,
}

impl<K> Record<K> {
    pub fn now(key: K, value: f64, weight: f64) -> Self {
        Self {
            key,
            value,
            weight,
        }
    }
}

struct HeavyHitter<K> {
    key: K,
    ratio: f64,
}

impl<K> HeavyHitter<K> {
    pub fn new(key: K, ratio: f64) -> Self {
        Self {
            key,
            ratio
        }
    }
}

impl<K> PartialEq for HeavyHitter<K> {
    fn eq(&self, other: &Self) -> bool {
        self.ratio == other.ratio
    }
}

impl<K> Eq for HeavyHitter<K> {}

impl<K> Ord for HeavyHitter<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ratio.partial_cmp(&other.ratio).unwrap_or(Ordering::Equal)
    }
}

impl<K> PartialOrd for HeavyHitter<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.ratio.partial_cmp(&other.ratio)
    }
}

pub struct ForwardDecayTracker<K> {
    landmark: Instant,
    events: VecDeque<Record<K>>,
}

impl<K> ForwardDecayTracker<K> where K: Hash + Eq + Clone {
    pub fn now() -> Self {
        Self {
            landmark: Instant::now(),
            events: VecDeque::new(),
        }
    }

    pub fn update(&mut self, key: K, value: f64) {
        let weight = Instant::now().duration_since(self.landmark).exponential_weight();
        self.events.push_back(Record::now(key, value, weight));
    }

    pub fn query(&self, threshold: f64) -> Vec<K> {
        let mut heap = BinaryHeap::new();
        let mut seen = HashSet::new();

        for record in self.events.iter() {
            if !seen.insert(&record.key) {
                continue;
            }

            let count: f64 = self.events.iter().filter(|r| r.key == record.key).map(|r| r.value).sum();
            let ratio = count / self.events.len() as f64;

            if ratio >= threshold {
                heap.push(HeavyHitter::new(record.key.clone(), ratio));
            }
        }

        let mut keys = Vec::with_capacity(heap.len());

        while let Some(hh) = heap.pop() {
            keys.push(hh.key);
        }

        keys
    }
}



struct NaiveHeavyHitters {
    counts: HashMap<String, f64>,
}

impl NaiveHeavyHitters {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }
}

impl NaiveHeavyHitters {
    fn update(&mut self, item: &str) {
        let count = self.counts.entry(item.to_string()).or_insert(0.0);
        *count += 1.0;
    }

    fn query(&self, threshold: f64) -> Vec<String> {
        let mut items: Vec<(&String, &f64)> = self.counts.iter().collect();
        items.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut heavy_hitters = Vec::new();
        let mut total_count = 0.0;
        for (item, count) in items {
            total_count += count;
            if *count >= threshold * total_count {
                heavy_hitters.push(item.clone());
            } else {
                break;
            }
        }
        heavy_hitters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay() {
        assert_eq!(Duration::from_secs_f64(0.0).exponential_weight(), 1.0);
        assert_eq!(Duration::from_secs_f64(1.0).exponential_weight(), 0.5);
        assert_eq!(Duration::from_secs_f64(2.0).exponential_weight(), 0.25);
        assert_eq!(Duration::from_secs_f64(3.0).exponential_weight(), 0.125);
        assert_eq!(Duration::from_secs_f64(4.0).exponential_weight(), 0.0625);
    }

    #[test]
    fn sample() {
        let mut hh = ForwardDecayTracker::now();
        hh.update("apple", 1.0);
        hh.update("apple", 1.0);
        hh.update("banana", 1.0);
        hh.update("orange", 1.0);
        hh.update("orange", 1.0);
        hh.update("orange", 1.0);

        assert_eq!(hh.query(0.5), vec!["orange"]);
        assert_eq!(hh.query(0.333), vec!["orange", "apple"]);
        assert_eq!(hh.query(0.166), vec!["orange", "apple", "banana"]);
    }

    #[test]
    fn naive_sample() {
        let mut hh = NaiveHeavyHitters::new();
        hh.update("apple");
        hh.update("apple");
        hh.update("banana");
        hh.update("orange");
        hh.update("orange");
        hh.update("orange");

        assert_eq!(hh.query(0.5), vec!["orange"]);
        assert_eq!(hh.query(0.333), vec!["orange", "apple"]);
        assert_eq!(hh.query(0.166), vec!["orange", "apple", "banana"]);
    }


    #[test]
    fn exponential_decay() {
        let t = Duration::from_secs_f64(1.0);
        let now = Duration::from_secs_f64(2.0);

        assert_eq!(t.exponential_weight() / now.exponential_weight(), (t.as_secs_f64() - now.as_secs_f64()).exponential_weight());
    }
}
