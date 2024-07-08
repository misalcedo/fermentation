use std::env::args;
use std::fs::read_to_string;
use std::time::{Duration, Instant};
use fermentation::ForwardDecay;
use fermentation::g::Exponential;
use fermentation::space_saving::BTreeSpaceSaving;

fn main() {
    let path = args().skip(1).next().expect("must pass an input file");
    let contents = read_to_string(path.as_str()).expect("unable to open file");
    let start = Instant::now();
    let decay = ForwardDecay::new(start, Exponential::rate(0.001, Duration::from_secs(60)));
    let mut ss = BTreeSpaceSaving::new(8, decay);

    let mut hits = 0;

    for e in contents.split_whitespace() {
        hits += 1;
        ss.hit(e);
    }

    let top = ss.top(2).expect("unable to guarantee top hitters");
    let frequent = ss.frequent(0.1);
    let end = Instant::now();

    println!("Elapse: {}", (end - start).as_secs_f64());
    println!("Top elements: {:?}", &top);

    for (index, e) in top.into_iter().enumerate() {
        println!("Element {index} is {} with {:?}", e, ss.get(e, end));
    }

    println!("Frequent elements: {:?}", frequent);
    println!("Total hits: {}, Decayed hits: {}", hits, ss.hits(end));
}
