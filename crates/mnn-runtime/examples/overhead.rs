//! Compare wrapper paths using a tiny known-result model (not a production-model benchmark).
use mnn_runtime::{Runtime, RuntimeConfig, Tensor};
use std::time::{Duration, Instant};

fn measure(mut run: impl FnMut(), iterations: u32) -> (Duration, Duration, Duration) {
    for _ in 0..100 {
        run();
    }
    let mut times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        run();
        times.push(start.elapsed());
    }
    times.sort_unstable();
    (
        times[times.len() / 2],
        times[times.len() * 95 / 100],
        times[times.len() * 99 / 100],
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::new(RuntimeConfig::new().with_threads(1))?;
    let model =
        runtime.load_bytes(include_bytes!("../tests/fixtures/packed_output3.mnn").as_slice())?;
    let input = Tensor::new("input", vec![1, 3, 1, 2], vec![1.0; 6])?;
    let borrowed = measure(
        || {
            std::hint::black_box(model.run(std::slice::from_ref(&input)).unwrap());
        },
        1000,
    );
    let mut outputs = model.run(std::slice::from_ref(&input))?;
    let reused = measure(
        || {
            let result = model
                .run_reusing(vec![input.clone()], std::mem::take(&mut outputs))
                .unwrap();
            outputs = std::hint::black_box(result.tensors);
        },
        1000,
    );
    println!("tiny model, current compilation profile; P50/P95/P99");
    println!("borrowed: {borrowed:?}; reused outputs: {reused:?}");
    println!(
        "Input cloning is included in both loops; run_owned transfers an existing Vec without that clone."
    );
    Ok(())
}
