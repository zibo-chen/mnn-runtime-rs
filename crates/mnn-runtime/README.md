# mnn-runtime

Safe, named multi-input/multi-output MNN inference for Rust applications.

This crate is the public application API from the
[`mnn-runtime-rs`](https://github.com/zibo-chen/mnn-runtime-rs) project. It uses
`mnn-runtime-sys` as its only native boundary, confines each native model to a
worker thread, and coordinates multiple model libraries through a process-wide
execution gate by default.

```rust,no_run
use mnn_runtime::{Runtime, Tensor};

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let runtime = Runtime::cpu()?;
let model = runtime.load_file("model.mnn")?;
let input = Tensor::new("input", vec![1, 3, 224, 224], vec![0.0; 3 * 224 * 224])?;
let outputs = model.run(&[input])?;
# Ok(())
# }
```

See the repository README for feature selection, architecture, and migration
instructions.

