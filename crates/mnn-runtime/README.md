# mnn-runtime

[![crates.io](https://img.shields.io/crates/v/mnn-runtime.svg)](https://crates.io/crates/mnn-runtime)
[![docs.rs](https://docs.rs/mnn-runtime/badge.svg)](https://docs.rs/mnn-runtime)

Safe, named multi-input/multi-output MNN inference for Rust applications,
including dynamic OCR inputs and actual per-run output shapes.

This crate is the public application API from the
[`mnn-runtime-rs`](https://github.com/zibo-chen/mnn-runtime-rs) project. It uses
`mnn-runtime-sys` as its only native boundary, confines each native model to a
worker thread, and coordinates multiple model libraries through a process-wide
execution gate by default.

```toml
[dependencies]
mnn-runtime = "0.1"
```

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


Use `Model::run_dynamic_owned` when width, height, or batch must change from
exported graph dimensions. Synchronous calls wait for queue capacity;
`try_submit_owned` / `try_submit_dynamic_owned` provide nonblocking admission.
Load metadata uses signed shapes and optional element counts. Use returned
`Tensor` shapes for dynamic results. `RuntimeConfig::with_gpu_cache_dir` and
`Model::save_cache` enable persistent GPU tuning caches. CUDA and MinGW static
C++ runtimes have dedicated features. Minimum Rust version: 1.75.
