# mnn-runtime-rs

`mnn-runtime-rs` is a small, safe Rust runtime for embedding Alibaba MNN in
applications that use more than one inference library. It provides named
multi-input/multi-output inference, a thread-confined native session, and one
shared native dependency path for OCR, gaze estimation, and future models.

The initial API is intentionally focused: static-shape `f32` vision models on
CPU, Metal, Core ML, OpenCL, OpenGL, or Vulkan. Dynamic shapes and additional
tensor element types are planned, but are rejected explicitly instead of being
handled unsafely.

## Why this project exists

If `ocr-rs` and a gaze crate each compile their own MNN wrapper, one executable
can end up with duplicate MNN symbols, separate global thread pools,
and incompatible feature selections. This workspace makes the dependency
graph converge on exactly one package:

```text
application
├── ocr-rs ───────────┐
├── screen-gaze-mnn ──┼── mnn-runtime ── mnn-runtime-sys (links="mnn") ── MNN
└── other models ─────┘
```

Cargo permits only one package with a given native `links` value.
`mnn-runtime-sys` is therefore both the project-owned version/feature boundary
and the single native link owner. It compiles one narrow bridge and links one
MNN library; it never embeds an additional runtime beside it.

## Quick start

```rust,no_run
use mnn_runtime::{Runtime, Tensor};

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let runtime = Runtime::cpu()?;
let model = runtime.load_file("model.mnn")?;

for input in model.info().inputs() {
    println!("input {}: {:?}", input.name(), input.shape());
}

let image = Tensor::new("input", vec![1, 3, 224, 224], vec![0.0; 3 * 224 * 224])?;
let outputs = model.run(&[image])?;
println!("{} values", outputs[0].data().len());
# Ok(())
# }
```

Inspect a model without writing code:

```bash
cargo run -p mnn-runtime --example inspect -- path/to/model.mnn
cargo run -p mnn-runtime --example smoke -- path/to/model.mnn
```

For models such as Intel Open Model Zoo gaze estimation, pass each graph input
as a separate named `Tensor`; packing unrelated inputs into one artificial
buffer is not required.

## Runtime behavior

- Every loaded model owns a dedicated worker thread. MNN interpreter/session
  handles never cross application threads, and destruction order is fixed.
- `Model` is cloneable and can be called from any thread. Calls to one model are
  processed in submission order.
- The default `ExecutionMode::Serialized` uses one process-wide gate. This
  avoids oversubscription and global-state races when OCR and gaze run in the
  same process. `ExecutionMode::Parallel` is an explicit opt-in for measured
  workloads.
- Inputs and outputs are selected by graph name and validated before entering
  native code.

## Features

The default build enables MNN's internal thread pool, static linking, verified
prebuilt downloads, and CPU inference.

| Feature | Backend or behavior |
| --- | --- |
| `prebuilt` | Use verified archives from `MNN-Prebuilds` where available |
| `build-from-source` | Force a pinned MNN source build |
| `static` / `dynamic` | Select native link mode (static is the default) |
| `metal` | Metal |
| `coreml` | Core ML (also enables Metal) |
| `opencl` | OpenCL |
| `opengl` | OpenGL |
| `vulkan` | Vulkan |
| `openmp` | OpenMP |
| `crt-static` | Static C runtime on Windows |

Feature selection belongs to the final application. Libraries should normally
declare `mnn-runtime` with `default-features = false`; the executable then
chooses one compatible backend/threading combination. Do not enable OpenMP and
MNN's internal thread pool together.

## Integrating existing libraries

1. Remove native MNN compilation and all `rustc-link-lib=MNN` output from the
   downstream crate's `build.rs`.
2. Remove direct MNN FFI modules and depend on `mnn-runtime` instead.
3. Keep model preprocessing/postprocessing in the domain crate; only graph
   execution belongs here.
4. Let the final executable choose features. For example:

   ```toml
   [dependencies]
   ocr-rs = { version = "...", default-features = false }
   screen-gaze-mnn = { version = "...", default-features = false }
   mnn-runtime = { version = "0.1", features = ["metal"] }
   ```

5. Confirm the graph with `cargo tree -i mnn-runtime-sys`; there should be one
   version and one native `links="mnn"` owner.

## Versioning and current native baseline

This first release targets MNN 3.6.0. On supported targets, the default build
downloads the current `MNN-Prebuilds` `dev` artifacts and validates a pinned
SHA-256 for every archive. Unsupported targets/backends fall back to the
official MNN 3.6.0 source archive, which is also checksum-pinned.

The checksum makes silent replacement fail closed, but the `dev` URL itself is
still mutable. Before publishing the crates, promote the validated assets to an
immutable release tag and update `PREBUILT_TAG`; after that, a clean build no
longer depends on mutable release state.

## License

Apache-2.0. MNN and the upstream Rust bindings are also Apache-2.0.
