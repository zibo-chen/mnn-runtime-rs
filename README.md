# mnn-runtime-rs

[![CI](https://github.com/zibo-chen/mnn-runtime-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/zibo-chen/mnn-runtime-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnn-runtime.svg)](https://crates.io/crates/mnn-runtime)
[![docs.rs](https://docs.rs/mnn-runtime/badge.svg)](https://docs.rs/mnn-runtime)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`mnn-runtime-rs` is a small, safe Rust runtime for embedding Alibaba MNN in
applications that use more than one inference library. It provides named
multi-input/multi-output inference, a thread-confined native session, and one
shared native dependency path for OCR, gaze estimation, and future models.

The API supports static and dynamic-shape `f32` vision models on CPU, Metal,
Core ML, OpenCL, OpenGL, Vulkan, or CUDA. Each request carries concrete input
shapes; results carry the shapes produced by that inference. Additional tensor
element types are not supported.

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

Add the public runtime crate to your application (Rust 1.75 or newer):

```toml
[dependencies]
mnn-runtime = "0.1"
```

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
- `Model` is cloneable and can be called from any thread. Accepted calls to one model are processed in submission order. Each model has
  a bounded queue (two waiting requests by default). Synchronous calls wait for
  capacity; only `try_submit_*` reports `QueueFull`.
- The default `ExecutionMode::Serialized` uses one process-wide gate owned by the sys crate for model creation, inference, and destruction. This
  avoids oversubscription and global-state races when OCR and gaze run in the
  same process. `ExecutionMode::Parallel` is an explicit opt-in for measured
  workloads.
- Inputs and outputs are selected by graph name and validated before entering
  native code.

## Features

The default build enables MNN's internal thread pool, static linking, verified
prebuilt downloads where the ABI/CRT matches, and CPU inference. Windows builds
using Rust's default dynamic CRT compile MNN from source with the same CRT.

| Feature | Backend or behavior |
| --- | --- |
| `prebuilt` | Select a verified CPU/GPU archive from `mnn-native-prebuilds` |
| `build-from-source` | Force a pinned MNN source build |
| `static` / `dynamic` | Select native link mode (static is the default) |
| `metal` | Metal |
| `coreml` | Core ML (also enables Metal) |
| `opencl` | OpenCL |
| `opengl` | OpenGL |
| `vulkan` | Vulkan |
| `cuda` | CUDA on Linux / Windows MSVC; requires the NVIDIA toolkit |
| `static-cpp-runtime` | Statically link the MinGW C++/GCC/pthread runtimes on Windows GNU |
| `openmp` | OpenMP |
| `crt-static` | Require static CRT; also set `RUSTFLAGS="-C target-feature=+crt-static"` |

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

This release uses MNN **3.6.1**, pinned to upstream commit
`d407447ed56c4121a11ccbd266dc184ca1ead0c2`, and the
[`mnn-3.6.1-d407447e-r1` native release](https://github.com/zibo-chen/mnn-native-prebuilds/releases/tag/mnn-3.6.1-d407447e-r1).
All 22 archives have embedded SHA-256 checksums. The build selects the smallest
profile containing every requested backend; additional backends in a combined
archive bring their link dependencies with them. The application still chooses
its inference device through `RuntimeConfig::with_backend`.

| Targets | Base profile | Additional prebuilt profiles |
| --- | --- | --- |
| Linux GNU x86_64 | CPU | Vulkan + OpenCL; CUDA 12 |
| Linux GNU aarch64 | CPU | Vulkan + OpenCL |
| Windows MSVC x86_64 | CPU | Vulkan + OpenCL; CUDA 12 |
| Windows MSVC i686 / aarch64 | CPU | Vulkan + OpenCL |
| macOS arm64 / x86_64 (universal) | CPU + Metal | Metal + Core ML |
| iOS arm64 device / simulator | CPU + Metal | Metal + Core ML |
| Android arm64-v8a / armeabi-v7a | CPU | Vulkan + OpenCL + OpenGL ES |

For example, `--features vulkan` or `--features opencl` selects the combined
Vulkan/OpenCL archive on Linux and Windows. On Android either feature selects
the Vulkan/OpenCL/OpenGL archive. `--features coreml` selects the Apple
Metal/Core ML archive. `--features cuda` selects CUDA 12 on Linux/Windows x86_64.
Combining CUDA with Vulkan/OpenCL has no matching archive and builds from source.

Linux prebuilts require glibc 2.39+ and a compatible libstdc++; use
`build-from-source` for older GNU systems or musl. Windows prebuilts require
Rust's static CRT (`RUSTFLAGS="-C target-feature=+crt-static"`); the default
dynamic CRT builds from source. Apple minimums are macOS 11 and iOS 13.
Android packages use NDK r27c, API 21 and `c++_static`.

iOS archives are static and have no MNN thread pool. Select them with
`default-features = false, features = ["prebuilt", "static", "metal"]`
(or `"coreml"`). Enabling `mnn-threadpool` or dynamic linking on iOS builds
from source. Other prebuilts require `mnn-threadpool` and exclude OpenMP.
Unsupported targets, feature combinations or CRT/threading policies fall back
to the checksum-pinned source commit. Set `MNN_REQUIRE_PREBUILT=1` to make a
missing compatible prebuilt an error instead; CI uses this to verify selection.

Vulkan/OpenCL packages dynamically load system drivers and need no SDK link
libraries. Runtime loader/driver availability is still required. Android's
combined package also links EGL/GLESv3. CUDA 12 packages require a local CUDA
Toolkit >=12.8,<13, its runtime/cuBLAS and NVIDIA drivers. They include cubins
for SM 75, 80, 86, 89, 90, 120 and compute 120 PTX. Use `build-from-source` for
other CUDA toolkits or custom `MNN_CUDA_ARCHS`.

For custom mirrors, `MNN_PREBUILT_BASE_URL` overrides the download directory.
`MNN_PREBUILT_TAG` must include the `mnn-` prefix; changing it also requires
`MNN_PREBUILT_SHA256` for the selected archive. `MNN_PREBUILT_CACHE_DIR` selects
a shared download cache. Custom archives must retain the release directory
layout and profile ABI/backend contract.

## License

Apache-2.0. MNN and the upstream Rust bindings are also Apache-2.0.

Maintained by [ChenZibo](https://github.com/zibo-chen) (`qw.54@163.com`).

## Publishing

Both crates share a version and are released by pushing a matching `v*` tag.
GitHub Actions runs CI, publishes `mnn-runtime-sys` followed by `mnn-runtime`,
and creates a GitHub Release after publishing succeeds. See
[PUBLISHING.md](PUBLISHING.md) for release setup, updating the native baseline,
and the required `CRATES_IO_TOKEN` secret.

## Allocation and scheduling control

`run_owned(inputs)` moves an existing `Vec<Tensor>` to the worker. `run_reusing`
consumes previously returned output tensors and resizes/reuses their allocations;
outputs are selected by those tensors' names. Both consume their buffers on an
error. Native host buffers are cached per model, with NC4HW4 converted to the
public contiguous NCHW layout. `TensorInfo::is_channel_last()` identifies NHWC.

```rust,no_run
# use mnn_runtime::{Model, Tensor};
# fn process(model: &Model, inputs: Vec<Tensor>, previous_outputs: Vec<Tensor>) -> mnn_runtime::Result<()> {
let result = model.run_reusing(inputs, previous_outputs)?;
println!("{:?}", result.timings);
# Ok(()) }
```

`try_submit_owned(inputs, deadline)` returns a `PendingRun`. Dropping or cancelling
that handle skips work that has not entered MNN; `wait_timeout` also cancels
queued work on expiry. An active native inference cannot be interrupted. A
superseded frame can be cancelled before submitting its replacement; cancelled
queue entries are reclaimed by the worker, so retry after `QueueFull` rather
than assuming cancellation immediately creates capacity. Dropping the last
`Model` waits for worker shutdown.

CPU thread counts must be 1..=32. MNN shares its internal pool: initialize the
largest required multithreaded model first. A later request exceeding the
existing pool returns `ThreadCountLimited` instead of silently reducing it.
`ModelInfo::effective_threads()` exposes the actual CPU count (`None` for GPU).
This checks the session configuration, not hardware utilization.

`GpuTuning::Auto` uses Fast on OpenCL and None on Vulkan. Wide/Heavy tuning are
explicit opt-ins; Fast/Normal are rejected for Vulkan. OpenCL defaults to Buffer
storage to avoid 2D image limits on wide tensors. Other backends never receive
OpenCL memory bits; CPU thread counts are not interpreted as GPU mode flags.

Run `cargo run -p mnn-runtime --example overhead` for a tiny-model latency
comparison. `RunTimings` separates queue/gate wait, resize, upload, session execution,
and download; GPU work can complete during download. Production optimization
needs cold/warm runs, P50/P95/P99, RSS and allocation measurements on real models.

For static OpenMP builds the build script links the compiler's runtime (`gomp`,
`omp`, or `vcomp`); `MNN_OPENMP_LIB` overrides it for custom toolchains. GNU Linux
prebuilt archives are never reused for musl. Intel iOS uses the simulator SDK;
Catalyst currently requires an external MNN installation.

## Dynamic inputs and outputs

```rust,no_run
use mnn_runtime::{Runtime, RuntimeConfig, Tensor};
# fn example() -> mnn_runtime::Result<()> {
let runtime = Runtime::new(RuntimeConfig::new().with_gpu_cache_dir("mnn-cache"))?;
let model = runtime.load_file("recognizer.mnn")?;
let name = model.info().inputs()[0].name();
for (batch, width) in [(1, 64), (8, 320), (1, 96)] {
    let input = Tensor::new(name, vec![batch, 3, 48, width], vec![0.0; batch * 3 * 48 * width])?;
    let outputs = model.run_dynamic_owned(vec![input])?;
    println!("actual output: {:?}", outputs[0].shape());
}
model.save_cache()?;
# Ok(()) }
```

- `run` / `run_owned` / `run_for_outputs` validate fixed graph dimensions while
  accepting concrete values for unresolved dimensions.
- `run_dynamic` / `run_dynamic_owned` / `run_dynamic_for_outputs` also permit
  resizing fixed dimensions, such as a recognition graph exported with batch 1.
  Rank must match; MNN must support the resulting operator shapes.
- `run_reusing` follows the dynamic input policy. It adjusts returned buffers to
  the actual output shapes and retains allocations when capacity is sufficient.
- `try_submit_dynamic_owned` adds nonblocking admission and cancellation to
  dynamic inference. All synchronous methods block for queue capacity.
- `ModelInfo` is an immutable snapshot taken at load time. `TensorInfo::shape()`
  returns signed dimensions (`&[i32]`), `element_count()` returns `Option<usize>`,
  and `concrete_shape()` fails for unresolved dimensions. A deferred output is
  `[-1]` until inference; use the **returned Tensor**, not load metadata, for
  result allocation. These metadata signatures change the initial 0.1 API.
- Concrete tensors require positive dimensions, rank at most 8, and an element
  count within MNN's signed allocation limits with room for packed channels.
  Rank-zero scalar tensors are supported. Zero-element tensors and non-f32
  tensors are outside this runtime's current scope.

Every worker validates all inputs before changing shapes, resizes the session
once, refreshes tensor pointers, copies inputs, runs inference, then allocates
outputs using their current metadata. Host buffers are allocated lazily and
recreated when shape/layout changes. Native model bytes stay alive for the
session lifetime: MNN 3.6 refuses to resize after `releaseModel()`. Rust's copy of
the model is dropped after initialization.

## Persistent GPU caches

`RuntimeConfig::with_gpu_cache_dir` enables per-model cache files. The key
includes model bytes, MNN version, backend, precision, power, memory, thread
count and backend tuning/memory flags. Caches load before session creation.
`Model::cache_file()` exposes the selected path; `save_cache()` queues a flush
on the model worker after earlier requests. Worker shutdown also attempts a
flush; use explicit saves to observe errors. CPU models may have empty cache
files because they do not generate GPU kernels. The caller chooses a cache
directory appropriate to the device and application deployment.

## OCR compatibility verification

See [the reproducible OCR validation harness](validation/ocr-rs/README.md).
It replaces only the MNN adapter in a temporary OCR source copy and runs the
existing model suite. It also captures outputs from the old and new engines in
separate processes, avoiding duplicate native MNN copies. The production
`ocr-rs` checkout is not modified by the harness.

For a future OCR migration, map `build-mnn-from-source` to `build-from-source`,
`mnn-static` to `static`, `mnn-dynamic` to `dynamic`, and `static-cpp-runtime` to
the same-named runtime feature. Disable runtime default features when selecting
dynamic linking, and explicitly select the desired threading/backend features.
`crt-static` is a separate MSVC/Rust CRT policy and is not a substitute for the
MinGW `static-cpp-runtime` feature. Declare Rust 1.75 or newer in downstream
packages. Preserve OCR's public configuration/enums/error types in its adapter.

CUDA source builds set `MNN_CUDA=ON`; compatible requests use CUDA 12 prebuilts.
Use `CUDA_TOOLKIT_ROOT_DIR`, `CUDA_PATH` or `CUDA_HOME` for the toolkit, and
optionally `MNN_CUDA_ARCHS` for CMake's `CUDA_ARCHS`. Linux installs/links
`libMNN_Cuda_Main.so` even when MNN itself is static. Prebuilt static MNN uses
`lib/cuda-static/libMNN_Cuda_Main.so`; dynamic MNN uses the copy in `lib/`.
Ship the matching side library and
make it discoverable by the dynamic loader along with CUDA's runtime/cuBLAS.
Source builds and external Linux CUDA installations use the companion in
`MNN_LIB_DIR`. Windows
CUDA requires MSVC; MinGW CUDA is not supported by this baseline. Static GPU
archives use whole-archive linking to retain backend registrations.
