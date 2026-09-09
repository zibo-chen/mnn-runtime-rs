# OCR compatibility validation

This harness tests the runtime without modifying the production OCR checkout.
`prepare.py` copies OCR source/tests into a new directory, links its model/image
assets, removes its native build dependencies and replaces its MNN adapter with
`adapter.rs`. OCR algorithms, public model APIs and model tests remain unchanged.
Missing assets fail preparation instead of allowing the suite to silently skip.

The adapter is a **validation fixture**, not a finished public OCR migration.
It covers the CPU/Metal NCHW model paths, configuration, shared runtime and raw
inference APIs exercised by the suites below. It does not implement the old
`SessionPool` or its `available()` semantics, nor general public layout overrides.
Those compatibility details remain work in the downstream OCR crate. They do
not require another native bridge: pool permits can be managed above `Model`,
and actual graph layout is exposed by `TensorInfo::is_channel_last()`.

From the runtime repository root, use a fresh destination:

```sh
python3 validation/ocr-rs/prepare.py /path/to/rust-paddle-ocr /tmp/ocr-runtime-validation
cargo test --manifest-path /tmp/ocr-runtime-validation/Cargo.toml --test all_models_tests
cargo tree --manifest-path /tmp/ocr-runtime-validation/Cargo.toml -i mnn-runtime-sys
```

The printed dependency tree must contain exactly one sys version. The validation
copy has no build.rs, C++ bridge or bindgen/cmake/cc build dependencies. Its model,
image, and fixture assets are required and are not included in this repository.

Capture identical raw inputs with old and new engines in separate processes:

```sh
python3 validation/ocr-rs/capture.py /path/to/rust-paddle-ocr /path/to/rust-paddle-ocr/models /tmp/ocr-baseline
python3 validation/ocr-rs/capture.py /tmp/ocr-runtime-validation /path/to/rust-paddle-ocr/models /tmp/ocr-runtime
python3 validation/ocr-rs/compare.py /tmp/ocr-baseline /tmp/ocr-runtime
```

`probe.rs` requires all 24 models. Detectors run 64×96 → 96×160 → 64×96 inputs.
Recognizers run height 48 with (batch, width) = (1,64) → (3,160) → (1,96) →
(1,64). Orientation runs 224×224. This gives 87 captures, including growth,
shrinkage, batch changes and repeated shapes, with deterministic nonzero input.
`compare.py` checks shape, count, finite values and every f32 value. Defaults are
absolute tolerance 1e-5 and relative tolerance 1e-4. CPU validation on the current
corpus produced bitwise-equal numeric values (maximum absolute error 0).

To test a GPU on a machine with that backend compiled and available:

```sh
MNN_GPU_TEST_BACKEND=metal cargo test --manifest-path /tmp/ocr-runtime-validation/Cargo.toml --features metal --test gpu_regressions -- --include-ignored
python3 validation/ocr-rs/capture.py /tmp/ocr-runtime-validation /path/to/rust-paddle-ocr/models /tmp/ocr-metal --backend metal
```

Replace Metal with OpenCL or Vulkan for the original GPU regression test. The
capture tool additionally accepts CUDA. Device validation must be run on the
corresponding supported hardware; enabling a feature is not execution evidence.

## Validation recorded on 2026-09-09

Environment: Apple Silicon, macOS 26.5.2, MNN 3.6.0, Rust 1.97.1.
OCR baseline: `a5324eead71a066d3569d12e89a8499bed92b779` (ocr-rs 2.4.1).
Runtime changes are based on `b665667e8269c74a691d2e127e1e9a761df2ab56`.

| Check | Result |
| --- | --- |
| Original OCR model suite through runtime adapter | 49 passed, 0 skipped |
| Separate-process CPU comparison | 24 models, 87 cases, 9,456,316 values; maximum absolute error 0 |
| Runtime workspace tests with Metal enabled | 24 passed |
| Original OCR dynamic/GPU regressions on Metal | 2 passed, including cache save/reload |
| Pinned upstream source build, dynamic linking | 20 runtime tests passed |
| Rust 1.85.0 check, workspace/all targets/locked | Passed |
| Clippy, workspace/all targets/Metal, warnings denied | Passed |
| Dependency ownership in validation copy | One mnn-runtime-sys, links = mnn |

CUDA toolkit execution, OpenCL/Vulkan device execution and MinGW linkage were
not tested on this Mac. Target/link planning has portable regression coverage;
actual GPU/platform runs remain necessary before claiming those devices are
validated. Existing CI continues to cover CPU/static/source/dynamic/MSRV builds.
