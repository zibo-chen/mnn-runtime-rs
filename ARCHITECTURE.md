# Architecture

## Crate boundary

- `mnn-runtime-sys` owns `links = "mnn"`, artifact verification, native linking,
  and a narrow C++ bridge. It is independently publishable.
- `mnn-runtime` owns the stable application API, validation, lifecycle, worker
  isolation, and cross-model scheduling policy.
- Domain crates own image preprocessing, model-specific tensor names,
  postprocessing, calibration, and product behavior.

This separation means a future native-binding replacement does not need to
change OCR or gaze APIs.

## Object lifetime

```text
Runtime (cheap clone)
  └── process execution gate + immutable configuration
       └── Model (cheap clone)
            └── command channel
                 └── dedicated worker thread
                      ├── Interpreter
                      └── Session (destroyed before Interpreter)
```

The worker design avoids asserting undocumented `Send` or `Sync` guarantees
for MNN raw pointers. It also turns unexpected worker termination into a Rust
error instead of exposing a dangling session.

## Current invariants

- A runtime has at least one worker thread.
- A model contains at least one input and one output.
- Every graph tensor exposed by the safe API is static-shape `f32`.
- Each inference call supplies every input exactly once.
- Requested outputs are known and unique.
- Tensor shape and element count are checked before native calls.
- Serialized runtimes created anywhere in the process share one gate.

## Planned extensions

1. Dynamic input resizing with per-request shape validation.
2. Typed `f16`, integer, and quantized tensors.
3. Cancellation/deadline-aware asynchronous submission.
4. Promote checksummed prebuilt native artifacts from `dev` to immutable tags.
5. Runtime metrics for queue wait, copy time, and inference time.
