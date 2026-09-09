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

- A runtime is a configuration factory; each loaded model owns one worker thread.
- A model contains at least one input and one output.
- Every graph tensor exposed by the safe API is static-shape `f32`.
- Each inference call supplies every input exactly once.
- Requested outputs are known and unique.
- Tensor shape and element count are checked before native calls.
- Serialized runtimes share the gate owned by the unique sys crate, including across high-level crate versions.
- Creation, inference and destruction participate in that gate; explicit Parallel runtimes opt out.
- Model bytes are released after native initialization; host buffers survive across requests.
- Admission is bounded and expired/cancelled requests are checked again after acquiring the gate.

## Planned extensions

1. Dynamic input resizing with per-request shape validation.
2. Typed `f16`, integer, and quantized tensors.
3. Fair, workload-aware scheduling across models and a nonblocking Future adapter.
4. Promote checksummed prebuilt native artifacts from `dev` to immutable tags.
5. Hardware-specific allocation/RSS/throughput baselines using the exposed stage timings.

## Scheduler decision

Keep model-affine workers in this iteration. Ownership remains explicit for
CPU and GPU objects, while shared sys coordination and bounded admission fix
the immediate resource problems. A local tiny-model debug-profile run measured
P50 15.75 us for the borrowed API and 12 us with reused outputs (100 warmups,
1000 samples; input cloning included in both). These are wrapper-path
observations on one machine, not a prediction for OCR or GPU inference. They do
not establish that a shared executor would improve mixed OCR/gaze tail latency.
A shared executor remains an option once real-model mixed-workload benchmarks
quantify fairness, model-load stalls and the cost of a single execution lane.
