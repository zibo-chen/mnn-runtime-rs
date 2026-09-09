# Changelog

## 0.1.0 - Unreleased

- Add a pinned single-owner MNN dependency boundary.
- Add safe static-shape `f32` named multi-input/multi-output inference.
- Add model worker isolation and deterministic native lifetime management.
- Add process-wide serialized execution with explicit parallel opt-in.
- Add model inspection example and migration documentation.


## Unreleased

- Normalize packed native tensors to contiguous public layouts and validate
  output copies. Reuse native staging buffers and release model byte copies.
- Add owned submission, reusable outputs, bounded queues, cancellation/deadlines,
  stage timings, layout metadata and actual CPU thread diagnostics.
- Move process coordination to the unique sys crate and cover native lifecycles.
- Separate GPU mode bits from CPU threads; use Fast OpenCL/None Vulkan tuning.
- Match Windows CRTs, choose Intel iOS simulator SDKs, restrict GNU prebuilts to
  GNU targets, and link OpenMP runtimes for static builds.
- Verify extraction identity when artifact checksums change; isolate temporary
  downloads and bound network retries/timeouts.
- Add numerical fixtures, queue/lifecycle tests, target selection tests and
  source/static/dynamic/OpenMP CI coverage.
