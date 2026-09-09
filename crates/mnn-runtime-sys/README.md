# mnn-runtime-sys

Publishable native boundary for
[`mnn-runtime`](https://crates.io/crates/mnn-runtime).

The crate declares `links = "mnn"`, compiles one small C++ bridge, and links
exactly one MNN library. By default it downloads a target-specific archive from
[`zibo-chen/mnn-native-prebuilds`](https://github.com/zibo-chen/mnn-native-prebuilds/releases/tag/mnn-3.6.1-d407447e-r1) and
verifies its SHA-256 before use. Unsupported targets or backends fall back to a
pinned MNN 3.6.1 source build. Existing installations can be selected with
`MNN_INCLUDE_DIR` and `MNN_LIB_DIR`.

Most users should depend on the safe `mnn-runtime` crate instead of calling this
crate directly.

The pinned release has 22 profiles covering CPU, Metal/Core ML, Vulkan/OpenCL,
Android OpenGL ES, and CUDA 12. Cargo features choose the smallest compatible
profile, with SHA-256 validation and matching static/shared link dependencies.
Windows prebuilts require static CRT; iOS prebuilts require static linking and
disabled `mnn-threadpool`. `MNN_REQUIRE_PREBUILT=1` rejects source fallback.
See the repository README for the complete platform and runtime requirements.

`Engine::resize_inputs` validates a complete indexed set of shapes before
resizing the session. Write every input after resizing, then run and query
outputs for actual dimensions before downloading values. Native model bytes
are retained for the engine lifetime. Host tensors are lazy and shape-aware.
`new_with_cache_file` loads a kernel cache before creating the session;
`save_cache` persists tuning information on the owning thread. Callers using
this crate directly are responsible for thread/process coordination.

The `cuda` feature supports Linux and Windows MSVC toolkits, and source builds
install Linux's separate `libMNN_Cuda_Main.so`. Prebuilt static CUDA uses
`lib/cuda-static/libMNN_Cuda_Main.so`, while shared CUDA uses the copy in `lib/`.
CUDA prebuilts require Toolkit >=12.8,<13 and external CUDA/cuBLAS runtimes.
`static-cpp-runtime` links the
MinGW C++/GCC/pthread archives on Windows GNU and is distinct from `crt-static`.
See the repository README for toolkit paths and deployment requirements.
