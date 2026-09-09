# mnn-runtime-sys

Publishable native boundary for
[`mnn-runtime`](https://crates.io/crates/mnn-runtime).

The crate declares `links = "mnn"`, compiles one small C++ bridge, and links
exactly one MNN library. By default it downloads a target-specific archive from
[`zibo-chen/MNN-Prebuilds`](https://github.com/zibo-chen/MNN-Prebuilds) and
verifies its SHA-256 before use. Unsupported targets or backends fall back to a
pinned MNN 3.6.0 source build. Existing installations can be selected with
`MNN_INCLUDE_DIR` and `MNN_LIB_DIR`.

Most users should depend on the safe `mnn-runtime` crate instead of calling this
crate directly.

