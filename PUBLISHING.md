# Publishing

The workspace contains two independently publishable crates. Publish them in
dependency order:

1. Promote the checked `MNN-Prebuilds` assets from mutable tag `dev` to an
   immutable release tag. Keep the archive names unchanged except for the tag.
2. Update `PREBUILT_TAG` and the SHA-256 table in
   `crates/mnn-runtime-sys/build.rs`.
3. Run the full checks and a clean package verification:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo package -p mnn-runtime-sys
   cargo publish -p mnn-runtime-sys --dry-run
   ```

4. Publish the native crate and wait for the crates.io index to contain that
   exact version:

   ```bash
   cargo publish -p mnn-runtime-sys
   cargo info mnn-runtime-sys@0.1.0
   ```

5. Package, dry-run, and publish the safe crate:

   ```bash
   cargo package -p mnn-runtime
   cargo publish -p mnn-runtime --dry-run
   cargo publish -p mnn-runtime
   ```

`mnn-runtime` intentionally declares both a local `path` and a registry
`version` for `mnn-runtime-sys`. Cargo uses the path in this workspace and the
version from crates.io in the packaged crate. Consequently its package check
cannot pass before the matching sys version is published; this ordering is a
Cargo registry requirement, not a hidden local dependency.

