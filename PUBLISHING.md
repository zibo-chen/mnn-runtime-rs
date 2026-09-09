# Publishing

The GitHub repository is `zibo-chen/mnn-runtime-rs`. The workspace publishes
`mnn-runtime-sys` and `mnn-runtime` under the same version and author,
`ChenZibo <qw.54@163.com>`, with the Apache-2.0 license.

The release workflow follows
[`ocr-rs`](https://github.com/zibo-chen/rust-paddle-ocr/blob/next/.github/workflows/publish.yml):
push a `v*` tag and authenticate using the repository secret `CRATES_IO_TOKEN`.
Here Cargo handles both packages in dependency order. Use Rust/Cargo **1.90 or
newer for publishing**; the library's minimum supported Rust version remains
1.75. Python 3.11+ is needed for the release metadata check.

## First-release setup

1. Create the public GitHub repository `zibo-chen/mnn-runtime-rs`, connect it as
   `origin`, and push the reviewed source and workflows to `main`.
2. Add `CRATES_IO_TOKEN` in **Settings → Secrets and variables → Actions**.
   The crates.io token must allow publishing both `mnn-runtime-sys` and
   `mnn-runtime`, including creating them for their first release. The workflow
   passes this secret to Cargo through `CARGO_REGISTRY_TOKEN`.
3. Verify the pinned native baseline below. Moving channel names such as `dev`
   are rejected by the release metadata check.

## Native baseline

The current baseline is MNN 3.6.1 at upstream commit
`d407447ed56c4121a11ccbd266dc184ca1ead0c2`, from
[`mnn-native-prebuilds/mnn-3.6.1-d407447e-r1`](https://github.com/zibo-chen/mnn-native-prebuilds/releases/tag/mnn-3.6.1-d407447e-r1).
`crates/mnn-runtime-sys/prebuilt.rs` records the release tag, all 22 profile
checksums/backend sets, and the separately verified source archive checksum.
Archive filenames and their root directories are `<release-tag>-<target>`;
the tag already includes the `mnn-` prefix.

When updating this baseline, verify the release's `index.json`, `SHA256SUMS`,
source commit, profile ABI/CRT/threadpool settings, minimum systems and runtime
dependencies before updating the table. Do not overwrite assets used by a
published crate version. Embedded checksums reject replacements; versioned
releases should remain available for consumers rebuilding older crate versions.

CI links real Rust executables against every profile, using both static and
dynamic libraries where supplied. It covers both macOS architectures, Linux
x86_64/aarch64, Windows x86_64/i686/aarch64, Android arm64/armv7 and iOS
device/simulator. Runnable desktop targets execute CPU tests. The Linux Vulkan
profile also runs inference on Mesa Lavapipe; physical GPU/NPU validation is
separate and must not be inferred from a successful link or CPU test.

Prebuilt jobs set `MNN_REQUIRE_PREBUILT=1` to reject accidental source fallback.
iOS uses static libraries without `mnn-threadpool`; Windows prebuilts require
`+crt-static`. CUDA jobs install Toolkit 12.8.1 and validate both companion
library locations. Additional jobs test explicit source builds and the default
Windows dynamic CRT configuration. Source builds disable optional KleidiAI
downloads; packaged ARM optimizations remain available in the prebuilts.

## Prepare a version

Update both `workspace.package.version` in the root `Cargo.toml` and
`dependencies.mnn-runtime-sys.version` in `crates/mnn-runtime/Cargo.toml`.
Refresh `Cargo.lock`, review the changes, and commit them before tagging.
For the initial version:

```bash
python3 scripts/check_release.py --tag v0.1.0
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo +1.75.0 check --workspace --all-targets --locked
DOCS_RS=1 RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
cargo publish --workspace --dry-run --locked
```

For a local preview of uncommitted changes, append `--allow-dirty` to the
dry-run command. Actual publishing in CI requires a clean tagged checkout.
Run `python3 scripts/check_release.py` without `--tag` to check metadata while
native artifact promotion is still pending.

Cargo 1.90+ can package and verify both interdependent crates before either is
published. Their normalized packages use a crates.io version dependency for
`mnn-runtime-sys`; the workspace's local `path` is not a published dependency.
Avoid `--all-features`: the static/dynamic and threadpool/OpenMP choices are
mutually exclusive. CI exercises supported combinations separately.

## Release

After the reviewed commit is on `main`, push its matching version tag:

```bash
git tag -a v0.1.0 -m 'Release v0.1.0'
git push origin v0.1.0
```

The **Publish to crates.io** workflow:

1. Requires the tag to match the committed workspace and dependency versions,
   and checks the native release tag.
2. Runs the reusable CI workflow, including Linux/macOS/Windows tests, Clippy,
   Rust 1.75 compatibility, source-build modes, package verification, and
   docs.rs-style documentation builds.
3. Builds and tests in release mode, then runs
   `cargo publish --workspace --locked`. Cargo verifies both packages and
   publishes `mnn-runtime-sys` before `mnn-runtime`, waiting for registry
   availability as needed.
4. Creates a GitHub Release with generated notes only after crates.io
   publication succeeds. Tags containing a prerelease suffix create a GitHub
   prerelease.

The workflow validates committed versions instead of rewriting manifests
after checkout. It retains Cargo's package verification and does not use
`--no-verify` or `--allow-dirty` for publishing.

## Recover a partial publication

Workspace publishing is not atomic. If the sys crate was uploaded before a
later failure, inspect crates.io and the workflow logs first. From the same
clean tagged checkout, publish only the missing crate with the same token:

```bash
cargo publish -p mnn-runtime --locked
```

If both crates were published and only GitHub Release creation failed, rerun
that failed job. Do not move the tag or try to replace an existing crates.io
version with different contents.
