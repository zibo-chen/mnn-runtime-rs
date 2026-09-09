#!/usr/bin/env python3
"""Link a real Rust consumer against both variants of a pinned native profile."""

import argparse
import os
from pathlib import Path
import platform
import subprocess


def run(command, env):
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", default="")
    parser.add_argument("--backend", help="Also require inference on this provisioned device")
    args = parser.parse_args()
    env = os.environ.copy()
    env["MNN_REQUIRE_PREBUILT"] = "1"
    env.setdefault("MNN_PREBUILT_CACHE_DIR", str(Path("target/native-downloads").resolve()))
    target = args.target
    ios = "-apple-ios" in target
    if "android" in target:
        ndk = Path(env["ANDROID_NDK_ROOT"])
        host = "darwin-x86_64" if platform.system() == "Darwin" else "linux-x86_64"
        tools = ndk / "toolchains/llvm/prebuilt" / host / "bin"
        clang_target = "armv7a-linux-androideabi" if target == "armv7-linux-androideabi" else target
        key = target.replace("-", "_")
        env[f"CC_{key}"] = str(tools / f"{clang_target}21-clang")
        env[f"CXX_{key}"] = str(tools / f"{clang_target}21-clang++")
        env[f"AR_{key}"] = str(tools / "llvm-ar")
        env[f"CARGO_TARGET_{key.upper()}_LINKER"] = env[f"CC_{key}"]
    if "windows-msvc" in target:
        env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -C target-feature=+crt-static"
    if env.get("CUDA_PATH"):
        root = Path(env["CUDA_PATH"])
        env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(root / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    host = next(line[6:] for line in subprocess.check_output(["rustc", "-vV"], text=True).splitlines() if line.startswith("host: "))
    can_run = target == host or (host == "x86_64-pc-windows-msvc" and target == "i686-pc-windows-msvc")
    for mode in (["static"] if ios else ["static", "dynamic"]):
        features = ["prebuilt", mode] + ([] if ios else ["mnn-threadpool"])
        features += [feature for feature in args.features.split(",") if feature]
        options = ["-p", "mnn-runtime", "--target", target, "--no-default-features", "--features", ",".join(features), "--locked"]
        run(["cargo", "build", *options, "--example", "inspect"], env)
        if can_run:
            run(["cargo", "test", *options], env)
        if args.backend:
            if not can_run:
                parser.error("backend inference requires a runnable target")
            env["MNN_TEST_BACKEND"] = args.backend
            run(["cargo", "test", *options, "--test", "backends", "--", "--ignored"], env)


if __name__ == "__main__":
    main()
