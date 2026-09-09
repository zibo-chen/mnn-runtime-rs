#!/usr/bin/env python3
"""Validate workspace release metadata (Python 3.11+, no dependencies)."""

import argparse
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path):
    with (ROOT / path).open("rb") as manifest:
        return tomllib.load(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Also enforce release-only requirements for this v* tag")
    args = parser.parse_args()

    workspace = read_manifest("Cargo.toml")["workspace"]["package"]
    version = workspace["version"]
    for name in ("mnn-runtime-sys", "mnn-runtime"):
        manifest = read_manifest(f"crates/{name}/Cargo.toml")
        package = manifest["package"]
        for field in ("version", "authors", "license", "repository", "homepage", "rust-version"):
            if package[field] != {"workspace": True}:
                parser.error(f"{name}: {field} must inherit workspace.package.{field}")
        if name == "mnn-runtime":
            dependency = manifest["dependencies"]["mnn-runtime-sys"]
            if dependency["version"] != version:
                parser.error(f"mnn-runtime-sys dependency must match workspace version {version}")

    build = (ROOT / "crates/mnn-runtime-sys/prebuilt.rs").read_text()
    checksums = re.findall(r'sha256: "([^"]*)"', build)
    if not checksums or any(not re.fullmatch(r"[a-f0-9]{64}", checksum) for checksum in checksums):
        parser.error("every prebuilt SHA-256 must contain exactly 64 hexadecimal characters")

    if args.tag:
        if args.tag != f"v{version}":
            parser.error(f"tag {args.tag!r} does not match workspace version v{version}")
        tag = re.search(r'const PREBUILT_TAG: &str = "([^"]+)";', build)
        if not tag or tag[1].lower() in ("dev", "latest", "main", "master", "nightly"):
            parser.error("pin PREBUILT_TAG to a versioned MNN release before publishing; see PUBLISHING.md")

    print(f"Release metadata OK: mnn-runtime-sys and mnn-runtime {version}")


if __name__ == "__main__":
    main()
