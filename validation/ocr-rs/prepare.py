#!/usr/bin/env python3
"""Create an isolated OCR validation copy. Never edits the input checkout."""
import argparse
import json
import re
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('ocr', type=Path)
parser.add_argument('destination', type=Path)
args = parser.parse_args()
source, destination = args.ocr.resolve(), args.destination.resolve()
if destination.exists():
    parser.error('destination must not exist')
suite = (source / 'tests/all_models_tests.rs').read_text()
required = set(re.findall(r'"((?:models/|res/|tests/fixtures/)[^"]+)"', suite))
missing = [path for path in sorted(required) if not (source / path).is_file()]
if missing:
    parser.error('missing OCR assets (tests would silently skip): ' + ', '.join(missing))
runtime = Path(__file__).resolve().parents[2]
for directory in ['src', 'tests']:
    shutil.copytree(source / directory, destination / directory)
for directory in ['models', 'res']:
    (destination / directory).symlink_to(source / directory, target_is_directory=True)
manifest = (source / 'Cargo.toml').read_text()
a = manifest.index('[build-dependencies]')
b = manifest.index('[dependencies]', a)
manifest = manifest[:a] + manifest[b:]
manifest = manifest.replace('[dependencies]', '[dependencies]\nmnn-runtime = { path = ' + json.dumps(str(runtime / 'crates/mnn-runtime')) + ' }')
# Do not carry examples/benches into the validation copy.
a = manifest.index('[[bench]]')
manifest = manifest[:a] + '\n[workspace]\n\n[profile.dev]\nopt-level = 3\n'
for feature in ['metal', 'opencl', 'opengl', 'vulkan', 'cuda', 'coreml', 'static-cpp-runtime']:
    manifest = manifest.replace(feature + ' = []', feature + ' = ["mnn-runtime/' + feature + '"]')
(destination / 'Cargo.toml').write_text(manifest)
shutil.copyfile(Path(__file__).with_name('adapter.rs'), destination / 'src/mnn/mod.rs')
print(destination)
