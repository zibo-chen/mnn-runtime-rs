#!/usr/bin/env python3
"""Run the same probe against either OCR checkout, in its own process."""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('ocr', type=Path)
parser.add_argument('models', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--backend', choices=['cpu', 'metal', 'opencl', 'vulkan', 'cuda'], default='cpu')
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
if any(output.glob('*.bin')):
    parser.error('output directory already contains captures')
with tempfile.TemporaryDirectory(prefix='mnn-ocr-probe-') as work:
    work = Path(work)
    (work / 'src').mkdir()
    (work / 'src/main.rs').write_text(Path(__file__).with_name('probe.rs').read_text())
    features = [] if args.backend == 'cpu' else [args.backend]
    (work / 'Cargo.toml').write_text(
        '[package]\nname = "mnn-ocr-probe"\nversion = "0.0.0"\nedition = "2021"\n'
        '[dependencies]\nocr-rs = { path = ' + json.dumps(str(args.ocr.resolve())) + ', features = ' + json.dumps(features) + ' }\n'
        '[workspace]\n[profile.dev]\nopt-level = 3\n')
    env = os.environ.copy()
    env['PROBE_BACKEND'] = args.backend
    env.setdefault('CARGO_TARGET_DIR', str(args.ocr.resolve() / 'target'))
    subprocess.run(['cargo', 'run', '--manifest-path', str(work / 'Cargo.toml'), '--',
                    str(args.models.resolve()), str(output)], env=env, check=True)
