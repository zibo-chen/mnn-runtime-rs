#!/usr/bin/env python3
"""Compare shape and every f32 value captured by probe.rs, without NumPy."""
import argparse
import array
import math
import struct
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('actual', type=Path)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-4)
args = parser.parse_args()

def read(path):
    raw = path.read_bytes()
    rank, = struct.unpack_from('<I', raw)
    shape = struct.unpack_from('<' + 'Q' * rank, raw, 4)
    values = array.array('f', raw[4 + rank * 8:])
    if sys.byteorder != 'little':
        values.byteswap()
    assert len(values) == math.prod(shape), path
    return shape, values

files = sorted(path.name for path in args.baseline.glob('*.bin'))
assert files and files == sorted(path.name for path in args.actual.glob('*.bin'))
max_error, elements, failures = 0.0, 0, []
for filename in files:
    shape, expected = read(args.baseline / filename)
    actual_shape, actual = read(args.actual / filename)
    assert shape == actual_shape, (filename, shape, actual_shape)
    for index, (reference, result) in enumerate(zip(expected, actual)):
        error = abs(reference - result)
        max_error = max(max_error, error)
        if not math.isfinite(result) or error > args.atol + args.rtol * abs(reference):
            if len(failures) < 10:
                failures.append((filename, index, reference, result))
        elements += 1
assert not failures, failures
print(f'{len(files)} cases, {elements} f32 values passed; maximum absolute error: {max_error:g}')
