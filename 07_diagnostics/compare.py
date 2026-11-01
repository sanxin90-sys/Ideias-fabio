"""
07 · compare.py · check the diagnostics
=======================================

IN ONE LINE
    Compares the new expC/expD/expF results with the originals in ../reference/.

OUTPUT   one line per file: how many numbers, and the largest absolute difference.
         0 means identical. Neural-network numbers may move in the last digits with
         another jax version or CPU; the kernel (expD) is exact.
RUN      cd 07_diagnostics ; python compare.py   (after expC, expD, expF)
"""
import json
import os

import numpy as np


def flat(d, p=''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f'{p}/{k}')
    elif isinstance(d, (int, float)) and d is not None:
        yield p, float(d)


for name in ('expC_results.json', 'expD_results.json', 'expF_results.json'):
    if not os.path.exists(name):
        print(f'{name}: not run yet'); continue
    new, ref = dict(flat(json.load(open(name)))), dict(flat(json.load(open(f'../reference/{name}'))))
    diff = max((abs(new[k] - ref[k]) for k in ref if k in new and np.isfinite(ref[k])), default=0)
    print(f'{name}: {len(ref)} numbers, largest absolute difference {diff:.2g}')
