"""Generate and cache the shower corpus for v6.

Runs the AllShowers flow-matching generator via v3's `GenerateShowers`
and writes the raw point clouds to disk. No detector / count / kernel
logic here — downstream scripts (01_build_dataset.py, ...) read this cache.

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 00_generate_data.py
"""
import os
import sys
from modules_v6.constants import SHOWER_CACHE, NUM_SHOWERS, BATCH_SIZE

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch._utils  # workaround: torch 2.x lazy submodule needed by torch.save on Py3.13

torch.set_float32_matmul_precision('highest')

import modules_v6  # noqa: F401 — triggers sys.path injection for v3 + v4
from modules.generate_showers import GenerateShowers

os.makedirs(SHOWER_CACHE, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print(f"output dir : {SHOWER_CACHE}")
print(f"num showers: {NUM_SHOWERS}")

generate_showers_instance = GenerateShowers(
    output_dir=SHOWER_CACHE, device=device, batch_size=BATCH_SIZE,
)
generate_showers_instance(num_samples=NUM_SHOWERS, save=True)
