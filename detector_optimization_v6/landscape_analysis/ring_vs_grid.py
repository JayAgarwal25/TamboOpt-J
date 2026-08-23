#!/usr/bin/env python3
"""
Coverage-shape test: does a boundary-only ring layout (zero interior
coverage) score differently than a uniformly-spaced grid?

Tests whether the utility landscape rewards full-area coverage (interior
matters) or only the envelope/edge of the available area (interior is wasted
detectors). Uses the existing frozen FNN+recon surrogate -- no training.
"""
import sys, os, json
import numpy as np
import torch

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (input/output locations)
import modules_v6  # injects v3/v4 paths

from common import Scorer
from modules_v6.detector_strategies_ne import (
    layout_grid, layout_uniform_random, layout_edge_ring,
)

SEED = 42
N_SEEDS = 5     # independent layout instantiations per strategy (jitter/phase differs)
N_BATCHES = 5   # independent fresh primary batches per layout (avoid batch-overfitting bias)

print("=" * 70)
print("Q3 follow-up: edge-ring vs. uniform grid coverage")
print("=" * 70)

# Batches are drawn per (layout seed) below rather than held fixed, so ask the
# scorer for none up front and use its draw() directly.
sc = Scorer(n_batches=0, device=torch.device("cpu"))
mountain = sc.mountain
fresh_batch = sc.draw
eval_U = sc.U_on


strategies = {
    "edge_ring":      layout_edge_ring,
    "grid":           layout_grid,
    "uniform_random": layout_uniform_random,
}

results = {name: [] for name in strategies}
# Every seed is scored on the SAME N_BATCHES primary batches, so the 25 values
# per strategy are 5 layouts x 5 shared batches, not 25 independent draws. The
# layout is what varies between strategies, so it is the unit of replication.
seed_means = {name: [] for name in strategies}
mean_r_last = {}
for name, fn in strategies.items():
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed * 777 + 1)
        x, y = fn(mountain, rng=rng)
        batch_Us = []
        for bseed in range(N_BATCHES):
            primary = fresh_batch(SEED + bseed)
            u, mean_r = eval_U(x, y, primary)
            batch_Us.append(u)
        results[name].extend(batch_Us)
        seed_means[name].append(float(np.mean(batch_Us)))
        mean_r_last[name] = mean_r
        print(f"  {name:15s} seed={seed}: U mean={np.mean(batch_Us):.3f}  "
              f"std={np.std(batch_Us):.3f}  mean_r={mean_r:.4f}")

print()
print("=" * 70)
print("SUMMARY (mean +/- std over all seeds x batches)")
print("=" * 70)
summary = {}
for name, vals in results.items():
    vals = np.array(vals)
    summary[name] = dict(mean=float(vals.mean()), std=float(vals.std()), n=len(vals),
                         seed_means=seed_means[name])
    print(f"  {name:15s}: U = {vals.mean():.3f} +/- {vals.std():.3f}  (n={len(vals)})")

out_path = os.path.join(_layouts.results_dir(), "ring_vs_grid_results.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {out_path}")

print()
g = np.array(seed_means["grid"])
e = np.array(seed_means["edge_ring"])
gap = float(g.mean() - e.mean())
# Standard error of the difference of the two layout-level means. The old
# version compared the gap against a hard-coded 5 utility units and called
# anything smaller "within noise", which was not a noise calculation: the
# layout-to-layout scatter here is well under one unit.
se = float(np.sqrt(g.var(ddof=1) / len(g) + e.var(ddof=1) / len(e)))
print(f"Grid - Edge-ring gap: {gap:+.3f} +/- {se:.3f} "
      f"({100.0 * gap / g.mean():+.1f}% of grid U)")
if abs(gap) < 2.0 * se:
    print("  => Grid and edge-ring are within noise of each other: coverage shape "
          "does not measurably matter at this detector count and area.")
elif gap > 0:
    print(f"  => Grid beats edge-only ring by {gap / se:.1f} standard errors: "
          "interior coverage matters, utility needs full-area evaluation rather "
          "than just the envelope.")
else:
    print(f"  => Edge-only ring beats grid by {abs(gap) / se:.1f} standard errors: "
          "utility mostly rewards boundary flux interception, and interior "
          "coverage may be wasted detectors.")
