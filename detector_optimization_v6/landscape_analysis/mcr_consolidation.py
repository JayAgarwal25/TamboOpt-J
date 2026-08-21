#!/usr/bin/env python3
"""
MCR-style (Model Class Reliance) consolidation across all 12 independently-
obtained L-BFGS optima (2 original: grid/center-scheme, plus 10 from the
5-seed x 2-scheme extension). Each optimum already has its own leave-one-out
dip sweep computed in its own native detector indexing (no new GPU/model
evaluation needed here). This script:

  1. Hungarian-aligns all 12 layouts to one reference (ds_center) via
     modules_v6.opt_core.align_to_reference, which generalizes to any K.
  2. Remaps each layout's dips into the shared reference-slot ordering.
  3. Per reference slot, computes the distribution of dip across the 12
     layouts (mean/std/min/max), how often that slot held one of that
     layout's own top-5 critical / bottom-5 redundant detectors (hit-rate
     out of 12), and the spatial spread of the 12 layouts' actual detector
     positions mapped to that slot (distance from the reference layout's own
     position at that slot).

Answers: is there a stable RANGE of criticality per physical region across
many independently-obtained good layouts, or is "which detectors are
critical" essentially different every time?
"""
import os
import sys
import json

import numpy as np
import torch

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (input/output locations)
import modules_v6  # noqa: F401
from modules_v6.opt_core import align_to_reference

# Results live beside the other run outputs, not next to the code.
HERE = _layouts.results_dir()
RUN_BASE = _layouts.RUNS
OUT_DIR = os.path.join(HERE, "mcr_consolidation")
os.makedirs(OUT_DIR, exist_ok=True)

N_HIGHLIGHT = 5

# (tag, layout run-dir suffix, dips JSON path, dips JSON key path)
LAYOUTS = [
    ("ds_center", "ds_center",
     os.path.join(HERE, "detector_removal_results.json"), ("leave_one_out", "dips")),
    ("ds_grid", "ds_grid",
     os.path.join(HERE, "detector_removal_cross_layout_check_results.json"), ("grid_dips",)),
]
for seed in (101, 202, 303, 404, 505):
    for scheme in ("grid", "center"):
        tag = f"ds_seed{seed}_{scheme}"
        LAYOUTS.append((
            tag, tag,
            os.path.join(HERE, "other_optimizers", f"seed{seed}_{scheme}", "detector_removal_results.json"),
            ("leave_one_out", "dips"),
        ))

assert len(LAYOUTS) == 12, f"expected 12 layouts, got {len(LAYOUTS)}"


def load_layout_xy(run_suffix):
    path = os.path.join(RUN_BASE, f"test_v6_run_04_optimize_lbfgs_ensemble_{run_suffix}", "layout_best.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    x = d["x"].float().reshape(-1).numpy()
    y = d["y"].float().reshape(-1).numpy()
    return np.stack([x, y], axis=-1), float(d["U"])


def load_dips(json_path, key_path):
    with open(json_path) as f:
        data = json.load(f)
    for k in key_path:
        data = data[k]
    return np.asarray(data, dtype=np.float64)


tags = [t for t, _, _, _ in LAYOUTS]
print("=" * 70)
print(f"MCR consolidation across {len(tags)} independent L-BFGS optima")
print("=" * 70)
print("Layouts:", tags)

positions = np.empty((12, 100, 2), dtype=np.float64)
dips_native = np.empty((12, 100), dtype=np.float64)
full_Us = np.empty(12, dtype=np.float64)

for k, (tag, run_suffix, json_path, key_path) in enumerate(LAYOUTS):
    xy, U = load_layout_xy(run_suffix)
    assert xy.shape == (100, 2), f"{tag}: unexpected shape {xy.shape}"
    dips = load_dips(json_path, key_path)
    assert dips.shape == (100,), f"{tag}: unexpected dips shape {dips.shape}"
    positions[k] = xy
    dips_native[k] = dips
    full_Us[k] = U
    print(f"  [{k:2d}] {tag:<16} U={U:.3f}  dip mean/std/max = "
          f"{dips.mean():.3f}/{dips.std():.3f}/{dips.max():.3f}")

REF_IDX = tags.index("ds_center")
print(f"\nReference layout: {tags[REF_IDX]} (index {REF_IDX})")

aligned_positions, perms = align_to_reference(positions, ref_idx=REF_IDX)

# aligned_dips[k, j] = dips_native[k][perms[k][j]]
aligned_dips = np.take_along_axis(dips_native, perms, axis=1)

# Per-layout own critical-5 / redundant-5 sets, in NATIVE indexing.
own_critical = np.empty((12, N_HIGHLIGHT), dtype=np.int64)
own_redundant = np.empty((12, N_HIGHLIGHT), dtype=np.int64)
for k in range(12):
    order = np.argsort(dips_native[k])
    own_critical[k] = order[-N_HIGHLIGHT:]
    own_redundant[k] = order[:N_HIGHLIGHT]

# For each layout k and reference slot j, does perms[k][j] (the native index
# occupying that slot) fall in that layout's own critical/redundant set?
critical_hit = np.zeros((12, 100), dtype=bool)
redundant_hit = np.zeros((12, 100), dtype=bool)
for k in range(12):
    crit_set = set(own_critical[k].tolist())
    redund_set = set(own_redundant[k].tolist())
    for j in range(100):
        native_idx = perms[k, j]
        critical_hit[k, j] = native_idx in crit_set
        redundant_hit[k, j] = native_idx in redund_set

critical_hit_rate = critical_hit.sum(axis=0)   # (100,) in [0, 12]
redundant_hit_rate = redundant_hit.sum(axis=0)

# Spatial spread: distance from each layout's aligned position at slot j to
# the reference layout's own position at slot j. Excludes the reference
# layout itself (its distance to itself is a trivial, always-zero row that
# would otherwise deflate the mean by exactly 12/11 -- caught by code review).
ref_pos = aligned_positions[REF_IDX]  # (100, 2)
dist_to_ref = np.sqrt(((aligned_positions - ref_pos[None, :, :]) ** 2).sum(axis=-1))  # (12, 100)
non_ref_mask = np.ones(12, dtype=bool)
non_ref_mask[REF_IDX] = False
dist_to_ref_non_ref = dist_to_ref[non_ref_mask]  # (11, 100)

slot_dip_mean = aligned_dips.mean(axis=0)
slot_dip_std = aligned_dips.std(axis=0)
slot_dip_min = aligned_dips.min(axis=0)
slot_dip_max = aligned_dips.max(axis=0)
slot_dist_mean = dist_to_ref_non_ref.mean(axis=0)
slot_dist_max = dist_to_ref_non_ref.max(axis=0)

print("\n" + "=" * 70)
print("SUMMARY (per reference slot, across 12 layouts)")
print("=" * 70)
n_never_critical = int((critical_hit_rate == 0).sum())
n_always_or_often_critical = int((critical_hit_rate >= 6).sum())
n_never_redundant = int((redundant_hit_rate == 0).sum())
n_always_or_often_redundant = int((redundant_hit_rate >= 6).sum())
print(f"critical hit-rate: {n_never_critical}/100 slots never critical, "
      f"{n_always_or_often_critical}/100 slots critical in >=6/12 layouts")
print(f"redundant hit-rate: {n_never_redundant}/100 slots never redundant, "
      f"{n_always_or_often_redundant}/100 slots redundant in >=6/12 layouts")
print(f"mean dip across all slots/layouts: {aligned_dips.mean():.4f}")
print(f"mean spatial spread (dist to ref, {non_ref_mask.sum()} non-reference layouts) "
      f"across all slots: {dist_to_ref_non_ref.mean():.1f}m (max observed: {dist_to_ref_non_ref.max():.1f}m)")

top_hit_slots = np.argsort(-critical_hit_rate)[:10]
print("\nTop-10 reference slots by critical hit-rate:")
for j in top_hit_slots:
    print(f"  slot {j:3d}: hit-rate={critical_hit_rate[j]}/12  "
          f"mean dip={slot_dip_mean[j]:.3f} std={slot_dip_std[j]:.3f}  "
          f"mean spatial spread={slot_dist_mean[j]:.1f}m")

# ---------------------------------------------------------------------------
# Follow-ups from independent second-opinion review (code + physics lenses):
# the single-reference hit-rate/spread result alone can't tell "no shared
# structure" apart from "the forced-1-to-1 alignment breaking down for most
# slots" -- these three checks were flagged as the highest-value ways to
# resolve that ambiguity without any new GPU/model evaluation.
# ---------------------------------------------------------------------------

# ---- 1. Permutation null baseline for hit-rate (label-shuffle, alignment fixed) ----
print("\n" + "=" * 70)
print("PERMUTATION NULL BASELINE for hit-rate (label-shuffle, alignment fixed)")
print("=" * 70)
# inv_perm[k, i] = slot j such that perms[k, j] == i
inv_perm = np.empty_like(perms)
for k in range(12):
    inv_perm[k, perms[k]] = np.arange(100)

N_PERM = 10000
rng_perm = np.random.default_rng(7)
null_hit_count = np.zeros((N_PERM, 100), dtype=np.int64)
for k in range(12):
    rand_vals = rng_perm.random((N_PERM, 100))
    rand_native = np.argsort(rand_vals, axis=1)[:, :N_HIGHLIGHT]   # (N_PERM, 5), no replacement
    slots = inv_perm[k][rand_native]                                # (N_PERM, 5)
    rows = np.repeat(np.arange(N_PERM), N_HIGHLIGHT)
    np.add.at(null_hit_count, (rows, slots.reshape(-1)), 1)

null_max_hit = null_hit_count.max(axis=1)
null_never_hit = (null_hit_count == 0).sum(axis=1)
null_hit_ge3 = (null_hit_count >= 3).sum(axis=1)

obs_max_hit = int(critical_hit_rate.max())
obs_never_hit = int((critical_hit_rate == 0).sum())
obs_hit_ge3 = int((critical_hit_rate >= 3).sum())

p_max_hit = float((null_max_hit >= obs_max_hit).mean())
p_never_hit_low = float((null_never_hit <= obs_never_hit).mean())
p_hit_ge3 = float((null_hit_ge3 >= obs_hit_ge3).mean())

print(f"Observed max hit-rate = {obs_max_hit}/12; null P(max >= observed) = {p_max_hit:.4f} "
      f"(null mean {null_max_hit.mean():.2f}, std {null_max_hit.std():.2f})")
print(f"Observed never-critical slots = {obs_never_hit}/100; null P(never-critical <= observed) = "
      f"{p_never_hit_low:.4f} (null mean {null_never_hit.mean():.1f}, std {null_never_hit.std():.1f})")
print(f"Observed slots with hit-rate>=3 = {obs_hit_ge3}; null P(count >= observed) = {p_hit_ge3:.4f} "
      f"(null mean {null_hit_ge3.mean():.2f}, std {null_hit_ge3.std():.2f})")

# ---- 2. Alignment-free direct spatial clustering test (real-world coords, no Hungarian) ----
print("\n" + "=" * 70)
print("ALIGNMENT-FREE SPATIAL CLUSTERING TEST (no Hungarian, real-world coords)")
print("=" * 70)


def pooled_nearest_cross_layout_dist(idx_sets):
    pts, layout_id = [], []
    for k in range(12):
        for idx in idx_sets[k]:
            pts.append(positions[k, idx])
            layout_id.append(k)
    pts = np.array(pts)
    layout_id = np.array(layout_id)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    same_layout = layout_id[:, None] == layout_id[None, :]
    d2_masked = np.where(same_layout, np.inf, d2)
    return np.sqrt(d2_masked.min(axis=1))


obs_critical_nn = pooled_nearest_cross_layout_dist([own_critical[k] for k in range(12)])
obs_redundant_nn = pooled_nearest_cross_layout_dist([own_redundant[k] for k in range(12)])

rng_boot = np.random.default_rng(20260712)
N_BOOTSTRAP = 500
null_means = np.empty(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    rand_idx_sets = [rng_boot.choice(100, size=N_HIGHLIGHT, replace=False) for _ in range(12)]
    null_means[b] = pooled_nearest_cross_layout_dist(rand_idx_sets).mean()

null_mean, null_std = null_means.mean(), null_means.std()
z_critical = (obs_critical_nn.mean() - null_mean) / null_std
z_redundant = (obs_redundant_nn.mean() - null_mean) / null_std
pctl_critical = float((null_means < obs_critical_nn.mean()).mean())
pctl_redundant = float((null_means < obs_redundant_nn.mean()).mean())


def _tightness_label(z):
    if z < -1:
        return "clusters TIGHTER than random"
    if z > 1:
        return "clusters LOOSER than random"
    return "indistinguishable from random"


print(f"Observed mean nearest cross-layout distance, CRITICAL points (n=60): {obs_critical_nn.mean():.1f}m")
print(f"Observed mean nearest cross-layout distance, REDUNDANT points (n=60): {obs_redundant_nn.mean():.1f}m")
print(f"Null (random 5-of-100 per layout, {N_BOOTSTRAP} draws): mean nearest cross-layout "
      f"distance = {null_mean:.1f}m +/- {null_std:.1f}m")
print(f"  CRITICAL: z={z_critical:+.2f} (percentile {pctl_critical*100:.1f}%) -- {_tightness_label(z_critical)}")
print(f"  REDUNDANT: z={z_redundant:+.2f} (percentile {pctl_redundant*100:.1f}%) -- {_tightness_label(z_redundant)}")

# ---- 3. Multi-reference sensitivity check ----
print("\n" + "=" * 70)
print("MULTI-REFERENCE SENSITIVITY CHECK")
print("=" * 70)


def per_slot_hit_rate(ref_idx):
    _, perms_r = align_to_reference(positions, ref_idx=ref_idx)
    hit = np.zeros(100, dtype=np.int64)
    for k in range(12):
        crit_set = set(own_critical[k].tolist())
        for j in range(100):
            if perms_r[k, j] in crit_set:
                hit[j] += 1
    return hit


REF2_IDX = tags.index("ds_grid")
REF3_IDX = tags.index("ds_seed202_center")
hit_ref2 = per_slot_hit_rate(REF2_IDX)
hit_ref3 = per_slot_hit_rate(REF3_IDX)


def top_slots_positions(hit_arr, ref_idx, k_top=10):
    top = np.argsort(-hit_arr)[:k_top]
    return positions[ref_idx, top], hit_arr[top]


top1_pos, top1_hit = top_slots_positions(critical_hit_rate, REF_IDX)
top2_pos, top2_hit = top_slots_positions(hit_ref2, REF2_IDX)
top3_pos, top3_hit = top_slots_positions(hit_ref3, REF3_IDX)

MATCH_THRESHOLD_M = 250.0


def match_fraction(pos_a, pos_b, thresh=MATCH_THRESHOLD_M):
    d = np.sqrt(((pos_a[:, None, :] - pos_b[None, :, :]) ** 2).sum(-1))
    return float((d.min(axis=1) < thresh).mean())


frac_1to2 = match_fraction(top1_pos, top2_pos)
frac_1to3 = match_fraction(top1_pos, top3_pos)
frac_2to3 = match_fraction(top2_pos, top3_pos)

print(f"Reference 1 ({tags[REF_IDX]}) top-10 hit-counts: {top1_hit.tolist()}")
print(f"Reference 2 ({tags[REF2_IDX]}) top-10 hit-counts: {top2_hit.tolist()}")
print(f"Reference 3 ({tags[REF3_IDX]}) top-10 hit-counts: {top3_hit.tolist()}")
print(f"Cross-reference consistency (fraction of one reference's top-10 physical locations "
      f"within {MATCH_THRESHOLD_M:.0f}m of another reference's top-10 locations):")
print(f"  ref1->ref2: {frac_1to2*100:.0f}%   ref1->ref3: {frac_1to3*100:.0f}%   ref2->ref3: {frac_2to3*100:.0f}%")

results = dict(
    layout_tags=tags,
    reference_layout=tags[REF_IDX],
    full_Us=full_Us.tolist(),
    per_slot=dict(
        dip_mean=slot_dip_mean.tolist(),
        dip_std=slot_dip_std.tolist(),
        dip_min=slot_dip_min.tolist(),
        dip_max=slot_dip_max.tolist(),
        critical_hit_rate=critical_hit_rate.tolist(),
        redundant_hit_rate=redundant_hit_rate.tolist(),
        spatial_spread_mean_m=slot_dist_mean.tolist(),
        spatial_spread_max_m=slot_dist_max.tolist(),
    ),
    own_critical_native=own_critical.tolist(),
    own_redundant_native=own_redundant.tolist(),
    permutation_null=dict(
        n_perm=N_PERM,
        obs_max_hit=obs_max_hit, null_max_hit_mean=float(null_max_hit.mean()),
        null_max_hit_std=float(null_max_hit.std()), p_max_hit=p_max_hit,
        obs_never_hit=obs_never_hit, null_never_hit_mean=float(null_never_hit.mean()),
        null_never_hit_std=float(null_never_hit.std()), p_never_hit_low=p_never_hit_low,
        obs_hit_ge3=obs_hit_ge3, null_hit_ge3_mean=float(null_hit_ge3.mean()),
        null_hit_ge3_std=float(null_hit_ge3.std()), p_hit_ge3=p_hit_ge3,
    ),
    alignment_free_clustering=dict(
        n_bootstrap=N_BOOTSTRAP,
        obs_critical_nn_mean_m=float(obs_critical_nn.mean()),
        obs_redundant_nn_mean_m=float(obs_redundant_nn.mean()),
        null_mean_m=float(null_mean), null_std_m=float(null_std),
        z_critical=float(z_critical), z_redundant=float(z_redundant),
        percentile_critical=pctl_critical, percentile_redundant=pctl_redundant,
    ),
    multi_reference_check=dict(
        ref2=tags[REF2_IDX], ref3=tags[REF3_IDX],
        match_threshold_m=MATCH_THRESHOLD_M,
        frac_ref1_to_ref2=frac_1to2, frac_ref1_to_ref3=frac_1to3, frac_ref2_to_ref3=frac_2to3,
    ),
)
out_json = os.path.join(OUT_DIR, "mcr_consolidation_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_json}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(critical_hit_rate, bins=np.arange(-0.5, 13.5, 1), color="tab:red",
                 alpha=0.7, label="critical hit-rate")
    axes[0].hist(redundant_hit_rate, bins=np.arange(-0.5, 13.5, 1), color="tab:blue",
                 alpha=0.5, label="redundant hit-rate")
    axes[0].set_xlabel("hit-count out of 12 layouts")
    axes[0].set_ylabel("number of reference slots")
    axes[0].set_title("Cross-layout hit-rate distribution (100 reference slots)")
    axes[0].legend()

    sc = axes[1].scatter(critical_hit_rate, slot_dip_mean, c=slot_dist_mean,
                          cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    axes[1].set_xlabel("critical hit-count (out of 12)")
    axes[1].set_ylabel("mean leave-one-out dip across 12 layouts")
    axes[1].set_title("Mean dip vs. cross-layout agreement")
    cbar = fig.colorbar(sc, ax=axes[1])
    cbar.set_label("mean spatial spread from reference (m)")

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "mcr_consolidation_summary.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    axes2[0].hist(null_max_hit, bins=np.arange(-0.5, 13.5, 1), color="gray", alpha=0.7,
                  label="null (label-shuffle)")
    axes2[0].axvline(obs_max_hit, color="tab:red", linewidth=2, label=f"observed ({obs_max_hit})")
    axes2[0].set_xlabel("max hit-rate across 100 slots")
    axes2[0].set_ylabel("permutation count")
    axes2[0].set_title(f"Null distribution of max hit-rate (p={p_max_hit:.4f})")
    axes2[0].legend()

    axes2[1].hist(null_means, bins=30, color="gray", alpha=0.7, label="null (random 5-of-100)")
    axes2[1].axvline(obs_critical_nn.mean(), color="tab:red", linewidth=2,
                      label=f"observed critical ({obs_critical_nn.mean():.0f}m)")
    axes2[1].axvline(obs_redundant_nn.mean(), color="tab:blue", linewidth=2,
                      label=f"observed redundant ({obs_redundant_nn.mean():.0f}m)")
    axes2[1].set_xlabel("mean nearest cross-layout distance (m)")
    axes2[1].set_ylabel("bootstrap count")
    axes2[1].set_title("Alignment-free clustering: observed vs. null")
    axes2[1].legend()

    fig2.tight_layout()
    out_png2 = os.path.join(OUT_DIR, "mcr_consolidation_null_tests.png")
    fig2.savefig(out_png2, dpi=150)
    print(f"[plot] wrote {out_png2}")
except Exception as exc:
    print(f"[plot] skipped ({exc!r})")
