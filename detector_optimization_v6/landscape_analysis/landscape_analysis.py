#!/usr/bin/env python3
"""
Landscape analysis for TAMBO v6 detector optimization.
Q1: Random baseline gap
Q2: GES vs L-BFGS layout agreement (Hungarian alignment)
Q3: Batch sensitivity of U
Q4: Theoretical ceiling from aleatoric floor
Q5: Gradient norm at GES optimum
Q6: Landscape width along random directions
"""
import sys, os, json, time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import modules_v6  # injects v3/v4 paths

from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.opt_core import utility_of_xy, load_models, W_THETA, W_PHI, W_E, W_DIV
from modules_v6.detector_strategies_ne import layout_uniform_random

DEVICE = torch.device("cpu")
RUN_BASE = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs"
SEED = 42
BATCH_SIZE = 512
PYTHON = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/envs/tambo/bin/python3"

print("="*70)
print("TAMBO v6 Landscape Analysis")
print("="*70)
print(f"Device: {DEVICE}")

# ── Load models
print("\nLoading models...")
t0 = time.time()
fnn, recon = load_models(DEVICE,
    fnn_folder=FNN_FOLDER,
    recon_dir=RECON_FOLDER + "_deepsets")
print(f"Models loaded in {time.time()-t0:.1f}s")

# ── Load mountain
print("Loading mountain geometry...")
mountain = load_tr_mountain(GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)

# ── Load primaries
print("Loading primaries...")
primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()
n_total = primary_all.shape[0]
print(f"  {n_total} primaries loaded")

# ── Reproduce GES fixed batch (seed=42)
g = torch.Generator().manual_seed(SEED)
idx_fixed = torch.randint(0, n_total, (BATCH_SIZE,), generator=g)
primary_fixed = primary_all[idx_fixed].to(DEVICE)

# ── Load best layouts
def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])

ges_x, ges_y, ges_U_saved = load_layout(
    f"{RUN_BASE}/test_v6_run_04_optimize_ges_combined/layout_best.pt")
lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(
    f"{RUN_BASE}/test_v6_run_04_optimize_lbfgs_ensemble_ds_combined/layout_best.pt")
print(f"GES best U (saved): {ges_U_saved:.4f}")
print(f"L-BFGS DS best U (saved): {lbfgs_U_saved:.4f}")

# ── Evaluation helper
@torch.no_grad()
def eval_U(x, y, primary=None):
    if primary is None:
        primary = primary_fixed
    U, r, comps = utility_of_xy(x.to(DEVICE), y.to(DEVICE), primary, fnn, recon)
    return float(U.item()), r, {k: float(v.item()) for k, v in comps.items()}

# ══════════════════════════════════════════════════════════════════════════════
# Q1: RANDOM BASELINES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q1: Random Baselines (8 uniformly random layouts)")
print("="*70)
random_Us = []
for i in range(8):
    rx, ry = layout_uniform_random(mountain, rng=np.random.default_rng(i * 1337 + 42))
    u, r_mean, comps = eval_U(rx.to(DEVICE), ry.to(DEVICE))
    random_Us.append(u)
    print(f"  random_{i:02d}: U={u:7.3f}  mean_r={float(r_mean.mean()):.4f}  "
          f"u_e={comps['u_e']:.3f}  u_theta={comps['u_theta']:.3f}  u_phi={comps['u_phi']:.3f}")

mu_rand = np.mean(random_Us)
sd_rand = np.std(random_Us)
print(f"\nRandom U:     mean={mu_rand:.3f}  std={sd_rand:.3f}  "
      f"range=[{min(random_Us):.2f}, {max(random_Us):.2f}]")
print(f"Optimized U:  {ges_U_saved:.3f}")
print(f"Gap:          {ges_U_saved - mu_rand:.3f}  ({(ges_U_saved - mu_rand) / abs(mu_rand) * 100:.1f}% above random mean)")

# ══════════════════════════════════════════════════════════════════════════════
# Q2: OPTIMIZER AGREEMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q2: GES vs L-BFGS DS Layout Agreement (Hungarian alignment)")
print("="*70)

# Verify U on the same fixed batch
ges_U_reeval, ges_r, ges_comps = eval_U(ges_x, ges_y)
lbfgs_U_reeval, lbfgs_r, lbfgs_comps = eval_U(lbfgs_x, lbfgs_y)
print(f"GES U re-evaluated on fixed batch:   {ges_U_reeval:.4f}")
print(f"L-BFGS DS U re-evaluated:            {lbfgs_U_reeval:.4f}")

ges_xy   = np.stack([ges_x.numpy(),   ges_y.numpy()],   axis=-1)   # (100, 2)
lbfgs_xy = np.stack([lbfgs_x.numpy(), lbfgs_y.numpy()], axis=-1)  # (100, 2)

diff = ges_xy[:, None, :] - lbfgs_xy[None, :, :]  # (100, 100, 2)
cost = (diff**2).sum(-1)                            # (100, 100)
row_ind, col_ind = linear_sum_assignment(cost)
displacement = np.sqrt(((ges_xy[row_ind] - lbfgs_xy[col_ind])**2).sum(-1))

print(f"\nHungarian alignment displacements:")
print(f"  mean   = {displacement.mean():.1f} m")
print(f"  median = {np.median(displacement):.1f} m")
print(f"  std    = {displacement.std():.1f} m")
print(f"  max    = {displacement.max():.1f} m")
print(f"  p25/p75 = {np.percentile(displacement, 25):.1f} / {np.percentile(displacement, 75):.1f} m")

if displacement.mean() < 50:
    verdict = "SAME basin (mean < 50m) — optimizers found essentially identical layout"
elif displacement.mean() < 200:
    verdict = "LIKELY SAME basin (50-200m) — minor positional spread, same physical region"
else:
    verdict = "DIFFERENT basins (> 200m) — genuinely distinct local optima with similar U"
print(f"\nVerdict: {verdict}")

# ══════════════════════════════════════════════════════════════════════════════
# Q3: BATCH SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q3: Batch Sensitivity (5 independent 512-primary batches, GES best layout)")
print("="*70)
batch_Us = []
for seed in [1, 2, 3, 4, 5]:
    g2 = torch.Generator().manual_seed(seed)
    idx2 = torch.randint(0, n_total, (BATCH_SIZE,), generator=g2)
    prim2 = primary_all[idx2].to(DEVICE)
    u, _, _ = eval_U(ges_x, ges_y, prim2)
    batch_Us.append(u)
    print(f"  seed={seed}: U={u:.4f}")

# Also evaluate fixed batch for reference
u_fixed, _, _ = eval_U(ges_x, ges_y)
print(f"  seed=42 (GES fixed): U={u_fixed:.4f}")

all_Us = batch_Us + [u_fixed]
print(f"\nAll 6 U values: mean={np.mean(all_Us):.4f}  std={np.std(all_Us):.4f}")
print(f"Batch std (5 fresh): {np.std(batch_Us):.4f}")
print(f"GES vs L-BFGS gap: {ges_U_saved - lbfgs_U_saved:.4f}")
if np.std(batch_Us) > 1.0:
    print("=> Std > 1.0: the 1.3pt GES vs L-BFGS gap IS within batch noise")
else:
    print("=> Std < 1.0: the GES vs L-BFGS gap MAY be real signal above noise")

# ══════════════════════════════════════════════════════════════════════════════
# Q4: THEORETICAL CEILING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q4: Theoretical Ceiling")
print("="*70)

floor_path = os.path.join(_V6, "aleatoric_floor_sigma200.json")
floor_data = json.load(open(floor_path))
floor_zmse  = floor_data["floor_zmse"]["total"]
max_r2      = floor_data["max_R2"]["total"]

# Actual mean_r at GES optimum
_, ges_r_val, ges_comps = eval_U(ges_x, ges_y)
mean_r = float(ges_r_val.mean().item())

# Perfect-reconstruction ceiling: zero angle error, zero log-E error
# U_E_ceil  = mean(r / (0^2 + 0.01)) = mean_r / 0.01 = 100 * mean_r
# U_ang_ceil = mean(r / (0^2 + 0.001)) = mean_r / 0.001 = 1000 * mean_r
u_e_ceil   = mean_r * 100.0
u_ang_ceil = mean_r * 1000.0
U_perfect  = (W_THETA * u_ang_ceil + W_PHI * u_ang_ceil + W_E * u_e_ceil) / W_DIV

print(f"GES best:     U={ges_U_reeval:.3f}  mean_r={mean_r:.4f}")
print(f"  components: u_theta={ges_comps['u_theta']:.4f}  u_phi={ges_comps['u_phi']:.4f}  u_e={ges_comps['u_e']:.4f}")
print()
print(f"Perfect-recon ceiling (mean_r={mean_r:.4f}):")
print(f"  u_ang_ceil = mean_r / 0.001 = {u_ang_ceil:.1f}")
print(f"  u_e_ceil   = mean_r / 0.01  = {u_e_ceil:.1f}")
print(f"  U_perfect  = ({W_THETA}*{u_ang_ceil:.1f} + {W_PHI}*{u_ang_ceil:.1f} + {W_E}*{u_e_ceil:.1f}) / {W_DIV:.0f}")
print(f"             = {U_perfect:.1f}")
print()
print(f"Current/Ceiling: {ges_U_reeval:.1f} / {U_perfect:.1f} = {ges_U_reeval/U_perfect*100:.1f}%")
print(f"Gap to ceiling:  {U_perfect - ges_U_reeval:.1f}")
print()
print(f"FNN aleatoric floor z-MSE:  {floor_zmse:.4f}")
print(f"Max achievable FNN R²:      {max_r2:.4f}")
print("Note: FNN floor limits per-detector signal quality; the recon ceiling")
print("      above assumes perfect reconstruction given any detector signal.")

# ══════════════════════════════════════════════════════════════════════════════
# Q5: GRADIENT NORM AT OPTIMUM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q5: Gradient Norm at GES Best Layout")
print("="*70)

x_dev = ges_x.clone().to(DEVICE).requires_grad_(True)
y_dev = ges_y.clone().to(DEVICE).requires_grad_(True)
U_val, _, _ = utility_of_xy(x_dev, y_dev, primary_fixed, fnn, recon)
U_val.backward()
gx = x_dev.grad.detach()
gy = y_dev.grad.detach()
g_full = torch.cat([gx, gy])

print(f"U at GES best: {U_val.item():.4f}")
print(f"||grad||_2:    {g_full.norm().item():.6f}")
print(f"||grad||_inf:  {g_full.abs().max().item():.6f}")
print(f"Mean |grad|:   {g_full.abs().mean().item():.6f}")
print(f"Grad-x stats:  min={gx.min().item():.5f}  max={gx.max().item():.5f}  "
      f"mean={gx.mean().item():.5f}")
print(f"Grad-y stats:  min={gy.min().item():.5f}  max={gy.max().item():.5f}  "
      f"mean={gy.mean().item():.5f}")

# For comparison, also compute gradient at a random layout
rx_rand, ry_rand = layout_uniform_random(mountain, rng=np.random.default_rng(0))
xr_dev = rx_rand.clone().to(DEVICE).requires_grad_(True)
yr_dev = ry_rand.clone().to(DEVICE).requires_grad_(True)
U_rand_val, _, _ = utility_of_xy(xr_dev, yr_dev, primary_fixed, fnn, recon)
U_rand_val.backward()
g_rand = torch.cat([xr_dev.grad.detach(), yr_dev.grad.detach()])
print(f"\nFor comparison — gradient at a random layout:")
print(f"  U={U_rand_val.item():.4f}  ||grad||_2={g_rand.norm().item():.6f}")
print(f"  Optimum/random grad ratio: {g_full.norm().item()/g_rand.norm().item():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Q6: LANDSCAPE WIDTH
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Q6: Landscape Width — U along random directions from GES best")
print("="*70)

DISTANCES = [10, 50, 100, 200, 500]
N_DIRS = 5
D = 2 * N_DETECTORS

base_U, _, _ = eval_U(ges_x, ges_y)
print(f"Base U at GES best: {base_U:.4f}")
print()

torch.manual_seed(12345)
dirs = [torch.randn(D) for _ in range(N_DIRS)]
dirs = [d / d.norm() for d in dirs]

results_width = {}
for dist in DISTANCES:
    Us_at_dist = []
    for d_vec in dirs:
        x_new = ges_x + d_vec[:N_DETECTORS] * dist
        y_new = ges_y + d_vec[N_DETECTORS:] * dist
        xp, yp = project_to_mountain_ne(mountain, x_new.cpu(), y_new.cpu())
        with torch.no_grad():
            u_dir, _, _ = utility_of_xy(xp.to(DEVICE), yp.to(DEVICE),
                                         primary_fixed, fnn, recon)
        Us_at_dist.append(float(u_dir.item()))
    results_width[dist] = Us_at_dist
    mean_u = np.mean(Us_at_dist)
    drop   = base_U - mean_u
    vals_str = "  ".join(f"{u:.2f}" for u in Us_at_dist)
    print(f"  d={dist:4d}m:  mean U={mean_u:.3f}  drop={drop:+.3f}  "
          f"[{vals_str}]")

print()
print("Landscape summary:")
for dist in DISTANCES:
    mean_u = np.mean(results_width[dist])
    frac_drop = (base_U - mean_u) / abs(base_U) * 100
    print(f"  d={dist:4d}m:  U drops by {base_U - mean_u:.3f} ({frac_drop:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Q1 Random gap:       {ges_U_saved - mu_rand:.1f} pts  (random mean={mu_rand:.1f}, optimized={ges_U_saved:.1f})")
print(f"Q2 Optimizer agreement: mean displacement={displacement.mean():.1f}m — {verdict}")
batch_sd = np.std(batch_Us)
print(f"Q3 Batch std:        {batch_sd:.3f}  (GES vs LBFGS gap={ges_U_saved - lbfgs_U_saved:.3f})")
print(f"Q4 Ceiling:          U_perfect={U_perfect:.1f}, current={ges_U_reeval:.1f} ({ges_U_reeval/U_perfect*100:.1f}%)")
print(f"Q5 Gradient norm:    ||g||={g_full.norm().item():.4f}  (random layout: {g_rand.norm().item():.4f})")
print(f"Q6 Landscape width:  U at 200m drops by {base_U - np.mean(results_width[200]):.3f} pts")
print()
print("Conclusions:")
if ges_U_saved - mu_rand > 20:
    print("  [STRUCTURED] Large gap from random => surrogate has real landscape structure")
else:
    print("  [FLAT?] Small gap from random => landscape may be genuinely flat")

if displacement.mean() < 200:
    print("  [SAME BASIN] Optimizers converged to same physical region")
else:
    print("  [MULTI-MODAL] Different basins with similar U => broad plateau")

if batch_sd > 1.0:
    print("  [NOISE LIMITED] Batch std > 1.0 => optimizer comparisons unreliable at this batch size")
else:
    print("  [STABLE] Batch std < 1.0 => U estimates are reliable")

frac_of_ceil = ges_U_reeval / U_perfect
if frac_of_ceil > 0.85:
    print(f"  [NEAR CEILING] {frac_of_ceil*100:.1f}% of perfect-recon ceiling => surrogate is the bottleneck")
elif frac_of_ceil > 0.70:
    print(f"  [MODERATE GAP] {frac_of_ceil*100:.1f}% of ceiling => some surrogate/layout optimization room")
else:
    print(f"  [FAR FROM CEILING] {frac_of_ceil*100:.1f}% => significant room for improvement")
