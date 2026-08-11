"""Optimize detector positions: pre-Adam perturbation, then L-BFGS ensemble.

Frequentist sibling of ``04_optimize_hmc_chains.py``. Instead of sampling a
posterior with NUTS, stage 2 runs **L-BFGS to a local optimum from each of the
K perturbed Adam warm-starts**, then summarizes the ensemble of K optimized
layouts with a per-position mean and std.

Per scheme:

1.  Sample the scheme's initial layout (`mountain.sample_initial_layout`) and
    create K = `N_CHAINS` Gaussian perturbations of it (std
    `INIT_OVERDISP_SIGMA`, projected back to the mountain).
2.  Run Adam (`N_ADAM_EPOCHS`) independently from each perturbed start → K
    Adam-best layouts.
3.  Run L-BFGS (`LBFGS_MAX_ITER`) from each Adam-best on a FIXED primary batch
    (deterministic objective for the line search) → K refined layouts.
4.  **Align** the K refined layouts so each output group corresponds to the
    same *physical position*, not the same detector index. Because the FNN /
    recon are permutation-equivariant, detector index i is not the same unit
    across runs — so we match each run's detectors to a reference layout by
    closest position (Hungarian / `linear_sum_assignment`). This makes the
    grouping network-input invariant.
5.  Per aligned group: **mean and std** of (x, y) across the K runs.

The "combined" run pools the K Adam-bests from every scheme, refines all of
them with L-BFGS, and aligns the full K * len(INIT_SCHEMES) ensemble.

Artifacts (per scheme + "combined") land in
``<OPT_FOLDER>_lbfgs_ensemble_{scheme}/``:

    layout_best.pt          highest-U L-BFGS layout (mountain-projected)
    layout_mean.pt          per-group mean position + std (aligned ensemble)
    layouts_all.pt          aligned (K, n_det, 2) + per-run U + source + perm
    optimize_log.json       Adam + L-BFGS logs + ensemble stats + config
    optimize_curves.png     all Adam chains U + all L-BFGS refinements U
    layout_ensemble.png     mountain top-down: ensemble points + mean + 1σ ellipses

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 04_optimize_lbfgs_ensemble.py
"""
import importlib.util
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.dual_surrogate import DualSpeciesSurrogate
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules.layout_optimization import LearnableXY
from modules_v4.tr_geometry      import load_tr_mountain
from modules_v6.tr_geometry_ne   import project_to_mountain_ne, sample_initial_layout_ne
from modules_v6.tr_surface_map_ne import SurfaceUpMap

# Shared optimizer core (objective, alignment, model loading, the gradient-turn
# diagnostic, constants) lives in modules_v6/opt_core.py; the figures live in
# plots/opt_plotting.py (loaded by path). utility_of_xy is NOT no_grad-wrapped,
# so Adam / L-BFGS backprop through it here.
from modules_v6.opt_core import (
    utility_of_xy, align_to_reference, consecutive_cos_distance, load_models,
    W_THETA, W_PHI, W_E, W_PR, W_DIV,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD, OFFMESH_PENALTY_W,
    PENALTY_ONSET_FRAC,
)
from modules_v6.tr_geometry_ne import _ne_max_gap
_plt_spec = importlib.util.spec_from_file_location(
    "opt_plotting", os.path.join(_HERE, "plots", "opt_plotting.py"))
_plt = importlib.util.module_from_spec(_plt_spec); _plt_spec.loader.exec_module(_plt)


# ── Config ───────────────────────────────────────────────────────────────────
INIT_SCHEMES         = ("grid", "center")
RUN_COMBINED         = False # True
COMBINED_SCHEME_NAME = "combined"
OPT_DIR_TEMPLATE     = OPT_FOLDER + "_lbfgs_ensemble_full_corpus_{scheme}"
# Recon dir to load (DeepSets recon from 03_train_recon_deepsets.py). Overridable
# with --recon_folder (exact path). utility_of_xy feeds recon (B, n_det, 4).
RECON_DIR            = RECON_FOLDER + "_deepsets"

# K perturbed restarts per scheme.
N_CHAINS            = 8
INIT_OVERDISP_SIGMA = 100.0  # metres — per-restart init spread around scheme init

# Adam warm-start
N_ADAM_EPOCHS       = 5_000
PRIMARIES_PER_STEP  = 5_000
ADAM_LR             = 1.0
GRAD_CLIP           = 100.0
ADAM_LOG_EVERY      = 100

# Gradient-direction diagnostic: window (in steps) for vector-averaging the raw
# gradients before the consecutive-step cosine distance. Averaging the gradient
# VECTORS over W steps cancels zero-mean minibatch noise before the (nonlinear)
# cosine, removing the noise-inflation bias instead of merely smoothing it.
# 1 = no averaging (raw, noisy).
GRAD_COS_WINDOW     = 10

# Detector-position trajectory logging. The optimizer used to persist only
# scalar U per step, so a layout animation had to be reconstructed by replaying
# Adam from the stored gradients. Recording (East, North) directly makes the
# path a first-class artifact: `trajectory.pt` per scheme, plus the same arrays
# inside the resume checkpoints so a preemption doesn't lose them.
#
# 1 = every step. The stride used to be 5 to keep the file small (a chain is
# 1000 frames x 200 floats x 4 B = 800 KB at stride 5, ~12 MB per scheme for
# K=15, next to a ~66 MB adam_resume.pt), but it is the binding limit on how
# smooth the layout animation can be: at stride 5 the L-BFGS line search moves
# the layout 402 m between CONSECUTIVE logged frames at p90, so the GIF
# teleports no matter how it is resampled. At stride 1 a single chain costs
# 5001 Adam frames + ~14.5k L-BFGS closure frames x 200 floats x 4 B = ~16 MB
# per scheme, which is nothing. Raise it again if K goes back up.
POS_LOG_EVERY = 1

# Off-mesh penalty weight, in units of U (0 = off, restoring the old behaviour
# where nothing in the objective knew the mountain existed). See
# opt_core.offmesh_penalty for why: L-BFGS is projected only ONCE per sweep
# chunk, so it spent 66% of its closures with detectors outside the 91.3 m snap
# radius, optimising a surrogate that has no training data out there — and one
# center run returned U=+16.18 with 28 detectors stacked on a single centroid
# after the end-of-chunk snap. The penalty is 0 inside the snap radius, so an
# in-band layout scores exactly as it did before.
OFFMESH_PENALTY = OFFMESH_PENALTY_W

_MESH_CACHE = {}


def _penalty_args(mountain):
    """(mesh_en on DEVICE, r0) for utility_of_xy, built once per mountain.

    r0 is PENALTY_ONSET_FRAC of the max_gap the projection uses — the wall
    starts INSIDE the snap radius, so the equilibrium it creates sits in-band
    rather than just outside it (see opt_core.PENALTY_ONSET_FRAC)."""
    key = id(mountain)
    if key not in _MESH_CACHE:
        cen = torch.as_tensor(mountain.centroids_ENU[:, :2], dtype=torch.float32)
        _MESH_CACHE[key] = (cen.to(DEVICE),
                            float(_ne_max_gap(mountain)) * PENALTY_ONSET_FRAC)
    return _MESH_CACHE[key]

# L-BFGS refinement (stage 2)
LBFGS_MAX_ITER       = 1_500
LBFGS_LR             = 1.0
LBFGS_HISTORY_SIZE   = 20
# LBFGS_BATCH_PRIMARIES = 15000
# LBFGS_BATCH_PRIMARIES = 8000
LBFGS_BATCH_PRIMARIES = 6000

# Cap on how many corpus primaries the L-BFGS sweep covers (0 = the whole
# corpus). The sweep visits ceil(n_primaries / LBFGS_BATCH_PRIMARIES) chunks per
# chain, so its cost is LINEAR in corpus size: 67 chunks at the 1M-row corpus
# (measured 6.35 h/scheme) becomes 477 at 7.14M rows (~45 h/scheme). That growth
# buys nothing statistically — the layout is only 2*N_DETECTORS = 200 free
# parameters, and the corpus was enlarged for the SURROGATE (Steps 2/3), which
# still trains on every row. A fixed seeded subsample keeps the objective's
# per-chunk determinism (what the line search needs) while holding the sweep at
# the measured cost.
LBFGS_MAX_SWEEP_PRIMARIES = 1_000_000

# Fixed scoring set: chunk-end candidates are re-scored here before any of them
# is called "best". Held out of the sweep pool, so it is the one objective every
# chunk's result is comparable on — the per-chunk U is not, each chunk having
# its own batch. Evaluated in SCORING_BATCH_PRIMARIES slices and combined as a
# size-weighted mean, which is exact: every term in U is a per-event mean, and
# the layout-only off-mesh penalty is identical in every slice.
SCORING_SET_PRIMARIES   = 20_000
SCORING_BATCH_PRIMARIES = 4_000

# Tolerance band for carrying a chunk's positions forward. A chunk that beats
# the best score is always accepted and becomes the new best; a chunk that lands
# within this fraction of it is still carried forward (the sweep keeps exploring
# from there) but does NOT become the best; anything worse is rolled back to the
# best-known layout. Strict improvement-only acceptance would throw away a chunk
# that found a better region but scored fractionally worse through scoring-set
# noise, and restart from the same point every time.
SCORING_ACCEPT_TOL = 0.05

# Composite weights (W_*) + reconstructability thresholds are imported from
# modules_v6/opt_core.py (shared across the 04 optimizers).
SEED   = 4200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Density heatmap colorbar upper limit (plots [0, DENSITY_VMAX]); keeps faint
# structure from being washed out by a few hot cells. <=0 → auto-scale.
DENSITY_VMAX = 0.2


def adam_warm_start(scheme: str,
                    mountain,
                    fnn: DualSpeciesSurrogate,
                    recon: torch.nn.Module,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    init_override):
    """N_ADAM_EPOCHS of Adam with mountain projection. Returns:
       (best_x, best_y, init_x, init_y, log, grad_hist, pos_hist).
       `init_override=(x, y)` is the (already mountain-projected) starting
       layout; `scheme` is a log label. `grad_hist` is a (N_ADAM_EPOCHS,
       2*n_det) CPU tensor of the flat parameter gradient at each step (for
       cross-run gradient diagnostics). `pos_hist` is a (n_frames, 2*n_det) CPU
       tensor of the flat (East|North) layout recorded every POS_LOG_EVERY
       steps AFTER the mountain projection — i.e. the positions the next step
       actually starts from — with the initial layout as frame 0."""
    e_init, n_init = init_override          # (East, North): first coord is East
    e_init = e_init.float()
    n_init = n_init.float()
    print(f"[adam] init {scheme}  E in [{e_init.min():.1f}, {e_init.max():.1f}]  "
          f"N in [{n_init.min():.1f}, {n_init.max():.1f}]")

    xy_module = LearnableXY(e_init, n_init, device=str(DEVICE)).to(DEVICE)
    optimizer = torch.optim.Adam(xy_module.parameters(), lr=ADAM_LR)

    log = []
    grad_hist = []
    # Frame 0 = the starting layout, so the trajectory is self-contained (no
    # need to pair it with perturbed_inits to know where it began).
    pos_hist = [torch.cat([e_init.reshape(-1), n_init.reshape(-1)]).clone()]
    best_u = -float("inf")
    best_x = e_init.clone()
    best_y = n_init.clone()

    for epoch in range(N_ADAM_EPOCHS):
        idx = torch.randint(0, n_total_primaries, (PRIMARIES_PER_STEP,))
        primary_batch = primary_all[idx].to(DEVICE)

        x_det, y_det = xy_module()
        _mesh, _r0 = _penalty_args(mountain)
        U, r, parts = utility_of_xy(x_det, y_det, primary_batch, fnn, recon,
                                    mesh_en=_mesh, penalty_w=OFFMESH_PENALTY,
                                    penalty_r0=_r0)
        loss = -U

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Flat gradient (x then y) before clipping — cosine is scale-invariant,
        # so clipping (uniform rescale) wouldn't change it anyway.
        grad_hist.append(
            torch.cat([xy_module.x.grad.detach().reshape(-1),
                       xy_module.y.grad.detach().reshape(-1)]).cpu()
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        # Project to mountain surface.
        with torch.no_grad():
            e_cpu = xy_module.x.detach().cpu()
            n_cpu = xy_module.y.detach().cpu()
            e_new, n_new = project_to_mountain_ne(mountain, e_cpu, n_cpu)
            xy_module.x.data.copy_(e_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(n_new.to(DEVICE).to(xy_module.y.dtype))
            # Post-projection layout: what the next step starts from. Always
            # record the last step too, so pos_hist ends at the final layout.
            if (epoch + 1) % POS_LOG_EVERY == 0 or epoch == N_ADAM_EPOCHS - 1:
                pos_hist.append(
                    torch.cat([e_new.reshape(-1), n_new.reshape(-1)]).float().clone())

        u_val = float(U.item())
        if u_val > best_u:
            best_u = u_val
            best_x = xy_module.x.detach().cpu().clone()
            best_y = xy_module.y.detach().cpu().clone()

        log.append(dict(
            epoch=epoch + 1, U=u_val, r_mean=float(r.mean().item()),
            u_theta=float(parts["u_theta"].item()),
            u_phi=float(parts["u_phi"].item()),
            u_e=float(parts["u_e"].item()),
            u_pr=float(parts["u_pr"].item()),
        ))
        if epoch == 0 or (epoch + 1) % ADAM_LOG_EVERY == 0 or epoch == N_ADAM_EPOCHS - 1:
            print(f"  [adam {epoch+1:4d}/{N_ADAM_EPOCHS}] U={u_val:+.3f}")

    print(f"[adam] best U={best_u:+.3f}")
    grad_hist = torch.stack(grad_hist, dim=0) if grad_hist else torch.zeros(0)
    pos_hist  = torch.stack(pos_hist,  dim=0) if pos_hist  else torch.zeros(0)
    return best_x, best_y, e_init, n_init, log, grad_hist, pos_hist


def _build_chain_inits(init_x: torch.Tensor, init_y: torch.Tensor,
                       K: int, generator: torch.Generator) -> torch.Tensor:
    """K overdispersed starts around (init_x, init_y). Returns (K, 2*n_det) on DEVICE."""
    base = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach()  # (D,)
    perturb = torch.randn(
        K, base.numel(), generator=generator, device="cpu",
    ).to(DEVICE) * INIT_OVERDISP_SIGMA
    return base.unsqueeze(0) + perturb                                        # (K, D)


def _perturbed_adam_runs(scheme: str, K: int, generator: torch.Generator,
                         mountain, fnn, recon, primary_all, n_total_primaries,
                         init_center=None, resume_path=None):
    """K pre-Adam perturbations of the scheme init → K Adam runs.

    Returns (adam_bests, adam_logs, perturbed_inits, adam_grads, adam_positions),
    each length K. adam_grads[k] is the (N_ADAM_EPOCHS, 2*n_det) per-step
    gradient history; adam_positions[k] is the (n_frames, 2*n_det) layout
    trajectory (see POS_LOG_EVERY).

    If init_center is a (N_DETECTORS, 2) tensor with (East, North) per detector,
    all K chains are warm-started from that layout with small N(0, 10m) per-chain
    diversity perturbations instead of using the normal scheme initialization.

    If `resume_path` is given, progress is checkpointed after every completed
    chain (gpu_requeue can preempt mid-ensemble — losing 14/15 already-finished
    5000-epoch chains to redo just the last one would be the expensive
    mistake). The file is intentionally NEVER deleted on success: it doubles as
    the persisted per-scheme Adam-chain cache that a "combined" run (or a
    resumed main() that finds this scheme's layout_best.pt already on disk)
    reads instead of recomputing.
    """
    if resume_path and os.path.exists(resume_path):
        _ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        adam_bests, adam_logs = _ck["adam_bests"], _ck["adam_logs"]
        perturbed_inits, adam_grads = _ck["perturbed_inits"], _ck["adam_grads"]
        # Older checkpoints predate position logging; treat them as empty so a
        # resumed run still completes (those chains just have no trajectory).
        adam_positions = _ck.get("adam_positions", [])
        if len(adam_positions) < len(adam_bests):
            adam_positions = adam_positions + [torch.zeros(0)] * (
                len(adam_bests) - len(adam_positions))
            print(f"[resume] {resume_path}: checkpoint predates position logging — "
                  f"{len(adam_bests)} chain(s) will have no trajectory")
        if len(adam_bests) >= K:
            print(f"[resume] {resume_path}: all {K} chains already done "
                  "(reusing cached Adam results, not recomputing)")
            return (adam_bests[:K], adam_logs[:K], perturbed_inits[:K],
                    adam_grads[:K], adam_positions[:K])
        print(f"[resume] {resume_path}: {len(adam_bests)}/{K} chains already done")
    else:
        adam_bests, adam_logs, perturbed_inits, adam_grads = [], [], [], []
        adam_positions = []
    k_start = len(adam_bests)

    if init_center is not None:
        e_t = init_center[:, 0].float()
        n_t = init_center[:, 1].float()
        # Small N(0, 10m) per-chain perturbations for diversity around the warm-start.
        base = torch.cat([e_t.to(DEVICE), n_t.to(DEVICE)], dim=0).detach()
        small_noise = torch.randn(K, base.numel(), generator=generator,
                                  device="cpu").to(DEVICE) * 10.0
        chains_init = base.unsqueeze(0) + small_noise
    else:
        e_np, n_np = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
        e_t = torch.as_tensor(e_np, dtype=torch.float32)
        n_t = torch.as_tensor(n_np, dtype=torch.float32)
        chains_init = _build_chain_inits(e_t, n_t, K, generator)              # (K, D)

    for k in range(K):
        if k < k_start:
            continue
        xk = chains_init[k, :N_DETECTORS].cpu()
        yk = chains_init[k, N_DETECTORS:].cpu()
        xk, yk = project_to_mountain_ne(mountain, xk, yk)
        perturbed_inits.append((xk.float().clone(), yk.float().clone()))
        print(f"\n[perturb→adam] scheme={scheme}  chain {k+1}/{K}")
        bx, by, _, _, log, ghist, phist = adam_warm_start(
            scheme=scheme, mountain=mountain, fnn=fnn, recon=recon,
            primary_all=primary_all, n_total_primaries=n_total_primaries,
            init_override=(xk, yk),
        )
        adam_bests.append((bx, by))
        adam_logs.append(log)
        adam_grads.append(ghist)
        adam_positions.append(phist)

        if resume_path:
            _tmp = resume_path + ".tmp"
            torch.save({"adam_bests": adam_bests, "adam_logs": adam_logs,
                       "perturbed_inits": perturbed_inits, "adam_grads": adam_grads,
                       "adam_positions": adam_positions}, _tmp)
            os.replace(_tmp, resume_path)
    return adam_bests, adam_logs, perturbed_inits, adam_grads, adam_positions


@torch.no_grad()
def score_on_set(x: torch.Tensor, y: torch.Tensor, scoring_set: torch.Tensor,
                 fnn: DualSpeciesSurrogate, recon: torch.nn.Module, mountain,
                 project: bool = True):
    """U of one layout on the fixed scoring set, in memory-safe slices.

    Projects first unless told otherwise: an unprojected layout is not one that
    can be built, so scoring it would rank a candidate by a value the snap is
    about to take away.
    """
    if project:
        x, y = project_to_mountain_ne(mountain, x.detach().cpu(), y.detach().cpu())
    x, y = x.to(DEVICE), y.to(DEVICE)
    mesh_en, pen_r0 = _penalty_args(mountain)
    step = SCORING_BATCH_PRIMARIES or scoring_set.shape[0]
    tot, n_tot = 0.0, 0
    for s in range(0, scoring_set.shape[0], step):
        batch = scoring_set[s:s + step].to(DEVICE)
        U, _, _ = utility_of_xy(x, y, batch, fnn, recon, mesh_en=mesh_en,
                                penalty_w=OFFMESH_PENALTY, penalty_r0=pen_r0)
        tot += float(U.item()) * batch.shape[0]
        n_tot += batch.shape[0]
    return (tot / max(n_tot, 1)), x.detach().cpu(), y.detach().cpu()


def lbfgs_refine(init_x: torch.Tensor,
                 init_y: torch.Tensor,
                 fnn: DualSpeciesSurrogate,
                 recon: torch.nn.Module,
                 primary_fixed: torch.Tensor,
                 mountain):
    """L-BFGS-maximize U from (init_x, init_y) on a fixed primary batch.

    Runs unconstrained (the line search needs a smooth objective), then
    projects the optimum back onto the mountain and re-scores it on the same
    fixed batch. Returns (x_proj, y_proj, U_proj, iter_log, grad_hist,
    pos_hist). grad_hist is a (n_closure_calls, 2*n_det) CPU tensor of the flat
    gradient at each closure evaluation (for cross-run gradient diagnostics);
    pos_hist is the matching (n_closure_calls, 2*n_det) layout at each
    evaluation, with the mountain-projected optimum appended as the last frame.

    Note these are CLOSURE evaluations, not accepted iterates: strong-Wolfe
    probes points it then rejects, so the path includes line-search trials and
    is not monotonic in U. It is also unprojected until that final frame (the
    optimizer runs free and projects once at the end), unlike the Adam
    trajectory which is projected every step."""
    mesh_en, pen_r0 = _penalty_args(mountain)
    xy = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0).detach().clone()
    xy.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [xy], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE, line_search_fn="strong_wolfe",
        tolerance_grad=1e-11,tolerance_change=1e-13,
    )

    iter_log = []
    grad_hist = []
    pos_hist = []

    class _NonFiniteLoss(Exception):
        pass

    def closure():
        optimizer.zero_grad()
        x_det = xy[:N_DETECTORS]
        y_det = xy[N_DETECTORS:]
        U, r, parts = utility_of_xy(x_det, y_det, primary_fixed, fnn, recon,
                                    mesh_en=mesh_en, penalty_w=OFFMESH_PENALTY,
                                    penalty_r0=pen_r0)
        loss = -U
        if not torch.isfinite(loss):
            raise _NonFiniteLoss
        loss.backward()
        grad_hist.append(xy.grad.detach().reshape(-1).cpu())   # (2*n_det,)
        # grad_hist was just appended, so len-1 is this closure's 0-based index.
        if (len(grad_hist) - 1) % POS_LOG_EVERY == 0:
            pos_hist.append(xy.detach().reshape(-1).float().cpu().clone())
        iter_log.append(dict(
            iter=len(iter_log), U=float(U.item()), r_mean=float(r.mean().item()),
            u_theta=float(parts["u_theta"].item()),
            u_phi=float(parts["u_phi"].item()),
            u_e=float(parts["u_e"].item()),
            u_pr=float(parts["u_pr"].item()),
        ))
        return loss

    try:
        optimizer.step(closure)
    except _NonFiniteLoss:
        print(f"  [lbfgs] non-finite loss after {len(iter_log)} closure calls — aborting step")

    # A diverged line search can leave NaN in xy; project_to_mountain passes
    # NaN through unsnapped, which would poison the ensemble alignment.
    # Fall back to the (already mountain-projected) Adam-best init.
    with torch.no_grad():
        if not torch.isfinite(xy).all():
            print("  [lbfgs] non-finite iterate — falling back to the Adam-best init")
            xy.data = torch.cat([init_x.to(DEVICE), init_y.to(DEVICE)], dim=0)

    # Project the optimum to the mountain and re-score on the same fixed batch.
    with torch.no_grad():
        x_cpu = xy[:N_DETECTORS].detach().cpu()
        y_cpu = xy[N_DETECTORS:].detach().cpu()
        x_proj, y_proj = project_to_mountain_ne(mountain, x_cpu, y_cpu)
        U_proj, _, _ = utility_of_xy(
            x_proj.to(DEVICE), y_proj.to(DEVICE), primary_fixed, fnn, recon,
            mesh_en=mesh_en, penalty_w=OFFMESH_PENALTY, penalty_r0=pen_r0,
        )
    # Final frame = the mountain-projected optimum actually returned, so the
    # trajectory ends where the next chunk (or the ensemble) picks up.
    pos_hist.append(torch.cat([x_proj.reshape(-1), y_proj.reshape(-1)]).float().cpu())
    grad_hist = torch.stack(grad_hist, dim=0) if grad_hist else torch.zeros(0)
    pos_hist  = torch.stack(pos_hist,  dim=0) if pos_hist  else torch.zeros(0)
    return (x_proj.float(), y_proj.float(), float(U_proj.item()),
            iter_log, grad_hist, pos_hist)


def select_best_on_common_batch(refined, score_fn):
    """Return (best_idx, scores) where scores[k] = score_fn(x_k, y_k).

    The chunk sweep leaves each chain's last-chunk utility, which is
    not comparable across chains (different data); selecting the
    reference layout requires one common batch."""
    scores = [float(score_fn(xp, yp)) for xp, yp in refined]
    return int(max(range(len(scores)), key=lambda i: scores[i])), scores


def _run_one_scheme(scheme: str,
                    mountain,
                    fnn: DualSpeciesSurrogate,
                    recon: torch.nn.Module,
                    primary_all: torch.Tensor,
                    n_total_primaries: int,
                    per_source):
    """Pre-computed Adam-bests → L-BFGS refine each → align ensemble → mean/std.

    `per_source` is {source_label: (adam_bests, adam_logs, perturbed_inits)}.
    A single entry = per-scheme run; multiple entries = the combined run."""
    opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
    os.makedirs(opt_dir, exist_ok=True)
    is_combined = len(per_source) > 1
    print("-" * 72)
    print(f"[run] scheme={scheme}"
          f"{'  (sources=' + str(list(per_source)) + ')' if is_combined else ''}  ->  {opt_dir}")

    # Flatten Adam-bests across all sources (track which source each came from).
    all_bests, all_adam_logs, all_adam_grads, source_per_run = [], [], [], []
    all_adam_pos = []
    for src, (bests, logs, _inits, agrads, apos) in per_source.items():
        for (bx, by), log, ag, ap in zip(bests, logs, agrads, apos):
            all_bests.append((bx, by))
            all_adam_logs.append(log)
            all_adam_grads.append(ag)
            all_adam_pos.append(ap)
            source_per_run.append(src)

    # Split the corpus into shuffled chunks of LBFGS_BATCH_PRIMARIES, so each
    # L-BFGS closure holds one batch while the sweep still covers the pool. The
    # same chunk order is reused for every Adam-best so all refinements stay
    # directly comparable.
    #
    # LBFGS_MAX_SWEEP_PRIMARIES caps the pool (see its definition): the sweep's
    # cost is linear in pool size and the 200-parameter layout doesn't need the
    # whole 7M-row corpus.
    #
    # SEEDED (it wasn't before): with per-chain resume, an unseeded permutation
    # would re-draw a different order — and, under the cap, a different SUBSET —
    # after a preemption, so chains refined before and after the restart would
    # be optimized against different data and the aligned ensemble would mix
    # them. Seeding makes the pool and chunk order identical across resubmits.
    g = torch.Generator().manual_seed(SEED)
    shuffled = torch.randperm(n_total_primaries, generator=g)
    n_score = min(int(SCORING_SET_PRIMARIES), n_total_primaries)
    scoring_idx, pool_idx = shuffled[:n_score], shuffled[n_score:]
    scoring_set = primary_all[scoring_idx].to(DEVICE)
    print(f"[lbfgs] scoring set={n_score} primaries (held out of the sweep), "
          f"evaluated in slices of {SCORING_BATCH_PRIMARIES}")

    n_sweep = pool_idx.numel()
    if LBFGS_MAX_SWEEP_PRIMARIES and LBFGS_MAX_SWEEP_PRIMARIES < n_sweep:
        n_sweep = int(LBFGS_MAX_SWEEP_PRIMARIES)
    perm = pool_idx[:n_sweep]
    n_chunks = math.ceil(n_sweep / LBFGS_BATCH_PRIMARIES)
    print(f"[lbfgs] sweep pool={n_sweep} of {n_total_primaries} primaries "
          f"-> {n_chunks} chunk(s) of {LBFGS_BATCH_PRIMARIES}"
          f"{'  (capped)' if n_sweep < n_total_primaries else ''}")
    primary_chunks = [
        primary_all[perm[i * LBFGS_BATCH_PRIMARIES:(i + 1) * LBFGS_BATCH_PRIMARIES]].to(DEVICE)
        for i in range(n_chunks)
    ]

    # Stage 2: L-BFGS refine every Adam-best, sweeping sequentially over ALL
    # chunks (i.e. the whole corpus) — several L-BFGS runs chained per Adam-best.
    #
    # gpu_requeue can preempt mid-ensemble; lbfgs_resume.pt checkpoints the
    # chains refined so far (after each FULL chain, not each chunk — a chunk is
    # cheap enough relative to a 15-chain ensemble that per-chain granularity is
    # the right cost/robustness trade), so a restart skips straight to the
    # first unrefined chain instead of redoing the whole ensemble.
    lbfgs_resume_path = os.path.join(opt_dir, "lbfgs_resume.pt")
    if os.path.exists(lbfgs_resume_path):
        _ck = torch.load(lbfgs_resume_path, map_location="cpu", weights_only=False)
        refined, lbfgs_logs = _ck["refined"], _ck["lbfgs_logs"]
        refined_U, all_lbfgs_grads = _ck["refined_U"], _ck["all_lbfgs_grads"]
        scoring_logs = _ck.get("scoring_logs", [])
        all_lbfgs_pos = _ck.get("all_lbfgs_pos", [])          # pre-logging checkpoints
        if len(all_lbfgs_pos) < len(refined):
            all_lbfgs_pos = all_lbfgs_pos + [torch.zeros(0)] * (
                len(refined) - len(all_lbfgs_pos))
        k_start = len(refined)
        print(f"[resume] {lbfgs_resume_path}: {k_start}/{len(all_bests)} chains already refined")
    else:
        refined, lbfgs_logs, refined_U, all_lbfgs_grads = [], [], [], []
        all_lbfgs_pos, scoring_logs = [], []
        k_start = 0

    for k, (bx, by) in enumerate(all_bests):
        if k < k_start:
            continue
        print(f"[lbfgs] refine {k+1}/{len(all_bests)}  (src={source_per_run[k]})  "
              f"over {n_chunks} chunk(s)")
        xp, yp = bx, by
        run_logs, run_grads, run_pos = [], [], []
        # Baseline is the Adam-best on the scoring set, so the refinement can
        # only ever hand back something that beats what it started from.
        best_S, best_sx, best_sy = score_on_set(bx, by, scoring_set, fnn, recon, mountain)
        best_at, n_kept, n_reverted = -1, 0, 0
        # Every scoring-set evaluation, at its position on the combined closure
        # timeline — the series the curves plot, since it is the only one whose
        # values are comparable to each other.
        run_scores = [dict(chunk=-1, closure=0, U_chunk=None, U_score=best_S,
                           best=True, kept=True)]
        print(f"  [lbfgs] refine {k} adam-best scores {best_S:+.3f} on the scoring set")
        for c, primary_chunk in enumerate(primary_chunks):
            xp, yp, Up, lg, ghist, phist = lbfgs_refine(
                xp, yp, fnn, recon, primary_chunk, mountain)
            run_logs.extend(lg)
            run_grads.extend(ghist)
            run_pos.extend(phist)          # chunks run back-to-back -> one path
            # Accept/reject on the scoring set, not on the chunk's own U: each
            # chunk optimizes a different batch, so its U ranks batches as much
            # as layouts. A chunk that did not improve the held-out score has its
            # positions rolled back, so the sweep can only ever walk uphill on
            # one fixed objective. Scored EVERY chunk — a chunk with no verdict
            # could only be accepted blindly, which is what this replaces.
            S, sx, sy = score_on_set(xp, yp, scoring_set, fnn, recon, mountain)
            floor = best_S - abs(best_S) * SCORING_ACCEPT_TOL
            is_best = S > best_S
            keep = is_best or S >= floor
            if is_best:
                best_S, best_sx, best_sy, best_at = S, sx, sy, c
                xp, yp = sx, sy
                mark = f"  scored {S:+.3f}  <- NEW BEST"
            elif keep:
                xp, yp = sx, sy
                n_kept += 1
                mark = (f"  scored {S:+.3f}  within {SCORING_ACCEPT_TOL:.0%} of "
                        f"{best_S:+.3f} — carried forward")
            else:
                xp, yp = best_sx.clone(), best_sy.clone()
                n_reverted += 1
                mark = f"  scored {S:+.3f}  below {floor:+.3f} — reverted"
            run_scores.append(dict(chunk=c, closure=len(run_logs),
                                   U_chunk=float(Up), U_score=float(S),
                                   best=bool(is_best), kept=bool(keep)))
            print(f"  [lbfgs] refine {k} chunk {c+1}/{n_chunks}  U={Up:+.3f}  "
                  f"({len(lg)} closure calls){mark}")
        refined.append((best_sx, best_sy))
        refined_U.append(best_S)
        scoring_logs.append(run_scores)
        lbfgs_logs.append(run_logs)
        all_lbfgs_grads.append(run_grads)
        all_lbfgs_pos.append(torch.stack(run_pos, dim=0) if run_pos else torch.zeros(0))
        print(f"  [lbfgs] refine {k} DONE  best U={best_S:+.3f} on the scoring set "
              f"from chunk {best_at + 1 if best_at >= 0 else 0} "
              f"({n_chunks - n_kept - n_reverted} new best, {n_kept} within "
              f"{SCORING_ACCEPT_TOL:.0%}, {n_reverted} reverted, of {n_chunks})")

        _tmp = lbfgs_resume_path + ".tmp"
        torch.save({"refined": refined, "lbfgs_logs": lbfgs_logs,
                   "refined_U": refined_U, "all_lbfgs_grads": all_lbfgs_grads,
                   "all_lbfgs_pos": all_lbfgs_pos,
                   "scoring_logs": scoring_logs}, _tmp)
        os.replace(_tmp, lbfgs_resume_path)

    # Per-run consecutive-step gradient cosine distance (Adam + L-BFGS phases),
    # W-step vector-averaged to suppress minibatch-noise inflation.
    adam_cos_per_run  = [consecutive_cos_distance(g, GRAD_COS_WINDOW).tolist()
                         for g in all_adam_grads]
    lbfgs_cos_per_run = [consecutive_cos_distance(g, GRAD_COS_WINDOW).tolist()
                         for g in all_lbfgs_grads]

    # Build the (K, n_det, 2) ensemble and align by closest position.
    layouts_xy = np.stack(
        [np.stack([xp.numpy(), yp.numpy()], axis=-1) for xp, yp in refined], axis=0,
    )                                                                # (K, n_det, 2)

    # refined_U[k] is chain k's utility on ITS OWN final chunk, which is not the
    # same primaries across chains, so argmax(refined_U) compares chains on
    # different data. Re-score every chain on one common batch (the first
    # seeded chunk) before picking the reference/best layout.
    def _score(xp, yp):
        with torch.no_grad():
            U, _, _ = utility_of_xy(xp.to(DEVICE), yp.to(DEVICE), primary_chunks[0], fnn, recon)
        return float(U.item())
    ref_idx, common_U = select_best_on_common_batch(refined, _score)    # best-U run = reference
    aligned, perms = align_to_reference(layouts_xy, ref_idx)
    mean_xy = aligned.mean(axis=0)                                   # (n_det, 2)
    std_xy  = aligned.std(axis=0)                                    # (n_det, 2)

    best_x, best_y = refined[ref_idx]
    best_src = source_per_run[ref_idx]
    print(f"[ensemble] K={len(refined)}  best U={common_U[ref_idx]:+.3f} (common batch)  "
          f"(run {ref_idx}, src={best_src})  "
          f"mean σx={std_xy[:,0].mean():.1f}m σy={std_xy[:,1].mean():.1f}m")

    # Minimum pairwise detector separation of the selected layout, and how many
    # pairs sit closer than 1m (optimized layouts have contained exactly
    # coincident detectors, which nothing previously reported).
    with torch.no_grad():
        _dx = best_x.unsqueeze(0) - best_x.unsqueeze(1)
        _dy = best_y.unsqueeze(0) - best_y.unsqueeze(1)
        _dist = torch.sqrt(_dx ** 2 + _dy ** 2)
        _n = _dist.shape[0]
        _upper_mask = torch.triu(torch.ones(_n, _n, dtype=torch.bool), diagonal=1)
        _offdiag = _dist[~torch.eye(_n, dtype=torch.bool)]
        best_layout_min_separation_m = float(_offdiag.min().item())
        best_layout_pairs_below_1m = int((_dist[_upper_mask] < 1.0).sum().item())
    print(f"[ensemble] best-layout min pairwise separation="
          f"{best_layout_min_separation_m:.3f}m  "
          f"pairs<1m={best_layout_pairs_below_1m}")

    # ── Persist artifacts ───────────────────────────────────────────────────
    torch.save({"x": best_x, "y": best_y, "U": common_U[ref_idx],
                "U_last_chunk": refined_U[ref_idx],
                "run": ref_idx, "source": best_src},
               os.path.join(opt_dir, "layout_best.pt"))
    torch.save({"mean_x": torch.as_tensor(mean_xy[:, 0]),
                "mean_y": torch.as_tensor(mean_xy[:, 1]),
                "std_x":  torch.as_tensor(std_xy[:, 0]),
                "std_y":  torch.as_tensor(std_xy[:, 1])},
               os.path.join(opt_dir, "layout_mean.pt"))
    torch.save({"aligned": torch.as_tensor(aligned),          # (K, n_det, 2)
                "perms": torch.as_tensor(perms),
                "utilities": torch.as_tensor(refined_U),
                "common_U": common_U,
                "source_per_run": source_per_run,
                "ref_idx": ref_idx},
               os.path.join(opt_dir, "layouts_all.pt"))

    # Layout trajectories, for post-hoc animation/inspection. Per chain:
    #   adam[k]  : (n_frames, 2*n_det) every POS_LOG_EVERY Adam steps,
    #              post-projection, frame 0 = the chain's starting layout.
    #   lbfgs[k] : (n_frames, 2*n_det) every POS_LOG_EVERY L-BFGS CLOSURE calls
    #              (line-search probes included, so not monotonic in U),
    #              concatenated across chunks, ending on the projected optimum.
    # Split x/y with t[:, :n_det] = East, t[:, n_det:] = North.
    torch.save({"adam": all_adam_pos,
                "lbfgs": all_lbfgs_pos,
                "source_per_run": source_per_run,
                "refined_U": refined_U,
                "ref_idx": ref_idx,
                "n_det": N_DETECTORS,
                "pos_log_every": POS_LOG_EVERY,
                "adam_epochs": N_ADAM_EPOCHS},
               os.path.join(opt_dir, "trajectory.pt"))

    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "scheme": scheme,
            "sources": list(per_source),
            "source_per_run": source_per_run,
            "ref_idx": ref_idx,
            "ref_source": best_src,
            "refined_U": refined_U,
            "common_U": common_U,
            "common_batch": "chunk0",
            "best_U": common_U[ref_idx],
            "best_U_last_chunk": refined_U[ref_idx],
            "best_layout_min_separation_m": best_layout_min_separation_m,
            "best_layout_pairs_below_1m": best_layout_pairs_below_1m,
            "ensemble_stats": dict(
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                max_std_x=float(std_xy[:, 0].max()),
                max_std_y=float(std_xy[:, 1].max()),
            ),
            "grad_cos_consecutive": dict(
                adam=adam_cos_per_run,    # per run: 1 - cos(g_t, g_{t-1}) over Adam steps
                lbfgs=lbfgs_cos_per_run,  # per run: same over L-BFGS closure calls
            ),
            "adam_logs": all_adam_logs,
            "lbfgs_logs": lbfgs_logs,
            "scoring_logs": scoring_logs,
            "config": dict(
                n_chains=N_CHAINS, init_overdisp_sigma=INIT_OVERDISP_SIGMA,
                n_adam_epochs=N_ADAM_EPOCHS, primaries_per_step=PRIMARIES_PER_STEP,
                adam_lr=ADAM_LR, grad_clip=GRAD_CLIP,
                lbfgs_max_iter=LBFGS_MAX_ITER, lbfgs_lr=LBFGS_LR,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
                offmesh_penalty=OFFMESH_PENALTY,
                lbfgs_batch_primaries=LBFGS_BATCH_PRIMARIES,
                scoring_set_primaries=SCORING_SET_PRIMARIES,
                scoring_batch_primaries=SCORING_BATCH_PRIMARIES,
                scoring_accept_tol=SCORING_ACCEPT_TOL,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plt.plot_curves_lbfgs(all_adam_logs, lbfgs_logs, all_adam_grads, all_lbfgs_grads,
                           os.path.join(opt_dir, "optimize_curves.png"),
                           grad_cos_window=GRAD_COS_WINDOW,
                           scoring_logs=scoring_logs)
    _plt.plot_components_lbfgs(all_adam_logs, lbfgs_logs,
                              os.path.join(opt_dir, "utility_components.png"))
    # Render the ensemble + density in the (North, Up) cross section. The
    # optimiser works in (East, North); SurfaceUpMap projects East -> Up =
    # g(North, East) so detectors sit ON the mountain profile (CPU is fine —
    # plotting is the last, read-only step). DENSITY_VMAX clamps the heatmap
    # colorbar to [0, DENSITY_VMAX]; <=0 auto-scales.
    try:
        surface = SurfaceUpMap.from_mountain(mountain).to("cpu")
        vmax = DENSITY_VMAX if DENSITY_VMAX and DENSITY_VMAX > 0 else None
        _plt.plot_ensemble(aligned, mean_xy, std_xy, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_ensemble.png"),
                       surface=surface, title_kind="L-BFGS ensemble")
        _plt.plot_density_heatmap(aligned, best_x, best_y,
                       mountain, os.path.join(opt_dir, "layout_density.png"),
                       surface=surface, vmax=vmax)
    except Exception as exc:
        print(f"[plot] ensemble/density skipped ({exc!r})")

    print(f"[done] scheme={scheme}  best U={common_U[ref_idx]:+.3f}  "
          f"σ̄=({std_xy[:,0].mean():.1f}, {std_xy[:,1].mean():.1f}) m  ({opt_dir})")
    if os.path.exists(lbfgs_resume_path):
        os.remove(lbfgs_resume_path)
    return dict(scheme=scheme, best_U=common_U[ref_idx],
                best_x=best_x, best_y=best_y,
                mean_std_x=float(std_xy[:, 0].mean()),
                mean_std_y=float(std_xy[:, 1].mean()),
                opt_dir=opt_dir)


def main():
    global N_CHAINS, N_ADAM_EPOCHS, LBFGS_MAX_ITER, DENSITY_VMAX, SEED
    global OFFMESH_PENALTY, INIT_SCHEMES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED,
                    help="Base seed for chain perturbations, per-scheme offsets, and the "
                         "fixed L-BFGS polishing batch (default 42, the original run's value).")
    ap.add_argument("--chains", type=int, default=N_CHAINS)
    ap.add_argument("--adam-epochs", type=int, default=N_ADAM_EPOCHS)
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_MAX_ITER)
    ap.add_argument("--schemes", type=str, default=",".join(INIT_SCHEMES),
                    help="comma-separated init schemes to run (choices: grid, "
                         "center, random; default: all of "
                         f"'{','.join(INIT_SCHEMES)}'). Each scheme writes its own "
                         "OPT_DIR_TEMPLATE folder, so passing ONE scheme per "
                         "process is how the schemes get split across parallel "
                         "jobs instead of running back-to-back in one.")
    ap.add_argument("--offmesh-penalty", type=float, default=OFFMESH_PENALTY,
                    help="weight (in units of U) of the differentiable off-mesh "
                         "penalty subtracted inside utility_of_xy. 0 disables "
                         "it, reproducing the old unconstrained behaviour. The "
                         "penalty is exactly 0 inside the snap radius, so an "
                         "in-band layout scores identically either way "
                         f"(default: {OFFMESH_PENALTY:g})")
    ap.add_argument("--density-vmax", type=float, default=DENSITY_VMAX,
                    help="density heatmap colorbar upper limit (plots [0, vmax]); "
                         "pass <=0 to auto-scale (default from config)")
    ap.add_argument("--save_best_layout", type=str, default=None,
                    help="Path to save the globally best layout as a "
                         "(N_DETECTORS, 2) tensor with (East, North) per detector.")
    ap.add_argument("--init_from", type=str, default=None,
                    help="Path to a (N_DETECTORS, 2) layout tensor (or dict with "
                         "'x'/'y' keys). Warm-starts all chains from this layout "
                         "with small N(0, 10m) diversity perturbations.")
    ap.add_argument("--fnn_folder", type=str, default=None,
                    help="Override FNN_FOLDER from constants.py (use a fine-tuned "
                         "checkpoint directory from the adaptive retraining loop).")
    ap.add_argument("--recon_folder", type=str, default=None,
                    help="Exact path to the recon dir to load (default: "
                         "RECON_FOLDER + '_deepsets', the 03_train_recon_deepsets.py output).")
    ap.add_argument("--opt_suffix", type=str, default="",
                    help="Suffix appended to the output directory name for each "
                         "scheme (e.g. '_r1' to get lbfgs_ensemble_r1_{scheme}/).")
    args = ap.parse_args()
    N_CHAINS, N_ADAM_EPOCHS, LBFGS_MAX_ITER = \
        int(args.chains), int(args.adam_epochs), int(args.lbfgs_iters)
    DENSITY_VMAX = float(args.density_vmax)
    SEED = int(args.seed)
    OFFMESH_PENALTY = float(args.offmesh_penalty)

    _valid = ("grid", "center", "random")   # sample_initial_layout_ne schemes
    INIT_SCHEMES = tuple(s.strip() for s in args.schemes.split(",") if s.strip())
    _bad = [s for s in INIT_SCHEMES if s not in _valid]
    if _bad or not INIT_SCHEMES:
        raise SystemExit(f"--schemes: unknown {_bad or '(empty)'}; "
                         f"choose from {_valid}")

    print("=" * 72)
    print("v6/04_optimize_lbfgs_ensemble.py — Adam warm-start + L-BFGS ensemble")
    print("=" * 72)
    print(f"device       : {DEVICE}")
    print(f"seed         : {SEED}")
    print(f"init schemes : {INIT_SCHEMES}")
    print(f"chains (K)   : {N_CHAINS}  (init σ={INIT_OVERDISP_SIGMA} m)")
    print(f"offmesh pen  : {OFFMESH_PENALTY:g}"
          + ("  (U keeps no knowledge of the mountain)" if not OFFMESH_PENALTY
             else f"  (onset at {PENALTY_ONSET_FRAC:g}x the snap radius)"))
    print(f"Adam epochs  : {N_ADAM_EPOCHS}  (primaries/step={PRIMARIES_PER_STEP})")
    print(f"L-BFGS       : max_iter={LBFGS_MAX_ITER}  batch={LBFGS_BATCH_PRIMARIES}")

    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    if args.opt_suffix:
        global OPT_DIR_TEMPLATE
        OPT_DIR_TEMPLATE = OPT_FOLDER + "_lbfgs_ensemble" + args.opt_suffix + "_{scheme}"

    if args.fnn_folder:
        import modules_v6.constants as _C
        _C.FNN_FOLDER = args.fnn_folder
        global FNN_FOLDER
        FNN_FOLDER = args.fnn_folder
        print(f"[fnn_folder] overriding FNN_FOLDER -> {args.fnn_folder}")

    if args.recon_folder:
        global RECON_DIR
        RECON_DIR = args.recon_folder
        print(f"[recon_folder] overriding recon dir -> {args.recon_folder}")

    fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_DIR)

    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    # Optional warm-start center layout (N_DETECTORS, 2) = (East, North).
    init_center = None
    if args.init_from:
        raw = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            e_c = raw["x"].float().reshape(-1)   # saved "x" is the first coord = East
            n_c = raw["y"].float().reshape(-1)
            init_center = torch.stack([e_c, n_c], dim=-1)
        else:
            init_center = raw.float()
        print(f"[init_from] loaded layout from {args.init_from}  "
              f"shape={tuple(init_center.shape)}")

    results = []
    per_scheme = {}     # scheme -> (adam_bests, adam_logs, perturbed_inits)
    for i, scheme in enumerate(INIT_SCHEMES):
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}"
              f"{'  (warm-start from --init_from)' if init_center is not None else ''}")
        print("=" * 72)
        opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
        os.makedirs(opt_dir, exist_ok=True)
        adam_resume_path = os.path.join(opt_dir, "adam_resume.pt")
        layout_best_path = os.path.join(opt_dir, "layout_best.pt")

        # gpu_requeue can preempt between schemes too (this is what forced a
        # full grid+center redo before this checkpoint existed): a scheme
        # whose layout_best.pt is already on disk is fully done — skip both
        # phases and just reload its cached Adam-chain results (still needed
        # for a "combined" run over multiple schemes).
        if os.path.exists(layout_best_path) and os.path.exists(adam_resume_path):
            print(f"[skip] scheme={scheme} already complete ({layout_best_path}); "
                  "reloading cached Adam results for combined use")
            _ck = torch.load(adam_resume_path, map_location="cpu", weights_only=False)
            _apos = _ck.get("adam_positions", [])      # pre-logging checkpoints
            _apos = _apos + [torch.zeros(0)] * (len(_ck["adam_bests"]) - len(_apos))
            per_scheme[scheme] = (_ck["adam_bests"], _ck["adam_logs"],
                                  _ck["perturbed_inits"], _ck["adam_grads"], _apos)
            # best_x/best_y/best_U/opt_dir are all the summary print below
            # reads; mean_std_x/y aren't recoverable without re-aligning the
            # ensemble, so they're reported as n/a for a skipped scheme.
            _lb = torch.load(layout_best_path, map_location="cpu", weights_only=False)
            results.append(dict(scheme=scheme, best_U=_lb["U"],
                                best_x=_lb["x"], best_y=_lb["y"],
                                mean_std_x=float("nan"), mean_std_y=float("nan"),
                                opt_dir=opt_dir))
            continue

        # Per-scheme seed offset so chains from different schemes are independent
        # when --init_from is provided (otherwise every scheme gets identical
        # perturbations from init_center, because the generator is reset to the
        # same state before each scheme). Seeding also makes the per-chain resume
        # above deterministic across resubmits, which is what the L-BFGS sweep
        # permutation already relies on.
        scheme_seed = SEED + i
        torch.manual_seed(scheme_seed); np.random.seed(scheme_seed)
        g = torch.Generator().manual_seed(scheme_seed)
        per_scheme[scheme] = _perturbed_adam_runs(
            scheme, N_CHAINS, g, mountain, fnn, recon, primary_all, n_total_primaries,
            init_center=init_center, resume_path=adam_resume_path,
        )
        results.append(_run_one_scheme(
            scheme, mountain, fnn, recon, primary_all, n_total_primaries,
            {scheme: per_scheme[scheme]},
        ))

    if RUN_COMBINED and len(per_scheme) > 1:
        print()
        print("=" * 72)
        print(f"init scheme: {COMBINED_SCHEME_NAME} (sources={list(per_scheme)})")
        print("=" * 72)
        results.append(_run_one_scheme(
            COMBINED_SCHEME_NAME, mountain, fnn, recon, primary_all, n_total_primaries,
            per_scheme,
        ))

    print()
    print("=" * 72)
    print("summary")
    print("=" * 72)
    for r in results:
        print(f"  {r['scheme']:<10}  best U={r['best_U']:+.3f}  "
              f"σ̄=({r['mean_std_x']:.1f}, {r['mean_std_y']:.1f}) m  ->  {r['opt_dir']}")

    # Save the globally best layout across all schemes if requested.
    if args.save_best_layout:
        best_r = max(results, key=lambda r: r["best_U"])
        best_layout = torch.stack([best_r["best_x"], best_r["best_y"]], dim=-1)
        torch.save(best_layout, args.save_best_layout)
        print(f"\n[save_best_layout] best U={best_r['best_U']:+.3f} "
              f"(scheme={best_r['scheme']})  ->  {args.save_best_layout}")


if __name__ == "__main__":
    main()