"""Measure the irreducible (aleatoric) noise floor of the Step-2 surrogate.

THEORY.md §10.4: the surrogate predicts (E, T) from the primary summary q and the
layout xy, but the true response is a function of the *full stochastic shower
point cloud*. Two showers with an identical primary produce different (E, T), so
the best any model can do is predict the conditional mean E[(E,T) | q, xy]; the
shower-to-shower variance is an irreducible floor on the z-scored val MSE.

The training corpus cannot reveal this floor directly: every shower has a unique
primary, and its 7 strategy-rows all share the SAME realization — there is no
(same q, different shower) pair in the data. So we *generate* it: sample a set of
primaries, draw M independent showers for each (the flow-matching generator is
stochastic), run the EXACT training kernel (`compute_labels_batch`) with a fixed
layout, and measure the within-primary variance of the (same log/z-transforms
the trainer uses) labels.

    floor_c (z-MSE units) = mean_{p,s,i} Var_realizations(y_{c}) / Var_corpus(y_c)

reported per channel (E, T), as the 0.5*(E+T) total the trainer logs, and
restricted to *fired* detectors (to compare against conditional-on-fired metrics).
floor = 1 - R²_max: a model whose val MSE equals the floor is Bayes-optimal.

Run from the v6 folder (needs a GPU for generation):

    cd TambOpt/detector_optimization_v6
    python compute_aleatoric_floor.py --n-prim 128 --m-real 64
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4 (and TAMBO-opt)
from modules_v6.fnn_surrogate import compute_labels_batch
from modules_v6.detector_strategies import _STRATEGIES, _STRATEGY_FNS
from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, RECENTER_TO_MOUNTAIN,
)
from modules_v4.tr_geometry    import load_tr_mountain
from modules_v4.tr_surface_map import SurfaceEastMap

# Low-level generator pieces (importing modules.generate_showers injects the
# TAMBO-opt path so `allshowers` is importable).
from modules.generate_showers import GenerateShowers  # noqa: F401  (path injection)
from allshowers.generate_showers import (
    sample_primary_particles, run_point_count_fm, run_allshowers,
    _DEFAULT_POINT_COUNT_MODEL, _DEFAULT_ALLSHOWERS_RUN_DIR,
)

T_LOG_SCALE = 1.0e8          # must match 02_train_fnn*.py
FIRE_EPS    = 1.0e-3         # log1p(E) above this ⇒ detector "fired" this shower
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constants.GEOMETRY_PATH went stale after the TAMBOSim v1 reorg
# (resources/basic_geometry.h5 → resources/geometry/colca_valley.h5; confirmed in
# TAMBOSim git). GEOMETRY_GROUP/DET_KEY still match. Prefer a local copy in this
# folder, then the new TAMBOSim path, then the (stale) constant.
GEOMETRY_PATH_RESOLVED = next(
    (p for p in (
        os.path.join(_HERE, "colca_valley.h5"),
        "/n/home05/zdimitrov/tambo/TAMBOSim/resources/geometry/colca_valley.h5",
        GEOMETRY_PATH,
    ) if os.path.exists(p)),
    GEOMETRY_PATH,
)


def _corpus_global_std():
    """Per-channel std of the corpus labels in the SAME space the trainer
    normalizes by: E.pt is already log1p(E); T.pt is raw → apply log1p(T*1e8)."""
    E = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    e_std = float(E.std()); e_mean = float(E.mean())
    del E
    T = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    T = torch.log1p(T * T_LOG_SCALE)
    t_std = float(T.std()); t_mean = float(T.mean())
    del T
    return e_mean, e_std, t_mean, t_std


def _generate_repeated_showers(n_prim, m_real, seed, gen_batch):
    """Sample n_prim primaries, repeat each m_real times, generate independent
    showers. Returns clouds (n_prim, m_real, P, 5) and the primaries dict."""
    prim = sample_primary_particles(n=n_prim, seed=seed)        # corpus ranges
    energies   = torch.repeat_interleave(prim["energies"],   m_real, dim=0)  # (n*m,1)
    directions = torch.repeat_interleave(prim["directions"], m_real, dim=0)  # (n*m,3)
    labels     = torch.repeat_interleave(prim["labels"],     m_real, dim=0)  # (n*m,)
    print(f"[gen] {n_prim} primaries × {m_real} realizations = {n_prim*m_real} showers")

    # PointCountFM runs on CPU (its compiled TorchScript has device constants
    # baked at trace time → CUDA raises a device-mismatch; production uses CPU).
    num_points = run_point_count_fm(
        model_path=_DEFAULT_POINT_COUNT_MODEL,
        energies=energies, directions=directions, labels=labels,
    )
    samples = run_allshowers(
        run_dir=_DEFAULT_ALLSHOWERS_RUN_DIR,
        energies=energies, directions=directions, labels=labels,
        num_points=num_points, num_timesteps=16, batch_size=gen_batch,
        solver="midpoint", device=str(DEVICE),
    ).float().cpu()                                            # (n*m, P, 5)

    # Sanity: realizations of one primary must actually differ.
    npl = num_points.reshape(n_prim, m_real, -1).sum(-1).float()   # (n_prim, m_real)
    within_cv = (npl.std(dim=1) / npl.mean(dim=1).clamp(min=1)).mean()
    print(f"[gen] mean within-primary CV of total hit count = {within_cv:.3f} "
          f"({'OK — realizations differ' if within_cv > 1e-3 else 'WARNING — near-identical!'})")

    P = samples.shape[1]
    return samples.reshape(n_prim, m_real, P, 5), prim


def _recenter(clouds_flat, mountain):
    """Per-shower recenter onto the mountain bbox centre — identical to
    build_training_pairs(recenter_to_mountain=True)."""
    mtn_cx = 0.5 * (mountain.n_min + mountain.n_max)
    mtn_cy = 0.5 * (mountain.u_min + mountain.u_max)
    mask  = (clouds_flat[:, :, 3] > 0).float()                 # (N, P)
    w_sum = mask.sum(dim=1).clamp(min=1.0)
    cx = (clouds_flat[:, :, 0] * mask).sum(dim=1) / w_sum
    cy = (clouds_flat[:, :, 1] * mask).sum(dim=1) / w_sum
    dx = (mtn_cx - cx).view(-1, 1)
    dy = (mtn_cy - cy).view(-1, 1)
    clouds_flat[..., 0] = clouds_flat[..., 0] + dx * mask
    clouds_flat[..., 1] = clouds_flat[..., 1] + dy * mask
    return clouds_flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prim",    type=int, default=128, help="distinct primaries")
    ap.add_argument("--m-real",    type=int, default=64,  help="realizations per primary")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=256, help="AllShowers gen batch")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "aleatoric_floor.json"))
    args = ap.parse_args()

    print("=" * 72)
    print("aleatoric floor — within-primary label variance / corpus variance")
    print("=" * 72)
    print(f"device={DEVICE}  recenter={RECENTER_TO_MOUNTAIN}  "
          f"n_prim={args.n_prim}  m_real={args.m_real}  strategies={len(_STRATEGIES)}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Corpus global std (the trainer's z-score denominators).
    e_mean, e_std, t_mean, t_std = _corpus_global_std()
    print(f"[corpus] log1p(E): mean={e_mean:.4f} std={e_std:.4f}   "
          f"log1p(T*1e8): mean={t_mean:.4f} std={t_std:.4f}")

    # Mountain + differentiable surface (East = f(N,Up)) — as in 01_build_dataset.
    print(f"[geometry] {GEOMETRY_PATH_RESOLVED}")
    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    surface = SurfaceEastMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)

    # Generate independent realizations per primary.
    t0 = time.time()
    clouds, _prim = _generate_repeated_showers(
        args.n_prim, args.m_real, args.seed, args.gen_batch)
    n_prim, m_real, P, _ = clouds.shape
    if RECENTER_TO_MOUNTAIN:
        clouds = _recenter(clouds.reshape(n_prim * m_real, P, 5), mountain
                           ).reshape(n_prim, m_real, P, 5)
    print(f"[gen] done in {time.time()-t0:.1f}s  P(max_points)={P}")

    # For each (primary, strategy): one fixed layout, M realizations → within-var.
    rng = np.random.default_rng(args.seed)
    var_E, var_T, fired_frac = [], [], []   # each appends (n_det,) per (p,s)
    t0 = time.time()
    for s_idx, (s_name, fn_name, kwargs) in enumerate(_STRATEGIES):
        fn = _STRATEGY_FNS[fn_name]
        for p in range(n_prim):
            x_det, y_det = fn(mountain, n_det=N_DETECTORS, rng=rng, **kwargs)
            x_det = x_det.float().to(DEVICE); y_det = y_det.float().to(DEVICE)
            cl = clouds[p].to(DEVICE)                          # (M, P, 5)
            E, T = compute_labels_batch(cl, x_det, y_det, surface)   # (M, n_det) raw
            E = torch.log1p(E)                                 # → training E space
            T = torch.log1p(T * T_LOG_SCALE)                   # → training T space
            var_E.append(E.var(dim=0, unbiased=True).cpu())    # (n_det,)
            var_T.append(T.var(dim=0, unbiased=True).cpu())
            fired_frac.append((E > FIRE_EPS).float().mean(dim=0).cpu())
        print(f"[kernel] strategy {s_idx+1}/{len(_STRATEGIES)} {s_name:<18} done")
    print(f"[kernel] all within-group variances in {time.time()-t0:.1f}s")

    var_E = torch.cat(var_E)              # (n_prim*n_strat*n_det,)
    var_T = torch.cat(var_T)
    fired = torch.cat(fired_frac) > 0.5   # detector fires in majority of realizations

    def _floor(v, std):  return float(v.mean()) / (std ** 2)
    floor_E, floor_T = _floor(var_E, e_std), _floor(var_T, t_std)
    floor_E_fired = _floor(var_E[fired], e_std) if fired.any() else float("nan")
    floor_T_fired = _floor(var_T[fired], t_std) if fired.any() else float("nan")
    floor_total = 0.5 * (floor_E + floor_T)

    res = dict(
        n_prim=n_prim, m_real=m_real, n_strategies=len(_STRATEGIES),
        recenter=RECENTER_TO_MOUNTAIN, fire_eps=FIRE_EPS,
        corpus_std=dict(E=e_std, T=t_std),
        within_group_std=dict(E=float(var_E.mean()**0.5), T=float(var_T.mean()**0.5)),
        fired_fraction=float(fired.float().mean()),
        floor_zmse=dict(
            E=floor_E, T=floor_T, total=floor_total,
            E_fired=floor_E_fired, T_fired=floor_T_fired,
        ),
        max_R2=dict(E=1 - floor_E, T=1 - floor_T, total=1 - floor_total),
    )
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 72)
    print("ALEATORIC FLOOR  (z-scored val-MSE units — directly comparable to val)")
    print("=" * 72)
    print(f"  fired fraction of (primary,strategy,detector) groups : {res['fired_fraction']:.3f}")
    print(f"  floor  E (all)   = {floor_E:.4f}   (max R² = {1-floor_E:.3f})")
    print(f"  floor  T (all)   = {floor_T:.4f}   (max R² = {1-floor_T:.3f})")
    print(f"  floor  total     = {floor_total:.4f}   = 0.5*(E+T)")
    print(f"  floor  E (fired) = {floor_E_fired:.4f}")
    print(f"  floor  T (fired) = {floor_T_fired:.4f}")
    print(f"\n  A surrogate whose val MSE reaches the floor is Bayes-optimal;")
    print(f"  no architecture/optimizer change can go below it.")
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
