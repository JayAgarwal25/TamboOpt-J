"""Measure the irreducible (aleatoric) noise floor of the Step-2 surrogate.

THEORY.md §10.4: the surrogate predicts (E, T) from the primary summary q and the
layout xy, but the true response is a function of the *full stochastic shower
point cloud*. Two showers with an identical primary produce different (E, T), so
the best any model can do is predict the conditional mean E[(E,T) | q, xy]; the
shower-to-shower variance is an irreducible floor on the z-scored val MSE.

The training corpus cannot reveal this floor directly: every shower has a unique
primary, and its 7 strategy-rows all share the SAME realization — there is no
(same q, different shower) pair in the data. So we *generate* it: sample a set of
primaries, draw M independent showers for each — PER SPECIES, with the SAME
per-species AllShowers checkpoints + staging + anti-clip that built the dual corpus
(00_generate_data_dual_species.py) — run the EXACT (North, East) training kernel
(`fnn_surrogate_ne.compute_labels_batch`) with a fixed layout, and measure the
within-primary variance of the (same log/z-transforms the trainer uses) labels.
Both species' components are pooled, matching the combined dual corpus.

    floor_c (z-MSE units) = mean_{p,s,i} Var_realizations(y_{c}) / Var_corpus(y_c)

reported per channel (E, T) and as the 0.5*(E+T) total the trainer logs. For the
*fired*-detector subset we report it both ways: normalized by the global corpus
variance (z-MSE units) AND by the fired-conditional corpus variance ("fired vs
fired" → conditional-on-fired R², since the global variance also includes the
non-fired detectors). floor = 1 - R²_max: a model whose val MSE equals
the floor is Bayes-optimal.

Generation needs a GPU (the AllShowers flow-matching sampler is impractically slow
on CPU). The path resolution is CWD-independent, so run it from anywhere, e.g.:

    cd TambOpt/detector_optimization_v6
    python plots/compute_aleatoric_floor.py --n-prim 128 --m-real 64
"""
import argparse
import json
import os
import sys
import time


# v6 folder = parent of this file's plots/ dir. File-relative (NOT cwd-relative) so
# the script imports modules_v6 no matter where it's launched from.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4 (and TAMBO-opt)
from modules_v6.fnn_surrogate_ne import compute_labels_batch
from modules_v6.detector_strategies_ne import _STRATEGIES, _STRATEGY_FNS
from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, RECENTER_TO_MOUNTAIN,
)
from modules_v4.tr_geometry       import load_tr_mountain
from modules_v6.tr_surface_map_ne import SurfaceUpMap

# Generation reuses 00_generate_data_dual_species.py VERBATIM — the SAME per-species
# AllShowers checkpoints + staging (pre_ln injection) + anti-clip re-roll that built
# the dual corpus. So the floor's labels are per-species components from the exact
# generators behind corpus_std: the numerator (generated within-shower variance) and
# the denominator (corpus variance) finally match. (00's filename starts with a digit
# → load it by path; importing only runs its module-level imports/config, not main().)
from modules.generate_showers import GenerateShowers  # noqa: F401  (path injection)
import importlib.util as _ilu
_spec00 = _ilu.spec_from_file_location(
    "gen00", os.path.join(_HERE, "00_generate_data_dual_species.py"))
gen00 = _ilu.module_from_spec(_spec00); _spec00.loader.exec_module(gen00)
sample_primary_particles = gen00.sample_primary_particles   # re-export

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

# Must match the NE dataset builder: fnn_surrogate_ne.build_training_pairs calls
# compute_labels_batch with its DEFAULT sigma (200 m), so use 200 to reproduce the
# exact training labels (the kernel's transverse smoothing length).
sigma_spatial = 200


def _describe(x: torch.Tensor) -> dict:
    """min / max / mean / std (+ z-scored extremes) of a 1-D tensor, so the std
    can be read against the value range. z_min/z_max = (extreme - mean)/std (how
    many σ below/above the mean each extreme sits); z_range = z_max - z_min is the
    full span in σ units. A std that is a large fraction of the range ⇒ the values
    are spread out relative to their own scale."""
    mn, mx = float(x.min()), float(x.max())
    mu, sd = float(x.mean()), float(x.std())
    z = (lambda v: (v - mu) / sd) if sd > 0 else (lambda v: float("nan"))
    return dict(min=mn, max=mx, mean=mu, std=sd,
                z_min=z(mn), z_max=z(mx),
                z_range=(z(mx) - z(mn)) if sd > 0 else float("nan"))


def _corpus_label_stats() -> dict:
    """Corpus label distributions in the trainer's log space (E = log1p(E),
    T = log1p(T*1e8)): full _describe (min/max/mean/std + z-range) over ALL
    detectors and over the FIRED subset (E > FIRE_EPS). The `std` fields are the
    floor's z-score denominators; the min/max/mean give the range to read the std
    against. (Fired std differs from global std because the global also includes
    the non-fired ≈zero detectors.)"""
    E = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    fired = E > FIRE_EPS                                   # E-based fired mask (corpus)
    out = {"E_all": _describe(E)}
    out["E_fired"] = _describe(E[fired]) if fired.any() else None
    del E                                                  # free E; keep `fired` for T
    T = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    T = torch.log1p(T * T_LOG_SCALE)
    out["T_all"]   = _describe(T)
    out["T_fired"] = _describe(T[fired]) if fired.any() else None
    del T, fired
    return out


def _generate_repeated_showers(n_prim, m_real, seed, gen_batch):
    """Sample n_prim primaries, repeat each m_real times, and generate independent
    realizations as PER-SPECIES components — using the same per-species AllShowers
    checkpoints + staging + anti-clip as 00_generate_data_dual_species.py (`SPECIES`,
    `stage_run_dir`, `_gen_chunk`), so the labels match the dual corpus.

    Returns ({species_name: clouds (n_prim, m_real, P, 5)}, primaries dict); every
    species is padded to the common target_P (the muon cap)."""
    if gen_batch:
        gen00.BATCH_SIZE = int(gen_batch)                  # GPU generate batch size
    prim = sample_primary_particles(                       # corpus ranges (match 00)
        e_min=10 ** gen00.LOG_E_MIN, e_max=10 ** gen00.LOG_E_MAX,
        zenith_min=gen00.ZENITH_MIN, zenith_max=gen00.ZENITH_MAX,
        azimuth_min=gen00.AZIMUTH_MIN, azimuth_max=gen00.AZIMUTH_MAX,
        n=n_prim, seed=seed,
    )
    energies   = torch.repeat_interleave(prim["energies"],   m_real, dim=0)  # (n*m,1)
    directions = torch.repeat_interleave(prim["directions"], m_real, dim=0)  # (n*m,3)
    labels     = torch.repeat_interleave(prim["labels"],     m_real, dim=0)  # (n*m,)
    event_ids  = torch.repeat_interleave(torch.arange(n_prim, dtype=torch.int64), m_real, dim=0)
    target_P   = max(cfg["max_points"] for cfg in gen00.SPECIES.values())
    print(f"[gen] {n_prim} primaries × {m_real} realizations "
          f"= {n_prim*m_real} showers/species  (species={list(gen00.SPECIES)})")

    out = {}
    for name, cfg in gen00.SPECIES.items():
        staged_dir, pcfm = gen00.stage_run_dir(name, cfg)
        g = gen00.Generator(run_dir=staged_dir, num_timesteps=gen00.NUM_TIMESTEPS,
                            compile=True, solver=gen00.SOLVER)
        g.max_points = int(cfg["max_points"])
        sh = gen00._gen_chunk(g, pcfm, cfg, energies, directions, labels, event_ids, target_P)
        samples = torch.as_tensor(sh.points, dtype=torch.float32)            # (n*m, target_P, 5)
        out[name] = samples.reshape(n_prim, m_real, target_P, 5)
        # Sanity: realizations of one primary must actually differ.
        cnt = (samples[:, :, 3] > 0).sum(dim=1).float().reshape(n_prim, m_real)
        cv  = float((cnt.std(dim=1) / cnt.mean(dim=1).clamp(min=1)).mean())
        print(f"[gen] {name:8s} {tuple(out[name].shape)}  within-primary CV(hit count)={cv:.3f}"
              f" ({'ok' if cv > 1e-3 else 'WARNING near-identical'})")
        del g, sh, samples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out, prim


def _recenter(clouds_flat, mountain):
    """Per-shower recenter onto the mountain bbox centre — identical to
    fnn_surrogate_ne.build_training_pairs(recenter_to_mountain=True). The shower
    transverse plane is (North, Up) even in the NE pipeline, so this intentionally
    stays on mountain.u_min/u_max (NOT east) — matching the NE builder exactly."""
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
    ap.add_argument("--gen-batch", type=int, default=32, help="AllShowers gen batch")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, f"aleatoric_floor_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    args = ap.parse_args()

    print("=" * 72)
    print("aleatoric floor — within-primary label variance / corpus variance")
    print("=" * 72)
    print(f"device={DEVICE}  recenter={RECENTER_TO_MOUNTAIN}  "
          f"n_prim={args.n_prim}  m_real={args.m_real}  strategies={len(_STRATEGIES)}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_float32_matmul_precision('high') # TODO maybe deactivate if perforamne is too bad

    # Corpus label distribution: the z-score denominators (std) AND the range
    # (min/max/mean/z-span) to read those stds against.
    corpus_stats = _corpus_label_stats()
    e_std = corpus_stats["E_all"]["std"]; t_std = corpus_stats["T_all"]["std"]
    e_std_fired = corpus_stats["E_fired"]["std"] if corpus_stats["E_fired"] else float("nan")
    t_std_fired = corpus_stats["T_fired"]["std"] if corpus_stats["T_fired"] else float("nan")
    print("[corpus] label distribution (log space)      min /    max /   mean /    std    (z-span)")
    for k in ("E_all", "E_fired", "T_all", "T_fired"):
        s = corpus_stats[k]
        if s is None:
            continue
        print(f"  {k:8s} {s['min']:8.3f} / {s['max']:7.3f} / {s['mean']:7.3f} / {s['std']:7.3f}   "
              f"z[{s['z_min']:+.2f}, {s['z_max']:+.2f}] = {s['z_range']:.2f} sigma")

    # Mountain + differentiable surface (Up = g(North, East)) — as in
    # 01_build_dataset_northeast.py / fnn_surrogate_ne.
    print(f"[geometry] {GEOMETRY_PATH_RESOLVED}")
    mountain = load_tr_mountain(
        GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)

    # Generate independent realizations per primary — PER SPECIES (dual corpus).
    t0 = time.time()
    clouds, _prim = _generate_repeated_showers(
        args.n_prim, args.m_real, args.seed, args.gen_batch)
    n_prim, m_real = args.n_prim, args.m_real
    P = next(iter(clouds.values())).shape[2]
    if RECENTER_TO_MOUNTAIN:
        for name in clouds:
            clouds[name] = _recenter(
                clouds[name].reshape(n_prim * m_real, P, 5), mountain
            ).reshape(n_prim, m_real, P, 5)
    print(f"[gen] done in {time.time()-t0:.1f}s  P(max_points)={P}  species={list(clouds)}")

    # For each (species component, strategy, primary): one fixed layout, M
    # realizations → within-var. Both species are pooled so the numerator matches
    # the combined dual corpus (per-species component rows) used for corpus_std.
    rng = np.random.default_rng(args.seed)
    var_E, var_T, fired_frac = [], [], []   # each appends (n_det,) per (species,s,p)
    gen_E_vals, gen_T_vals = [], []         # all generated labels → value range
    t0 = time.time()
    for s_idx, (s_name, fn_name, kwargs) in enumerate(_STRATEGIES):
        fn = _STRATEGY_FNS[fn_name]
        for sp_clouds in clouds.values():           # pool both species' components
            for p in range(n_prim):
                x_det, y_det = fn(mountain, n_det=N_DETECTORS, rng=rng, **kwargs)
                x_det = x_det.float().to(DEVICE); y_det = y_det.float().to(DEVICE)
                cl = sp_clouds[p].to(DEVICE)                   # (M, P, 5)
                E, T = compute_labels_batch(cl, x_det, y_det, surface, sigma_spatial=sigma_spatial)
                E = torch.log1p(E)                             # → training E space
                T = torch.log1p(T * T_LOG_SCALE)               # → training T space
                var_E.append(E.var(dim=0, unbiased=True).cpu())
                var_T.append(T.var(dim=0, unbiased=True).cpu())
                fired_frac.append((E > FIRE_EPS).float().mean(dim=0).cpu())
                gen_E_vals.append(E.reshape(-1).cpu())
                gen_T_vals.append(T.reshape(-1).cpu())
        print(f"[kernel] strategy {s_idx+1}/{len(_STRATEGIES)} {s_name:<18} done")
    print(f"[kernel] all within-group variances in {time.time()-t0:.1f}s")

    var_E = torch.cat(var_E)              # (n_prim*n_strat*n_det,)
    var_T = torch.cat(var_T)
    fired = torch.cat(fired_frac) > 0.5   # detector fires in majority of realizations

    # Generated-label distribution — the range to read within_group_std against
    # (a within-group σ that is a large fraction of the value range ⇒ noise-dominated;
    #  if it exceeds the corpus z-span, the generator is over-dispersed vs the corpus).
    gen_stats = dict(E=_describe(torch.cat(gen_E_vals)), T=_describe(torch.cat(gen_T_vals)))
    wg_E = float(var_E.mean() ** 0.5); wg_T = float(var_T.mean() ** 0.5)
    print("[generated] label distribution (log space)   min /    max /   mean /    std   (within-group σ)")
    for k, wg in (("E", wg_E), ("T", wg_T)):
        s = gen_stats[k]; rng = s["max"] - s["min"]
        pct = 100.0 * wg / rng if rng > 0 else float("nan")
        print(f"  {k:8s} {s['min']:8.3f} / {s['max']:7.3f} / {s['mean']:7.3f} / {s['std']:7.3f}   "
              f"within-σ={wg:.3f} ({pct:.1f}% of range, corpus std={e_std if k=='E' else t_std:.3f})")

    def _floor(v, std):  return float(v.mean()) / (std ** 2)
    floor_E, floor_T = _floor(var_E, e_std), _floor(var_T, t_std)
    # Fired detectors only, under two normalizations:
    #   *_fired          : fired within-var / ALL-corpus var (z-MSE units, same global
    #                      denominator as the 'all' floor — comparable to the trainer's z-score).
    #   *_fired_vs_fired : fired within-var / FIRED-corpus var → conditional-on-fired R²,
    #                      i.e. fired compared against the fired-signal spread, not the
    #                      global spread (which also includes non-fired detectors).
    floor_E_fired = _floor(var_E[fired], e_std) if fired.any() else float("nan")
    floor_T_fired = _floor(var_T[fired], t_std) if fired.any() else float("nan")
    floor_E_fired_vs_fired = _floor(var_E[fired], e_std_fired) if fired.any() else float("nan")
    floor_T_fired_vs_fired = _floor(var_T[fired], t_std_fired) if fired.any() else float("nan")
    floor_total = 0.5 * (floor_E + floor_T)
    floor_total_fired_vs_fired = 0.5 * (floor_E_fired_vs_fired + floor_T_fired_vs_fired)

    res = dict(
        n_prim=n_prim, m_real=m_real, n_strategies=len(_STRATEGIES),
        species=list(gen00.SPECIES),
        recenter=RECENTER_TO_MOUNTAIN, fire_eps=FIRE_EPS,
        corpus_std=dict(E=e_std, T=t_std),
        corpus_std_fired=dict(E=e_std_fired, T=t_std_fired),
        # Full label ranges (min/max/mean/std + z-span) to read the stds against.
        label_stats=dict(corpus=corpus_stats, generated=gen_stats),
        within_group_std=dict(E=float(var_E.mean()**0.5), T=float(var_T.mean()**0.5)),
        within_group_std_fired=dict(
            E=float(var_E[fired].mean()**0.5) if fired.any() else float("nan"),
            T=float(var_T[fired].mean()**0.5) if fired.any() else float("nan"),
        ),
        fired_fraction=float(fired.float().mean()),
        floor_zmse=dict(
            E=floor_E, T=floor_T, total=floor_total,
            E_fired=floor_E_fired, T_fired=floor_T_fired,                  # fired var / ALL-corpus var
            E_fired_vs_fired=floor_E_fired_vs_fired,                       # fired var / FIRED-corpus var
            T_fired_vs_fired=floor_T_fired_vs_fired,
            total_fired_vs_fired=floor_total_fired_vs_fired,
        ),
        max_R2=dict(
            E=1 - floor_E, T=1 - floor_T, total=1 - floor_total,
            E_fired_vs_fired=1 - floor_E_fired_vs_fired,
            T_fired_vs_fired=1 - floor_T_fired_vs_fired,
            total_fired_vs_fired=1 - floor_total_fired_vs_fired,
        ),
        STRATEGIES=[(s_name, fn_name, kwargs) for s_name, fn_name, kwargs in _STRATEGIES],
        sigma_spatial=sigma_spatial,
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
    print(f"  floor  E (fired)              = {floor_E_fired:.4f}   [fired var / all-corpus var]")
    print(f"  floor  T (fired)              = {floor_T_fired:.4f}   [fired var / all-corpus var]")
    print(f"  floor  E (fired vs fired)     = {floor_E_fired_vs_fired:.4f}   (max R²|fired = {1-floor_E_fired_vs_fired:.3f})")
    print(f"  floor  T (fired vs fired)     = {floor_T_fired_vs_fired:.4f}   (max R²|fired = {1-floor_T_fired_vs_fired:.3f})")
    print(f"  floor  total (fired vs fired) = {floor_total_fired_vs_fired:.4f}   (max R²|fired = {1-floor_total_fired_vs_fired:.3f})")
    print(f"\n  A surrogate whose val MSE reaches the floor is Bayes-optimal;")
    print(f"  no architecture/optimizer change can go below it.")
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
