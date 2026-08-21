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
(`dataset_builder.compute_labels_batch`) with a fixed layout, and measure the
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

    cd TambOpt
    python plots/training/aleatoric_floor.py --n-prim 128 --m-real 64
"""
import argparse
import json
import os
import sys
import time


# v6 folder = parent of this file's plots/ dir. File-relative (NOT cwd-relative) so
# the script imports modules no matter where it's launched from.
# `_HERE` is this file's own directory; `_V6` the repo root, found by walking up
# to the _pathfix.py marker instead of counting parents. Counting is what broke
# the old plots/single_species/ scripts: written for plots/*.py, they resolved
# the "repo root" to plots/ and could not import `modules` at all.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.data import compute_labels_batch, place_clouds_enu
from modules.layouts.strategies import _STRATEGIES, _STRATEGY_FNS
from modules.showers import load_tau_primaries
from modules.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, SIGMA_SPATIAL,
    TRAINING_DATASET_FOLDER,
    USE_TAU_PRIMARIES, TAU_WHOLESKY_PATH, LOG_E_MIN, LOG_E_MAX,
)
from modules.geometry       import load_tr_mountain
from modules.geometry import SurfaceUpMap

# Generation reuses 00_generate_data_dual_species.py VERBATIM — the SAME per-species
# AllShowers checkpoints + staging (pre_ln injection) + anti-clip re-roll that built
# the dual corpus. So the floor's labels are per-species components from the exact
# generators behind corpus_std: the numerator (generated within-shower variance) and
# the denominator (corpus variance) finally match. (00's filename starts with a digit
# → load it by path; importing only runs its module-level imports/config, not main().)
from modules.showers.generate import GenerateShowers  # noqa: F401 — triggers TAMBO-opt sys.path injection in modules.showers.generate
import importlib.util as _ilu
_spec00 = _ilu.spec_from_file_location(
    "gen00", os.path.join(_V6, "scripts", "00_generate_data_dual_species.py"))
gen00 = _ilu.module_from_spec(_spec00); _spec00.loader.exec_module(gen00)
sample_primary_particles = gen00.sample_primary_particles   # re-export

T_LOG_SCALE = 1.0e8          # must match 02_train_fnn*.py
FIRE_EPS    = 1.0e-3         # log1p(E) above this ⇒ detector "fired" this shower
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mesh path resolution lives in constants.GEOMETRY_PATH_RESOLVED. This script used
# to re-derive it here with a colca_valley.h5 fallback, which is now actively wrong:
# GEOMETRY_GROUP is "malata", so falling back to colca_valley.h5 would raise a
# missing-group KeyError rather than degrade gracefully.

# Must match the NE dataset builder: dataset_builder.build_training_pairs calls
# compute_labels_batch with its DEFAULT sigma (SIGMA_SPATIAL), so use the same
# constant to reproduce the exact training labels (the kernel's transverse
# smoothing length).
sigma_spatial = SIGMA_SPATIAL


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
    # Per-species denominators: species_ids.pt is row-aligned with E/T (0=electron
    # block, 1=muon block, in gen00.SPECIES order), so mask by it for a per-species
    # corpus std — electron and muon have different label spreads.
    species = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt"))
    sp_names = list(gen00.SPECIES)
    E = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    fired = E > FIRE_EPS                                   # E-based fired mask (corpus)
    out = {"E_all": _describe(E)}
    out["E_fired"] = _describe(E[fired]) if fired.any() else None
    out["E_by_species"] = {n: (_describe(E[species == i]) if (species == i).any() else None)
                           for i, n in enumerate(sp_names)}
    del E                                                  # free E; keep `fired` for T
    T = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    T = torch.log1p(T * T_LOG_SCALE)
    out["T_all"]   = _describe(T)
    out["T_fired"] = _describe(T[fired]) if fired.any() else None
    out["T_by_species"] = {n: (_describe(T[species == i]) if (species == i).any() else None)
                           for i, n in enumerate(sp_names)}
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
    if USE_TAU_PRIMARIES:
        # Draw the SAME kind of primaries Step 0 does — real taus carrying a
        # physical ENU decay position, not synthetic direction/energy draws. The
        # position matters here: it is what Step 1 places the cloud at, so the
        # floor's labels only match the corpus if the primaries carry one.
        prim = load_tau_primaries(
            TAU_WHOLESKY_PATH, e_min=10 ** LOG_E_MIN, e_max=10 ** LOG_E_MAX,
            n=n_prim, seed=seed,
        )
        print(f"[gen] real tau primaries from {os.path.basename(TAU_WHOLESKY_PATH)}"
              f"  ({int(prim['energies'].shape[0])} drawn)")
    else:
        prim = sample_primary_particles(                   # corpus ranges (match 00)
            e_min=10 ** gen00.LOG_E_MIN, e_max=10 ** gen00.LOG_E_MAX,
            zenith_min=gen00.ZENITH_MIN, zenith_max=gen00.ZENITH_MAX,
            azimuth_min=gen00.AZIMUTH_MIN, azimuth_max=gen00.AZIMUTH_MAX,
            n=n_prim, seed=seed,
        )
    # load_tau_primaries caps at the number of in-band events, which can be < n_prim.
    n_prim = int(prim["energies"].shape[0])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prim",    type=int, default=128, help="distinct primaries")
    ap.add_argument("--m-real",    type=int, default=64,  help="realizations per primary")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=32, help="AllShowers gen batch")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_V6, f"aleatoric_floor_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    args = ap.parse_args()

    print("=" * 72)
    print("aleatoric floor — within-primary label variance / corpus variance")
    print("=" * 72)
    print(f"device={DEVICE}  tau_primaries={USE_TAU_PRIMARIES}  "
          f"n_prim={args.n_prim}  "
          f"m_real={args.m_real}  strategies={len(_STRATEGIES)}")
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
    # 01_build_dataset_northeast.py / dataset_builder.
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
    # NOT args.n_prim: with real tau primaries the in-band count can be smaller.
    n_prim, m_real = int(_prim["energies"].shape[0]), args.m_real
    P = next(iter(clouds.values())).shape[2]

    # Place each realization exactly as Step 1 does, so the generated labels live
    # in the same frame as the corpus labels that form the z-score denominator.
    # Placement is always the real ENU decay vertices from tau_wholesky.jl; skipping
    # it would leave the clouds in the generator's native frame near the origin, so
    # numerator and corpus denominator would describe different geometries.
    if _prim.get("positions") is None:
        raise RuntimeError(
            "no decay positions on the primaries — placement requires the real ENU "
            "vertices from tau_wholesky.jl (re-run 00_generate_data_dual_species.py)")
    pos  = torch.as_tensor(_prim["positions"],  dtype=torch.float32)    # (n_prim,3)
    dirs = torch.as_tensor(_prim["directions"], dtype=torch.float32)    # (n_prim,3)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    # Every realization of primary p shares that primary's vertex + direction.
    pos_flat  = torch.repeat_interleave(pos,  m_real, dim=0)            # (n_prim*m,3)
    dirs_flat = torch.repeat_interleave(dirs, m_real, dim=0)
    for name in clouds:
        clouds[name] = place_clouds_enu(
            clouds[name].reshape(n_prim * m_real, P, 5), pos_flat, dirs_flat,
            east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
        ).reshape(n_prim, m_real, P, 5)
    print(f"[place] real ENU decay vertices from tau_wholesky.jl at {n_prim} primaries "
          f"(east_entry={EAST_ENTRY:g}, dx={LAYER_EAST_DX:g})")
    print(f"[gen] done in {time.time()-t0:.1f}s  P(max_points)={P}  species={list(clouds)}")

    # For each (species component, strategy, primary): one fixed layout, M
    # realizations → within-var. Both species are pooled so the numerator matches
    # the combined dual corpus (per-species component rows) used for corpus_std.
    rng = np.random.default_rng(args.seed)
    var_E, var_T, fired_frac = [], [], []   # each appends (n_det,) per (species,s,p)
    var_E_sp = {name: [] for name in clouds}   # same, kept separated per species
    var_T_sp = {name: [] for name in clouds}
    gen_E_vals, gen_T_vals = [], []         # all generated labels → value range
    t0 = time.time()
    for s_idx, (s_name, fn_name, kwargs) in enumerate(_STRATEGIES):
        fn = _STRATEGY_FNS[fn_name]
        for sp_name, sp_clouds in clouds.items():   # per species (also pooled below)
            for p in range(n_prim):
                x_det, y_det = fn(mountain, n_det=N_DETECTORS, rng=rng, **kwargs)
                x_det = x_det.float().to(DEVICE); y_det = y_det.float().to(DEVICE)
                cl = sp_clouds[p].to(DEVICE)                   # (M, P, 5)
                E, T = compute_labels_batch(cl, x_det, y_det, surface, sigma_spatial=sigma_spatial)
                E = torch.log1p(E)                             # → training E space
                T = torch.log1p(T * T_LOG_SCALE)               # → training T space
                vE = E.var(dim=0, unbiased=True).cpu()
                vT = T.var(dim=0, unbiased=True).cpu()
                var_E.append(vE);            var_T.append(vT)
                var_E_sp[sp_name].append(vE); var_T_sp[sp_name].append(vT)
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

    # Per-species floors: each species' within-var normalized by THAT species'
    # corpus std, so electron and muon get separate levels (a shared denominator
    # would distort them since their label spreads differ).
    floor_by_species = {}
    for name in clouds:
        vE_sp = torch.cat(var_E_sp[name]); vT_sp = torch.cat(var_T_sp[name])
        es_sp = corpus_stats["E_by_species"][name]["std"]
        ts_sp = corpus_stats["T_by_species"][name]["std"]
        fE_sp, fT_sp = _floor(vE_sp, es_sp), _floor(vT_sp, ts_sp)
        floor_by_species[name] = dict(E=fE_sp, T=fT_sp, total=0.5 * (fE_sp + fT_sp),
                                      corpus_std=dict(E=es_sp, T=ts_sp))

    res = dict(
        n_prim=n_prim, m_real=m_real, n_strategies=len(_STRATEGIES),
        species=list(gen00.SPECIES),
        tau_primaries=USE_TAU_PRIMARIES,
        n_prim_used=n_prim, fire_eps=FIRE_EPS,
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
        floor_zmse_by_species=floor_by_species,
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
    print("\n  per-species floor (each vs its own corpus std):")
    for name, fl in floor_by_species.items():
        print(f"    {name:8s}  E={fl['E']:.4f}  T={fl['T']:.4f}  total={fl['total']:.4f}"
              f"   (max R² total = {1-fl['total']:.3f})")
    print(f"\n  A surrogate whose val MSE reaches the floor is Bayes-optimal;")
    print(f"  no architecture/optimizer change can go below it.")
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
