"""v5 detector layout optimization via (mu+lambda)-ES.

Scientific context
------------------
Jeffrey's question: does a gradient-free evolutionary algorithm converge to
the same optimum as v6's gradient-based L-BFGS/DE?

This script answers that using the SAME frozen dual-species surrogate and the
SAME composite utility U as v6/04_optimize_lbfgs_ensemble.py — the only
difference is the optimizer.  No .backward() is ever called on detector
positions.

Algorithm: (mu+lambda)-ES
--------------------------
  Population : MU=20 complete 100-detector layouts, each a (100, 2) North+Up array.
  Generation :
    1. For each of MU parents, produce LAMBDA=5 offspring:
       - with probability P_CROSSOVER: Hungarian-align with a random other parent
         and swap ~50% of detector pairs, then mutate.
       - otherwise: mutate only.
    2. Evaluate U for all MU + MU*LAMBDA = 120 candidates (forward pass only).
    3. Keep the top MU by U  →  next generation's parents.
  Sigma      : anneals geometrically from 200 m (gen 0) to 20 m (gen 199).
  Termination: 200 generations or 30-generation U plateau.

Multi-restart: 5 independent restarts with seeds 0–4, each starting from MU
freshly sampled random layouts.  The best layout across all restarts is saved
as layout_best.pt for direct comparison with v6's result.

Output layout:
    <RUN_OUTPUT_DIR>/<timestamp>/
        es_config.json          full hyperparameter record
        restart_<k>/
            es_log.json         per-generation {gen, U_best, U_mean, U_std, sigma}
            layout_best.pt      {x, y, U} for this restart's best layout
            U_curve.png         U_best vs generation plot
        summary.json            cross-restart statistics
        layout_best.pt          overall best layout (same format as v6)
        layout_ensemble.png     all restart best-layouts overlaid on mountain

Run:
    cd TambOpt/detector_optimization_v5
    python 01_run_evolution.py [--seed SEED] [--output-dir DIR] [--n-gen N]
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6   = os.path.normpath(os.path.join(_HERE, "..", "detector_optimization_v6"))
for _p in (_HERE, _V6):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import modules_v5   # injects v3 + v4 onto sys.path
import modules_v6   # injects v3 + v4 (idempotent) + makes v6 modules importable

from modules_v6.dual_surrogate  import load_dual_surrogate
from modules_v6.reconstruction  import build_recon_from_ckpt
from modules_v4.tr_geometry     import load_tr_mountain

from modules_v5.constants import (
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    FNN_FOLDER, RECON_FOLDER,
    N_DETECTORS,
    ES_MU, ES_LAMBDA, ES_N_GEN, ES_N_RESTART,
    ES_PLATEAU_TOL, ES_PLATEAU_EPS,
    ES_SIGMA_INIT, ES_SIGMA_FINAL, ES_CROSSOVER_P,
    N_EVAL_PRIMARIES,
    RUN_OUTPUT_DIR,
)
from modules_v5.ev_es_operators import (
    anneal_sigma,
    sample_layout,
    mutate_and_project,
    crossover_layouts,
    evaluate_population,
    sample_primaries,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
    """Load frozen dual-species FNN and reconstruction network from v6 checkpoints."""
    fnn = load_dual_surrogate(FNN_FOLDER, DEVICE)

    recon_path = os.path.join(RECON_FOLDER, "recon.pt")
    recon_ckpt = torch.load(recon_path, map_location=DEVICE, weights_only=False)
    recon = build_recon_from_ckpt(recon_ckpt, N_DETECTORS, DEVICE)
    recon.eval()
    print(
        f"[load] recon  model={recon_ckpt.get('config', {}).get('model_type', 'mlp')}  "
        f"epoch={recon_ckpt.get('epoch', '?')}  val={recon_ckpt.get('val_total', '?')}"
    )
    return fnn, recon


# ── Single-restart ES ─────────────────────────────────────────────────────────

def run_one_restart(
    restart_idx: int,
    mountain,
    fnn,
    recon,
    primary_batch: torch.Tensor,
    base_seed: int,
    n_gen: int,
    out_dir: str,
) -> dict:
    """Run one independent (mu+lambda)-ES restart.

    Args:
        restart_idx  : 0-indexed restart number (used for logging and seeding).
        mountain     : MountainData from load_tr_mountain.
        fnn, recon   : frozen surrogate models on DEVICE.
        primary_batch: (N_EVAL_PRIMARIES, 5) tensor on DEVICE — fixed across all gens.
        base_seed    : integer seed; restart_idx is added so each restart is distinct.
        n_gen        : max generations for this restart.
        out_dir      : directory to write per-restart artifacts.

    Returns:
        dict with keys: best_U, best_xy, history, n_gen_run.
    """
    seed = base_seed + restart_idx
    rng  = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"[restart {restart_idx}]  seed={seed}  max_gen={n_gen}  "
          f"mu={ES_MU}  lambda={ES_LAMBDA}  "
          f"sigma {ES_SIGMA_INIT:.0f}→{ES_SIGMA_FINAL:.0f} m")
    print(f"{'='*72}")

    # ── Initialise parents ────────────────────────────────────────────────────
    parents = [sample_layout(mountain, rng, scheme="random") for _ in range(ES_MU)]

    # ── Evaluate initial population ───────────────────────────────────────────
    fitnesses = evaluate_population(parents, fnn, recon, primary_batch)
    order     = np.argsort(fitnesses)[::-1]
    parents   = [parents[i] for i in order]
    fitnesses = fitnesses[order]

    best_U   = float(fitnesses[0])
    best_xy  = parents[0].copy()
    history  = []
    plateau  = 0

    print(f"[restart {restart_idx}] init  U_best={best_U:+.4f}  "
          f"U_mean={fitnesses.mean():+.4f}  U_std={fitnesses.std():.4f}")

    # ── Generational loop ─────────────────────────────────────────────────────
    for gen in range(n_gen):
        sigma = anneal_sigma(gen, n_gen)
        t0    = time.time()

        # --- produce offspring ---
        offspring = []
        for parent in parents:
            for _ in range(ES_LAMBDA):
                if rng.random() < ES_CROSSOVER_P:
                    other = parents[int(rng.integers(ES_MU))]
                    child = crossover_layouts(parent, other, rng)
                    child = mutate_and_project(child, sigma, mountain, rng)
                else:
                    child = mutate_and_project(parent, sigma, mountain, rng)
                offspring.append(child)

        # --- evaluate all candidates (mu + mu*lambda) ---
        candidates     = parents + offspring                          # 120 total
        cand_fitnesses = evaluate_population(candidates, fnn, recon, primary_batch)

        # --- (mu+lambda) selection: keep top mu ---
        top_idx   = np.argsort(cand_fitnesses)[::-1][:ES_MU]
        parents   = [candidates[i] for i in top_idx]
        fitnesses = cand_fitnesses[top_idx]

        gen_best_U  = float(fitnesses[0])
        gen_mean_U  = float(fitnesses.mean())
        gen_std_U   = float(fitnesses.std())

        # --- plateau tracking ---
        if gen_best_U > best_U + ES_PLATEAU_EPS:
            best_U  = gen_best_U
            best_xy = parents[0].copy()
            plateau = 0
        else:
            plateau += 1

        history.append({
            "gen":          gen,
            "U_best":       gen_best_U,
            "U_mean":       gen_mean_U,
            "U_std":        gen_std_U,
            "sigma":        sigma,
            "plateau":      plateau,
            "elapsed_s":    time.time() - t0,
        })

        if (gen % 20 == 0) or (gen == n_gen - 1):
            print(f"  [r{restart_idx} gen {gen:3d}/{n_gen}]  "
                  f"U_best={gen_best_U:+.4f}  U_mean={gen_mean_U:+.4f}  "
                  f"σ={sigma:.1f}m  plateau={plateau}")

        if plateau >= ES_PLATEAU_TOL:
            print(f"  [r{restart_idx}] plateau at gen {gen} — stopping early")
            break

    # ── Artifacts ─────────────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, "es_log.json")
    with open(log_path, "w") as f:
        json.dump({"restart": restart_idx, "seed": seed, "history": history}, f, indent=2)

    best_path = os.path.join(out_dir, "layout_best.pt")
    torch.save({
        "x": torch.as_tensor(best_xy[:, 0]),
        "y": torch.as_tensor(best_xy[:, 1]),
        "U": best_U,
        "restart": restart_idx,
        "seed": seed,
        "n_gen_run": len(history),
    }, best_path)

    _plot_u_curve(history, restart_idx, os.path.join(out_dir, "U_curve.png"))

    print(f"[restart {restart_idx}] done  best_U={best_U:+.4f}  "
          f"gens_run={len(history)}")
    return {"best_U": best_U, "best_xy": best_xy, "history": history,
            "n_gen_run": len(history), "seed": seed}


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_u_curve(history: list, restart_idx: int, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gens    = [h["gen"]    for h in history]
        u_best  = [h["U_best"] for h in history]
        u_mean  = [h["U_mean"] for h in history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(gens, u_best, label="U_best", linewidth=1.5)
        ax.plot(gens, u_mean, label="U_mean", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Utility U")
        ax.set_title(f"Restart {restart_idx} — (mu+lambda)-ES convergence")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  [plot] {path}")
    except Exception as exc:
        print(f"  [plot] skipped ({exc!r})")


def _plot_ensemble(results: list, mountain, run_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for i, r in enumerate(results):
            xy = r["best_xy"]
            ax.scatter(xy[:, 0], xy[:, 1], s=15, color=colors[i], alpha=0.6,
                       label=f"restart {i}  U={r['best_U']:+.3f}")

        ax.set_xlabel("North [m]")
        ax.set_ylabel("Up [m]")
        ax.set_title("v5 ES — best layout per restart")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        path = os.path.join(run_dir, "layout_ensemble.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] {path}")
    except Exception as exc:
        print(f"[plot] ensemble skipped ({exc!r})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v5 (mu+lambda)-ES detector layout optimizer")
    parser.add_argument("--seed",       type=int, default=0,
                        help="base random seed; restart k uses seed+k")
    parser.add_argument("--output-dir", type=str, default=RUN_OUTPUT_DIR,
                        help="root output directory")
    parser.add_argument("--n-gen",      type=int, default=ES_N_GEN,
                        help="max generations per restart")
    parser.add_argument("--n-restart",  type=int, default=ES_N_RESTART,
                        help="number of independent restarts")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 72)
    print("v5 (mu+lambda)-ES  |  reusing v6 dual-species surrogate")
    print("=" * 72)
    print(f"device      : {DEVICE}")
    print(f"output      : {run_dir}")
    print(f"n_restarts  : {args.n_restart}")
    print(f"n_gen/restart: {args.n_gen}")
    print(f"base_seed   : {args.seed}")
    print(f"mu / lambda : {ES_MU} / {ES_LAMBDA}  (candidates/gen: {ES_MU + ES_MU * ES_LAMBDA})")
    print(f"sigma       : {ES_SIGMA_INIT:.0f} → {ES_SIGMA_FINAL:.0f} m (geometric)")
    print(f"plateau_tol : {ES_PLATEAU_TOL} gens")
    print(f"crossover_p : {ES_CROSSOVER_P}")
    print()

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "base_seed": args.seed, "n_gen": args.n_gen, "n_restart": args.n_restart,
        "ES_MU": ES_MU, "ES_LAMBDA": ES_LAMBDA,
        "ES_SIGMA_INIT": ES_SIGMA_INIT, "ES_SIGMA_FINAL": ES_SIGMA_FINAL,
        "ES_CROSSOVER_P": ES_CROSSOVER_P,
        "ES_PLATEAU_TOL": ES_PLATEAU_TOL, "ES_PLATEAU_EPS": ES_PLATEAU_EPS,
        "N_EVAL_PRIMARIES": N_EVAL_PRIMARIES,
        "FNN_FOLDER": FNN_FOLDER, "RECON_FOLDER": RECON_FOLDER,
        "N_DETECTORS": N_DETECTORS,
        "device": str(DEVICE), "timestamp": timestamp,
    }
    with open(os.path.join(run_dir, "es_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Load geometry ─────────────────────────────────────────────────────────
    print("[load] mountain geometry …")
    t0 = time.time()
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    print(f"  done ({time.time()-t0:.1f}s)  "
          f"N=[{mountain.n_min:.0f}, {mountain.n_max:.0f}]  "
          f"Up=[{mountain.u_min:.0f}, {mountain.u_max:.0f}]")

    # ── Load surrogate ────────────────────────────────────────────────────────
    print("[load] surrogate models …")
    fnn, recon = load_models()

    # ── Fixed primary batch (same across all restarts for comparable U) ───────
    primary_batch = sample_primaries(N_EVAL_PRIMARIES, seed=args.seed).to(DEVICE)
    print(f"[init] primary_batch  shape={tuple(primary_batch.shape)}  device={DEVICE}")

    # ── Run restarts ──────────────────────────────────────────────────────────
    results    = []
    t_total    = time.time()

    for r_idx in range(args.n_restart):
        r_dir   = os.path.join(run_dir, f"restart_{r_idx:02d}")
        result  = run_one_restart(
            restart_idx   = r_idx,
            mountain      = mountain,
            fnn           = fnn,
            recon         = recon,
            primary_batch = primary_batch,
            base_seed     = args.seed,
            n_gen         = args.n_gen,
            out_dir       = r_dir,
        )
        results.append(result)

    # ── Cross-restart summary ─────────────────────────────────────────────────
    all_U     = np.array([r["best_U"] for r in results])
    best_idx  = int(np.argmax(all_U))
    best_r    = results[best_idx]

    summary = {
        "best_U":       float(all_U[best_idx]),
        "best_restart": best_idx,
        "best_seed":    best_r["seed"],
        "mean_U":       float(all_U.mean()),
        "std_U":        float(all_U.std()),
        "all_U":        all_U.tolist(),
        "n_gen_run":    [r["n_gen_run"] for r in results],
        "total_elapsed_s": time.time() - t_total,
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Overall best layout — same format as v6's layout_best.pt for direct comparison.
    best_xy = best_r["best_xy"]
    torch.save({
        "x": torch.as_tensor(best_xy[:, 0]),
        "y": torch.as_tensor(best_xy[:, 1]),
        "U": float(all_U[best_idx]),
        "restart": best_idx,
        "seed": best_r["seed"],
    }, os.path.join(run_dir, "layout_best.pt"))

    _plot_ensemble(results, mountain, run_dir)

    print()
    print("=" * 72)
    print(f"[done]  best_U={all_U[best_idx]:+.4f}  (restart {best_idx})")
    print(f"        mean_U={all_U.mean():+.4f}  std_U={all_U.std():.4f}")
    print(f"        total time: {(time.time()-t_total)/60:.1f} min")
    print(f"        output: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
