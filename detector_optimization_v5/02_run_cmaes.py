"""v5b detector layout optimization via sep-CMA-ES + multi-fidelity evaluation.

Scientific context
------------------
Same scientific question as 01_run_evolution.py — gradient-free vs gradient-based —
but using a more principled ES: sep-CMA-ES (diagonal covariance adaptation).

Where (mu+lambda)-ES uses a fixed isotropic mutation (same sigma in every direction),
CMA-ES learns the *shape* of the fitness landscape: after each generation it updates
a covariance matrix that elongates the mutation ellipsoid along directions that
historically produced improvement.  This is the ES analogue of using curvature.

For d=200 the full CMA-ES covariance matrix (200x200) is borderline expensive to
update each generation.  sep-CMA-ES (CMA_diagonal=True in pycma) uses only the
diagonal, which costs O(d) instead of O(d^2) and retains most of the benefit.

Multi-fidelity
--------------
U evaluations are cheap (forward pass through two frozen NNs) but not free, and
noise from a small primary batch dominates when sigma is large anyway.  We use:
  - explore phase (sigma > SIGMA_MF_THRESHOLD):  N_EVAL_PRIMARIES_EXPLORE = 64
  - refine phase  (sigma <= SIGMA_MF_THRESHOLD): N_EVAL_PRIMARIES_REFINE  = 512
Both batches are fixed per-restart (seeded) so U values within each phase are
internally consistent.  The plateau counter resets naturally when switching phases
because the 512-primary U is more accurate and tends to register improvement.

CMA-ES constraint handling
--------------------------
CMA-ES operates in unconstrained R^200.  Each candidate solution (flat 200-D vector)
is reshaped to (100, 2) and projected onto the mountain surface before evaluation.
We tell() CMA-ES with the *original* (unprojected) flat vectors so its internal
covariance model is not distorted by the non-Euclidean projection geometry.  This
is standard practice for constrained ES — the algorithm learns to stay on-mountain
gradually as off-mountain mutations consistently underperform.

Output layout (same as 01_run_evolution.py for direct comparison)
    <RUN_OUTPUT_DIR>/cmaes_<timestamp>/
        cmaes_config.json
        restart_<k>/
            cmaes_log.json
            layout_best.pt
            U_curve.png
        summary.json
        layout_best.pt
        layout_ensemble.png

Run:
    cd TambOpt/detector_optimization_v5
    python 02_run_cmaes.py [--seed SEED] [--output-dir DIR] [--n-gen N]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6   = os.path.normpath(os.path.join(_HERE, "..", "detector_optimization_v6"))
for _p in (_HERE, _V6):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cma
import numpy as np
import torch

import modules_v5
import modules_v6

from modules_v6.dual_surrogate import load_dual_surrogate
from modules_v6.reconstruction  import build_recon_from_ckpt
from modules_v4.tr_geometry     import load_tr_mountain

from modules_v5.constants import (
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    FNN_FOLDER, RECON_FOLDER,
    N_DETECTORS,
    ES_SIGMA_INIT,
    ES_PLATEAU_TOL, ES_PLATEAU_EPS,
    N_EVAL_PRIMARIES_EXPLORE, N_EVAL_PRIMARIES_REFINE, SIGMA_MF_THRESHOLD,
    CMAES_POPSIZE, CMAES_N_GEN, CMAES_N_RESTART,
    RUN_OUTPUT_DIR,
)
from modules_v5.ev_es_operators import (
    project_layout,
    sample_layout,
    evaluate_single_layout,
    sample_primaries,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models():
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


# ── Single-restart CMA-ES ─────────────────────────────────────────────────────

def run_one_restart(
    restart_idx: int,
    mountain,
    fnn,
    recon,
    base_seed: int,
    n_gen: int,
    out_dir: str,
) -> dict:
    seed = base_seed + restart_idx
    rng  = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"[restart {restart_idx}]  seed={seed}  max_gen={n_gen}  "
          f"popsize={CMAES_POPSIZE}  sep-CMA-ES  "
          f"sigma_init={ES_SIGMA_INIT:.0f}m")
    print(f"  multi-fidelity: {N_EVAL_PRIMARIES_EXPLORE} primaries (σ>{SIGMA_MF_THRESHOLD}m) "
          f"→ {N_EVAL_PRIMARIES_REFINE} primaries (σ≤{SIGMA_MF_THRESHOLD}m)")
    print(f"{'='*72}")

    # Fixed primary batches — one per fidelity level, both seeded for reproducibility.
    pb_explore = sample_primaries(N_EVAL_PRIMARIES_EXPLORE, seed=seed).to(DEVICE)
    pb_refine  = sample_primaries(N_EVAL_PRIMARIES_REFINE,  seed=seed).to(DEVICE)

    # Initial layout: random mountain sample, flattened to R^(2*N_DETECTORS).
    x0_layout = sample_layout(mountain, rng, scheme="random")
    x0 = x0_layout.flatten().astype(np.float64)

    opts = cma.CMAOptions()
    opts.set({
        'maxiter':       n_gen,
        'popsize':       CMAES_POPSIZE,
        'CMA_diagonal':  True,   # sep-CMA-ES: diagonal covariance (O(d) update)
        'seed':          seed,
        'verbose':       -9,     # suppress pycma stdout
    })
    es = cma.CMAEvolutionStrategy(x0, float(ES_SIGMA_INIT), opts)

    best_U  = -np.inf
    best_xy = x0_layout.copy()
    history = []
    plateau = 0
    gen     = 0

    while not es.stop() and gen < n_gen:
        t0        = time.time()
        solutions = es.ask()   # list of CMAES_POPSIZE flat float64 vectors

        # Multi-fidelity: cheap batch when exploring (large sigma), accurate when refining.
        primary_batch = pb_explore if es.sigma > SIGMA_MF_THRESHOLD else pb_refine
        n_prim_used   = N_EVAL_PRIMARIES_EXPLORE if es.sigma > SIGMA_MF_THRESHOLD else N_EVAL_PRIMARIES_REFINE

        # Project each solution to the mountain and evaluate U.
        fitnesses    = []
        proj_layouts = []
        for sol in solutions:
            xy_proj = project_layout(sol, mountain)
            u, _    = evaluate_single_layout(xy_proj, fnn, recon, primary_batch)
            fitnesses.append(u)
            proj_layouts.append(xy_proj)

        # Tell CMA-ES with the ORIGINAL (unprojected) solutions — negated because
        # pycma minimizes but we want to maximize U.
        es.tell(solutions, [-u for u in fitnesses])

        gen_best_U = max(fitnesses)
        gen_mean_U = float(np.mean(fitnesses))
        gen_std_U  = float(np.std(fitnesses))

        if gen_best_U > best_U + ES_PLATEAU_EPS:
            best_U  = gen_best_U
            best_xy = proj_layouts[int(np.argmax(fitnesses))].copy()
            plateau = 0
        else:
            plateau += 1

        history.append({
            "gen":        gen,
            "U_best":     gen_best_U,
            "U_mean":     gen_mean_U,
            "U_std":      gen_std_U,
            "sigma":      es.sigma,
            "n_primaries": n_prim_used,
            "plateau":    plateau,
            "elapsed_s":  time.time() - t0,
        })

        if gen % 20 == 0 or gen == n_gen - 1:
            print(f"  [r{restart_idx} gen {gen:3d}/{n_gen}]  "
                  f"U_best={gen_best_U:+.4f}  U_mean={gen_mean_U:+.4f}  "
                  f"σ={es.sigma:.1f}m  n_prim={n_prim_used}  plateau={plateau}")

        if plateau >= ES_PLATEAU_TOL:
            print(f"  [r{restart_idx}] plateau at gen {gen} — stopping early")
            break

        if es.stop():
            print(f"  [r{restart_idx}] CMA-ES stop condition: {es.stop()}")
            break

        gen += 1

    # Re-evaluate the best layout with the full 512-primary batch so the reported
    # U is on the same scale as the (mu+lambda)-ES result (which always used 512).
    best_U_final, _ = evaluate_single_layout(best_xy, fnn, recon, pb_refine)
    print(f"[restart {restart_idx}] done  best_U={best_U_final:+.4f} (re-eval 512 prim)  "
          f"gens_run={len(history)}")

    with open(os.path.join(out_dir, "cmaes_log.json"), "w") as f:
        json.dump({"restart": restart_idx, "seed": seed, "history": history}, f, indent=2)

    torch.save({
        "x": torch.as_tensor(best_xy[:, 0]),
        "y": torch.as_tensor(best_xy[:, 1]),
        "U": best_U_final,
        "restart": restart_idx,
        "seed": seed,
        "n_gen_run": len(history),
    }, os.path.join(out_dir, "layout_best.pt"))

    _plot_u_curve(history, restart_idx, os.path.join(out_dir, "U_curve.png"))

    return {"best_U": best_U_final, "best_xy": best_xy,
            "history": history, "n_gen_run": len(history), "seed": seed}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_u_curve(history, restart_idx, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gens   = [h["gen"]    for h in history]
        u_best = [h["U_best"] for h in history]
        u_mean = [h["U_mean"] for h in history]
        sigmas = [h["sigma"]  for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(gens, u_best, label="U_best", linewidth=1.5)
        ax1.plot(gens, u_mean, label="U_mean", linewidth=1.0, linestyle="--", alpha=0.7)
        ax1.set_ylabel("Utility U")
        ax1.set_title(f"Restart {restart_idx} — sep-CMA-ES + multi-fidelity")
        ax1.legend()
        ax2.semilogy(gens, sigmas, color="tab:orange", linewidth=1.2)
        ax2.axhline(SIGMA_MF_THRESHOLD, color="gray", linestyle=":", linewidth=0.8,
                    label=f"fidelity switch ({SIGMA_MF_THRESHOLD}m)")
        ax2.set_ylabel("σ [m]")
        ax2.set_xlabel("Generation")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  [plot] {path}")
    except Exception as exc:
        print(f"  [plot] skipped ({exc!r})")


def _plot_ensemble(results, mountain, run_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for i, r in enumerate(results):
            xy = r["best_xy"]
            ax.scatter(xy[:, 0], xy[:, 1], s=15, color=colors[i], alpha=0.6,
                       label=f"restart {i}  U={r['best_U']:+.3f}")
        ax.set_xlabel("North [m]")
        ax.set_ylabel("East [m]")
        ax.set_title("v5b sep-CMA-ES — best layout per restart")
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
    parser = argparse.ArgumentParser(description="v5b sep-CMA-ES detector layout optimizer")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=RUN_OUTPUT_DIR)
    parser.add_argument("--n-gen",      type=int, default=CMAES_N_GEN)
    parser.add_argument("--n-restart",  type=int, default=CMAES_N_RESTART)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.output_dir, f"cmaes_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 72)
    print("v5b sep-CMA-ES + multi-fidelity  |  reusing v6 dual-species surrogate")
    print("=" * 72)
    print(f"device       : {DEVICE}")
    print(f"output       : {run_dir}")
    print(f"n_restarts   : {args.n_restart}")
    print(f"n_gen/restart: {args.n_gen}")
    print(f"popsize      : {CMAES_POPSIZE}")
    print(f"sigma_init   : {ES_SIGMA_INIT:.0f} m  (CMA-ES adapts from here)")
    print(f"plateau_tol  : {ES_PLATEAU_TOL} gens")
    print()

    config = {
        "algorithm": "sep-CMA-ES",
        "base_seed": args.seed, "n_gen": args.n_gen, "n_restart": args.n_restart,
        "CMAES_POPSIZE": CMAES_POPSIZE,
        "ES_SIGMA_INIT": ES_SIGMA_INIT,
        "ES_PLATEAU_TOL": ES_PLATEAU_TOL, "ES_PLATEAU_EPS": ES_PLATEAU_EPS,
        "N_EVAL_PRIMARIES_EXPLORE": N_EVAL_PRIMARIES_EXPLORE,
        "N_EVAL_PRIMARIES_REFINE": N_EVAL_PRIMARIES_REFINE,
        "SIGMA_MF_THRESHOLD": SIGMA_MF_THRESHOLD,
        "FNN_FOLDER": FNN_FOLDER, "RECON_FOLDER": RECON_FOLDER,
        "N_DETECTORS": N_DETECTORS,
        "device": str(DEVICE), "timestamp": timestamp,
    }
    with open(os.path.join(run_dir, "cmaes_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("[load] mountain geometry …")
    t0 = time.time()
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    print(f"  done ({time.time()-t0:.1f}s)")

    print("[load] surrogate models …")
    fnn, recon = load_models()

    results  = []
    t_total  = time.time()

    for r_idx in range(args.n_restart):
        r_dir  = os.path.join(run_dir, f"restart_{r_idx:02d}")
        result = run_one_restart(
            restart_idx = r_idx,
            mountain    = mountain,
            fnn         = fnn,
            recon       = recon,
            base_seed   = args.seed,
            n_gen       = args.n_gen,
            out_dir     = r_dir,
        )
        results.append(result)

    all_U    = np.array([r["best_U"] for r in results])
    best_idx = int(np.argmax(all_U))
    best_r   = results[best_idx]

    summary = {
        "algorithm":    "sep-CMA-ES",
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
