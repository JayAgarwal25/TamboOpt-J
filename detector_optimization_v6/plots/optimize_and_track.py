"""Optimize the detector layout from the grid through a given recon, and TRACK
both surrogate-U (what the optimizer climbs) and true-U (kernel labels, monitored
on the side) at each checkpoint. Answers: as we climb the surrogate, does the real
utility follow, or is the surrogate a mirage?

Faithful to the production optimizer's core: Adam, lr 1.0, mountain projection each
step, PRIMARIES_PER_STEP random primaries per gradient step (04_optimize_lbfgs).
It is the single Adam-from-grid climb (not the full ensemble + L-BFGS), which is
what this diagnostic question needs.

    python plots/optimize_and_track.py --recon_dir <dir> --steps 5000
"""
import argparse, json, os, sys, time
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np, torch
import modules_v6  # noqa: F401
from modules_v6.opt_core import load_models, utility_of_xy
from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6 import run_world
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX,
    N_PLANES
)

# Bound in main() from the resolved run world, never at import.
TRAINING_DATASET_FOLDER = None
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_etu", os.path.join(_HERE, "plots", "eval_true_utility.py"))
_etu = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_etu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1.0)               # matches 04 ADAM_LR
    ap.add_argument("--primaries-per-step", type=int, default=256) # matches 04
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-events", type=int, default=512)           # fixed monitoring batch
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    run_world.add_run_world_args(ap)
    args = ap.parse_args()

    global TRAINING_DATASET_FOLDER
    W = run_world.resolve(args, need_write=False)
    TRAINING_DATASET_FOLDER = W.dataset_folder
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mtn).to(dev)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev)
    kernel_fnn = _etu.KernelDualLabels(elec, muon, surf, dev)   # true-U labeller (fixed events)
    fnn, recon = load_models(dev, fnn_folder=W.fnn_folder, recon_dir=W.recon_dir)

    prim_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float().to(dev)
    prim_mon = prim_all[:B]                                       # fixed monitoring batch

    # init layout = grid, snapped to the mountain, as learnable (East, North).
    e0, n0 = _etu.grid_layout(mtn)
    e = e0.detach().clone().to(dev).requires_grad_(True)
    n = n0.detach().clone().to(dev).requires_grad_(True)
    opt = torch.optim.Adam([e, n], lr=args.lr)

    print("=" * 72)
    print(f"optimize+track  recon_dir={args.recon_dir}  steps={args.steps}  lr={args.lr}")
    print(f"monitor batch: {B} events (same events, surrogate vs true)")
    print("=" * 72)

    @torch.no_grad()
    def monitor(step):
        sU = utility_of_xy(e, n, prim_mon, fnn, recon)[0].item()
        tU = utility_of_xy(e, n, prim_mon, kernel_fnn, recon)[0].item()
        print(f"[{step:5d}] surrogate_U={sU:+9.3f}   true_U={tU:+9.3f}")
        return {"step": step, "surrogate_U": sU, "true_U": tU}

    traj = [monitor(0)]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, prim_all.shape[0], (args.primaries_per_step,), device=dev)
        U = utility_of_xy(e, n, prim_all[idx], fnn, recon)[0]
        opt.zero_grad(set_to_none=True)
        (-U).backward()
        opt.step()
        with torch.no_grad():   # snap outliers back onto the mountain (as production does)
            e_p, n_p = project_to_mountain_ne(mtn, e.detach().cpu(), n.detach().cpu())
            e.data.copy_(e_p.to(dev)); n.data.copy_(n_p.to(dev))
        if step % args.eval_every == 0 or step == args.steps:
            traj.append(monitor(step))

    out = args.out or os.path.join(os.path.dirname(args.recon_dir or "."),
                                   "optimize_track.json")
    with torch.no_grad():
        layout = torch.stack([e.detach().cpu(), n.detach().cpu()], dim=-1)   # (n_det, 2) E,N
    torch.save(layout, os.path.splitext(out)[0] + "_layout.pt")
    with open(out, "w") as f:
        json.dump({"recon_dir": args.recon_dir, "steps": args.steps, "lr": args.lr,
                   "trajectory": traj}, f, indent=2)
    s0, sN = traj[0]["surrogate_U"], traj[-1]["surrogate_U"]
    t_0, tN = traj[0]["true_U"], traj[-1]["true_U"]
    print(f"\n[done] {time.time()-t0:.0f}s   surrogate_U {s0:+.2f} -> {sN:+.2f}   "
          f"true_U {t_0:+.2f} -> {tN:+.2f}   -> {out}")


if __name__ == "__main__":
    main()
