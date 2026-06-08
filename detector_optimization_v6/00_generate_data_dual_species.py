"""Generate a DUAL-species (electron + muon) shower corpus from the new best checkpoints.

Sibling of `00_generate_data.py`. The original script used a single AllShowers +
PointCountFM model (sampling both pdg classes internally). The new best
checkpoints (hhanif/tambo_simulations_for_training/best_ckpts) are **per species**
— a separate AllShowers run-dir and PointCountFM `compiled.pt` for electrons and
for muons, with different point-cloud caps (electron 4096, muon 25088). This
script generates each species with its own model pair and writes them into one
cached corpus (electron rows padded up to the muon point cap).

Generation is **streamed in chunks**: the HDF5 file is preallocated once
(`create_empty_file`) and each chunk is written at its row offset
(`save_batch`), so peak RAM is one chunk (`--chunk` showers), not the whole
corpus — counts can scale to disk limits without the save-step OOM that a single
in-RAM tensor would hit at the 25088-point muon cap. Species are written as
contiguous blocks (electron then muon); the training shower-level split
randperms showers, so no global shuffle is needed at generation time.

Two wrinkles handled here that a plain path-swap would miss:
  1. The AllShowers `Generator` loads `weights/best.pt`, but in the new run-dirs
     `weights/` is empty — the trained weights live in `checkpoints/best_epoch_*.pt`.
     We STAGE a usable run-dir (conf.yaml + preprocessing/trafos.pt + weights/best.pt)
     into fast local storage, extracting the flow state_dict robustly.
  2. `max_num_points` is not set in conf (loader would default to 6016, truncating
     muons). We set `generator.max_points` explicitly per species.

PointCountFM (`compiled.pt`) runs on CPU — its TorchScript has device constants
baked at trace time and raises a device-mismatch on CUDA (same reason
`00_generate_data.py`/`compute_aleatoric_floor.py` keep it on CPU). AllShowers
runs on GPU.

Run (heavy — submit via SLURM for production sizes):

    cd TambOpt/detector_optimization_v6
    python 00_generate_data_dual_species.py --n-electron 250000 --n-muon 250000

ASSUMPTIONS TO VERIFY (flagged inline): the generator `label` is sampled 0/1 as
the original pipeline did; the saved `pdg` is set to a SPECIES id (electron=0,
muon=1) so the downstream corpus distinguishes species. Adjust SPECIES below if
the new models expect a different label convention.
"""
import argparse
import glob
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch._utils  # noqa: F401 — torch 2.x lazy submodule needed by torch.save on Py3.13

torch.set_float32_matmul_precision("high")

import showerdata
import modules_v6  # noqa: F401 — sys.path injection for v3 + v4 (and TAMBO-opt)
from modules_v6.constants import SHOWER_CACHE, RUN_LOCATION

# Low-level generator pieces (importing modules.generate_showers injects TAMBO-opt path).
from modules.generate_showers import GenerateShowers  # noqa: F401  (path injection)
from allshowers.generate_showers import (
    sample_primary_particles, run_point_count_fm,
)
from allshowers.generator import Generator, generate

# ── Config ───────────────────────────────────────────────────────────────────
BEST = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/best_ckpts"

# Per-species model paths + point-cloud caps + saved-pdg id.
SPECIES = {
    "electron": dict(
        allshower_run=os.path.join(BEST, "20260519_185649_Electron-Allshower"),
        pcfm_compiled=os.path.join(BEST, "20260521_040716_Electron-PointCountFM", "compiled.pt"),
        max_points=4096,
        pdg=0,                       # saved species id
    ),
    "muon": dict(
        allshower_run=os.path.join(BEST, "20260520_160031_Muons-Allshower"),
        pcfm_compiled=os.path.join(BEST, "20260521_043912_Muon-PointCountFM", "compiled.pt"),
        max_points=4096,   # TODO: capped for now (true muon cap during training ~25088)
        pdg=1,
    ),
}

NUM_TIMESTEPS = 16
SOLVER        = "midpoint"
GEN_BATCH     = 30                  # AllShowers gen batch (memory-bound; 30 matches production)
CHUNK_SIZE    = 2000               # showers per streamed write-batch (bounds peak RAM)
STAGE_ROOT    = os.path.join(RUN_LOCATION, "allshowers_staged")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State-dict wrapper keys to probe when checkpoints/best_epoch_*.pt is not already
# a flat tensor dict (the Generator wants the raw flow state_dict).
_WRAP_KEYS = ("model", "model_state_dict", "state_dict", "ema", "ema_model",
              "flow", "net", "network", "weights")


def _extract_state_dict(obj):
    """Return a flat {name: tensor} state_dict from a loaded checkpoint, whether
    it is already raw or wrapped under a common key."""
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj                                   # already a raw state_dict
    if isinstance(obj, dict):
        for k in _WRAP_KEYS:
            v = obj.get(k)
            if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                print(f"  [stage] extracted state_dict from wrapper key '{k}'")
                return v
        # Last resort: first dict-of-tensors value at the top level.
        for k, v in obj.items():
            if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                print(f"  [stage] extracted state_dict from key '{k}'")
                return v
        raise RuntimeError(
            f"could not find a state_dict inside checkpoint; top-level keys = {list(obj.keys())}")
    raise RuntimeError(f"unexpected checkpoint object type: {type(obj)}")


def stage_run_dir(name, cfg):
    """Build a Generator-loadable run-dir in fast local storage. Idempotent.

    Copies conf.yaml + preprocessing/trafos.pt, extracts the flow state_dict from
    checkpoints/best_epoch_*.pt → weights/best.pt, and copies the PointCountFM
    compiled.pt. Returns (staged_run_dir, staged_pcfm_path)."""
    src = cfg["allshower_run"]
    dst = os.path.join(STAGE_ROOT, name)
    weights_pt = os.path.join(dst, "weights", "best.pt")
    pcfm_dst   = os.path.join(dst, "pcfm_compiled.pt")

    if os.path.exists(weights_pt) and os.path.exists(pcfm_dst) \
            and os.path.exists(os.path.join(dst, "conf.yaml")) \
            and os.path.exists(os.path.join(dst, "preprocessing", "trafos.pt")):
        print(f"[stage] {name}: already staged at {dst}")
        return dst, pcfm_dst

    print(f"[stage] {name}: staging {src} -> {dst}")
    os.makedirs(os.path.join(dst, "weights"), exist_ok=True)
    os.makedirs(os.path.join(dst, "preprocessing"), exist_ok=True)
    shutil.copy2(os.path.join(src, "conf.yaml"), os.path.join(dst, "conf.yaml"))
    shutil.copy2(os.path.join(src, "preprocessing", "trafos.pt"),
                 os.path.join(dst, "preprocessing", "trafos.pt"))

    ckpts = sorted(glob.glob(os.path.join(src, "checkpoints", "best_epoch_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints/best_epoch_*.pt in {src}")
    print(f"[stage] {name}: loading {os.path.basename(ckpts[-1])}")
    raw = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    sd = _extract_state_dict(raw)
    torch.save(sd, weights_pt)
    print(f"[stage] {name}: wrote weights/best.pt ({len(sd)} tensors)")

    shutil.copy2(cfg["pcfm_compiled"], pcfm_dst)
    print(f"[stage] {name}: copied PointCountFM compiled.pt")
    return dst, pcfm_dst


def _pad_points(samples, target_P):
    """Zero-pad a (N, P, 5) tensor up to target_P points (padding rows = 0, which
    the kernel ignores via its energy mask)."""
    N, P, C = samples.shape
    if P == target_P:
        return samples
    out = torch.zeros((N, target_P, C), dtype=samples.dtype)
    out[:, :P, :] = samples
    return out


def _gen_chunk(gen, pcfm, cfg, n, target_P):
    """Generate one chunk of n showers for a species → a showerdata.Showers,
    padded to target_P. Bounded memory (only this chunk is held)."""
    prim = sample_primary_particles(n=n)        
    energies, directions = prim["energies"], prim["directions"]
    # ASSUMPTION: generator label sampled 0/1 as the original pipeline did.
    labels = prim["labels"]

    # Stage 1 — PointCountFM on CPU (TorchScript device-baked → CUDA mismatches).
    num_points = run_point_count_fm(
        model_path=pcfm, energies=energies, directions=directions, labels=labels,
    )
    
    # Stage 2 — AllShowers on GPU (max_points already set on gen).
    samples = generate(
        generator=gen, energies=energies, num_points=num_points,
        angles=directions, batch_size=GEN_BATCH, device=str(DEVICE), labels=labels,
    ).float().cpu()                                        # (n, sp_max_points, 5)
    samples = _pad_points(samples, target_P)
    
    # Saved pdg = species id so the downstream corpus distinguishes electron/muon.
    pdg = torch.full((n,), int(cfg["pdg"]), dtype=torch.int64)
    
    return showerdata.Showers(
        points=samples.numpy(), energies=energies.numpy(),
        pdg=pdg.numpy(), directions=directions.numpy(),
    )


def main():
    ap = argparse.ArgumentParser()
    # Streamed in chunks → peak RAM is one chunk, not the whole corpus, so these
    # counts can scale freely (disk is the only limit). Muons are capped at 25088
    # points; the file is preallocated at that P and electrons are padded up.
    ap.add_argument("--n-electron", type=int, default=2)   # TEMP: smoke-run default
    ap.add_argument("--n-muon",     type=int, default=2)   # TEMP: smoke-run default
    ap.add_argument("--chunk",      type=int, default=CHUNK_SIZE,
                    help="showers per streamed write-batch (bounds peak RAM)")
    ap.add_argument("--out", type=str, default=None,
                    help="output .pt path (default: <SHOWER_CACHE>/cashed_showers_dual_<total>.pt)")
    args = ap.parse_args()

    os.makedirs(SHOWER_CACHE, exist_ok=True)
    os.makedirs(STAGE_ROOT, exist_ok=True)

    counts = {"electron": args.n_electron, "muon": args.n_muon}
    active = [n for n in ("electron", "muon") if counts[n] > 0]
    total = sum(counts[n] for n in active)
    
    out_path = args.out or os.path.join(SHOWER_CACHE, f"cashed_showers_dual_{total}.pt")
    target_P = max(SPECIES[n]["max_points"] for n in active)

    print("=" * 72)
    print("v6/00_generate_data_dual_species.py — electron + muon mixed corpus (streamed)")
    print("=" * 72)
    print(f"device      : {DEVICE}")
    print(f"electrons   : {args.n_electron}  (max_points={SPECIES['electron']['max_points']})")
    print(f"muons       : {args.n_muon}  (max_points={SPECIES['muon']['max_points']})")
    print(f"chunk       : {args.chunk}  -> peak RAM ≈ "
          f"{args.chunk * target_P * 5 * 4 / 1e9:.2f} GB/chunk")
    print(f"output      : {out_path}  (preallocated {total}×{target_P}×5, "
          f"≈{total * target_P * 5 * 4 / 1e9:.1f} GB on disk)")

    t0 = time.time()
    # Preallocate the HDF5 once; write each chunk at its row offset (bounded RAM).
    showerdata.create_empty_file(out_path, shape=(total, target_P, 5), overwrite=True)

    offset = 0
    for i, name in enumerate(active):
        cfg, n = SPECIES[name], counts[name]
        print("=" * 72)
        print(f"[{name}] {n} showers  (max_points={cfg['max_points']}, "
              f"{(n + args.chunk - 1) // args.chunk} chunks)")
        print("=" * 72)
        staged_dir, pcfm = stage_run_dir(name, cfg)
        gen = Generator(run_dir=staged_dir, num_timesteps=NUM_TIMESTEPS,
                        compile=("cuda" in str(DEVICE)), solver=SOLVER)
        gen.max_points = int(cfg["max_points"])

        done = 0
        while done < n:
            c = min(args.chunk, n - done)
            sh = _gen_chunk(gen, pcfm, cfg, c, target_P)
            showerdata.save_batch(sh, out_path, start=offset)
            offset += c
            done += c
            del sh
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{name}] wrote {done}/{n}  (file offset {offset}/{total})  "
                  f"{time.time()-t0:.0f}s")
        del gen
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Species are written as contiguous blocks (electron then muon); the training
    # shower-level split randperms showers, so no global shuffle is needed here.
    print(f"[done] {offset} showers (electron pdg=0, muon pdg=1) in "
          f"{time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
