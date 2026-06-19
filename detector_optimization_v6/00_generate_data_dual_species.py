"""Generate a paired electron+muon shower corpus using per-species AllShowers checkpoints.

Corpus layout: electron rows [0, N), muon rows [N, 2N); row i and row N+i share
the same primary (same energy, direction, and EM/hadronic label). The corpus `pdg`
field stores the EM/hadronic class (0 or 1) — the generator's conditioning input,
NOT the e/µ species. Species (0=electron, 1=muon) is written to a separate sidecar
`<corpus>_species.pt` for Step-1/2 routing.

Key design decisions:
- Per-species models have different point caps (electron 4096, muon 25088); electrons
  are zero-padded to the muon cap so the file has a uniform shape.
- Generation streams in chunks; the file is preallocated once (`create_empty_file`)
  and each chunk appended at its row offset (`save_batch`) — peak RAM is one chunk.
- `Generator`/`generate` are called directly (not the `GenerateShowers` wrapper)
  because the wrapper hardcodes max_points, staging, and full-corpus RAM return.
- Each per-species staged run-dir has `pre_ln: true` injected into conf.yaml —
  the May checkpoints use pre-LN transformer blocks; loading them into a post-LN
  model silently generates blobs (shared state_dict keys, no error).
- PointCountFM runs on CPU (TorchScript device constants baked at trace time).
- Anti-clip re-roll: if PCFM predicts > cap points, re-roll (up to MAX_PCFM_RETRIES)
  before the expensive GPU generate step to reduce blob artefacts from truncation.
"""
import argparse
import glob
import os
import shutil
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch._utils  # noqa: F401 — torch 2.x lazy submodule needed by torch.save on Py3.13

torch.set_float32_matmul_precision("high")

import showerdata
import modules_v6  # noqa: F401 — sys.path injection for v3 + v4 (and TAMBO-opt)
from modules_v6.constants import (
    LOG_E_MIN, LOG_E_MAX,
    ZENITH_MIN, ZENITH_MAX, AZIMUTH_MIN, AZIMUTH_MAX,
    SHOWER_CACHE, RUN_LOCATION, NUM_SHOWERS,
)

# Low-level generator pieces (importing modules.generate_showers injects TAMBO-opt path).
from modules.generate_showers import GenerateShowers  # noqa: F401  (path injection)
from allshowers.generate_showers import (
    sample_primary_particles, run_point_count_fm,
)
from allshowers.generator import Generator, generate

# ── Config ───────────────────────────────────────────────────────────────────
BEST = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/checkpoints"

# Per-species model paths + point-cloud caps. The species id (electron=0,
# muon=1) is the BLOCK INDEX in this dict (electron first, muon second) and is
# written to the Step-0 species sidecar — it is no longer stored in the corpus
# `pdg` field (which now carries the EM/hadronic class fed to the generator).
SPECIES = {
    "electron": dict(
        allshower_run=os.path.join(BEST, "20260519_185649_Electron-Allshower"),
        pcfm_compiled=os.path.join(BEST, "20260521_040716_Electron-PointCountFM", "compiled.pt"),
        max_points=4096,
    ),
    "muon": dict(
        allshower_run=os.path.join(BEST, "20260520_160031_Muons-Allshower"),
        pcfm_compiled=os.path.join(BEST, "20260521_043912_Muon-PointCountFM", "compiled.pt"),
        max_points=25088,
    ),
}

NUM_TIMESTEPS = 16
SOLVER        = "midpoint"
GEN_BATCH     = 30                  # AllShowers gen batch (memory-bound; 30 matches production)
CHUNK_SIZE    = 2000               # showers per streamed write-batch (bounds peak RAM)
STAGE_ROOT    = os.path.join(RUN_LOCATION, "allshowers_staged")
DEVICE        = torch.device("cuda")

# Anti-clip resampling (mainly muons). PointCountFM predicts a per-shower total
# point count; if it exceeds the species cap (`max_points`, 25088 for muons) the
# generator TRUNCATES the tail — losing points turns a rod into a diffuse blob.
# Because the clip is decided from num_points BEFORE the expensive AllShowers
# generate, we cheaply re-roll the (stochastic) PointCountFM for over-cap showers.
# Each re-roll replaces the previous draw, so a shower that comes back under the
# threshold keeps that draw and stops; one that stays over cap for the whole
# budget simply keeps its LAST re-roll and is truncated as before. MAX_CLIP_FRAC
# = tolerable fraction of predicted points lost to the cap (0.0 = re-roll on any
# clipping). Set MAX_PCFM_RETRIES = 0 to disable.
MAX_CLIP_FRAC    = 0.10
MAX_PCFM_RETRIES = 10

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

    os.makedirs(os.path.join(dst, "weights"), exist_ok=True)
    os.makedirs(os.path.join(dst, "preprocessing"), exist_ok=True)

    # conf.yaml is ALWAYS rewritten (even when already staged) so older staged
    # dirs pick up the injection: these May checkpoints were trained with
    # pre-LN transformer blocks (verified 2026-06-10 — post-LN loads silently
    # but generates blobs), so the staged conf requests pre_ln explicitly.
    with open(os.path.join(src, "conf.yaml")) as f:
        conf = yaml.safe_load(f)
    conf["model"]["pre_ln"] = True
    with open(os.path.join(dst, "conf.yaml"), "w") as f:
        yaml.safe_dump(conf, f, sort_keys=False)

    if os.path.exists(weights_pt) and os.path.exists(pcfm_dst) \
            and os.path.exists(os.path.join(dst, "preprocessing", "trafos.pt")):
        print(f"[stage] {name}: already staged at {dst} (conf.yaml re-patched)")
        return dst, pcfm_dst

    print(f"[stage] {name}: staging {src} -> {dst}")
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


def resample_overclip(pcfm, energies, directions, labels, num_points, cap,
                      max_clip_frac=MAX_CLIP_FRAC, max_retries=MAX_PCFM_RETRIES):
    """Re-roll PointCountFM for showers whose predicted total point count would be
    clipped by more than `max_clip_frac` at `cap` — truncation turns a rod into a
    diffuse blob. PointCountFM samples noise per call, so re-running it for the
    over-cap subset yields fresh counts; each re-roll REPLACES the previous draw
    (several retries → keep the last). Only the failed subset is re-rolled, and
    only the cheap CPU stage is touched (the GPU generate runs once afterward).
    Mutates and returns `num_points`. No-op when max_retries <= 0.

    Shared by the generation pipeline (`_gen_chunk`) and the angle-grid plots so
    both apply the identical anti-clip policy."""
    if max_retries <= 0:
        return num_points

    cap = int(cap)
    n = int(num_points.shape[0])

    def _clip_frac(npts):
        totals = npts.sum(1).to(torch.float32)             # (m,) predicted total
        return (totals - cap).clamp(min=0.0) / totals.clamp(min=1.0)

    clip_frac = _clip_frac(num_points)
    for attempt in range(1, max_retries + 1):
        bad = clip_frac > max_clip_frac
        nbad = int(bad.sum())
        if nbad == 0:
            break
        idx = torch.nonzero(bad, as_tuple=False).flatten()
        print(f"  [anti-clip {attempt}/{max_retries}] re-rolling PointCountFM "
              f"for {nbad}/{n} shower(s) clipping >{max_clip_frac:.0%} (cap {cap})")
        new_np = run_point_count_fm(
            model_path=pcfm, energies=energies[idx], directions=directions[idx],
            labels=labels[idx],
        )
        num_points[idx] = new_np                           # keep the latest draw
        clip_frac = _clip_frac(num_points)

    # Report only showers still ABOVE the re-roll threshold after the budget
    # (these were actually re-rolled and kept their last draw). Showers over the
    # cap but within max_clip_frac were intentionally never re-rolled — that
    # truncation is tolerated, so don't flag it as a retry failure.
    still = int((clip_frac > max_clip_frac).sum())
    if still:
        print(f"  [anti-clip] {still}/{n} shower(s) still clip >{max_clip_frac:.0%} "
              f"(cap {cap}) after {max_retries} retries — truncated (kept last draw)")
    return num_points


def _gen_chunk(gen, pcfm, cfg, energies, directions, labels, shower_ids, target_P,
               max_clip_frac=MAX_CLIP_FRAC, max_retries=MAX_PCFM_RETRIES):
    """Generate one chunk of showers for a species from PRE-SAMPLED primaries
    → a showerdata.Showers, padded to target_P. Bounded memory (only this
    chunk is held). Primaries (energies, directions, labels, shower_ids) come in
    as slices of the corpus-wide arrays so both species blocks share them (paired
    events).

    `labels` is the per-event EM/hadronic primary class (0/1) — the generator's
    conditioning input (both per-species models were trained on both classes)
    and the value stored as the corpus `pdg` field. `shower_ids` is the
    paired-event id, so the electron row e and muon row n_pairs+e share it."""
    labels = labels.to(torch.int64)

    # Stage 1 — PointCountFM on CPU (TorchScript device-baked → CUDA mismatches).
    num_points = run_point_count_fm(
        model_path=pcfm, energies=energies, directions=directions, labels=labels,
    )

    # Anti-clip re-roll for over-cap showers (mainly muons): re-roll only the
    # failed subset on the cheap CPU stage so the single GPU generate below sees
    # counts that mostly fit the cap (truncation → blob). See resample_overclip.
    num_points = resample_overclip(
        pcfm, energies, directions, labels, num_points,
        cap=int(cfg["max_points"]), max_clip_frac=max_clip_frac, max_retries=max_retries,
    )

    # Stage 2 — AllShowers on GPU (max_points already set on gen).
    samples = generate(
        generator=gen, energies=energies, num_points=num_points,
        angles=directions, batch_size=GEN_BATCH, device=str(DEVICE), labels=labels,
    ).float().cpu()                                        # (n, sp_max_points, 5)
    samples = _pad_points(samples, target_P)

    # Underflow guard: the inverse energy trafo (exp of a latent) can emit
    # EXACTLY 0.0 for extreme negative latents (float32 underflow, ~1-in-1e8
    # per point — guaranteed at production corpus sizes). showerdata requires
    # real points contiguous at the front: its ragged save slices
    # [:num_points] (an interior zero silently drops the last real point) and
    # a zero at slot 0 raises "Padding should be in the end". Stable-partition
    # each shower: e>0 rows first in original order, zero rows (including the
    # padding) at the end.
    key = (samples[:, :, 3] <= 0).to(torch.int8)           # 0 = real, 1 = zero/pad
    order = torch.argsort(key, dim=1, stable=True)         # (n, target_P)
    samples = torch.gather(
        samples, 1, order.unsqueeze(-1).expand(-1, -1, samples.shape[2]))
    
    # Saved pdg = the EM/hadronic primary class (the generator's conditioning
    # label). The e/µ species is recorded in the Step-0 species sidecar, not here.
    pdg = labels

    return showerdata.Showers(
        points=samples.numpy(), energies=energies.numpy(),
        pdg=pdg.numpy(), directions=directions.numpy(),
        shower_ids=shower_ids.numpy(),
    )


def main():
    ap = argparse.ArgumentParser()
    # Streamed in chunks → peak RAM is one chunk, not the whole corpus, so the
    # pair count can scale freely (disk is the only limit). Muons are capped at
    # 25088 points; the file is preallocated at that P and electrons are padded up.
    ap.add_argument("--n-pairs", type=int, default=NUM_SHOWERS,
                    help="number of paired events; the corpus holds 2*n_pairs rows "
                         "(electron block rows 0..N-1, muon block rows N..2N-1, "
                         "row i and row N+i share the same primary)")
    ap.add_argument("--seed", type=int, default=0,
                    help="primary-sampling seed (deterministic corpus)")
    ap.add_argument("--chunk", type=int, default=CHUNK_SIZE,
                    help="showers per streamed write-batch (bounds peak RAM)")
    ap.add_argument("--out", type=str, default=None,
                    help="output .pt path (default: <SHOWER_CACHE>/cashed_showers_dual_<2*n_pairs>.pt)")
    ap.add_argument("--resume-at-row", type=int, default=0,
                    help="continue a crashed run into the EXISTING output file, "
                         "skipping rows < this (use the last logged 'file offset'). "
                         "--n-pairs/--seed must match the original run so the "
                         "regenerated primaries pair with the rows already on disk.")
    args = ap.parse_args()

    os.makedirs(SHOWER_CACHE, exist_ok=True)
    os.makedirs(STAGE_ROOT, exist_ok=True)

    n_pairs = int(args.n_pairs)
    total   = 2 * n_pairs

    out_path = args.out or os.path.join(SHOWER_CACHE, f"cashed_showers_dual_{total}.pt")
    target_P = max(cfg["max_points"] for cfg in SPECIES.values())

    # Sample the primaries ONCE — both species blocks reuse them, so row i and
    # row n_pairs+i are the two components of one physical event.
    prim = sample_primary_particles(
        e_min=10**LOG_E_MIN, e_max=10**LOG_E_MAX, 
        zenith_min=ZENITH_MIN, zenith_max=ZENITH_MAX,
        azimuth_min=AZIMUTH_MIN, azimuth_max=AZIMUTH_MAX,
        n=n_pairs, seed=args.seed
        )
    energies_all, directions_all = prim["energies"], prim["directions"]
    # Per-event EM/hadronic primary class (0/1), randomly sampled by
    # sample_primary_particles. Fed to BOTH generator stages and stored as the
    # corpus `pdg`; both species blocks reuse it so paired rows share the class.
    labels_all = prim["labels"]
    # Paired-event ids: the electron row e and the muon row n_pairs+e are the two
    # components of one physical event, so both blocks reuse this arange and get
    # the SAME shower_id (mirrors the matched per-species training files).
    event_ids_all = torch.arange(n_pairs, dtype=torch.int64)

    print("=" * 72)
    print("v6/00_generate_data_dual_species.py — paired electron + muon corpus (streamed)")
    print("=" * 72)
    print(f"device      : {DEVICE}")
    print(f"pairs       : {n_pairs} events -> {total} rows (seed={args.seed})")
    for name in SPECIES:
        print(f"{name:12s} : max_points={SPECIES[name]['max_points']}")
    print(f"chunk       : {args.chunk}  -> peak RAM ≈ "
          f"{args.chunk * target_P * 5 * 4 / 1e9:.2f} GB/chunk")
    print(f"output      : {out_path}  (preallocated {total}×{target_P}×5, "
          f"≈{total * target_P * 5 * 4 / 1e9:.1f} GB on disk)")

    t0 = time.time()
    resume = int(args.resume_at_row)
    if resume > 0:
        # Continue into the existing preallocated file. Primaries are seeded,
        # so the regenerated slices pair exactly with the rows already on disk
        # (as long as --n-pairs/--seed match the original run).
        if not (0 < resume < total):
            raise SystemExit(f"--resume-at-row {resume} outside (0, {total})")
        if not os.path.exists(out_path):
            raise SystemExit(f"--resume-at-row given but {out_path} does not exist")
        print(f"[resume] continuing into existing file from row {resume}/{total}")
    else:
        # Preallocate the HDF5 once; each chunk is written at its row offset.
        showerdata.create_empty_file(out_path, shape=(total, target_P, 5), overwrite=True)

    # e/µ species sidecar (the "tag"): electron block rows [0, n_pairs) = 0,
    # muon block rows [n_pairs, 2*n_pairs) = 1. Fully determined by n_pairs, so
    # written unconditionally (resume-safe); the corpus `pdg` now holds the
    # EM/hadronic class instead. Row-aligned with the corpus.
    species_ids = torch.cat([
        torch.zeros(n_pairs, dtype=torch.int64),
        torch.ones(n_pairs, dtype=torch.int64),
    ])
    # Derived from out_path (not the constant) so a custom --out keeps the
    # sidecar paired — the Step-1 builders apply the same `<corpus>_species.pt`
    # rule to whatever corpus path they are pointed at.
    species_path = os.path.splitext(out_path)[0] + "_species.pt"
    torch.save(species_ids, species_path)
    print(f"[species] wrote sidecar {species_path} "
          f"({n_pairs} electron + {n_pairs} muon rows)")

    for i, (name, cfg) in enumerate(SPECIES.items()):
        block_start = i * n_pairs              # this species' rows: [block_start, block_start + n_pairs)
        done = min(max(resume - block_start, 0), n_pairs)   # rows of this block already on disk
        if done >= n_pairs:
            print(f"[{name}] block already on disk "
                  f"(rows {block_start}..{block_start + n_pairs - 1}) — skipping")
            continue
        print("=" * 72)
        print(f"[{name}] {n_pairs - done} of {n_pairs} showers  "
              f"(max_points={cfg['max_points']}, "
              f"{(n_pairs - done + args.chunk - 1) // args.chunk} chunks"
              f"{f', resuming at block row {done}' if done else ''})")
        print("=" * 72)
        staged_dir, pcfm = stage_run_dir(name, cfg)
        gen = Generator(run_dir=staged_dir, num_timesteps=NUM_TIMESTEPS,
                        compile=True, solver=SOLVER)
        gen.max_points = int(cfg["max_points"])

        while done < n_pairs:
            c = min(args.chunk, n_pairs - done)
            sh = _gen_chunk(
                gen, pcfm, cfg,
                energies_all[done:done + c], directions_all[done:done + c],
                labels_all[done:done + c],
                event_ids_all[done:done + c],
                target_P,
            )
            showerdata.save_batch(sh, out_path, start=block_start + done)
            done += c
            del sh
            torch.cuda.empty_cache()
            print(f"[{name}] wrote {done}/{n_pairs}  "
                  f"(file offset {block_start + done}/{total})  "
                  f"{time.time()-t0:.0f}s")
        del gen
        torch.cuda.empty_cache()

    # Species are contiguous blocks sharing primaries (electron then muon); the
    # training shower-level split randperms showers, so no global shuffle here.
    print(f"[done] {total} rows = {n_pairs} paired events "
          f"(electron rows 0..{n_pairs-1}, muon rows {n_pairs}..{total-1}; "
          f"corpus pdg = EM/hadronic class, e/µ species in {species_path}) "
          f"in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
