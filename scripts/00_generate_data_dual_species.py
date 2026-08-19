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
import json
import os
import shutil
import sys
import time

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import torch
import torch._utils  # noqa: F401 — torch 2.x lazy submodule needed by torch.save on Py3.13

torch.set_float32_matmul_precision("high")

import showerdata
import modules_v6  # noqa: F401 — sys.path injection for v3 + v4 (and TAMBO-opt)
from modules_v6.constants import (
    LOG_E_MIN, LOG_E_MAX,
    ZENITH_MIN, ZENITH_MAX, AZIMUTH_MIN, AZIMUTH_MAX,
    SHOWER_CACHE, RUN_LOCATION, NUM_SHOWERS, BATCH_SIZE,
    USE_TAU_PRIMARIES, TAU_WHOLESKY_PATH, DUAL_SHOWER_CACHE_PATH,
    HOLDOUT_FRAC, HOLDOUT_SEED, HELDOUT_SHOWER_CACHE_PATH,
)
from modules_v6.tau_showers import load_tau_primaries

# Low-level generator pieces (importing modules.generate_showers injects TAMBO-opt path).
from modules_v6.legacy_core.generate_showers import GenerateShowers  # noqa: F401 — triggers TAMBO-opt sys.path injection in legacy_core.generate_showers
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
        angles=directions, batch_size=BATCH_SIZE, device=str(DEVICE), labels=labels,
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


def _load_progress(progress_path):
    """{"electron": rows_done, "muon": rows_done}, or {} if no checkpoint yet."""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {}


def _save_progress(progress_path, state):
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, progress_path)


def _generate_corpus(tag, out_path, energies_all, directions_all, labels_all,
                     event_ids_all, positions_all, target_P, chunk_size):
    """Generate one full paired electron+muon corpus at `out_path`, resuming
    automatically from `<out_path>.progress.json` if a prior attempt was
    preempted mid-run (gpu_requeue). `tag` is a log label ("train"/"holdout").

    Progress is tracked per-species (rows written so far); a chunk is only
    counted done AFTER `showerdata.save_batch` returns, so a kill mid-chunk
    just re-does that one chunk on restart — no partial rows on disk get
    silently treated as complete."""
    n_pairs = int(energies_all.shape[0])
    total   = 2 * n_pairs
    progress_path = out_path + ".progress.json"

    if os.path.exists(out_path):
        state = _load_progress(progress_path)
        print(f"[{tag}] resuming into existing {out_path}  progress={state}")
    else:
        state = {}
        showerdata.create_empty_file(out_path, shape=(total, target_P, 5), overwrite=True)
        print(f"[{tag}] preallocated {out_path} ({total}x{target_P}x5, "
              f"≈{total * target_P * 5 * 4 / 1e9:.1f} GB)")

    # e/µ species + real-position sidecars: cheap, derived purely from n_pairs/
    # positions_all, so always safe to (re)write regardless of resume state.
    species_ids = torch.cat([torch.zeros(n_pairs, dtype=torch.int64),
                             torch.ones(n_pairs, dtype=torch.int64)])
    species_path = os.path.splitext(out_path)[0] + "_species.pt"
    torch.save(species_ids, species_path)
    if positions_all is not None:
        positions_dual = torch.cat([positions_all, positions_all], dim=0)
        positions_path = os.path.splitext(out_path)[0] + "_positions.pt"
        torch.save(positions_dual, positions_path)
        print(f"[{tag}/positions] wrote sidecar {positions_path} {tuple(positions_dual.shape)}")
    print(f"[{tag}/species] wrote sidecar {species_path} "
          f"({n_pairs} electron + {n_pairs} muon rows)")

    t0 = time.time()
    for i, (name, cfg) in enumerate(SPECIES.items()):
        block_start = i * n_pairs
        done = int(state.get(name, 0))
        if done >= n_pairs:
            print(f"[{tag}/{name}] block already complete ({done}/{n_pairs}) — skipping")
            continue
        print("=" * 72)
        print(f"[{tag}/{name}] {n_pairs - done} of {n_pairs} showers  "
              f"(max_points={cfg['max_points']}"
              f"{f', resuming at row {done}' if done else ''})")
        print("=" * 72)
        staged_dir, pcfm = stage_run_dir(name, cfg)
        gen = Generator(run_dir=staged_dir, num_timesteps=NUM_TIMESTEPS,
                        compile=True, solver=SOLVER)
        gen.max_points = int(cfg["max_points"])

        while done < n_pairs:
            c = min(chunk_size, n_pairs - done)
            sh = _gen_chunk(
                gen, pcfm, cfg,
                energies_all[done:done + c], directions_all[done:done + c],
                labels_all[done:done + c], event_ids_all[done:done + c], target_P,
            )
            showerdata.save_batch(sh, out_path, start=block_start + done)
            done += c
            state[name] = done
            _save_progress(progress_path, state)
            del sh
            torch.cuda.empty_cache()
            print(f"[{tag}/{name}] wrote {done}/{n_pairs}  "
                  f"(file offset {block_start + done}/{total})  {time.time()-t0:.0f}s")
        del gen
        torch.cuda.empty_cache()

    print(f"[{tag}/done] {total} rows = {n_pairs} paired events "
          f"(electron rows 0..{n_pairs-1}, muon rows {n_pairs}..{total-1}) "
          f"in {time.time()-t0:.0f}s -> {out_path}")
    if os.path.exists(progress_path):
        os.remove(progress_path)


def main():
    ap = argparse.ArgumentParser()
    # Streamed in chunks → peak RAM is one chunk, not the whole corpus, so the
    # pair count can scale freely (disk is the only limit). Muons are capped at
    # 25088 points; the file is preallocated at that P and electrons are padded up.
    ap.add_argument("--n-pairs", type=int, default=0,
                    help="number of paired events BEFORE the holdout split (see "
                         "HOLDOUT_FRAC); 0 = all available (all in-band tau "
                         "events with USE_TAU_PRIMARIES, else NUM_SHOWERS "
                         "synthetic).")
    ap.add_argument("--seed", type=int, default=0,
                    help="primary-sampling seed (deterministic corpus)")
    ap.add_argument("--chunk", type=int, default=CHUNK_SIZE,
                    help="showers per streamed write-batch (bounds peak RAM)")
    ap.add_argument("--out", type=str, default=None,
                    help="MAIN (train-pool) output .pt path (default: "
                         "DUAL_SHOWER_CACHE_PATH from constants). The holdout "
                         "corpus always goes to HELDOUT_SHOWER_CACHE_PATH — it "
                         "isn't overridable, so eval_true_utility.py can always "
                         "find it.")
    args = ap.parse_args()

    os.makedirs(SHOWER_CACHE, exist_ok=True)
    os.makedirs(STAGE_ROOT, exist_ok=True)

    target_P = max(cfg["max_points"] for cfg in SPECIES.values())

    # Sample the primaries ONCE — both species blocks reuse them, so row i and
    # row n_pairs+i are the two components of one physical event.
    if USE_TAU_PRIMARIES:
        # Real tau primaries (energy + direction + physical ENU decay position),
        # filtered to the generator's trained energy band. n_pairs follows the
        # number of in-band events (capped by --n-pairs if given).
        prim = load_tau_primaries(
            TAU_WHOLESKY_PATH, e_min=10**LOG_E_MIN, e_max=10**LOG_E_MAX,
            n=int(args.n_pairs), seed=args.seed,
        )
        positions_all = prim["positions"]                 # (M, 3) ENU (East, North, Up)
    else:
        n_req = int(args.n_pairs) or NUM_SHOWERS
        prim = sample_primary_particles(
            e_min=10**LOG_E_MIN, e_max=10**LOG_E_MAX,
            zenith_min=ZENITH_MIN, zenith_max=ZENITH_MAX,
            azimuth_min=AZIMUTH_MIN, azimuth_max=AZIMUTH_MAX,
            n=n_req, seed=args.seed,
        )
        positions_all = None                              # synthetic: no real position (Step 1 re-centers)

    energies_all, directions_all = prim["energies"], prim["directions"]
    # Per-event EM/hadronic primary class (0/1); both species blocks reuse it so paired rows share the class.
    labels_all = prim["labels"]
    n_total = int(energies_all.shape[0])
    event_ids_all = torch.arange(n_total, dtype=torch.int64)   # global event id (informational)

    # ── Holdout split — PHYSICAL EVENT level, before any generation ──────────
    # Deterministic seeded permutation of the n_total events (independent of
    # --seed, which only drives the EM/hadronic label draw): HOLDOUT_FRAC go to
    # a separate corpus that Steps 1-4 never read, reserved for
    # eval_true_utility.py. Sorting each half keeps rows in file order (cheap,
    # not required for correctness, just tidier logs/debugging).
    g = torch.Generator().manual_seed(HOLDOUT_SEED)
    perm = torch.randperm(n_total, generator=g)
    n_holdout = max(1, int(round(HOLDOUT_FRAC * n_total)))
    holdout_idx = perm[:n_holdout].sort().values
    train_idx   = perm[n_holdout:].sort().values
    n_train     = n_total - n_holdout

    out_path     = args.out or DUAL_SHOWER_CACHE_PATH
    heldout_path = HELDOUT_SHOWER_CACHE_PATH

    print("=" * 72)
    print("v6/00_generate_data_dual_species.py — paired electron + muon corpus (streamed)")
    print("=" * 72)
    print(f"device      : {DEVICE}")
    print(f"primaries   : {f'REAL tau ({os.path.basename(TAU_WHOLESKY_PATH)}, ENU position + direction)' if USE_TAU_PRIMARIES else 'synthetic sample_primary_particles'}")
    if USE_TAU_PRIMARIES:
        print(f"            : {n_total} in-band taus in [1e{LOG_E_MIN:g}, 1e{LOG_E_MAX:g}] GeV "
              f"(East {positions_all[:,0].min():.0f}..{positions_all[:,0].max():.0f}, "
              f"North {positions_all[:,1].min():.0f}..{positions_all[:,1].max():.0f}, "
              f"Up {positions_all[:,2].min():.0f}..{positions_all[:,2].max():.0f} m)")
    print(f"total events: {n_total}  (seed={args.seed})")
    print(f"holdout     : {n_holdout} events ({100*HOLDOUT_FRAC:.1f}%, seed={HOLDOUT_SEED}) "
          f"-> {heldout_path}  [eval_true_utility.py ONLY]")
    print(f"train pool  : {n_train} events -> {out_path}  [Steps 1-4]")
    for name in SPECIES:
        print(f"{name:12s} : max_points={SPECIES[name]['max_points']}")
    print(f"chunk       : {args.chunk}  -> peak RAM ≈ "
          f"{args.chunk * target_P * 5 * 4 / 1e9:.2f} GB/chunk")

    def _slice(idx):
        pos = positions_all[idx] if positions_all is not None else None
        return energies_all[idx], directions_all[idx], labels_all[idx], event_ids_all[idx], pos

    e_tr, d_tr, l_tr, id_tr, pos_tr = _slice(train_idx)
    e_ho, d_ho, l_ho, id_ho, pos_ho = _slice(holdout_idx)

    _generate_corpus("train",   out_path,     e_tr, d_tr, l_tr, id_tr, pos_tr, target_P, args.chunk)
    _generate_corpus("holdout", heldout_path, e_ho, d_ho, l_ho, id_ho, pos_ho, target_P, args.chunk)


if __name__ == "__main__":
    main()
