"""Step-1 dataset builder: layouts, plane-aware labels, training tensors.

Detectors are placed by horizontal map coordinates (North, East) — the ENU
convention of the h5 files — with `surface` giving Up = g(North, East). The
kernel is unchanged: North is still its transverse axis and the defined East its
depth. `encode_primary` / `compute_normalization` are re-exported from
`surrogates.fnn` for convenience.
"""

import os
import time
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch

from ..showers import GetCounts_planeaware

from ..constants import EAST_ENTRY, LAYER_EAST_DX, N_DETECTORS, PRIMARY_DIM, SIGMA_SPATIAL
from ..layouts.strategies import (_STRATEGIES, _STRATEGY_FNS)
from ..surrogates import encode_primary, compute_normalization  # noqa: F401  (re-export)
from ..surrogates.fnn import _load_species_sidecar


def _positions_sidecar_path(shower_cache_path: str) -> str:
    """Path of the Step-0 ENU decay-position sidecar paired with a dual corpus
    .pt (`…_dual.pt` -> `…_dual_positions.pt`). Written by
    00_generate_data_dual_species.py for real-primary (tau) runs; row-aligned
    with the corpus, columns (East, North, Up) in metres."""
    base, ext = os.path.splitext(shower_cache_path)
    return base + "_positions" + ext


def _load_positions_sidecar(shower_cache_path: str, keep_idx):
    """Load the Step-0 ENU decay-position sidecar, indexed by `keep_idx`.

    No synthetic fallback: a missing sidecar is an error, not a silent change of
    geometry."""
    path = _positions_sidecar_path(shower_cache_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"decay-position sidecar not found: {path}\n"
            "Placement requires the real ENU decay vertices from tau_wholesky.jl. "
            "Re-run 00_generate_data_dual_species.py to write the sidecar.")
    return torch.load(path)[torch.as_tensor(keep_idx)].float()          # (N, 3) East,North,Up


# ── Real-position shower placement (the C8 geometry) ─────────────────────────

def place_clouds_enu(clouds:   torch.Tensor,
                     positions: torch.Tensor,
                     dirs:      torch.Tensor,
                     east_entry:    float = EAST_ENTRY,
                     layer_east_dx: float = LAYER_EAST_DX) -> torch.Tensor:
    """Place native AllShowers clouds into site-local ENU at their real decay vertex.

        East = decay_E + x    North = decay_N + y    Up = decay_U + layer*dx*d_Up

    Native cols [x, y, layer_index, energy, time] -> the kernel's (North, Up, z_cont).
    THE placement the pipeline uses; shared with the notebooks and the
    aleatoric-floor script so nobody re-derives the algebra.

    **Cols 0/1 are HORIZONTAL (East, North) offsets, not plane-local transverse
    coordinates** — the longitudinal development is already in them, and adding it
    twice swings showers kilometres off-axis (trigger rate 2% vs 91%). The evidence,
    and the lost-z limitation this inherits from C8, are in docs/THEORY.md §11.4.

    Args:
        clouds    : (M, P, 5) native clouds — MODIFIED IN PLACE and returned.
        positions : (M, 3) decay vertices [East, North, Up] in metres.
        dirs      : (M, 3) unit tau travel directions [East, North, Up].
    """
    pe, pn, pu = (positions[:, i].view(-1, 1) for i in range(3))
    dU = dirs[:, 2].view(-1, 1)

    mask = clouds[:, :, 3] > 0                    # energy-carrying points only
    xh   = clouds[:, :, 0]                        # East  offset from the decay [m]
    yh   = clouds[:, :, 1]                        # North offset from the decay [m]
    s    = clouds[:, :, 2] * layer_east_dx        # depth along the axis [m]

    # Cols 0/1 already contain the longitudinal development — add the vertex only.
    E = pe + xh
    N = pn + yh
    U = pu + s * dU                               # lateral Up absent from the C8 output

    clouds[..., 0] = torch.where(mask, N, clouds[..., 0])
    clouds[..., 1] = torch.where(mask, U, clouds[..., 1])
    clouds[..., 2] = torch.where(mask, (east_entry - E) / layer_east_dx,
                                 clouds[..., 2])
    return clouds


def cloud_to_enu(clouds:        torch.Tensor,
                 east_entry:    float = EAST_ENTRY,
                 layer_east_dx: float = LAYER_EAST_DX):
    """Inverse of `place_clouds_enu`: kernel coords -> ENU points.

    Inverts the kernel's own gauge, ``East = east_entry - z_cont * layer_east_dx``,
    so placed clouds can be drawn in the same frame as the mountain and detectors.
    Zero-energy points are dropped — they were never placed, so their position is
    meaningless rather than merely uninteresting.

    Args:
        clouds : (..., P, 5) placed cloud(s).
    Returns:
        (M, 3) [East, North, Up] for the energy-carrying points.
    """
    c = clouds.detach().cpu().numpy() if isinstance(clouds, torch.Tensor) else np.asarray(clouds)
    c = c.reshape(-1, c.shape[-1])
    m = c[:, 3] > 0
    north, up = c[m, 0], c[m, 1]
    east = east_entry - c[m, 2] * layer_east_dx
    return np.stack([east, north, up], axis=1)


# ── Label computation (batched over showers, one shared layout per batch) ────

@torch.no_grad()
def compute_labels_batch(clouds:   torch.Tensor,
                         e_det:    torch.Tensor,    # East  (defined; detector col 0)
                         n_det:    torch.Tensor,    # North (detector col 1)
                         surface,                    # SurfaceUpMap: (North, East) → Up
                         east_entry:    float = EAST_ENTRY,
                         layer_east_dx: float = LAYER_EAST_DX,
                         sigma_spatial: float = SIGMA_SPATIAL) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the plane-aware kernel on a batch of showers sharing one layout.

    Detectors are (East, North), matching the ENU h5 convention; the kernel is
    unchanged and still reads North as its transverse axis and East as the depth.
    This wrapper only maps the pair onto those roles.

    Args:
        clouds : (B, max_points, 5) AllShowers point clouds.
        e_det  : (n_det,) East  — defined; sets depth / z_cont.
        n_det  : (n_det,) North — the kernel's transverse axis.
    Returns:
        E, T : (B, n_det) local intensities and weighted-average times.
    """
    up     = surface(n_det, e_det)                       # Up = g(North, East)
    z_cont = (east_entry - e_det) / layer_east_dx        # depth from defined East

    dummy_flux = torch.tensor([0.0], device=clouds.device)
    E, T = GetCounts_planeaware(
        clouds, n_det, up, z_cont,                       # transverse plane (North, Up) — kernel unchanged
        SmearN_fn=None,
        fluxB_e=dummy_flux,
        TimeAverage_vectorized_fn=None,
        sigma=sigma_spatial,
    )
    return E, T


# ── Dataset builder stages ───────────────────────────────────────────────────

class _CorpusMeta(NamedTuple):
    """Per-shower metadata for the kept rows; clouds are streamed separately."""
    dirs:      torch.Tensor    # (n_showers, 3) unit tau travel direction E,N,U
    positions: torch.Tensor    # (n_showers, 3) ENU decay vertices [m]
    primaries: torch.Tensor    # (n_showers, PRIMARY_DIM) encoded primaries
    species:   torch.Tensor    # (n_showers,) 0=electron, 1=muon
    n_file:    int             # rows in the corpus file
    per_sp:    int             # file rows per species block
    k_sp:      int             # rows kept per species
    n_showers: int             # 2 * k_sp


def _load_corpus_metadata(mountain, shower_cache_path: str,
                          max_showers: int = 0) -> _CorpusMeta:
    """Read the corpus metadata (no point clouds) and encode the primaries.

    The corpus is two equal species blocks back to back, so `max_showers` keeps
    the first `k_sp` of EACH block, not the first `max_showers` rows.
    """
    import showerdata

    meta   = showerdata.load_inc_particles(shower_cache_path)
    n_file = meta.pdg.shape[0]
    per_sp = n_file // 2                                  # electron / muon block size
    keep   = n_file if not max_showers else min(int(max_showers), n_file)
    k_sp   = keep // 2                                    # rows kept per species

    keep_idx = np.concatenate([np.arange(0, k_sp),
                               np.arange(per_sp, per_sp + k_sp)])
    dirs   = torch.as_tensor(meta.directions[keep_idx], dtype=torch.float32)
    energs = torch.as_tensor(meta.energies[keep_idx],   dtype=torch.float32)
    pdg    = torch.as_tensor(meta.pdg[keep_idx],        dtype=torch.long)
    # Real decay vertices: drive both the placement and the primary's rel_E/N/U.
    positions_all = _load_positions_sidecar(shower_cache_path, keep_idx)  # (N,3) E,N,U
    # Array centre from the mesh, not a constant, so it tracks the geometry.
    array_center = torch.as_tensor(mountain.centroids_ENU,
                                   dtype=torch.float32).mean(dim=0)       # (3,) E,N,U
    primaries_all = encode_primary(dirs, energs, pdg,
                                   positions_all, array_center)  # (n_showers, PRIMARY_DIM)

    # e/µ species per kept shower from the Step-0 sidecar (same keep_idx as the
    # metadata). Corpus `pdg` is the EM/hadronic class, so the Step-2 split keys
    # on this sidecar, not on the pdg feature.
    species_all = _load_species_sidecar(shower_cache_path, keep_idx)   # (N,)

    return _CorpusMeta(dirs=dirs, positions=positions_all, primaries=primaries_all,
                       species=species_all, n_file=n_file, per_sp=per_sp,
                       k_sp=k_sp, n_showers=2 * k_sp)


def _build_chunk_list(k_sp: int, per_sp: int, load_chunk: int, batch_size: int):
    """Flat streaming plan, plus the `load_chunk` actually used.

    Flat and precomputed so resume is just "skip the first N".

    `load_chunk` is rounded down to a whole number of `batch_size` sub-batches
    HERE, before the list is built, and the rounded value is returned so no caller
    can build the list off the raw one: a short final sub-batch changes the number
    of per-chunk layout rng draws, and so changes the dataset.

    Returns:
        load_chunk : rounded chunk size.
        chunk_list : list of (tag, file_start, ds_start, c_lo, c_hi).
    """
    load_chunk = max(int(batch_size), (int(load_chunk) // int(batch_size)) * int(batch_size))
    chunk_list = []
    for tag, file_start, ds_start in (("e", 0, 0), ("mu", per_sp, k_sp)):
        for c_lo in range(0, k_sp, load_chunk):
            chunk_list.append((tag, file_start, ds_start, c_lo, min(c_lo + load_chunk, k_sp)))
    return load_chunk, chunk_list


class _ResumeState:
    """The accumulating out_* tensors plus their resume checkpoint.

    The interval is WALL-CLOCK, not every-N-chunks: a checkpoint is the whole
    out_* set (~11.8 GB at 750k events), so a per-N policy would scale write
    volume with corpus size. See docs/THEORY.md §11.5.

    The layout `rng` is NOT checkpointed, so a resumed run draws fresh layouts
    for the remaining chunks — a valid draw from the same strategies, just not
    bit-identical to an uninterrupted run.
    """

    def __init__(self, n_pairs: int, n_det: int,
                 path: Optional[str] = None,
                 every_s: float = 1800.0,
                 verbose: bool = True):
        self.n_pairs = n_pairs
        self.n_det   = n_det
        self.path    = path
        self.every_s = every_s
        self.verbose = verbose
        self.chunks_done = 0
        self._last_t = time.time()

    def restore(self, n_chunks: int) -> None:
        """Reload a compatible checkpoint, else allocate the out_* tensors fresh."""
        if self.path and os.path.exists(self.path):
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            if ckpt.get("n_pairs") == self.n_pairs:
                self.primary, self.xy = ckpt["out_primary"], ckpt["out_xy"]
                self.E, self.T = ckpt["out_E"], ckpt["out_T"]
                self.strat, self.species = ckpt["out_strat"], ckpt["out_species"]
                self.chunks_done = int(ckpt["chunks_done"])
                if self.verbose:
                    print(f"[resume] {self.path}: {self.chunks_done}/{n_chunks} chunks "
                          "already done")
            elif self.verbose:
                print(f"[resume] {self.path} shape mismatch (n_pairs "
                      f"{ckpt.get('n_pairs')} != {self.n_pairs}) — ignoring, starting fresh")

        if self.chunks_done == 0:
            self.primary = torch.empty((self.n_pairs, PRIMARY_DIM), dtype=torch.float32)
            self.xy      = torch.empty((self.n_pairs, self.n_det, 2), dtype=torch.float32)
            self.E       = torch.empty((self.n_pairs, self.n_det),    dtype=torch.float32)
            self.T       = torch.empty((self.n_pairs, self.n_det),    dtype=torch.float32)
            self.strat   = torch.empty((self.n_pairs,), dtype=torch.int64)
            self.species = torch.empty((self.n_pairs,), dtype=torch.int64)

        self._last_t = time.time()

    def maybe_checkpoint(self, chunk_i: int, n_chunks: int) -> None:
        """Write a checkpoint if the interval has elapsed.

        Skips the final chunk — it is deleted right after the loop anyway.
        """
        if not self.path or (time.time() - self._last_t) < self.every_s:
            return
        if chunk_i + 1 >= n_chunks:
            return
        t_ck = time.time()
        tmp = self.path + ".tmp"
        torch.save({
            "n_pairs": self.n_pairs, "chunks_done": chunk_i + 1,
            "out_primary": self.primary, "out_xy": self.xy,
            "out_E": self.E, "out_T": self.T,
            "out_strat": self.strat, "out_species": self.species,
        }, tmp)
        os.replace(tmp, self.path)
        self._last_t = time.time()
        if self.verbose:
            print(f"[resume] checkpointed {chunk_i + 1}/{n_chunks} chunks "
                  f"in {self._last_t - t_ck:.1f}s -> {self.path}")

    def cleanup(self) -> None:
        """Drop the checkpoint once the build has completed."""
        if self.path and os.path.exists(self.path):
            os.remove(self.path)

    def tensors(self):
        return self.primary, self.xy, self.E, self.T, self.strat, self.species


def _label_chunk(clouds_chunk, meta: _CorpusMeta, state: _ResumeState,
                 mountain, surface, rng, *,
                 ds_start: int, c_lo: int, csz: int,
                 batch_size: int, n_det: int, device) -> None:
    """Label one loaded chunk under every strategy, writing into `state`.

    One layout per (strategy, sub-batch), shared by that sub-batch. The rng is
    threaded in from the caller: draw order is chunk -> strategy -> sub-batch,
    and reordering it changes every label in the dataset.
    """
    n_showers = meta.n_showers
    for s_idx, (s_name, fn_name, kwargs) in enumerate(_STRATEGIES):
        fn = _STRATEGY_FNS[fn_name]
        for sb_lo in range(0, csz, batch_size):
            sb_hi = min(sb_lo + batch_size, csz)
            B = sb_hi - sb_lo

            e_det, n_det_xy = fn(mountain, n_det=n_det, rng=rng, **kwargs)  # (East, North)
            e_det     = e_det.float().to(device)
            n_det_xy  = n_det_xy.float().to(device)

            clouds = clouds_chunk[sb_lo:sb_hi].to(device)
            E, T = compute_labels_batch(clouds, e_det, n_det_xy, surface)
            E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
            T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

            ds_lo = ds_start + c_lo + sb_lo
            ds_hi = ds_start + c_lo + sb_hi
            dst = slice(s_idx * n_showers + ds_lo, s_idx * n_showers + ds_hi)
            state.primary[dst]  = meta.primaries[ds_lo:ds_hi]
            state.xy[dst, :, 0] = e_det.cpu().unsqueeze(0).expand(B, -1)     # East  (col 0)
            state.xy[dst, :, 1] = n_det_xy.cpu().unsqueeze(0).expand(B, -1)  # North (col 1)
            state.E[dst] = E.cpu()
            state.T[dst] = T.cpu()
            state.strat[dst] = s_idx
            state.species[dst] = meta.species[ds_lo:ds_hi]


def _load_clouds(shower_cache_path: str, meta: _CorpusMeta, chunk, east_entry, layer_east_dx):
    """Stream one chunk of point clouds and place it at its real decay vertices.

    Returns the placed clouds and the count of non-finite points zeroed
    (float32 energy overflow in the corpus).
    """
    import showerdata

    _tag, file_start, ds_start, c_lo, c_hi = chunk
    sub = showerdata.load(shower_cache_path,
                          start=file_start + c_lo, stop=file_start + c_hi)
    clouds_chunk = torch.as_tensor(sub.points, dtype=torch.float32)
    del sub

    bad = ~torch.isfinite(clouds_chunk).all(dim=-1)
    n_bad = int(bad.sum())
    if n_bad:
        clouds_chunk[bad] = 0.0

    # One shared implementation, also used by the notebooks and the floor script.
    clouds_chunk = place_clouds_enu(
        clouds_chunk,
        meta.positions[ds_start + c_lo: ds_start + c_hi],   # (csz,3) decay E,N,U
        meta.dirs[ds_start + c_lo: ds_start + c_hi],        # (csz,3) unit dir
        east_entry=east_entry, layer_east_dx=layer_east_dx)
    return clouds_chunk, n_bad


def build_training_pairs(mountain, surface,
                         shower_cache_path: str,
                         batch_size:        int = 20,
                         max_showers:       int = 0,
                         seed:              int = 0,
                         device:            torch.device = torch.device("cpu"),
                         verbose:           bool = True,
                         east_entry:        float = EAST_ENTRY,
                         layer_east_dx:     float = LAYER_EAST_DX,
                         load_chunk:        int = 4096,
                         resume_path:       Optional[str] = None,
                         resume_every_s:    float = 1800.0):
    """Build (primary, xy, E, T) training tensors from the cached shower corpus.

    Runs as four stages — `_load_corpus_metadata`, `_build_chunk_list`, then per
    chunk `_load_clouds` (stream + place at the real ENU decay vertex) and
    `_label_chunk` (strategies × sub-batches) — accumulating into a
    `_ResumeState`, which also owns the gpu_requeue resume checkpoint. Point
    clouds are never loaded whole: peak RAM is one chunk, not the corpus.

    Layouts and labels use the (North, East) convention (`detector_strategies`
    plus `compute_labels_batch` above, with `surface` a SurfaceUpMap).

    Returns:
        primaries : (N_pairs, PRIMARY_DIM) float32
        xy        : (N_pairs, 100, 2) float32   columns = (East, North)
        E         : (N_pairs, 100) float32
        T         : (N_pairs, 100) float32
        strategy_ids : (N_pairs,)  int64 — index into `_STRATEGIES`
        species_ids  : (N_pairs,)  int64 — e/µ component (0=electron, 1=muon)
    """
    meta    = _load_corpus_metadata(mountain, shower_cache_path, max_showers)
    n_strat = len(_STRATEGIES)
    n_det   = N_DETECTORS

    load_chunk, chunk_list = _build_chunk_list(meta.k_sp, meta.per_sp,
                                               load_chunk, batch_size)
    if verbose:
        print(f"[load] streaming {meta.n_showers} rows ({meta.k_sp}/species) of "
              f"{meta.n_file} in chunks of {load_chunk}; peak RAM ≈ one chunk, "
              "not the corpus")

    state = _ResumeState(meta.n_showers * n_strat, n_det,
                         path=resume_path, every_s=resume_every_s, verbose=verbose)
    state.restore(len(chunk_list))

    rng = np.random.default_rng(seed)
    n_sanitized = 0

    for chunk_i, chunk in enumerate(chunk_list):
        if chunk_i < state.chunks_done:
            continue
        tag, _file_start, ds_start, c_lo, c_hi = chunk

        clouds_chunk, n_bad = _load_clouds(shower_cache_path, meta, chunk,
                                           east_entry, layer_east_dx)
        n_sanitized += n_bad

        _label_chunk(clouds_chunk, meta, state, mountain, surface, rng,
                     ds_start=ds_start, c_lo=c_lo, csz=c_hi - c_lo,
                     batch_size=batch_size, n_det=n_det, device=device)

        del clouds_chunk
        if verbose:
            print(f"[build] {tag} rows {c_lo}-{c_hi}/{meta.k_sp} done "
                  f"(×{n_strat} strategies)")
        state.maybe_checkpoint(chunk_i, len(chunk_list))

    state.cleanup()

    if verbose:
        print(f"[place] real ENU decay positions from tau_wholesky.jl "
              f"(N={meta.positions.shape[0]}; East/North offsets + Up along the axis, "
              f"east_entry={east_entry:g}, dx={layer_east_dx:g})")
    if verbose and n_sanitized:
        print(f"[sanitize] zeroed {n_sanitized} non-finite points (float32 energy overflow)")

    return state.tensors()
