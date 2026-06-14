"""(North, East) label computation + dataset builder.

v6-local mirror of `compute_labels_batch` and `build_training_pairs` from
modules_v6/fnn_surrogate.py — diff those two functions against the ones here to
see the convention change. The signatures and bodies are kept identical except:

  - `surface` is a `SurfaceUpMap` (North, East)→Up instead of SurfaceEastMap;
  - the kernel's y-coordinate is the *extrapolated* Up = surface(x_det, y_det),
    while `z_cont` comes directly from the **defined** East (= y_det);
  - layouts come from `detector_strategies_ne` (so `xy = (North, East)`).

`encode_primary` / `compute_normalization` are reused unchanged from
`fnn_surrogate` (re-exported for convenience).
"""

from typing import Tuple

import numpy as np
import torch

from modules_v4.tr_plane_kernel import GetCounts_planeaware

from .constants import EAST_ENTRY, LAYER_EAST_DX, N_DETECTORS, PRIMARY_DIM
from .detector_strategies_ne import (_STRATEGIES, _STRATEGY_FNS)
from .fnn_surrogate import encode_primary, compute_normalization  # noqa: F401  (re-export)


# ── Label computation (batched over showers, one shared layout per batch) ────

@torch.no_grad()
def compute_labels_batch(clouds:   torch.Tensor,
                         x_det:    torch.Tensor,    # North
                         y_det:    torch.Tensor,    # East (defined)
                         surface,                    # SurfaceUpMap: (North, East) → Up
                         east_entry:    float = EAST_ENTRY,
                         layer_east_dx: float = LAYER_EAST_DX,
                         sigma_spatial: float = 200.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run v4's plane-aware kernel on a batch of showers sharing one layout.

    (North, East) convention: `y_det` is the *defined* East, `surface` returns
    the extrapolated height Up, and `z_cont` comes straight from the East.

    Args:
        clouds : (B, max_points, 5) AllShowers point clouds.
        x_det  : (n_det,) North coordinates.
        y_det  : (n_det,) East coordinates (defined).
        surface: `SurfaceUpMap` — differentiable Up = g(North, East).

    Returns:
        E : (B, n_det) local intensities (v4 kernel `local_intensity`).
        T : (B, n_det) weighted-average times (v4 kernel `et`).
    """
    up     = surface(x_det, y_det)                       # extrapolated height (was: east)
    z_cont = (east_entry - y_det) / layer_east_dx        # depth from defined East (was: from east)

    dummy_flux = torch.tensor([0.0], device=clouds.device)
    E, T = GetCounts_planeaware(
        clouds, x_det, up, z_cont,                       # transverse plane (North, Up); was (x_det, y_det)
        SmearN_fn=None,
        fluxB_e=dummy_flux,
        TimeAverage_vectorized_fn=None,
        sigma=sigma_spatial,
    )
    return E, T


def build_training_pairs(mountain, surface,
                         shower_cache_path: str,
                         batch_size:        int = 20,
                         max_showers:       int = 0,
                         seed:              int = 0,
                         device:            torch.device = torch.device("cpu"),
                         verbose:           bool = True,
                         recenter_to_mountain: bool = False):
    """Build (primary, xy, E, T) training tensors from the cached shower corpus.

    Identical to fnn_surrogate.build_training_pairs except layouts/labels use the
    (North, East) convention (detector_strategies_ne + the NE compute_labels_batch
    above, with `surface` a SurfaceUpMap). Stored `xy = (North, East)`. The shower
    recentering is unchanged — the shower transverse plane is still (North, Up).

    Returns:
        primaries : (N_pairs, 5)   float32
        xy        : (N_pairs, 100, 2) float32   columns = (North, East)
        E         : (N_pairs, 100) float32
        T         : (N_pairs, 100) float32
        strategy_ids : (N_pairs,)  int64 — index into `_STRATEGIES`
    """
    import showerdata

    # The dual corpus is two equal species blocks: electron rows [0, per_sp) then
    # muon rows [per_sp, 2*per_sp). Loading ALL points dense is ~501 GB; instead
    # load only the kept prefix of EACH block (max_showers total rows, split
    # evenly so both species stay represented — 02 splits them by pdg). Metadata
    # (dir/energy/pdg) is tiny, so it is read in full first.
    meta   = showerdata.load_inc_particles(shower_cache_path)
    n_file = meta.pdg.shape[0]
    per_sp = n_file // 2
    keep   = n_file if not max_showers else min(int(max_showers), n_file)
    k_sp   = keep // 2                                   # rows kept per species

    # Load each species prefix into a preallocated tensor (peak ≈ kept + 1 block,
    # never the full corpus).
    e = showerdata.load(shower_cache_path, start=0, stop=k_sp)
    P, C = e.points.shape[1], e.points.shape[2]
    points = torch.empty((2 * k_sp, P, C), dtype=torch.float32)   # (2*k_sp, P, 5)
    points[:k_sp] = torch.as_tensor(e.points, dtype=torch.float32)
    del e
    m = showerdata.load(shower_cache_path, start=per_sp, stop=per_sp + k_sp)
    points[k_sp:] = torch.as_tensor(m.points, dtype=torch.float32)
    del m

    keep_idx = np.concatenate([np.arange(0, k_sp),
                               np.arange(per_sp, per_sp + k_sp)])
    dirs   = torch.as_tensor(meta.directions[keep_idx], dtype=torch.float32)  # (2*k_sp, 3)
    energs = torch.as_tensor(meta.energies[keep_idx],   dtype=torch.float32)  # (2*k_sp, 1)
    pdg    = torch.as_tensor(meta.pdg[keep_idx],        dtype=torch.long)     # (2*k_sp,)
    n_showers = points.shape[0]
    if verbose:
        print(f"[load] kept {k_sp}/{per_sp} per species "
              f"-> {n_showers} rows of {n_file} (DATASET_FRACTION applied via max_showers)")

    if recenter_to_mountain:
        # Shower transverse plane is (North, Up) — unchanged from the original.
        mtn_cx = 0.5 * (mountain.n_min + mountain.n_max)
        mtn_cy = 0.5 * (mountain.u_min + mountain.u_max)
        mask    = (points[:, :, 3] > 0).float()                  # (N, P)
        w_sum   = mask.sum(dim=1).clamp(min=1.0)                 # (N,)
        cx = (points[:, :, 0] * mask).sum(dim=1) / w_sum         # (N,)
        cy = (points[:, :, 1] * mask).sum(dim=1) / w_sum
        dx = (mtn_cx - cx).view(-1, 1)                           # (N, 1)
        dy = (mtn_cy - cy).view(-1, 1)
        # points is freshly allocated above (we own it) — recenter in place.
        points[..., 0] = points[..., 0] + dx * mask
        points[..., 1] = points[..., 1] + dy * mask
        if verbose:
            print(f"[recenter] shifted clouds to mountain center "
                  f"({mtn_cx:.1f}, {mtn_cy:.1f})")

    primaries_all = encode_primary(dirs, energs, pdg)   # (N, 5)

    n_strat = len(_STRATEGIES)
    n_pairs = n_showers * n_strat
    n_det   = N_DETECTORS

    out_primary = torch.empty((n_pairs, PRIMARY_DIM), dtype=torch.float32)
    out_xy      = torch.empty((n_pairs, n_det, 2),     dtype=torch.float32)   # (North, East)
    out_E       = torch.empty((n_pairs, n_det),        dtype=torch.float32)
    out_T       = torch.empty((n_pairs, n_det),        dtype=torch.float32)
    out_strat   = torch.empty((n_pairs,),              dtype=torch.int64)

    rng = np.random.default_rng(seed)

    for s_idx, (s_name, fn_name, kwargs) in enumerate(_STRATEGIES):
        fn = _STRATEGY_FNS[fn_name]
        n_batches = (n_showers + batch_size - 1) // batch_size
        if verbose:
            print(f"[build] strategy {s_idx+1}/{n_strat}  {s_name:<18}  "
                  f"{n_batches} batches of {batch_size}")

        for b in range(n_batches):
            lo = b * batch_size
            hi = min(lo + batch_size, n_showers)
            B  = hi - lo

            # Fresh layout for this batch — (North, East)
            x_det, y_det = fn(mountain, n_det=n_det, rng=rng, **kwargs)
            x_det = x_det.float().to(device)
            y_det = y_det.float().to(device)

            # Shared-layout kernel call (batch slice moved to device)
            clouds = points[lo:hi].to(device)              # (B, P, 5)
            E, T = compute_labels_batch(
                clouds, x_det, y_det, surface,
            )

            # Slot into CPU output arrays
            dst = slice(s_idx * n_showers + lo, s_idx * n_showers + hi)
            out_primary[dst] = primaries_all[lo:hi]
            out_xy[dst, :, 0] = x_det.cpu().unsqueeze(0).expand(B, -1)
            out_xy[dst, :, 1] = y_det.cpu().unsqueeze(0).expand(B, -1)
            out_E[dst] = E.cpu()
            out_T[dst] = T.cpu()
            out_strat[dst] = s_idx

    return out_primary, out_xy, out_E, out_T, out_strat
