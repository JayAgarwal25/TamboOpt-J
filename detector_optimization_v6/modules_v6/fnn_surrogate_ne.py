"""(North, East) label computation + dataset builder.

v6-local mirror of `compute_labels_batch` and `build_training_pairs` from
modules_v6/fnn_surrogate.py — diff those two functions against the ones here to
see the convention change. The signatures and bodies are kept identical except:

  - `surface` is a `SurfaceUpMap` (North, East)→Up instead of SurfaceEastMap;
  - the kernel's y-coordinate is the *extrapolated* Up = surface(x_det, y_det),
    while `z_cont` comes directly from the **defined** East (= y_det);
  - layouts come from `detector_strategies_ne` (so `xy = (East, North)`, matching
    the ENU convention of the h5 data files).

`encode_primary` / `compute_normalization` are reused unchanged from
`fnn_surrogate` (re-exported for convenience).
"""

import os
from typing import Tuple

import numpy as np
import torch

from modules_v4.tr_plane_kernel import GetCounts_planeaware

from .constants import EAST_ENTRY, LAYER_EAST_DX, N_DETECTORS, PRIMARY_DIM, SIGMA_SPATIAL
from .detector_strategies_ne import (_STRATEGIES, _STRATEGY_FNS)
from .fnn_surrogate import (encode_primary, compute_normalization,  # noqa: F401  (re-export)
                            _load_species_sidecar)


def _positions_sidecar_path(shower_cache_path: str) -> str:
    """Path of the Step-0 ENU decay-position sidecar paired with a dual corpus
    .pt (`…_dual.pt` -> `…_dual_positions.pt`). Written by
    00_generate_data_dual_species.py for real-primary (tau) runs; row-aligned
    with the corpus, columns (East, North, Up) in metres."""
    base, ext = os.path.splitext(shower_cache_path)
    return base + "_positions" + ext


def _load_positions_sidecar(shower_cache_path: str, keep_idx):
    """Load the Step-0 ENU decay-position sidecar and index it by `keep_idx`, or
    return None if absent (synthetic corpus → Step 1 re-centers to the mountain
    instead of placing at a real position)."""
    path = _positions_sidecar_path(shower_cache_path)
    if not os.path.exists(path):
        return None
    return torch.load(path)[torch.as_tensor(keep_idx)].float()          # (N, 3) East,North,Up


# ── Label computation (batched over showers, one shared layout per batch) ────

@torch.no_grad()
def compute_labels_batch(clouds:   torch.Tensor,
                         e_det:    torch.Tensor,    # East  (defined; detector col 0)
                         n_det:    torch.Tensor,    # North (detector col 1)
                         surface,                    # SurfaceUpMap: (North, East) → Up
                         east_entry:    float = EAST_ENTRY,
                         layer_east_dx: float = LAYER_EAST_DX,
                         sigma_spatial: float = SIGMA_SPATIAL) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run v4's plane-aware kernel on a batch of showers sharing one layout.

    Detectors are (East, North) — matching the ENU h5 convention. The KERNEL is
    unchanged: it still sees North as the transverse coordinate and East as the
    depth (z_cont). This wrapper just maps the (East, North) pair onto those
    physical roles: `surface` extrapolates Up = g(North, East), and z_cont comes
    from the defined East.

    Args:
        clouds : (B, max_points, 5) AllShowers point clouds.
        e_det  : (n_det,) East coordinates (defined; sets depth / z_cont).
        n_det  : (n_det,) North coordinates (kernel transverse axis).
        surface: `SurfaceUpMap` — differentiable Up = g(North, East).

    Returns:
        E : (B, n_det) local intensities (v4 kernel `local_intensity`).
        T : (B, n_det) weighted-average times (v4 kernel `et`).
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


def build_training_pairs(mountain, surface,
                         shower_cache_path: str,
                         batch_size:        int = 20,
                         max_showers:       int = 0,
                         seed:              int = 0,
                         device:            torch.device = torch.device("cpu"),
                         verbose:           bool = True,
                         recenter_to_mountain: bool = False,
                         east_entry:        float = EAST_ENTRY,
                         layer_east_dx:     float = LAYER_EAST_DX,
                         load_chunk:        int = 4096):
    """Build (primary, xy, E, T) training tensors from the cached shower corpus.

    Identical to fnn_surrogate.build_training_pairs except layouts/labels use the
    (North, East) convention (detector_strategies_ne + the NE compute_labels_batch
    above, with `surface` a SurfaceUpMap). Stored `xy = (North, East)`.

    **Shower placement.** If a Step-0 ENU decay-position sidecar
    (`<corpus>_positions.pt`, real tau primaries) is present, each cloud is
    reconstructed in ENU from the C8 shower geometry (decay_locations/
    c8_air_shower.cpp): the native cloud origin (transverse x=y=0, layer 0) is the
    shower start = the tau decay, and the shower develops ALONG the true tau
    direction. For a point with native transverse (x, y) and integer layer L:

        s         = L * layer_east_dx                 # depth along axis (500 m/plane, C8)
        ENU_point = decay + s * d + x * e1 + y * e2

    where d is the tau ENU direction and {e1, e2} span the plane perpendicular to
    it (e1 = (cosθcosφ, cosθsinφ, -sinθ), e2 = (-sinφ, cosφ, 0), matching the C8
    ObservationPlane x/y axes). Each point is then written in kernel coords
    (col0=North, col1=Up, col2 = (east_entry - East)/layer_east_dx); east_entry
    cancels against the detector z_cont, leaving physical relative depth. This
    replaces the earlier translate-only, East-longitudinal simplification — the
    shower is now oriented along the real tau direction. If no position sidecar
    exists, behaviour falls back to `recenter_to_mountain` (synthetic corpus): the
    shower transverse centroid is shifted to the mountain (North, Up) center and
    the depth is left untouched.

    **Bounded-memory streaming.** Point clouds are never loaded whole — only the
    tiny metadata (dir/energy/pdg) is read up front; clouds are streamed in
    chunks of `load_chunk` showers (loaded → used for all strategies → freed).
    See fnn_surrogate.build_training_pairs for full documentation.

    Returns:
        primaries : (N_pairs, 5)   float32
        xy        : (N_pairs, 100, 2) float32   columns = (East, North)
        E         : (N_pairs, 100) float32
        T         : (N_pairs, 100) float32
        strategy_ids : (N_pairs,)  int64 — index into `_STRATEGIES`
        species_ids  : (N_pairs,)  int64 — e/µ component (0=electron, 1=muon)
    """
    import showerdata

    # Metadata only (tiny); dense point clouds are streamed in chunks below.
    meta   = showerdata.load_inc_particles(shower_cache_path)
    n_file = meta.pdg.shape[0]
    per_sp = n_file // 2                                  # electron / muon block size
    keep   = n_file if not max_showers else min(int(max_showers), n_file)
    k_sp   = keep // 2                                    # rows kept per species
    n_showers = 2 * k_sp

    keep_idx = np.concatenate([np.arange(0, k_sp),
                               np.arange(per_sp, per_sp + k_sp)])
    dirs   = torch.as_tensor(meta.directions[keep_idx], dtype=torch.float32)
    energs = torch.as_tensor(meta.energies[keep_idx],   dtype=torch.float32)
    pdg    = torch.as_tensor(meta.pdg[keep_idx],        dtype=torch.long)
    primaries_all = encode_primary(dirs, energs, pdg)    # (n_showers, 5)

    # Real ENU decay positions (tau primaries) if the Step-0 sidecar exists;
    # else None → fall back to the synthetic recenter-to-mountain behaviour.
    positions_all = _load_positions_sidecar(shower_cache_path, keep_idx)  # (N,3) E,N,U or None
    place_real = positions_all is not None

    if recenter_to_mountain and not place_real:
        mtn_cx = 0.5 * (mountain.n_min + mountain.n_max)
        mtn_cy = 0.5 * (mountain.u_min + mountain.u_max)

    # e/µ species per kept shower from the Step-0 sidecar (same keep_idx as the
    # metadata). Corpus `pdg` is the EM/hadronic class, so the Step-2 split keys
    # on this sidecar, not on the pdg feature.
    species_all = _load_species_sidecar(shower_cache_path, keep_idx)   # (N,)

    n_strat = len(_STRATEGIES)
    n_pairs = n_showers * n_strat
    n_det   = N_DETECTORS

    out_primary = torch.empty((n_pairs, PRIMARY_DIM), dtype=torch.float32)
    out_xy      = torch.empty((n_pairs, n_det, 2),     dtype=torch.float32)
    out_E       = torch.empty((n_pairs, n_det),        dtype=torch.float32)
    out_T       = torch.empty((n_pairs, n_det),        dtype=torch.float32)
    out_strat   = torch.empty((n_pairs,),              dtype=torch.int64)
    out_species = torch.empty((n_pairs,),              dtype=torch.int64)

    load_chunk = max(int(batch_size), (int(load_chunk) // int(batch_size)) * int(batch_size))
    if verbose:
        print(f"[load] streaming {n_showers} rows ({k_sp}/species) of {n_file} "
              f"in chunks of {load_chunk}; peak RAM ≈ one chunk, not the corpus")

    rng = np.random.default_rng(seed)
    n_sanitized = 0

    for tag, file_start, ds_start in (("e", 0, 0), ("mu", per_sp, k_sp)):
        for c_lo in range(0, k_sp, load_chunk):
            c_hi = min(c_lo + load_chunk, k_sp)
            csz  = c_hi - c_lo

            sub = showerdata.load(shower_cache_path,
                                  start=file_start + c_lo, stop=file_start + c_hi)
            clouds_chunk = torch.as_tensor(sub.points, dtype=torch.float32)
            del sub

            bad = ~torch.isfinite(clouds_chunk).all(dim=-1)
            nb  = int(bad.sum())
            if nb:
                clouds_chunk[bad] = 0.0
                n_sanitized += nb

            if place_real:
                # Reconstruct each cloud in ENU from the C8 shower geometry
                # (decay_locations/c8_air_shower.cpp): origin at the decay, shower
                # developed ALONG the true tau direction, sampled at planes
                # layer_east_dx (=500 m) apart. See the build_training_pairs docstring.
                pos_chunk = positions_all[ds_start + c_lo: ds_start + c_hi]    # (csz,3) E,N,U decay
                dir_chunk = dirs[ds_start + c_lo: ds_start + c_hi]             # (csz,3) E,N,U unit
                pe = pos_chunk[:, 0].view(-1, 1)                              # decay East
                pn = pos_chunk[:, 1].view(-1, 1)                              # decay North
                pu = pos_chunk[:, 2].view(-1, 1)                              # decay Up
                dE = dir_chunk[:, 0].view(-1, 1)
                dN = dir_chunk[:, 1].view(-1, 1)
                dU = dir_chunk[:, 2].view(-1, 1)
                # Transverse basis in the plane perpendicular to d (the C8
                # ObservationPlane x/y axes from the tau zenith/azimuth):
                #   e1 = (cosθcosφ, cosθsinφ, -sinθ),  e2 = (-sinφ, cosφ, 0).
                theta = torch.arccos(dU.clamp(-1.0, 1.0))
                phi   = torch.atan2(dN, dE)
                st, ct = torch.sin(theta), torch.cos(theta)
                sp, cp = torch.sin(phi),   torch.cos(phi)
                e1E, e1N, e1U = ct * cp, ct * sp, -st                         # (csz,1) each
                e2E, e2N      = -sp, cp                                       # e2U = 0
                mask = clouds_chunk[:, :, 3] > 0                              # (csz,P) bool
                xt = clouds_chunk[:, :, 0]                                    # native transverse x [m]
                yt = clouds_chunk[:, :, 1]                                    # native transverse y [m]
                s  = clouds_chunk[:, :, 2] * layer_east_dx                    # depth along axis [m]
                # ENU position of every point: decay + s*d + xt*e1 + yt*e2.
                E = pe + s * dE + xt * e1E + yt * e2E
                N = pn + s * dN + xt * e1N + yt * e2N
                U = pu + s * dU + xt * e1U                                    # e2 has no Up component
                # Write kernel coords: col0=North, col1=Up, col2 = layer via East depth.
                clouds_chunk[..., 0] = torch.where(mask, N, clouds_chunk[..., 0])
                clouds_chunk[..., 1] = torch.where(mask, U, clouds_chunk[..., 1])
                clouds_chunk[..., 2] = torch.where(mask, (east_entry - E) / layer_east_dx,
                                                   clouds_chunk[..., 2])
            elif recenter_to_mountain:
                mask  = (clouds_chunk[:, :, 3] > 0).float()
                w_sum = mask.sum(dim=1).clamp(min=1.0)
                cx = (clouds_chunk[:, :, 0] * mask).sum(dim=1) / w_sum
                cy = (clouds_chunk[:, :, 1] * mask).sum(dim=1) / w_sum
                dx = (mtn_cx - cx).view(-1, 1)
                dy = (mtn_cy - cy).view(-1, 1)
                clouds_chunk[..., 0] = clouds_chunk[..., 0] + dx * mask
                clouds_chunk[..., 1] = clouds_chunk[..., 1] + dy * mask

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
                    out_primary[dst]  = primaries_all[ds_lo:ds_hi]
                    out_xy[dst, :, 0] = e_det.cpu().unsqueeze(0).expand(B, -1)     # East  (col 0)
                    out_xy[dst, :, 1] = n_det_xy.cpu().unsqueeze(0).expand(B, -1)  # North (col 1)
                    out_E[dst] = E.cpu()
                    out_T[dst] = T.cpu()
                    out_strat[dst] = s_idx
                    out_species[dst] = species_all[ds_lo:ds_hi]

            del clouds_chunk
            if verbose:
                print(f"[build] {tag} rows {c_lo}-{c_hi}/{k_sp} done "
                      f"(×{n_strat} strategies)")

    if verbose:
        if place_real:
            print(f"[place] real ENU decay positions "
                  f"(N={positions_all.shape[0]}; transverse North/Up + East→layer "
                  f"shift, east_entry={east_entry:g}, dx={layer_east_dx:g})")
        elif recenter_to_mountain:
            print(f"[recenter] shifted clouds to mountain center ({mtn_cx:.1f}, {mtn_cy:.1f})")
    if verbose and n_sanitized:
        print(f"[sanitize] zeroed {n_sanitized} non-finite points (float32 energy overflow)")

    return out_primary, out_xy, out_E, out_T, out_strat, out_species
