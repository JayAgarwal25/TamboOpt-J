"""Shower generation and parametrization functions.

Extracted from SWGOLO7_optimization.ipynb cells 4-6, 15, 22.
"""

import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt


def ReadShowers(path_g, path_p):
    """Read fitted parameter blocks for gamma and proton showers from text files.

    Parameters:
        path_g (str): path to gamma-fit text file.
        path_p (str): path to proton-fit text file.

    Returns:
        tuple: (PXmg_p, PXeg_p, PXmp_p, PXep_p) each a torch.Tensor of shape (4,3,3),
               or None on error.
    """
    def _read_blocks(path, n_blocks, allow_zero_row1=False):
        blocks = []
        for b in range(n_blocks):
            block = np.loadtxt(path, skiprows=b * 3, max_rows=3)
            for i in range(3):
                if block[i, 0] * block[i, 1] * block[i, 2] == 0:
                    if not (allow_zero_row1 and i == 1):
                        warnings.warn("Encountered 0")
                        return None
            blocks.append(block)
        return blocks

    # Gamma: electrons (4 blocks), then muons (4 blocks, allow zero in row 1)
    eg_blocks = _read_blocks(path_g, 4, allow_zero_row1=False)
    if eg_blocks is None:
        return None
    mg_blocks = []
    for b in range(4):
        block = np.loadtxt(path_g, skiprows=(4 + b) * 3, max_rows=3)
        for i in range(3):
            if block[i, 0] * block[i, 1] * block[i, 2] == 0 and i != 1:
                warnings.warn("Encountered 0")
                return None
        mg_blocks.append(block)

    # Proton: electrons (4 blocks), then muons (4 blocks, allow zero in row 1)
    ep_blocks = _read_blocks(path_p, 4, allow_zero_row1=False)
    if ep_blocks is None:
        return None
    mp_blocks = []
    for b in range(4):
        block = np.loadtxt(path_p, skiprows=(4 + b) * 3, max_rows=3)
        for i in range(3):
            if block[i, 0] * block[i, 1] * block[i, 2] == 0 and i != 1:
                warnings.warn("Encountered 0")
                return None
        mp_blocks.append(block)

    PXmg_p = torch.tensor(mg_blocks)
    PXeg_p = torch.tensor(eg_blocks)
    PXmp_p = torch.tensor(mp_blocks)
    PXep_p = torch.tensor(ep_blocks)

    return PXmg_p, PXeg_p, PXmp_p, PXep_p


_cached_stats = {}

def _get_stats(stats_path, plane=20):
    """Load and cache per-plane/per-channel mean/std for denormalization."""
    if stats_path not in _cached_stats:
        stats = torch.load(stats_path, weights_only=True)
        _cached_stats[stats_path] = stats
    stats = _cached_stats[stats_path]
    return stats['mean'][plane], stats['std'][plane]  # each (3,)


def denormalize_shower(images, stats_path, plane=20):
    """Denormalize generator output using per-plane/per-channel statistics. Original data is scaled [-1 1] using mean and std, accessible through _getstats().

    Parameters:
        images (torch.Tensor): standardized images from generator, shape (N, 24, C, H, W).
        stats_path (str): path to standardization_stats_train_only.pt file.
        plane (int): which plane to extract and denormalize.

    Returns:
        torch.Tensor: denormalized image for the given plane, shape (N, H, W, C).
    """
    plane_data = images[:, plane, :, :, :]  # (N, C, H, W)
    plane_data = (plane_data + 1) / 2  # map from [-1, 1] to [0, 1]
    return plane_data.permute(0, 2, 3, 1)  # (N, H, W, C)


def GenerateShowers(x_det, y_det, generate_showers_fn, GetCounts_differentiable_fn,
                    log=False, number_of_showers=1, device='cpu', use_cache=False,
                    output_dir=None):
    """Randomly generate showers with energy, angle, and core position.

    Shower generation is delegated to a ``GenerateShowers`` class instance
    (from ``modules.generate_showers``), which returns point-cloud tensors.
    The downstream logic (core-position estimation, detector counts, logging)
    operates directly on those point clouds.

    Parameters:
        x_det (torch.Tensor): detector x positions.
        y_det (torch.Tensor): detector y positions.
        generate_showers_fn: GenerateShowers instance; called as
            ``generate_showers_fn(num_samples=N, save=False)`` and returns
            ``(samples, energies, directions, labels)`` where
            ``samples`` has shape ``(N, max_points, 5)`` with columns
            ``[x, y, layer_index, energy, time]``.
        GetCounts_differentiable_fn (callable): callable for differentiable
            count extraction; signature ``(samples, x_det, y_det) -> (N, T)``.
        log (bool, optional): if True, plot generated showers. Defaults to False.
        number_of_showers (int, optional): number of showers to generate. Defaults to 1.
        device (str, optional): torch device to move tensors to. Defaults to 'cpu'.
        use_cache (bool, optional): if True, cache generated showers and reuse them
            on subsequent calls. Defaults to False.
        output_dir (str, optional): directory for reading/writing the cache file.
            Defaults to None.

    Returns:
        tuple: (N, T, X0, Y0, energies, directions, labels)
    """

    cache_path = f"{output_dir}/cached_showers_{number_of_showers}.pt"
    if use_cache and output_dir is not None and os.path.exists(cache_path):
        print(f"Loading cached showers from {cache_path}")
        samples, energies, directions, labels = torch.load(cache_path)
    else:
        samples, energies, directions, labels = generate_showers_fn(
            num_samples=number_of_showers, save=False
        )
        samples    = samples.to(device)
        energies   = energies.to(device)
        directions = directions.to(device)
        labels     = labels.to(device)

        if use_cache and output_dir is not None:
            torch.save((samples, energies, directions, labels), cache_path)
            print(f"Saved generated showers to {cache_path}")

    # samples: (N, max_points, 5) — columns: x, y, layer_index, energy, time
    point_x = samples[:, :, 0]  # (N, max_points)
    point_y = samples[:, :, 1]  # (N, max_points)
    point_e = samples[:, :, 3]  # (N, max_points) — particle energy

    # compute energy-weighted shower core position
    weight_sum = point_e.sum(dim=1).clamp(min=1e-8)          # (N,)
    X0 = (point_x * point_e).sum(dim=1) / weight_sum         # (N,)
    Y0 = (point_y * point_e).sum(dim=1) / weight_sum         # (N,)

    # plot showers if logging is enabled
    if log:
        ncols = 5
        nrows = (number_of_showers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = np.atleast_2d(axes).flatten()
        for i in range(number_of_showers):
            sc = axes[i].scatter(
                point_y[i].detach().cpu().numpy(),
                point_x[i].detach().cpu().numpy(),
                c=point_e[i].detach().cpu().numpy(),
                s=2, cmap='viridis',
            )
            fig.colorbar(sc, ax=axes[i], label='energy')
            axes[i].set_xlabel('y [m]')
            axes[i].set_ylabel('x [m]')
            axes[i].set_title(f'Shower {i}')
        for i in range(number_of_showers, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()

    N, T = GetCounts_differentiable_fn(samples, x_det, y_det)

    return N, T, X0, Y0, energies, directions, labels
