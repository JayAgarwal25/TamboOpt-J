"""Shower generation and parametrization functions.

Extracted from SWGOLO7_optimization.ipynb cells 4-6, 15, 22.
"""

import warnings
import numpy as np
import torch


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


def GenerateShowers(x, y, generator, scaler, GetCounts_differentiable_fn, SmearN_fn,
                    fluxB_e, log=False, number_of_showers=1, stats_path=None):
    """Randomly generate showers with energy, angle, and core position.

    Parameters:
        x (torch.Tensor): detector x positions.
        y (torch.Tensor): detector y positions.
        generator: PlaneDiffusionEvaluator instance.
        scaler: PlaneFNNGenerator instance.
        GetCounts_differentiable_fn: callable for differentiable count extraction.
        SmearN_fn: callable for detector smearing.
        fluxB_e: background flux tensor.
        log (bool): if True, plot generated showers.
        number_of_showers (int): number of showers to generate.
        stats_path (str): path to standardization stats file for denormalization.

    Returns:
        tuple: (N, T, X0, Y0, energy, sin_z, cos_z, sin_a, cos_a)
    """
    import matplotlib.pyplot as plt

    p_energy = torch.exp(
        torch.rand(number_of_showers) * (torch.log(torch.tensor(1.0)) - torch.log(torch.tensor(1e-5)))
        + torch.log(torch.tensor(1e-5))
    )

    azimuth = torch.rand(number_of_showers) * 2 * torch.pi
    zenith = torch.rand(number_of_showers) * torch.pi

    sin_z = torch.sin(zenith)
    cos_z = torch.cos(zenith)
    sin_a = torch.sin(azimuth)
    cos_a = torch.cos(azimuth)

    class_id = torch.arange(3).repeat((number_of_showers + 2) // 3)[:number_of_showers].float()

    generator.test_conditions = torch.stack([p_energy, class_id, sin_z, cos_z, sin_a, cos_a], dim=1)
    scaler.test_conditions = torch.stack([p_energy, class_id, sin_z, cos_z, sin_a, cos_a], dim=1)

    outputs_arr = generator.generate_samples(num_conditions=number_of_showers, batch_size=5000)
    output_images = outputs_arr['images']
    # if stats_path is not None:
    #     shower_rgb = denormalize_shower(output_images, stats_path, plane=20)
    # else:
    shower_rgb = output_images[:, 20, :, :, :].permute(0, 2, 3, 1)

    outputs_arr_bboxes = scaler.generate_samples(num_conditions=number_of_showers)
    bboxes = outputs_arr_bboxes['bboxes'][:, 20, :]
    location_means = torch.prod(shower_rgb[:, :, :, :2], dim=3)

    if log:
        ncols = 5
        nrows = (number_of_showers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = np.atleast_2d(axes).flatten()
        for i in range(number_of_showers):
            bbox = bboxes[i]
            extent = [bbox[2].item(), bbox[3].item(), bbox[1].item(), bbox[0].item()]
            im = axes[i].imshow(location_means[i], extent=extent)
            fig.colorbar(im, ax=axes[i])
            axes[i].set_xlabel('y [m]')
            axes[i].set_ylabel('x [m]')
            axes[i].set_title(f'Shower {i}')
        for i in range(number_of_showers, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()

        for ch, ch_name in [(0, 'Channel 0'), (1, 'Channel 1'), (2, 'Channel 2')]:
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
            axes = np.atleast_2d(axes).flatten()
            for i in range(number_of_showers):
                bbox = bboxes[i]
                extent = [bbox[2].item(), bbox[3].item(), bbox[1].item(), bbox[0].item()]
                im = axes[i].imshow(shower_rgb[i, :, :, ch], extent=extent)
                fig.colorbar(im, ax=axes[i])
                axes[i].set_xlabel('y [m]')
                axes[i].set_ylabel('x [m]')
                axes[i].set_title(f'Shower {i}')
            for i in range(number_of_showers, len(axes)):
                axes[i].set_visible(False)
            fig.suptitle(ch_name, fontsize=16)
            plt.tight_layout()
            plt.show()

    i_indices = torch.arange(32, dtype=torch.float32)
    j_indices = torch.arange(32, dtype=torch.float32)
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')

    X0 = torch.sum(i_grid * location_means, dim=(1, 2)) / torch.sum(location_means, dim=(1, 2))
    X0 = X0 * (bboxes[:, 1] - bboxes[:, 0]) / 32 + bboxes[:, 0]
    Y0 = torch.sum(j_grid * location_means, dim=(1, 2)) / torch.sum(location_means, dim=(1, 2))
    Y0 = Y0 * (bboxes[:, 3] - bboxes[:, 2]) / 32 + bboxes[:, 2]

    N, T = GetCounts_differentiable_fn(shower_rgb, x, y, bboxes)

    return N, T, X0, Y0, p_energy, sin_z, cos_z, sin_a, cos_a