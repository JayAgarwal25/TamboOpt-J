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


def GenerateShowers(x, y, generator, scaler, GetCounts_differentiable_fn, SmearN_fn,
                    fluxB_e, log=False, number_of_showers=1):
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
    shower_rgb = output_images[:, 20, :, :, :]
    shower_rgb = shower_rgb.permute(0, 2, 3, 1)

    outputs_arr_bboxes = scaler.generate_samples(num_conditions=number_of_showers)
    bboxes = outputs_arr_bboxes['bboxes'][:, 20, :]

    if log:
        for i in range(number_of_showers):
            plt.figure()
            plt.imshow(shower_rgb[i])
            print('BBox:', bboxes[i])

    location_means = torch.prod(shower_rgb[:, :, :, :2], dim=3)
    i_indices = torch.arange(32, dtype=torch.float32)
    j_indices = torch.arange(32, dtype=torch.float32)
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')

    X0 = torch.sum(i_grid * location_means, dim=(1, 2)) / torch.sum(location_means, dim=(1, 2))
    X0 = X0 * (bboxes[:, 1] - bboxes[:, 0]) / 32 + bboxes[:, 0]
    Y0 = torch.sum(j_grid * location_means, dim=(1, 2)) / torch.sum(location_means, dim=(1, 2))
    Y0 = Y0 * (bboxes[:, 3] - bboxes[:, 2]) / 32 + bboxes[:, 2]

    N, T = GetCounts_differentiable_fn(shower_rgb, x, y, bboxes)

    return N, T, X0, Y0, p_energy, sin_z, cos_z, sin_a, cos_a