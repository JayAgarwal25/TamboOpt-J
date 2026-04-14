"""Evolutionary operators for v5: fitness scoring, pruning, mutation.

All operators act on a `Population` (see ev_population.py) in place and are
purely physical / gradient-based — there is no multi-individual population
or crossover.  The design is a single-individual evolutionary strategy:

    while population.size > target_final:
        scores = compute_detector_fitness(...)   ← per-detector saliency via
                                                    gradient of U w.r.t. a mask
        prune_weakest(population, scores, k)     ← keep top-k by score
        mutate_positions(population, sigma, f)   ← perturb survivors

Fitness rationale
-----------------
We want each generation's pruning decision to reflect *marginal contribution
to the global utility U*, not raw signal strength.  A brute-force leave-one-
out evaluation would require N forward passes per generation — infeasible
for N = 10,000.

Instead we exploit the DeepSets sum-pool architecture: introduce a per-
detector multiplicative gate `mask` (init 1.0, `requires_grad=True`) before
pooling, compute U, and take **one** backward pass.  The result,
`∂U/∂mask[i]`, is exactly the first-order Taylor coefficient of U with
respect to scaling detector i's embedding contribution.  Large positive
values → detector is actively helping → keep.  Negative or near-zero →
detector is redundant or harmful → prune.

The first-order approximation is most accurate when the model has been
trained to tolerate partial masks.  The main notebook achieves this by
randomly dropping 10–90% of detectors per training batch during NN
training, so the saliency signal stays meaningful across the full
10k → 90 range.
"""

from typing import Callable, Tuple, Optional

import torch

# Imported via modules_v5 sys.path injection
from modules.utility_functions import reconstructability, U_PR, U_E, U_angle
from modules.reconstruction    import DenormalizeLabels

from modules_v5.ev_population import Population, build_input_batch


# ─────────────────────────────────────────────────────────────────────────────
# Fitness scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_utility(
    inputs_raw:       torch.Tensor,   # (B, N, 7)  unnormalized — for r_score
    inputs_norm:      torch.Tensor,   # (B, N, 7)  normalized   — for model
    mask:             torch.Tensor,   # (B, N) or (1, N)
    model:            torch.nn.Module,
    energies:         torch.Tensor,
    th:               torch.Tensor,
    ph:               torch.Tensor,
    reconstruct_threshold: float = 10.0,
    w_angle:          float = 1e2,
    w_energy:         float = 1e8,
    w_pr:             float = 5e5,
    w_norm:           float = 1e3,
) -> torch.Tensor:
    """Compute the scalar utility U used as the fitness of the current layout.

    Mirrors v4's cell-40 formula exactly:
        U = (w_angle * U_angle_theta + w_angle * U_angle_phi
             + w_energy * U_E + w_pr * U_PR) / w_norm
    """
    preds = model(inputs_norm, mask=mask)                     # (B, 3)
    preds_e, preds_th, preds_phi = DenormalizeLabels(
        preds[:, 0], preds[:, 1], preds[:, 2]
    )
    # N_int is at feature index 3 of the unnormalized input
    r_score = reconstructability(
        inputs_raw[:, :, 3], reconstruct_threshold=reconstruct_threshold
    )
    U = (
        w_angle  * U_angle(preds_th,  th, r_score) +
        w_angle  * U_angle(preds_phi, ph, r_score) +
        w_energy * U_E(preds_e, energies, r_score) +
        w_pr     * U_PR(r_score)
    ) / w_norm
    return U


def compute_detector_fitness(
    population:      Population,
    model:           torch.nn.Module,
    shower_fn:       Callable,
    n_samples:       int,
    input_mean:      torch.Tensor,
    input_std:       torch.Tensor,
    reconstruct_threshold: float = 10.0,
    w_angle:         float = 1e2,
    w_energy:        float = 1e8,
    w_pr:            float = 5e5,
    w_norm:          float = 1e3,
) -> Tuple[torch.Tensor, float]:
    """Return (fitness_scores, U_value) for the current population.

    fitness_scores : (N_det,) tensor on `population.device`.  A larger value
                     means "removing this detector would hurt the utility more".
                     Ranking is by raw value (not absolute value): detectors
                     that hurt the utility have negative scores and are pruned
                     first.

    U_value        : scalar python float, the current utility (for logging).

    Args:
        population  : current Population (variable size).
        model       : DeepSetsReconstruction (or any Module that accepts
                      `inputs, mask=...` and returns (B, 3)).
        shower_fn   : callable with signature
                          shower_fn(x_det, y_det, z_cont, log=False,
                                    number_of_showers=..., use_cache=...)
                      returning
                          (N, T, X0, Y0, energies, sin_z, cos_z, sin_a, cos_a, labels)
                      This is the v4-style `generate_showers` wrapper defined
                      in the main notebook (it closes over the GenerateShowers
                      instance, the y-shift state, and GetCounts_planeaware).
        n_samples   : number of showers to evaluate per generation.
        input_mean,
        input_std   : frozen (7,) tensors from initial NN training.
    """
    device = population.x.device
    model.eval()

    # 1. Generate showers at the current detector layout.  The z_cont closure
    #    inside shower_fn captures `population.z_cont`, so gradients do not
    #    need to flow back through the shower generator.
    with torch.no_grad():
        out = shower_fn(
            population.x, population.y, population.z_cont,
            log=False,
            number_of_showers=n_samples,
            use_cache=True,
        )
    N_list, T_list, X0, Y0, energies, sin_z, cos_z, sin_a, cos_a, _ = out
    N_list = N_list.to(device); T_list = T_list.to(device)
    X0 = X0.to(device); Y0 = Y0.to(device)
    energies = energies.to(device)
    th = torch.atan2(sin_z, cos_z).to(device)
    ph = torch.atan2(sin_a, cos_a).to(device)

    # 2. Build the (B, N, 7) feature tensor.  No grad on positions (those are
    #    not learnable in v5) — detach for safety.
    with torch.no_grad():
        inputs_raw = build_input_batch(population, N_list, T_list, X0, Y0)
    inputs_norm = (inputs_raw - input_mean) / input_std

    # 3. Gate.  A single (1, N) mask broadcast over the batch — any per-
    #    detector scaling affects every event the same way, which is the
    #    correct semantics for "remove this detector globally".
    B, N, _ = inputs_norm.shape
    mask = torch.ones(1, N, device=device, requires_grad=True)
    mask_batched = mask.expand(B, -1)

    # 4. Forward + utility + one backward pass → per-detector gradient.
    model.zero_grad(set_to_none=True)
    U = _compute_utility(
        inputs_raw, inputs_norm, mask_batched, model,
        energies, th, ph,
        reconstruct_threshold=reconstruct_threshold,
        w_angle=w_angle, w_energy=w_energy, w_pr=w_pr, w_norm=w_norm,
    )
    U.backward()

    fitness = mask.grad.detach().squeeze(0)  # (N_det,)
    model.zero_grad(set_to_none=True)
    return fitness, float(U.detach().cpu())


# ─────────────────────────────────────────────────────────────────────────────
# Pruning
# ─────────────────────────────────────────────────────────────────────────────

def prune_weakest(
    population:    Population,
    fitness_scores: torch.Tensor,
    target_size:    int,
) -> None:
    """In-place: keep the top `target_size` detectors by fitness score.

    Ranks by raw fitness (larger = keep), not absolute value.  Detectors
    with negative gradient saliency (i.e. scaling them up hurts U) are
    pruned first, which is the intended "natural selection" semantics.
    """
    k = min(int(target_size), population.size)
    if k <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if k == population.size:
        return
    topk_idx = torch.topk(fitness_scores, k, largest=True).indices
    population.apply_indices(topk_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Mutation
# ─────────────────────────────────────────────────────────────────────────────

def mutate_positions(
    population: Population,
    sigma:      float = 50.0,
    frac:       float = 0.5,
    max_gap:    Optional[float] = None,
) -> None:
    """In-place: add Gaussian position noise to a random subset of detectors.

    Survivors wandering off the mountain footprint are snapped onto the
    nearest centroid via `MountainData.project_to_mountain`, so the
    population always stays on a valid surface.  `z_cont` is refreshed
    afterwards to reflect the new positions.

    Args:
        sigma   : stddev of the Gaussian perturbation [m].
        frac    : fraction of detectors to perturb per call (default 0.5).
        max_gap : passed through to project_to_mountain (None = use the
                  default 2×mean-nearest-neighbour heuristic).
    """
    if frac <= 0.0 or sigma <= 0.0:
        return
    N = population.size
    k = max(1, int(round(N * float(frac))))
    device = population.x.device

    idx = torch.randperm(N, device=device)[:k]
    dx = torch.randn(k, device=device) * sigma
    dy = torch.randn(k, device=device) * sigma

    with torch.no_grad():
        population.x[idx] = population.x[idx] + dx
        population.y[idx] = population.y[idx] + dy
        # Snap any detector that drifted off the mountain back onto it.
        new_x, new_y = population.mountain.project_to_mountain(
            population.x, population.y, max_gap=max_gap
        )
        population.x = new_x.contiguous()
        population.y = new_y.contiguous()

    population.refresh_z_cont()
