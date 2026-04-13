# Gradient Path Analysis — `SWGOLO7_optimization_tr_same_10_center_init_20260413_100000.ipynb`

**Question:** does the gradient path break anywhere in the v4 TR optimization loop?

**Answer: No — the gradient path does NOT break.** All autograd dependencies are
intact end-to-end. The run is not converging for *objective-shape* reasons
(flat/saturated loss terms), not because of a broken chain.

---

## 1. Verified gradient path

```
xy_module.x / .y  (nn.Parameter, requires_grad=True)
       │
       ▼
surface(x, y)   ← F.grid_sample(bilinear, padding_mode="border")   ✓ differentiable
       │
east_det ── z_cont = (east_entry − east_det) / layer_east_dx        ✓ linear
       │
generate_showers(x, y, z_cont)
       │  samples are loaded from the cached fixture (no grad) but
       │  x_det, y_det, z_cont are forwarded into the kernel.
       ▼
GetCounts_planeaware(samples, x_d, y_d, z_cont)
       │  kernel = exp(−d²/(2σ²)) · relu(1 − |layer_p − z_cont|)
       ▼
(N_list, T_list)                                                    ✓ differentiable in x, y, z_cont
       │
torch.stack([x_exp, y_exp, z_cont_exp, N_list, T_list], dim=2)      ✓ all 5 features carry grad
       │
(inputs − input_mean) / input_std → model.eval() → DenormalizeLabels
       │
U = (1e2·U_θ + 1e2·U_φ + 1e8·U_E + 5e5·U_PR) / 1e3
       │
Loss = −U
Loss.backward()
```

### Runtime confirmation

The loop prints `p.grad.norm()` for each learnable parameter every epoch. Example
from the recorded notebook output (2003 epochs saved in `Utilities.txt`):

```
Epoch   0   x: grad_norm=5.70   y: grad_norm=9.66
Epoch 999   x: grad_norm=3.45   y: grad_norm=5.84
```

Non-zero, finite, no NaN, across every single recorded epoch. If any link were
broken (a `detach()`, an in-place on a leaf, a `.data` assignment in-graph, a
`torch.no_grad()` contamination, a dtype mismatch), these would be `None` or
zero.

### Per-link sanity checks that passed

| Link | Status | Note |
|---|---|---|
| `LearnableXY.forward()` | ✓ | returns `self.x, self.y` (Parameters). |
| `SurfaceEastMap.forward` | ✓ | `F.grid_sample` is differentiable w.r.t. the sampling grid. |
| `(east_entry − east_det) / layer_east_dx` | ✓ | affine, preserves grad. |
| `generate_showers` closure | ✓ | captures `z_cont`, calls `GetCounts_planeaware` with the live `(x_d, y_d, z_cont)`. |
| `_apply_y_shift` | ✓ | operates on a `samples.clone()`; shift is a Python float; samples have no grad anyway. |
| `ComputeShowerDetection` | ✓ | passes `x_det, y_det` through unchanged; minibatch loop uses `torch.cat`. |
| `GetCounts_planeaware` kernel | ✓ | spatial Gaussian × `relu(1 − |Δ|)` are both differentiable almost everywhere. |
| `torch.stack(..., dim=2).float()` | ✓ | `.float()` is a no-op on float32 and preserves grad otherwise. |
| `(inputs − mean) / std` | ✓ | `input_std[input_std < 1e-8] = 1.0` rules out div-by-zero. |
| `model.eval(); model(inputs.view(...))` | ✓ | `eval()` does not detach. |
| `DenormalizeLabels` | ✓ | linear. |
| `reconstructability` | ✓ | soft (sigmoid), no hard thresholds. |
| `U_angle`, `U_E`, `U_PR` | ✓ | elementwise arithmetic. |
| `torch.nn.utils.clip_grad_norm_` | ✓ | finite grad norms, no NaN mask. |
| `project_to_mountain` | ✓ | runs under `with torch.no_grad():`, copies to `.data`. |

---

## 2. Issues that DO hurt the run (objective-shape, not gradient wiring)

These are why `U_total` does not improve even though gradients flow.

### 2a. `U_PR` is a literal constant

```
column "U_PR" of Utilities.txt  →  1581138.88  for every one of 2003 epochs
```

With `reconstruct_threshold=10` and 90 detectors, `n ≈ 90 ≫ 10`, so
`r = sigmoid(5·(n − 10)) ≈ 1` for every event. Then
`U_PR = sqrt(sum(r) + 1e-6) = sqrt(10) ≈ 3.162` and
`5e5·U_PR = 1 581 138.88`. Because the sigmoid is saturated, its derivative
w.r.t. `n` (and hence `N_int`) is ~0 — **this term contributes no gradient**.

`U_PR` accounts for **~82 % of `U_total`**, so the useful signal from
`U_θ + U_φ` is buried under a frozen offset.

### 2b. `U_E` is numerically zero

```
column "U_E" of Utilities.txt  →  0.00  in every row
```

`U_E = sum(r / ((E_pred − E_true)² + 0.01))`. Because `DenormalizeLabels`
returns energy in **GeV** (1e5–1e8), `Δ²` is ~1e10–1e14 per event, so each
term is ~1e-10. Even with the `1e8` weight and 10 events, the sum is
~1e-2, which prints as `0.00`. **This term is effectively dead** — no
gradient contribution.

### 2c. Only `U_θ` and `U_φ` drive learning

Actual `Utilities.txt` trajectory:

```
epoch    0 : U_θ = 261 913   U_φ =  1 467   U_E = 0   U_PR = 1 581 139
epoch   99 : U_θ = 338 321   U_φ =  1 190   U_E = 0   U_PR = 1 581 139
epoch  499 : U_θ = 338 320   U_φ =  1 416   U_E = 0   U_PR = 1 581 139
epoch  999 : U_θ = 338 287   U_φ = 29 685   U_E = 0   U_PR = 1 581 139
epoch 1499 : U_θ = 336 218   U_φ = 20 515   U_E = 0   U_PR = 1 581 139
epoch 1999 : U_θ = 330 882   U_φ = 16 190   U_E = 0   U_PR = 1 581 139
```

- `U_θ`: rapid gain in the first ~100 epochs (+29 %), then plateau, then mild
  regression (−2 % by epoch 1999).
- `U_φ`: very noisy — swings from 1 000 to 30 000 and back. Classic SGD
  oscillation in a narrow, non-convex basin.
- `U_E`, `U_PR`: pinned.

### 2d. Fine-tune step is a silent no-op

```python
ft_dataloader = DataLoader(ft_dataset, batch_size=32, shuffle=True,
                           drop_last=True, num_workers=0)
```

With `Nfinetune=10` and `drop_last=True`, this yields **zero batches**. The NN
is never retrained during optimization. Every `Fine-tune at epoch N` print
executes zero parameter updates.

### 2e. NN was trained on 10 samples

`Nevents=Nvalidation=Ntest=Nbatch=Nfinetune=10`. Cell 27 shows `val_loss`
flatlining at `0.0868` from epoch 80 onward — the NN has memorised the ten
training instances. The reconstructor's predictions on new detector positions
are essentially unchanged, which explains why `U_θ` plateaus so quickly.

### 2f. Detectors barely move

| File | N-spread | U-spread | z_cont range | mean (N, Up, z) |
|---|---|---|---|---|
| `Layout_0.txt` | 70 m | 73 m | [11.16, 12.33] | (−16.45, 3135.49, 11.898) |
| `Layout_100.txt` | 79 m | 85 m | [11.03, 12.38] | (−16.52, 3135.24, 11.893) |
| `Layout_500.txt` | 79 m | 84 m | [11.03, 12.38] | (−16.57, 3135.05, 11.892) |
| `Layout_1000.txt` | 81 m | 86 m | [11.04, 12.39] | (−16.32, 3135.17, 11.892) |

Per-detector displacement `Layout_0 → Layout_1000`: mean 5.3 m, max 13.2 m.
`z_cont` per-detector delta range `[−0.203, +0.217]` layers (≈ ±30 m East).
Detectors move almost exclusively tangent to the mountain surface; they do
not redistribute in East (and therefore not in `z_cont`).

### 2g. Other observations (not bugs)

- `NUM_FEATURES = 5` (not 7) — the commented-out 7-feature path with `x0, y0`
  is disabled. The active path uses `[x, y, z_cont, N_int, T_int]`. The
  feature-index-3 indexing for `reconstructability` is consistent.
- `east_entry = 1500`, `layer_east_dx = 150` — overridden from the
  `CLAUDE.md` defaults (`-212`, `307`). This stretches the mountain surface
  into `z_cont ∈ [0, 23.5]` and puts the initial cluster at `z_cont ≈ 11.9`.
- `use_cache=True` — the "same 10 showers" are reused every epoch, so the
  gradient is deterministic given detector positions. Noise in `U_φ` therefore
  comes from the SGD trajectory, not from shower sampling.
- `Loss = −U`; optimizer minimizes Loss → should increase U. `U_total`
  nudges from 1844 → 1967 over 2000 epochs (+7 %). That increase is entirely
  in `U_θ + U_φ`, swamped by the constant `U_PR` offset in the total.

---

## 3. Recommendations (ranked by expected impact)

1. **De-saturate `U_PR`.** Either:
   - raise `reconstruct_threshold` so `n ≈ 90` is not deep in the sigmoid
     saturation regime, or
   - drop the `5e5` coefficient so `U_PR` is no longer ~82 % of `U_total`, or
   - remove `U_PR` from the objective entirely if it never provides gradient
     signal at the chosen `Nunits`.

2. **Rescale `U_E`.** Use `(log10(E_pred) − log10(E_true))²` or normalize
   `E_pred`/`E_true` to O(1) before the reciprocal. GeV-scale input makes the
   term numerically zero.

3. **Fix the fine-tune dataloader.** Set `drop_last=False`, or
   `batch_size=min(32, Nfinetune)`. Otherwise delete the fine-tune branch so
   the logging reflects reality.

4. **Grow `Nevents`/`Nbatch` past 10.** At the current size the NN memorizes
   ten shower instances and the gradient through the kernel is a
   near-degenerate estimate.

5. **Instrument `project_to_mountain`.** Log how many detectors are snapped
   back per epoch. If it's frequent, relax `max_gap` or gate the projection
   behind an "off-surface" check. The stable `z_cont` range across 2000
   epochs suggests the projection is doing a lot of clamping.

6. **Rethink the optimizer.** SGD(lr=0.5, momentum=0.3) is bouncing in a
   narrow basin — `U_φ` swings by 25× in consecutive epoch windows. Try
   Adam(lr=1e-3) or SGD with warmup + smaller lr.

---

## 4. Bottom line

The gradient reaches `xy_module.x / .y` correctly every epoch — this is not a
wiring bug. The run stalls because:

- ~82 % of the objective (`U_PR`) is a saturated constant with zero gradient,
- ~0 % of the objective (`U_E`) is numerically zero because of GeV-scale energy,
- the remaining ~18 % (`U_θ + U_φ`) learns for the first ~100 epochs and then
  oscillates in place,
- the fine-tune that was meant to keep the NN in sync with the moving layout
  never executes because of `drop_last=True` on a dataset of 10.

Fix items 1–4 above before suspecting autograd.
