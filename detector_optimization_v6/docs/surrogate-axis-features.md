# Stage-2 surrogate: give it the response geometry

## Why

The surrogate under-disperses. In log1p space — the space it is fitted in — it is
nearly unbiased (log-mean 1.721 vs the kernel's 1.914) but its spread is short:
**log-std 2.27 vs 3.17**, with `corr(log K, log S) = 0.79`. It smooths a sharp
function.

The sharpness is geometric. The kernel's per-detector response
(`modules_v4/tr_plane_kernel.py`) is

    spatial = exp(-(dN^2 + dU^2) / (2 sigma^2)),   plane = relu(1 - |layer - z_d|)

with **North and Up** the Gaussian coordinates and East the layer index — see
`compute_labels_batch`: *"the KERNEL still sees North as the transverse
coordinate and East as the depth (z_cont)"*.

The model's token is `[q(8), x_i, y_i]` — East and North only. So **`u_d` is
absent**: Up is one of the two coordinates the Gaussian is evaluated in, and the
model never receives it. To recover it the network would have to memorise the
mountain surface `g(N, E)`, a 964 m relief field, from (E, N) alone.

## Change

Three per-detector features:

| feature | definition | status |
|---|---|---|
| `u_d`   | `g(N_d, E_d)` via `SurfaceUpMap` | **new information** |
| `s_d`   | `v . d_hat` — along-axis depth | reparameterisation |
| `rho_d` | `\|\| v - s_d d_hat \|\|` — impact parameter | reparameterisation |

with `v = r_d - P_vertex`, `r_d = (E_d, N_d, u_d)`.

`s_d` and `rho_d` add no information — direction and vertex are already in the
primary encoding (`[dir(3), log_e_norm, pdg, rel_E/N/U]`, cols 0:3 and 5:8). They
are a change of coordinates that makes a sharp function easy to represent rather
than hard. `u_d` is the only strictly new input, and the highest-confidence part
of this change. It is also required to compute the other two in 3-D.

`rel_E/N/U` is relative to `array_center` while `xy` is absolute, so the model
needs `array_center` as a registered buffer, stored in the checkpoint.

Token width: 10 -> 13. Existing checkpoints are invalidated.

## Use the perpendicular foot, not the kernel's frame

The obvious alternative is offsets in the kernel's own (North, Up) plane,
`dN_d` and `dU_d`, evaluated where the axis crosses the detector's East slice.
Do not: that requires

    s_d = (E_d - E_vertex) / d_E

which blows up as `d_E -> 0`, i.e. for showers travelling North-South. The corpus
is whole-sky in azimuth so a large fraction sit there. It is the same singularity
`place_clouds_enu` documents for `1/d_U` ("unusable at |d_U| p5 = 0.027").

The perpendicular foot divides by nothing, is defined for every direction, and is
the same function of geometry at any azimuth — so the network learns one response
instead of memorising a pattern per arrival angle. It still has `d_hat` in the
primary and can compose the two to recover the kernel's frame-dependent
behaviour.

## What retrains

| stage | action | why |
|---|---|---|
| 00 | reuse | corpus unchanged; stage 2 never opens it |
| 01 | reuse | features are computed in-model from `(primary, xy)` |
| 02 | **retrain** | the change |
| 03 | **retrain** | the recon trains on `forward_sample` draws from the frozen surrogate, so its entire input distribution moves |
| 04 | rerun | uses both |

A run-08 `pipeline_status.json` must therefore list **only** 00 and 01 as done.

## How to tell whether it worked

The current diagnostics cannot show it. `fnn_*_target_vs_pred.png` and the val
metric both live in log space, which is exactly where the model already looks
good. Add first:

* linear-space ratio `expm1(pred)/expm1(target)` vs target
* cumulative fraction of true flux recovered vs target rank

Then:

1. **log-std of the prediction rises toward 3.17** (currently 2.27) — the direct
   measure of the under-dispersion this change targets.
2. **the artifact gap in `eval_true_utility.py` shrinks** (currently +10.38
   ensemble, +8.57 activation-center). This is the guard against the real risk
   here: cleaner geometric coordinates could make the surrogate artificially
   *invertible*, letting the recon read the primary out of a parameterisation
   artifact rather than out of physics. A growing gap means that happened.

## Known ceiling — test before investing

Part of the spread is irreducible: the point cloud is a stochastic sample, so two
events with identical primaries give different showers. The model's own predicted
log-variance (5.54, std 2.35) combines with its mean spread (2.27) in quadrature
to 3.27, reproducing the kernel's 3.17 to 3% — which suggests it is already close
to correctly calibrated.

Group events with near-identical (E, direction, vertex) and measure the
**kernel's** within-group log-spread. That is the floor no feature work can beat.
If it comes back near 2.35, this change is near its ceiling already and further
feature work is not worth it.
