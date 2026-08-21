# `eval/` — measuring what the pipeline actually produces

Everything here is run by hand. None of it is wired into a SLURM stage, so
nothing will notice if one of these breaks. They answer "is the number we are
about to report true", which is why they are kept apart from `plots/`, whose job
is to draw figures the pipeline emits on its own.

All of them compare a learned component against the ground-truth kernel on the
SAME events and the SAME layout, so a difference is model error and not a
difference of sample.

## Resolution and utility

| file | question it answers |
|---|---|
| `eval_recon_resolution.py` | how far off is the reconstructed direction, kernel-fed vs surrogate-fed |
| `eval_classical_baseline.py` | what does a non-learned plane fit achieve on the same events |
| `diag_layout_collapse.py` | how much of an optimized layout's gain survives a minimum-separation floor |

## Detection and per-detector behaviour

| file | question it answers |
|---|---|
| `eval_detection_stats.py` | which detectors light up, kernel vs surrogate, plus efficiency vs energy and vs decay distance |
| `diag_fnn_vs_kernel.py` | does the surrogate reproduce the kernel's per-detector response |
| `diag_recon_outputs.py` | what the reconstruction emits, for one batch, in physical units |
| `diag_residual_bias_variance.py` | is the surrogate's error a bias or a spread, binned |

## Optimizer behaviour

| file | question it answers |
|---|---|
| `optimize_and_track.py` | how a layout moves under the optimizer, with the utility tracked along the path |

## Conventions

Detector coordinates are `(East, North)`. `eval_true_utility.py` stays in
`plots/` because the pipeline drives it; several scripts here import it as a
library for its corpus loading, layout helpers and scoring, so that its
definition of the utility and theirs cannot drift apart.
