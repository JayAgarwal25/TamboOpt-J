# Design exploration — closing the surrogate artifact gap in v6 layout optimization

Problem source: `detector_optimization_v6/tmp.txt` + `plots/eval_true_utility.py` evidence.
Six independent design explorers (different reasoning methods, isolated contexts) produced the
approaches below. Date: 2026-07-26.

---

## 1. Problem statement

The 04 layout optimizer maximizes composite utility U through a frozen FNN surrogate + frozen
recon. Scoring the *same* layouts with the plane-aware kernel (ground truth the surrogate
approximates) shows:

- surrogate-U ≈ 170 vs true-U ≈ 36 on the same layout → **artifact gap ≈ +134**
- full-corpus run, optimized vs plain grid: ΔU_surrogate **+6.9**, ΔU_true **−1.4** → verdict ARTIFACT
- center-start run: ΔU_true +27.4 (real, but 5.7× overstated by the surrogate)

Aggravators: stage-1 surrogate R²≈0 on held-out labels (likely an *aleatoric floor* — the 8-dim
primary cannot determine the 25k-point cloud realization); L-BFGS's determinism forced fixed-batch
objectives (batch overfitting); the recon is not permutation-invariant over detector ordering
(part of the objective is a labeling artifact — `grid_score_diagnostics`: grid U≈136 vs
relabeled ≈30).

User TODOs: (1) fix/neutralize the artifact gap; (2) use the full dataset efficiently in the
optimizer; (3) explain/fix why the full-corpus run raised surrogate-U but lowered true-U.

### Named defaults (the obvious moves)

1. Train a better surrogate, rerun 04.
2. Skip the surrogate — optimize directly on the kernel.
3. Add a trust region / uncertainty penalty around the surrogate.

### Reframings

- **Step-back:** offline model-based optimization against a learned proxy — Goodhart / reward
  hacking. The optimizer finds the proxy's errors faster than its truths.
- **Inversion:** the worst design is "optimize a frozen wrong proxy infinitely hard on a tiny
  fixed batch" — nearly the current system. Suspect assumption: that a neural surrogate is
  needed at all for layout gradients.
- **Distant analogy:** the layout is an adversarial example against the surrogate; sensor
  placement is a submodular problem with greedy guarantees.
- **Constraint removal:** with free compute you'd differentiate the kernel over the corpus and
  never train a surrogate.

---

## 2. Measured facts the exploration surfaced (checkable, load-bearing)

1. **The kernel is cheap.** Constraint-first explorer's arithmetic: full-corpus kernel pass
   ≈ 5 TFLOP forward (~15 fwd+bwd) — seconds on an A100, only ~6–10× the surrogate's own cost.
   *The surrogate was never a meaningful speedup; it manufactures the +134 artifact at ~1/6 the
   kernel's price.*
2. **The kernel is already differentiable in principle.** `compute_labels_batch`
   (`modules_v6/fnn_surrogate_ne.py:142`) is pure torch; the `@torch.no_grad()` decorator is the
   only blocker to exact ∂U/∂(E,N).
3. **The adapter already exists.** `KernelDualLabels` in `plots/eval_true_utility.py` has the
   FNN's exact call signature and feeds the *unmodified* `utility_of_xy` — kernel-in-the-loop is
   one promotion away.
4. **The clouds are stored above kernel resolution.** The Gaussian kernel (σ_spatial) cannot
   resolve structure ≪ σ; voxel/superpoint compaction at ~σ/4 shrinks the corpus to a few GB —
   GPU-resident — with a measurable fidelity knob.
5. **TODO 3, mechanism:** the recon was trained on kernel labels but is fed surrogate outputs at
   optimization time — smooth mean-fields, out-of-distribution for the recon. "Surrogate-U"
   measures how well the recon decodes the surrogate's mean field, not layout physics; more
   optimization pressure digs deeper into that composite's idiosyncrasies, anti-correlated with
   truth. Deeper: even true-U through a *frozen* recon is biased — the recon is co-adapted to
   stage-01's strategy layouts.
6. **Directional validity is regime-dependent.** Center-start gains (+27.4 true) show surrogate
   gradients are roughly right *far from* good layouts; the artifact dominates *near* the grid
   optimum. (This is the textbook trust-region regime.)

---

## 3. Solution landscape

Two axes organize the six approaches:

- **x: what computes labels inside the optimizer** — learned surrogate ↔ ground-truth kernel ↔ no labels at all (estimator-free objective)
- **y: how the recon is treated** — frozen ↔ retrained once (canonical/invariant) ↔ co-trained in the loop ↔ removed from the objective

```
                     recon frozen          recon retrained/co-trained     recon out of objective
 surrogate in loop   (current system ✗)    D. KATR (trust region)         —
 kernel in loop      B1. KDA-1000          A1. KODO   A2. σ-Sketch        —
                                           B2. DEMOTE-AND-ANCHOR
 no labels in loop   —                     —                              C. Fisher-Greedy
```

**Distance audit.** A1/A2 share a core mechanism (full-batch deterministic L-BFGS through a
grad-enabled kernel on a compressed, GPU-resident corpus) — variants. B1/B2 share a core
mechanism (stochastic minibatch Adam through the grad-enabled kernel) — variants. C and D are
each mechanistically distinct. The four kernel-direct convergences were diagnosed as Function
Lock, but are backed by measured arithmetic (fact 1), so the convergence is evidence, not
priming; D was spawned with kernel-in-the-loop declared off-limits to cover the
surrogate-retained territory.

---

## 4. Approaches

### A1. KODO — Kernel-Only Direct Optimization *(constraint-first explorer)*

**Core mechanism.** Voxel-compact the corpus once (~15 m N-bins × exact layer, energy-weighted
sums) into a ~3–6 GB GPU-resident tensor; run full-corpus L-BFGS *directly through* a
grad-enabled `compute_labels_batch`. Alternating bilevel loop: recompute kernel labels at the
live layout (~seconds), fine-tune a strictly permutation-invariant DeepSets recon on them
(1–2 min), take M L-BFGS steps on (E,N), leashed by a per-detector trust region (~σ/2), with a
σ-annealing homotopy (σ 200 m → 50 m) restoring the smoothing the surrogate accidentally
provided.

**Enables.** Artifact gap excised, not closed — surrogate-U ≡ true-U by construction. True
full-batch gradients: no batch overfitting, no seed sensitivity (TODO 2 answered by residency).
Recon trained at the live layout kills the off-distribution failure (TODO 3). Coarse-to-fine
optimization the frozen pipeline could never express.

**Sacrifices.** The surrogate's accidental smoothing (σ-anneal is load-bearing); bilevel
stability risk (recon chasing a moving layout); ~2–4 days of voxel-compactor + exactness-test
engineering; only valid while the Gaussian kernel *is* the accepted truth. Trap: the kernel's
time channel divides by padded P — the compactor must reproduce that normalization exactly.

**Choose if** the kernel remains the project's definition of ground truth and the corpus stays
≲1M events.

```
00 corpus → 01v voxel compaction (one pass, fidelity-gated) → GPU-resident ~4 GB
   outer iter k:  kernel labels @ L_k (~3 s) → fine-tune perm-inv recon (1–2 min)
                  → full-corpus L-BFGS through kernel→recon→U (trust region, σ-anneal)
02 surrogate: deleted from the optimization path.
```

### A2. σ-Sketch — surrogate-free optimization on kernel-sketched clouds *(backward-from-ideal explorer)*

**Core mechanism.** Same kernel-direct full-batch L-BFGS spine as KODO; differs in the
compression (per-layer energy-conserving pooling to ~256 superpoints per cloud at σ/4
resolution, gated by a fidelity report: label R² > 0.99 vs full clouds) and in the recon
treatment (retrain **once** with canonical detector ordering — sort by East, tie-break North —
instead of bilevel co-training). `eval_true_utility.py` inverts roles: it becomes the auditor of
the *sketch* (final layouts scored on full clouds).

**Enables.** The published plot ("true-U vs iteration, monotone") falls straight out of the
optimizer's own log; the only exploitable error is sketch error — measured, with a knob (cell
size), unlike surrogate error which is unmeasurable at the point of use.

**Sacrifices.** Ordering canonicalization is weaker protection than strict invariance or an
in-loop gate; frozen-recon co-adaptation bias remains (A1/B2 address it); corpus-as-flux
assumption becomes explicit.

**Choose if** the fidelity gate passes at ~256 superpoints (σ-smoothing argument says it will;
knob: more superpoints — even 2k is a 12× win) and you want the smallest delta from today's 04
structure. If no affordable sketch is faithful, a surrogate re-earns its place — but must be
trained to R² ≫ 0 first.

### B1. KDA-1000 — Kernel-Direct Ascent with a metered truth ledger *(forward build-up explorer)*

**Core mechanism.** Delete the surrogate from the loop; run Adam on true-U through the
grad-enabled kernel with stratified ~512-event minibatches and importance-subsampled ~2k-point
cloud sketches (unbiased reweighting; events outside the 70–100° band / >1 km decays dropped —
the measured 46% keep-rate). A hard ledger meters ~1,000 total ground-truth evals: ~800 gradient
steps, ~150 candidate selections, ~50 paired audits on the *unsketched* kernel. Every reported
ΔU_true carries ledger provenance. Drop L-BFGS (its determinism was the Goodhart amplifier);
K=3–5 restarts warm-started free from surrogate-Adam (unmetered, untrusted).

**Enables.** Artifact gap = 0 by construction; full-corpus coverage at 1/300th naive cost;
trust as a first-class artifact (audited paired comparisons); fits one 7-h slice; preemption-
resumable via the ledger. Hygiene gate: average the objective over random detector permutations
if the perm-invariance check fails.

**Sacrifices.** ~800 steps vs 6,500 today — less fine-polish; 3–5 restarts vs 15 — weaker global
search; stage-02 investment written off from the optimization path.

**Choose if** a sketched kernel backward on ~512 events runs in ≤ a few seconds on the A100
(the eval script's forward already nearly proves it) **and** surrogate held-out R² remains ≈0 —
then every surrogate-loop alternative is strictly optimizing noise.

### B2. DEMOTE-AND-ANCHOR — kernel-anchored stochastic descent, surrogate as control variate *(adversarial explorer)*

**Core mechanism.** Same stochastic kernel-Adam spine as B1, plus three defenses the others
lack: (i) the surrogate returns *only* as an optional control variate on the gradient
(`g = g_kernel_mb − g_surr_mb + g_surr_bigbatch`) with an online variance-reduction check that
auto-disables it — its errors can cost variance, never bias; (ii) the recon is co-trained
two-timescale on kernel labels at the current layout **with ~50 m positional jitter** (prevents
memorizing the 60 coordinates), closing the frozen-recon co-adaptation channel; (iii) an in-loop
accept/revert gate: every K steps, score current vs last-accepted layout with the
`eval_true_utility` machinery on a held-out event split, using both the frozen production recon
(historical comparability) and the co-trained recon (the actual objective).

**Enables.** Everything B1 enables, plus a debiased objective (recon no longer frozen-wrong) and
certified monotonicity (revert on regression). Degrades gracefully: switch off every optional
part and plain minibatch kernel-Adam remains — still artifact-free.

**Sacrifices.** Per-step cost 10–50× surrogate steps; streaming/priority-sampling code to write
(83k placed clouds ≈ 40+ GB — CSR ragged store, pinned-memory gather); co-trained recon breaks
comparability with historical runs (mitigated by the dual-recon gate); recon co-training is
itself a hackable channel (contained by jitter + held-out gating).

**Choose if** you believe R²≈0 is an aleatoric floor (check `compute_aleatoric_floor.py`
first — this is the fork in the road). If the floor turns out low, the adversarial-retraining
idea becomes viable again — but this design's gate and demoted-surrogate roles are strictly
safer and reuse the same code.

### C. Fisher-Greedy Kernel Placement *(analogical-transfer explorer — sensor placement / optimal experimental design)*

**Core mechanism.** Reframe from 120-dim continuous ascent to **lazy stochastic greedy selection
of 60 sites from ~3,000 mesh-snapped candidates**, maximizing a sum of two monotone submodular
set functions evaluated directly against the kernel: a concave-of-modular coverage term (the
physics of `reconstructability` — sigmoid-of-modular is not submodular below threshold, so use
`min(n_soft, k)` / probabilistic coverage) plus a log-det Fisher-information term
`Σ_events log det(εI + Σ_d J_dᵀJ_d)` with per-site Fisher vectors J_d = ∂(E_d,T_d)/∂(primary)
computed **from the kernel** via autograd — estimator-free (bounds any future recon via
Cramér–Rao), provably submodular. One-time candidate-bank build (kernel over all candidate
sites, chunked, preemption-safe); marginal gains are then rank-updates on cached matrices —
milliseconds, full corpus per round. Recon trained *after* the layout is fixed. Optional short
kernel-gradient L-BFGS polish of the chosen 60.

**Enables.** No learned model in the objective — divergence between claimed and real ΔU cannot
occur. (1−1/e) near-optimality certificate w.r.t. the discretization; an *ordered* detector list
with a marginal-gain curve (what detector #61 is worth — directly useful for costing the array);
greedy state (a 60-list + priority queue) is trivially preemption-safe. The recon's
permutation-sensitivity stops being an objective artifact because nothing optimizes against
ordering anymore. Honest refusal: the recon-quality terms of U are *not* submodular and not even
a set function under an order-sensitive recon — so it replaces them rather than pretending.

**Sacrifices.** Fisher log-det is a proxy for the composite U (rewards information content, not
U's saturating shapes) — needs weight calibration against `eval_true_utility` on a handful of
layouts; the energy Fisher row needs an approximation; discretization gap (patched by polish);
throws away the surrogate infrastructure.

**Choose if** you value a defensible layout with a certificate and a marginal-gain curve over
squeezing the last unit of a proxy — and your own permutation diagnostic already says the
recon-based terms are artifact-dominated. Avoid if you have evidence the recon terms encode
physics Fisher information misses.

### D. KATR — Kernel-Anchored Trust Region *(composition gap-fill explorer; surrogate STAYS in the gradient path)*

**Core mechanism.** Classical multifidelity trust-region model management (Alexandrov/TRMM,
TuRBO/MBPO family): the frozen FNN+recon stack is a *low-fidelity gradient generator*. L-BFGS
maximizes an ensemble lower-confidence-bound (S=4–5 reseeded stage-02 surrogates:
`mean_s U_s − κ·std_s − leash(ρ)`), each round leashed to ≤ ρ meters/detector from the
incumbent; the round's top candidate is accepted or rejected by **one no-grad kernel-oracle
call** (`KernelDualLabels`, full corpus) via the trust-region ratio test — ρ doubles on verified
gains, halves on artifact-mined ones. A residual ledger of (layout, U_true − U_surr) pairs
additively corrects the objective. Per-event fidelity never needed: the oracle anchors the
corpus-averaged 120-dim → scalar map. New code ≈ one ~150-line driver (`05_anchor_loop.py`);
04's existing `--init_from`/`--fnn_folder` hooks already anticipate it.

**Enables.** Monotone true-U by construction (adoption requires kernel confirmation; the
+6.9/−1.4 failure is impossible as an *accepted* outcome). Honest stalling: pure-artifact
gradients collapse ρ and the loop reports "no trustworthy direction" instead of a fake gain.
The aleatoric floor stops mattering. Exploits the measured fact that surrogate gradients are
directionally right far from optima (center-start +27.4). Survives a future where ground truth
becomes non-differentiable CORSIKA re-simulation — the only approach here that does.

**Sacrifices.** Sequential rounds (~30–50 oracle calls); local search only (run from both grid
and center incumbents); total movement bounded by Σρ; S× stage-02 training cost.

**Choose if** ground truth is (or will become) an expensive non-differentiable oracle with a
small call budget — i.e., precisely when approaches A–C's kernel-in-the-loop assumption breaks.
If even ~30 oracle calls is over budget, go direct layout→U_true GP (TuRBO proper) and demote
the FNN to warm-starts.

---

## 5. Shared assumptions across all approaches (blind spots)

1. **The plane-aware Gaussian kernel = ground truth.** Every approach optimizes or verifies
   against `compute_labels_batch`. If the kernel misrepresents real detector response (timing
   model, plane weighting, σ_spatial), all six inherit the error. Only KATR is structurally
   ready for a truth upgrade.
2. **The composite U is the right physics objective.** Weights (W_THETA=W_PHI=1e2, W_E=2.5e2),
   the saturating U_angle/U_E shapes, and the inert reconstructability gate (r saturates at 1;
   u_pr computed but omitted) were taken as given. Only Fisher-Greedy partially questions it —
   by substitution, not by physics review.
3. **The corpus = the flux.** All optimize on corpus events; generalization to the true tau flux
   is assumed, not tested (importance weights/energy spectrum reweighting unaddressed).
4. **60 detectors, fixed count, positions-only.** Nobody optimized count, per-detector cost, or
   detector type; only Fisher-Greedy's marginal-gain curve touches it as a byproduct.
5. **Stages 00/01 (corpus generation, placement) are correct.** All build on
   `place_clouds_enu` + the C8 geometry as-is.

## 6. Unexplored territory

- **Redesigning U itself** — a physics review of the composite (the W_PR=5e5 term is dead code
  in the composite; the gate never gates). Any approach above optimizes whatever U is; if U is
  wrong, all six succeed at the wrong thing.
- **Temporal/staged designs** — deploy a sub-array, learn from real data, extend (the temporal
  explorer method was not spawned; plausibly relevant for a real observatory build-out).
- **Global gradient-free search on the sketched kernel** — the repo's existing DE optimizers
  (`04_optimize_differential_evolution*.py`) pointed at a compacted kernel objective would test
  whether the surrogate's smoothness was ever needed for *global* structure.
- **Truth upgrade path** — per-layout re-simulation as the oracle (KATR-ready, otherwise open).
- **Flux reweighting / robustness** — optimizing expected U under flux-model uncertainty rather
  than the empirical corpus measure.

## 7. Orchestrator's note on sequencing (not a seventh approach)

The approaches disagree about the loop but agree about three preparatory facts worth settling
first, cheap, in this order:

1. Un-`no_grad` `compute_labels_batch` (flagged variant) and time one fwd+bwd minibatch on the
   A100 — decides A/B feasibility with numbers (fact 1's arithmetic, verified).
2. Run the voxel/superpoint compaction fidelity check on ~512 events — decides A1 vs A2 vs B1's
   compression knob.
3. Re-run the per-species aleatoric floor (`plots/compute_aleatoric_floor.py`) against stage-2
   val loss — decides whether any surrogate-centric path (D, or "just train it better") is even
   alive. This is B2's explicitly named fork in the road.

All three fit in one interactive GPU session and none commits to a design.
