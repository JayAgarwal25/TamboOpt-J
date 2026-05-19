# Orchestrator diary

## 2026-05-12 12:50 — picked 0000 as parent for 0001
First expansion of the tree. Only node `0000` is `done` (val_mse=0.7062,
val_mse_E=0.658, val_mse_T=0.893). UCB1 trivially picks the baseline.
Allocated `0001` via `new_node.sh 0000`. No axis_hint passed to the
Researcher — with zero mutation history, no axis is over-explored.
Side observation for context (not enforced): `val_mse_T` (0.893) is
the bottleneck vs `val_mse_E` (0.658); the Researcher may organically
land on a loss-axis lead (L1 uncertainty weighting or L2 Huber-on-T)
or a stabler-training lead, but the choice is theirs.

## 2026-05-12 13:43 — 0001 done: OneCycleLR beats baseline
val_mse_total: 0.6978 (baseline 0000: 0.7062, **-1.2%**).
val_mse_E: 0.5994 (baseline 0.6583, **-9.0%** — the bigger win).
val_mse_T: 0.8873 (baseline 0.8931, -0.6% — T remains the bottleneck).
Adam best 0.7433 at epoch 95 (baseline Adam best 0.7757, **-4.2%**); L-BFGS
added another -6.1% on top. Wall 1629s (baseline 1830s — faster too).
The risk flagged by the Researcher (NaN during warmup) did not trigger.
Saturation pattern persists: adam_best_epoch is still 95/100, suggesting
OneCycleLR is helping but there is still room. T-channel is now the
clear bottleneck (val_mse_T / val_mse_E = 1.48). Next iteration should
probably target T directly (L2 Huber-on-T or L1 uncertainty weighting).

## 2026-05-12 13:55 — picked 0001 as parent for 0002
UCB top-2: 0001 s=0.126, 0000 s=0.118. 0001 has zero children and lowest
val_mse so far. Allocated 0002. Axis_hint to Researcher: prefer `loss`
or `architecture` to widen axis coverage (we've only sampled `optimizer`
so far). The most direct lever for T (the current bottleneck,
val_mse_T/E=1.48) is the loss axis — refs/notes.md L1 (uncertainty
weighting) or L2 (Huber on T). But leave the final call to the
Researcher.

## 2026-05-12 14:30 — 0002 done; mixed-unit caveat resolved
0002's training-objective metrics (E=MSE, T=Huber, total=0.2817) are
NOT comparable to 0000/0001 because the loss formulation changed. Wrote
`scripts/eval_pure_mse.py` and recomputed val MSE on the saved fnn.pt
for both channels. Updated `runs/0002/metrics.json` (original
training-objective fields preserved under `*_train_objective` suffix)
and re-ran `append_row.py` to refresh the index.

**Pure-MSE result for 0002:** total=0.7183, E=0.5115, T=0.9251.
- E improved 14.7% vs 0001 (0.5994 → 0.5115). The MSE+Huber blend
  shifted optimization budget toward E.
- T regressed 4.3% (0.8873 → 0.9251). Huber's flat-gradient regime in
  the tail did exactly what the risk flagged: under-weighted the T
  tail, which apparently contained real signal rather than outliers.
- Net total regressed 2.9%. 0001 remains the best.

Lesson: Huber-on-T with delta=1.0 is too aggressive; the T tail is
mostly signal. Future loss-axis mutations should either (a) tune delta
much higher (e.g. 3.0 = 3-sigma), (b) try uncertainty-weighted MSE
(refs/notes.md#L1) instead of Huber, or (c) reweight without changing
the loss family (e.g. modify the 0.5/0.5 blend in `mse_normalized`).
Also: I should integrate the pure-MSE recompute into `run_eval.py` so
future loss-mutations auto-produce apples-to-apples metrics. Tracked
as a follow-up; running it manually for now.

## 2026-05-12 14:32 — picked 0001 as parent for 0003
UCB top-3: 0001 s=0.340, 0000 s=0.331, 0002 s=0.319. 0001 still best.
0002's regression confirms Huber-on-T with delta=1.0 is not the right
loss-axis lever; the search should diversify. axis_hint to Researcher:
**prefer `architecture`** (untried). Loss axis just failed; optimizer
already explored. Strong leads in refs/notes.md: A1 (GELU+LayerNorm),
A3 (wider net hidden=768), A2 (residuals). Researcher picks among
these or another well-motivated architecture lead.

## 2026-05-12 14:35 — user steer: target tight diagonal in pred-vs-target scatter
User observation: both E and T scatter plots are far from y=x. Translating
to the metric: baseline z-scored val_mse_E≈0.66 (R²≈0.34) and val_mse_T≈0.89
(R²≈0.11). The model is only marginally better than predicting the channel
mean, especially on T. 1-2% improvements from optimizer/loss tweaks won't
move the plots meaningfully.

**Durable guidance for the Researcher (post-0003):** prefer larger-magnitude
levers over small refinements. Concretely:
- A3 (wider hidden=768 or 1024) — direct capacity bump.
- A4 (Deep Sets, weights shared per detector + pooled context) — bakes in
  permutation equivariance, the augmentation is approximating it expensively.
- X1 (Set Transformer over per-detector tokens) — strict upper bound on A4;
  adds learnable pairwise interactions.
- A2 (residual blocks 4-6 deep) — lets us safely deepen the net.

Optimizer/loss tweaks are diminishing-returns territory for now; only return
to them once the architecture is competitive (val_mse_total < ~0.4).

## 2026-05-12 17:48 — picked 0003 as parent for 0004
UCB top-3: 0003 s=0.94 (no children → big bonus), 0000 s=0.47, 0002 s=0.45.
0003 chosen by UCB. Strategic note: 0003's Adam-only val (0.7108) is the
best Adam result so far; the architecture is competitive even though
L-BFGS overfit at the end. Capacity-bump on top of this conditioning win
should compound.

axis_hint to Researcher: **A3 (wider hidden=768 or 1024)** strongly
preferred per user-driven steer recorded above. Falling back to A4 (Deep
Sets) or A2 (residuals) acceptable. NOT optimizer/loss again.

## 2026-05-12 17:50 — infra fix: clean patch.diff generation
Discovered that `new_node.sh` left `.git/` directories inside work dirs
after applying parent patches, and `train.sh`'s `diff -ruN` swept the
`.git/` tree into `patch.diff`. This broke `new_node.sh` for any
grandchild (it tried to `git apply` a patch full of `.git/COMMIT_EDITMSG`
hunks, which failed silently and left the child's work dir as raw
baseline). Discovered when 0004 came up with no patches.

Fix:
1. `train.sh`: `diff -ruN` now excludes `.git`, `__pycache__`, `output`,
   `train.log`, `metrics.json`.
2. `new_node.sh`: `rm -rf .git` after `git apply` succeeds (or fails).
3. Regenerated `runs/0003/patch.diff` cleanly (79 lines, just the two
   real edits).
4. Deleted the empty 0004 directory.
5. Reallocated as 0005 (work dir verified to contain both
   parent patches). Will proceed with 0005 as the next node.

Note for the future: 0002's patch.diff is also polluted but 0002 has
no children yet and is no longer the most-favored UCB parent
(currently 0001 / 0003), so I'm not regenerating it pre-emptively.
If/when 0002 becomes a parent, regenerate first.

## 2026-05-12 18:25 — 0005 done; picked 0005 as parent for 0006
0005 (hidden=1024 on top of 0003) finalized at val_mse_total=0.6989 —
essentially tied with 0001's 0.6978 (within 0.16%). Big E gain (0.536,
-10.5% vs 0001) but L-BFGS overfit again (final 0.832; saved Adam-best
0.6989). Pattern across 0003 & 0005: capacity helps Adam, L-BFGS
overfits the GELU+LayerNorm net. MLP family is plateauing.

UCB top scores: 0005 s=1.077 (best score, no children), 0001 s=0.56,
0003 s=0.55. UCB picks 0005.

**Strong axis_hint for 0006: out-of-genre or architecture (structural).**
- The MLP family has converged on val_mse≈0.69-0.71. Further width or
  depth tweaks will not break this plateau.
- The right inductive bias is permutation-equivariance across detectors:
  A4 (Deep Sets) or X1 (Set Transformer). Both reframe the 100 detectors
  as a set rather than a flat 200-d vector.
- A4 is the lower-risk option: shared per-detector MLP + pooled context.
- X1 (Set Transformer) is the upper bound; needs more code.
- Falling back: A2 (residual blocks 4-6 deep) would also break the
  current MLP plateau if Researcher prefers minimal-risk.

Also patched scripts/plot_target_vs_pred.py and scripts/eval_pure_mse.py
to read `hidden` from ckpt['config'] (was hard-coded to 512). Future
nodes with arbitrary hidden width will now plot/recompute fine.

## 2026-05-12 21:05 — 0006 done (Deep Sets, MAJOR WIN); picked 0006 as parent for 0007
**0006 = 0.5459 — new best by 21.8% vs 0001 (0.6978).** E dropped 40%
(0.599 → 0.359), T dropped 17% (0.887 → 0.733). The permutation-
equivariance inductive bias was exactly the right lever. User's
2026-05-12 14:35 steer toward structural changes proved decisive.

Caveats:
- L-BFGS crashed with CUDA OOM (full-batch forward = 21.5 GiB; 100
  detectors × 450k samples × token activations blew past the 40 GiB
  A100). Adam-best ckpt recovered via eval_pure_mse.py; metrics.json
  manually patched (status: failed → done) and index refreshed.
- Adam best_val was at epoch 55/100 — there's clear headroom for more
  Adam training or a less-aggressive cosine decay.

UCB top scores: 0006 s=1.33 (best score, no children → big bonus),
0005 s=1.18, 0001 s=0.63. UCB picks 0006.

**axis_hint for 0007:** Two equally good directions:
(a) **Widen Deep Sets** — psi/phi hidden currently = `max(32, hidden//8) = 128`
    at `hidden=1024`. Bumping to `hidden//4 = 256` or `hidden//2 = 512`
    doubles/quadruples per-token capacity within the same equivariant
    template. Adam saturated at epoch 55, suggesting capacity-limited.
(b) **X1 Set Transformer** — strict upper bound on A4. Adds learnable
    pairwise interactions between detector tokens (attention) where
    Deep Sets only has mean-pooling. Bigger rewrite, bigger payoff.
(c) **Skip L-BFGS** or curtail it for this family — every node with
    >~700k params has had L-BFGS overfit. For Deep Sets, the OOM also
    prevented L-BFGS from running. Could add a `LBFGS_MAX_ITER=0` or
    `--skip-lbfgs` knob via the loss-axis. (training-loop axis)

Researcher picks. Falling back to (a) is lower-risk; (b) is the bigger
ceiling. The L-BFGS OOM is an infra issue to address separately —
should chunk the full-batch closure forward into mini-batches with
gradient accumulation.

## 2026-05-13 10:30 — 0007 done; expand 5 complete
0007 (Deep Sets 4× width on top of 0006) finalized at val_mse_total=0.5476 —
**0.3% WORSE than parent 0006 (0.5459)**. Researcher's risk #1 falsified
exactly: capacity is NOT the bottleneck. Adam-best epoch shifted EARLIER
(35 vs 55) — wider model converged faster to a worse minimum. Two
hypotheses, consistent with risk-card analysis:
1. The OneCycleLR schedule is the bottleneck (not width), saturating Adam
   before optimum is reached.
2. The mean-pool inductive bias is the bottleneck — pairwise interactions
   (X1 Set Transformer) are needed to capture shower-wide T structure.

T channel basically unchanged (0.733 → 0.736; +0.5%), E ~tied (0.359 → 0.359).
The wider context vector did NOT help T — strongly supports hypothesis 2.

Also: 0007 wall=4474s (~75 min) — way over the 30-min eval cap due to
~57s/epoch for the wider model (vs 12s/epoch for 0006). Future Deep-Sets-
family nodes need to consider compute budget; deepening or attention may
be slower still.

L-BFGS OOM again (85 GiB attempted at ~1M params × 100 detectors × full
batch). The OOM is now a structural blocker for any non-tiny Deep Sets
node. Two infra fixes worth queuing as orthogonal mutations (not part of
this expand session):
- (i) Chunk the L-BFGS closure forward into mini-batches with gradient
  accumulation — keeps L-BFGS but trades wall time for memory.
- (ii) Add `LBFGS_MAX_ITER=0` knob to skip L-BFGS for set-architecture
  nodes that don't benefit from it.

**Final report:**
- Best: 0006 (Deep Sets, val_mse_total=0.5459, -22.7% vs baseline)
- Lineage: 0006 → 0005 (hidden=1024) → 0003 (GELU+LayerNorm) → 0001 (OneCycleLR) → 0000 (baseline)
- Next direction (post-expand-5): **X1 Set Transformer** on top of 0006,
  paired with the L-BFGS-OOM fix (skip or chunk). Capacity exhausted at
  this width; need attention for the T-channel.

## 2026-05-13 11:00 — /search expand 5 (radical changes), iter 1/5: picked 0006 as parent for 0008
UCB top-2: 0006 s=1.407, 0007 s=1.405 (essentially tied — 0006 wins by
val_mse since both have one child). Picked 0006 because it's the
genuine best and 0007 confirmed width-saturation. User explicitly
steered toward **radical structural changes** + **arxiv literature
search** + freedom to grow/shrink the network.

Plan for 5 iterations (subject to UCB updates):
- iter 1/5 (0008): X1 Set Transformer (self-attention over detector
  tokens) — the natural next step per 0007's residual diagnostic
  (T-channel did not improve from wider mean-pool).
- iter 2/5: GNN over k-NN detector layout (per refs/notes.md#X2 —
  exploits spatial xy locality).
- iter 3/5: Equivariant CNN / spatial conv over a sparse 2D image of
  the detector array.
- iter 4/5: Skip-L-BFGS / chunked-closure infra fix (training-loop)
  to unlock L-BFGS on Deep Sets family.
- iter 5/5: TBD based on prior results — likely an MoE per-detector-
  cluster specialist or a physics-informed loss.

axis_hint for Researcher: **STRONGLY prefer X1 Set Transformer or
another attention-based set-equivariant model**. Authoritative refs:
Lee et al. 2019 (Set Transformer), Vaswani et al. 2017, Zaheer 2017
(Deep Sets ⊂ Set Transformer). Also encourage arxiv searches for:
"detector array surrogate", "Cherenkov detector neural network",
"shower neural network reconstruction", "point cloud regression
neural network", "equivariant set neural network".

Network-size guidance:
- 0007 at ~934k params hit the inductive-bias wall (not capacity).
- Set Transformer at ~74k-300k params is appropriate.
- DO consider compute budget: 0007 took 75 min (4474s) which exceeded
  the 30-min target; aim for a model that hits ~25-30 min total.

L-BFGS OOM: kept as a known infra failure mode; Adam-best fallback
recovers metrics. Will address as orthogonal iter 4/5 mutation.

## 2026-05-13 11:05 — switched to SEARCH_PARALLEL=3; burst 1: 0008/0009/0010 from 0006
User explicitly asked for 3 parallel agents. Allocated 0009, 0010
(0008 already allocated). All 3 from parent 0006 (current best). Three
distinct axis hints to maximize information per burst:

- **0008: X1 Set Transformer** — replace Deep Sets mean-pool with
  multi-head self-attention between detector tokens. The natural
  pairwise-interaction upgrade per 0007's residual diagnostic.
  Refs: Lee et al. 2019 (arXiv:1810.00825), Vaswani et al. 2017.
- **0009: GNN over k-NN xy-distance graph** (refs/notes.md#X2).
  Exploits the spatial 2D detector layout. Refs: Hamilton et al.
  GraphSAGE (1706.02216) or Veličković et al. GAT (1710.10903).
  Implement via torch.scatter (no torch_geometric dep needed).
- **0010: training-loop infra fix** — chunk the L-BFGS closure
  forward pass into mini-batches with gradient accumulation, OR
  add a `LBFGS_MAX_ITER=0` skip knob. Unblocks the OOM that has
  prevented L-BFGS on every Deep Sets node (0006, 0007 both lost
  the L-BFGS gain phase). Orthogonal: applies to all Deep Sets
  descendants.

GPU sharing: 40 GiB A100, each Adam phase needs ~3-5 GiB, three
simultaneously = ~9-15 GiB. Fits. L-BFGS full-batch OOMs are
likely on 0008/0009 — Adam-best fallback recovers; 0010 should
be the only one that successfully runs L-BFGS.

## 2026-05-13 11:50 — burst 2 (5 more parallel): 0011-0015 from 0006
User confirmed GPU has headroom for 5 more (total 8 concurrent).
Allocated 0011-0015, all from parent 0006 (best). Each gets a distinct
radical-axis hint:

- **0011: ISAB Set Transformer w/ inducing points** — scales SAB via
  learned inducing points (Lee et al. 2019 §3.2). O(n·k) attention
  vs SAB's O(n²). Tests whether SAB's full attention is what helped
  0008 vs a leaner alternative.
- **0012: Cross-attention with primary as query** — primary (5-d)
  attends to all detector tokens. Models how the shower direction/
  energy modulates the per-detector readout. Different from
  detector-self-attention.
- **0013: Residual Deep Sets (4 stacked blocks)** — depth instead of
  width per refs/notes.md#A2. Tests whether residual depth in the
  equivariant template breaks the 0.5459 plateau.
- **0014: Learned uncertainty weighting** (refs/notes.md#L1) — two
  log-sigma scalars per E/T channel; loss = exp(-s)·MSE + s. Directly
  addresses T being 2× harder than E (val_mse_T/E=2.04 on 0006). With
  Deep Sets as the substrate, this is the right time to revisit the
  loss axis (per the durable 2026-05-12 14:35 steer: "only return to
  loss once architecture is competitive — done").
- **0015: AdamW + weight decay + larger LR_MAX** (refs/notes.md#O1) —
  0007's wider net saturated Adam at epoch 35; 0006 at 55. Decoupled
  weight decay regularizes wide nets, and bumping LR_MAX from 1e-4 to
  3e-4 may push Adam past its current floor in the Deep Sets family.

Decision rule: any mutation that improves on 0006's 0.5459 by >5%
becomes a strong candidate for further iteration; gut threshold for
"good enough to be the new line" is val_mse_total < 0.40 (the user's
durable threshold for "architecture is competitive").

## 2026-05-13 12:21 — burst-2 cancelled; ALL recovered Adam-best are tied with 0006
User cancelled 0008/0009/0011/0014/0015 mid-training; recovered their Adam-best
checkpoints via `eval_pure_mse.py`. Striking result: **every recovered val_mse
is within 0.5% of 0006's 0.5459**, regardless of architecture/loss/optimizer:

| Node | E | T | total | Δ vs 0006 |
|------|------|------|--------|-----------|
| **0008 SAB Transformer**   | 0.3561 | 0.7301 | **0.5431** | **-0.5%** |
| 0015 AdamW+3x LR           | 0.3577 | 0.7320 | 0.5448 | -0.2% |
| 0009 GNN k-NN              | 0.3584 | 0.7312 | 0.5448 | -0.2% |
| 0011 ISAB                  | 0.3580 | 0.7325 | 0.5452 | -0.1% |
| **0006 Deep Sets (parent)**| 0.3591 | 0.7327 | **0.5459** | 0.0% |
| 0014 uncertainty wt         | 0.3578 | 0.7343 | 0.5460 | +0.0% |
| 0007 4x wider Deep Sets    | 0.3589 | 0.7363 | 0.5476 | +0.3% |

Three observations:
1. **The 0006 plateau is a noise floor or data ceiling**, not an
   architecture limit. Pairwise attention (0008), spatial locality (0009),
   inducing points (0011), uncertainty weighting (0014), and 3x peak LR
   (0015) ALL produce within ±0.3% of each other. This is shocking
   tightness across radically different inductive biases.
2. **T-channel is stuck at 0.730-0.736** across every method — this is
   the harder floor. E settled at 0.356-0.359. T/E ratio ~2.05.
3. The Adam-best ckpts were saved at unknown epochs (killed early),
   but 0008/0009/0011 were running ~80 min wall vs 0006's 28 min solo;
   under heavy GPU contention they likely got ~40 min of effective
   compute, comparable to a full solo Adam phase.

**Hypothesis for the floor:** the labels (E, T) have inherent
stochasticity from the upstream Cherenkov shower simulation; the
remaining ~0.55 z-scored MSE may be the irreducible Bayes risk for
this dataset, not a model-capacity issue. To verify: estimate label
noise via repeated showers with the same primary, OR add more
training data, OR fundamentally change the input features.

**Next directions worth trying (not now — user paused):**
- (a) **More data / better features**: add per-detector neighbour
  features, or aggregate features at the shower level before per-
  detector prediction.
- (b) **Probabilistic head**: predict mean + variance for (E, T),
  use negative log-likelihood instead of MSE. Lets the model report
  its own uncertainty.
- (c) **Multi-seed ensemble**: train 5 Deep Sets with different
  seeds, average predictions. If val plateaus < 0.55 on the ensemble,
  it's noise-floor; if it stays at 0.55, it's a true ceiling for the
  architecture family.
- (d) **Investigate data**: compute the per-detector residual
  distribution on 0006 to see whether T errors are concentrated on
  specific detector positions / shower types (could reveal a feature
  gap).

## 2026-05-13 17:55 — 0017/0018/0019 all regressed: zero-inflation handling HURTS on the leaderboard metric
Three approaches to handle T zero-inflation, all from 0006 (Deep Sets, 10% data):

| Node | Approach | val_total | val_E | val_T | Δ vs 0006 |
|------|---------|-----------|-------|-------|-----------|
| 0006 | baseline | 0.546 | 0.359 | 0.733 | — |
| 0017 | soft reweight (log1p) | 1.049 | 0.463 | 1.636 | +92% (T +123%) |
| 0018 | hit gate + BCE | 0.830 | 0.432 | 1.228 | +52% (T +68%) |
| 0019 | hard mask only | 0.939 | 0.450 | 1.428 | +72% (T +95%) |

All three REGRESSED on the apples-to-apples pure-MSE leaderboard.

**Crucial revised understanding:** The val_mse_T=0.73 floor on 0006 is
dominated by the 96.6% **zero-target positions** where the model
correctly predicts near-zero. The 3.4% non-zero positions contribute
proportionally LESS to the total MSE despite their large individual
errors. When we mask/reweight to focus on non-zero positions, the
model loses gradient signal at zero positions, those predictions
degrade, and the overall metric WORSENS even though the model gets
"better at the hard cases".

This is the opposite of what the diagnosis predicted. Zero-inflation
is NOT a model-fitting bottleneck; the model already does well on
the bulk. The remaining val_mse_T floor reflects irreducible
difficulty of the non-zero T values — either:
- Label noise in the upstream simulation
- A feature gap (the model needs info beyond primary + xy to predict T)
- Genuine stochasticity in Cherenkov timing

The right next step is probably one of:
- Inspect TRAIN vs VAL T residual distribution (is this train/val noise
  or a genuine ceiling?)
- Add features: per-detector neighbour info, propagation time of flight
  from shower axis to detector, etc.
- Multi-seed ensemble to estimate label-noise floor
- Predict T uncertainty (Gaussian NLL) as a diagnostic

## 2026-05-14 11:02 — spawned 5 children of 0022 (S1-S5 design)

Design doc: docs/designs/next-5-iterations.md. All five branch from 0022 (new SOTA at val_mse 0.472, log-T canonical). Launched in tmux session `tambo_train`, one window per node, staggered 30s, each running `scripts/run_node_logT.sh <id> <T_LOG_SCALE>` (pipeline: train.sh + eval_pure_mse_log_t_canonical + append_row).

- 0023 (S1, optimizer): AdamW(wd=1e-4) + OneCycleLR max_lr 3e-4. Single-knob, lowest risk.
- 0024 (S2, T_LOG_SCALE): 1e8 -> 1e9. Hit-median maps to log1p(0.62), peak-gradient region. Note: NOT directly comparable to 0022's metric (different z-space).
- 0025 (S3, hurdle): phi outputs (E, gate, amount); T_pred = sigmoid(gate)*amount in log-T. BCE + masked MSE replaces standard T MSE.
- 0026 (S4, capacity): channel widths hidden//8 -> hidden//4 (psi/phi/token/context). ~4x params.
- 0027 (S5, stacked): AdamW+wd + OneCycle 3e-4 + Kendall task weighting, all on top of 0022.

Shared assumptions: parent=0022, Deep Sets, 10% data, Adam->L-BFGS pipeline. Unexplored deferred: Set Transformer under log-T, alternate target transforms (sqrt/arcsinh), L-BFGS removal, full-data SOTA re-eval.

## 2026-05-14 (later) — spawned regularization branch from 0026 (R2 + R4)

Design doc: docs/designs/regularization-from-0026.md. 0026's train log shows clean overfit inflection at epoch 53 (val best=0.4672, train continues to fall through epoch 100 with val rising by 0.005). Sibling 0023 documented L-BFGS amplification (val 0.46->0.51 over 228 iters). Five regularization approaches explored; user picked R4 + R2.

- 0028 (R4, strip recipe): remove L-BFGS, OneCycleLR, gradient clip. Constant LR=1e-4, patience=10 early stop. ~150 lines removed, 6 added. Lowest novelty, most evidence-supported.
- 0029 (R2, SWA): AveragedModel shadow from epoch 30; post-Adam, if SWA val < Adam best, save SWA's state_dict as fnn.pt for L-BFGS handoff. ~30 lines. AveragedModel's use_buffers=False default preserves set_normalization buffers.

Both branch from 0026 (current 2nd-place SOTA at 0.4672). Launched in tmux session `tambo_reg`, staggered 30s. Pipeline run_node_logT.sh -> train.sh + eval_pure_mse_log_t_canonical + append_row.

Hypothesis: R4 will win on val_mse_total since it directly cuts the documented L-BFGS overfit; R2 will reduce the train/val gap during Adam but may have its gain re-eaten by L-BFGS.

## 2026-05-14 (later) — baseline patched + 0030 spawned (L-BFGS-best + HISTORY=5)

User question revealed a systemic bug: across 0010/0023/0024, L-BFGS hit a min 20-50 iters in (0.003-0.006 below Adam-best) then drifted up; the recipe only checked LAST iter and threw the gain away. 0026 OOM'd at L-BFGS iter 0 so we never got even the Adam-vs-L-BFGS comparison on the widened model.

Changes:
1. `baseline/02_train_fnn.py` — patched the L-BFGS closure to track lbfgs_best_val and save fnn.pt whenever val improves inside the closure. Post-loop logic simplified (fnn.pt already holds best). This is a strict-positive change for all future nodes built fresh from baseline.

2. New_node.sh side effect: 0026's patch.diff failed 3-way merge on the patched baseline (line 454 area conflicts with 0026's inherited save block). Workaround for now: manually rebuild new nodes from existing parents via `cp -R runs/<parent>/work runs/<new>/work` then port the baseline's L-BFGS-best patch manually. TODO: fix new_node.sh to handle baseline drift, or regenerate 0026's patch.diff against new baseline.

3. 0030 spawned from 0026 work-dir (manually rebuilt). Has 0026's mutations + new save-best logic + HISTORY_SIZE=5. Solo on GPU; expect: Adam best ~0.4672 then L-BFGS to actually run (vs 0026's OOM), capture per-iter min, hopefully <0.4670.
