# FNN surrogate — development log

Chronological record of the architecture/hyperparameter search for the Step-2
forward surrogate (`02_train_fnn.py`). Metric is z-scored validation MSE
(`val_mse_total`, lower = better), split into E and T channels. The search ran
on a 10% data subset for fast iteration; production uses the full 3.5M-pair
corpus, so absolute numbers below are **not** directly comparable to production
runs — read them as relative deltas within the search. Durable conclusions are
distilled into THEORY.md §10.

## Baseline and first wins (2026-05-12)

- **0000 baseline**: val 0.7062 (E 0.658, **T 0.893** — T is the bottleneck).
- **0001 OneCycleLR**: 0.6978 (E −9%, T flat). Best MLP so far. Adam still
  saturates at epoch 95/100 → schedule, not just optimizer, has headroom.
- **0002 Huber-on-T (δ=1.0)**: regressed (T +4.3%). The T tail is *signal*, not
  outliers; clipping it under-weights real structure. Lesson: don't clip T.

**User steer (decisive):** both E and T pred-vs-target scatters sit far off the
`y=x` diagonal — R²_E ≈ 0.34, R²_T ≈ 0.11; the model barely beats predicting the
channel mean. 1–2% optimizer/loss tweaks won't move the plots. Prefer
large-magnitude levers (capacity, **Deep Sets**, Set Transformer) until
`val_mse_total < ~0.4`.

## MLP plateau → Deep Sets breakthrough (2026-05-12/13)

- **0003 GELU+LayerNorm**, **0005 hidden=1024**: ~0.69–0.71. The flat-MLP family
  plateaus regardless of width/conditioning.
- **0006 Deep Sets** (per-detector shared MLP + pooled context): **0.5459,
  −22.7% vs 0001.** E −40% (0.599→0.359), T −17% (0.887→0.733). The
  permutation-equivariance inductive bias was exactly the right lever — the
  augmentation had been approximating it expensively. L-BFGS OOM'd on the set
  model (full-batch forward); later fixed by chunking the closure.
- **0007 wider Deep Sets (4×)**: no improvement (0.5476). **Capacity is not the
  bottleneck** — the wider model converged faster to a no-better minimum.

## The cross-method floor (2026-05-13)

A burst of radically different set-equivariant / loss / optimizer variants from
0006 — Set Transformer (SAB), GNN over k-NN xy graph, ISAB inducing points,
primary→detector cross-attention, learned uncertainty weighting, AdamW+3× LR —
**all landed within ±0.3% of 0.546**:

| Node | method | total | Δ vs 0006 |
|------|--------|-------|-----------|
| 0008 | SAB Transformer | 0.5431 | −0.5% |
| 0015 | AdamW + 3× LR | 0.5448 | −0.2% |
| 0009 | GNN k-NN | 0.5448 | −0.2% |
| 0011 | ISAB | 0.5452 | −0.1% |
| 0006 | Deep Sets (parent) | 0.5459 | — |
| 0014 | uncertainty weighting | 0.5460 | +0.0% |
| 0007 | 4× wider Deep Sets | 0.5476 | +0.3% |

Shocking tightness across distinct inductive biases ⇒ the floor is a **data/label
ceiling**, not an architecture limit. T sticks at 0.73–0.736, E at 0.356–0.359
(T/E ≈ 2.05). Leading hypothesis: irreducible label noise from the upstream
shower simulation, or a feature gap for T (the model may need info beyond
primary + xy — e.g. time-of-flight from shower axis).

## Zero-inflation handling backfires (2026-05-13)

Three ways to focus the loss on the rare non-zero T positions — soft reweight,
BCE hit-gate, hard mask — **all regressed** on the apples-to-apples pure-MSE
metric (+52% to +92%). The 96.6% zero-target positions dominate the MSE; masking
removes their gradient signal, those predictions degrade, and total MSE worsens
even as the hard cases improve. **Zero-inflation is not the fitting bottleneck** —
the model already handles the zero bulk well.

## log-T target + L-BFGS fix (2026-05-14)

- **log-T canonical target** (`T ← log1p(T·1e8)`): new SOTA **0.472** (0022),
  later 0.4672 (0026 widened). This is now the production T treatment (§6).
- **L-BFGS systematic overfit**: across nodes, L-BFGS hit its val min 20–50 iters
  in (0.003–0.006 below Adam-best) then drifted up; the recipe only kept the
  *last* iter and threw the gain away. **Fix (now in baseline):** save `fnn.pt`
  whenever the in-closure val improves, so the checkpoint is the global best.
- Regularization branch from 0026 (strip-recipe: drop L-BFGS/OneCycle/clip,
  constant LR + early stop; and SWA shadow averaging) explored to counter the
  documented L-BFGS overfit.

## Path-c schedule de-risk on production flat MLP — FAILED (2026-06-04)

Ran on the **full-data production flat MLP** (`slurm-19088112.out`): LR-range
test → AdamW(wd=1e-5) + dropout=0 + raised LR_MIN + L-BFGS capped at 800, plus a
fix to a latent OneCycle `final_div_factor` bug (floor was hitting ~1e-8). The LR
test diverged past ~3e-3 and fell back to LR_MAX=5e-4.

**Result: regressed.** val_total 0.6009 (L-BFGS best iter 666; Adam was 0.6646)
— worse than the prior recipe's 0.40, and the honest conditional metrics are
damning:

- E R² (all detectors) 0.447 — flattering (dominated by trivially-correct empties)
- **E R² (fired only) −0.140** — magnitude is *worse* than predicting the fired mean
- fire precision 0.415 / recall 0.989 — the model **over-fires**, leaking energy
  onto empty detectors
- fired pred/target std 0.692 — clear magnitude compression (predict-the-mean)

Schedule/optimizer tuning on the flat MLP cannot fix this; it re-confirms the
0006-era plateau. **All path-c changes were reverted.** Standing recommendation:
**path (a) — rewrite the FNN as a pointwise DeepSets `φ(q, xᵢ, yᵢ)→(Eᵢ, Tᵢ)`**
(permutation-equivariant by construction, deletes the augmentation, ~34× fewer
params, ~100× more effective samples). This matches the 2026-05 search's verdict
that Deep Sets is the only lever that broke the MLP plateau.
