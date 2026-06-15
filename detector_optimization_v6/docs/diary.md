# FNN surrogate — development log

Chronological record of the architecture/hyperparameter search for the Step-2
forward surrogate (`02_train_fnn.py`), plus (from 2026-06) pipeline-level
events that changed what the surrogate is trained on. Metric is z-scored validation MSE
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

## Checkpoint ↔ transformer pairing — silent blob regressions (2026-06-10)

Two days lost to the same invisible bug, hit from both directions. TAMBO-opt's
checkpoint generations were trained with DIFFERENT transformer encoder blocks
that share **identical state-dict keys**, so a checkpoint loads into the wrong
variant without any error and just generates diffuse blobs instead of rod-like
showers:

- old `all_showers` (Apr 3) → **post-LN**: `x = LN(x + attn(x))`
- new per-species e/µ models (May 19–20) → **pre-LN**: `x = x + attn(LN(x))`

The 06-09 dual-species plots were blobs (new ckpts on old post-LN code); after
pulling Hamza's pre-LN code the OLD checkpoint's plots turned to blobs instead.
Verified both directions by swapping `transformer.py` and regenerating angle
grids; Hamza's own `ml_electron_test.h5` (rods matching sim) pinned the new
ckpts to his code. **Fix is per-checkpoint, not global**: `transformer.py`
takes `pre_ln: bool = False`, and the dual-species staging injects
`pre_ln: true` into the staged conf.yaml. Two companions from the same pass:
`generator.py` needs `with_time` (all current ckpts are time models,
`dim_inputs[0]==4` — the pull stripped it → AttributeError), and `generate()`
must run under `no_grad` (each batch otherwise retains its ODE autograd graph:
39 GB OOM on an A100 at the 4096-pt electron cap; hopeless at muon 25088).
**Lesson: identical tensor shapes + different semantics = silent garbage; pin
architecture flags inside the checkpoint's conf, never in the code version.**

## Dual-species paired pipeline (2026-06-11)

Provenance check on the per-species training files changed the architecture:
`combined_electrons.h5` / `combined_muons.h5` are the SAME 130k simulated
showers matched row-for-row (identical energies/directions/ids), split by
**secondary component** — primaries are tau decay daughters (e±, π; no muon or
tau primaries), and muons survive the rock while EM is absorbed (caps 25088 vs
4096). So "species" = component of one event, and a single event deposits BOTH.
Also: both models were trained with conditioning **label 0 only** (training
`pdg` all-zero) — sampling labels 0/1 at generation (the old wrapper default)
conditions half the corpus on an untrained embedding.

Pipeline rewired end-to-end (committed `f7e2698..ca4f5f3`):

- **00**: paired corpus — primaries sampled once, electron rows `0..N-1` +
  muon rows `N..2N-1` share them; streamed chunked HDF5 writes; labels always 0.
- **02**: TWO parallel DeepSets (`fnn_electron.pt` / `fnn_muon.pt`),
  per-species norm stats (µ counts ≫ e counts; shared stats would crush the
  e-channel loss).
- **`modules_v6/dual_surrogate.py`**: physical combination — counts add,
  times average count-weighted, in expm1/log1p space so all downstream units
  are unchanged; differentiable through both branches (unit-checked:
  100@10µs + 300@30µs → 400@25µs exactly).
- **03**: recon trains on the COMBINED predictions (complete events);
  stats computed from the data, not borrowed from a checkpoint.
- **04 (lbfgs + DE)**: optimize against the combined response; backprop flows
  through both models.

Full smoke chain (50 pairs → 01 → 02 → 03 → 04) passed on an A100. Open items:
muon PointCountFM predicts above the 25088 cap routinely (mean ~27k, seen 48k —
silent truncation); `build_training_pairs` still loads the whole corpus in RAM
(250 GB at 500k pairs — needs streaming before production).

## DE (North, East) validation — naming + x0-vs-bounds (2026-06-11)

`sample_initial_layout_ne` returns (North, **East**) but the DE script unpacked
into `N_np, U_np` — copy-paste from the L-BFGS sibling. Inside the script the
value is treated as East everywhere (bounds, projection, plots), so renamed to
`E_np`/`E_t`; cosmetic. The real crash: `project_to_mountain_ne` is a
*tolerance test* (keep anything within `max_gap` ≈ 170 m of a centroid), not a
bbox clamp, so σ=1000 m perturbed starts legitimately sit up to ~max_gap
OUTSIDE the tight centroid bbox — and SciPy requires `x0` inside `bounds`.
**Fix: widen the DE bounds by `_ne_max_gap(mountain)`** (candidates are
mountain-projected before scoring anyway). Verified with a 2-chain mini-run.
**Standing caveat**: no NE-trained Steps 2–3 exist — the DE script currently
scores (North, Up)-trained surrogates on East inputs (Up ∈ [2442, 3886] vs
East ∈ [−2019, 1182], disjoint), so its utilities are meaningless until the NE
chain (01_NE on the paired corpus → 02/03 retrained) is built.

## Production corpus run 21376182 — float-underflow vs the padding validator (2026-06-12)

First 500k-pair production run died 2.5 h in: electron block COMPLETE (~1.8 h),
muon at 20k/500k, then `showerdata.Showers()` raised "Padding should be in the
end of the shower points." while packaging an already-generated chunk.

**Root cause — a tail event the smoke runs could never catch.** The inverse
energy trafo is `exp(latent)`: mathematically positive, but float32 exp
underflows to EXACTLY 0.0 for extreme negative flow latents (~1-in-1e8 per
point; the run had generated ~5×10⁸ points). The validator is a slot-0 proxy
check, so an underflow zero only crashes when it lands at a shower's FIRST
point — and worse, *interior* zeros pass silently and then lose data: the
ragged save slices `[:num_points]` with `num_points = count_nonzero(e)`, so
one interior zero silently drops the shower's LAST real point.

**Fixes (verified by unit tests, no GPU needed):**

- **Stable partition in `_gen_chunk`**: key on `e ≤ 0`, `argsort(stable=True)`,
  then a `gather` that moves whole 5-feature points — real points first in
  original order, all zero rows (underflows + padding) at the end. Satisfies
  the validator AND kills the silent last-point drop. Checked against the real
  showerdata validator (rejects the broken input, accepts the partitioned one,
  `num_points` correct, x-values travel with their energies).
- **`--resume-at-row`**: continue a crashed run into the existing preallocated
  file from the last logged "file offset". Completed blocks are skipped
  outright (model never loads); a partial block resumes at its offset; seeded
  primaries re-pair exactly. Guards reject out-of-range rows / missing file.
  Unit-tested with mocked generators: fresh-run offsets, mid-muon resume
  (electron skipped, writes at rows 7,9 of a 10-row toy), mid-electron resume,
  and both guard paths.
- **`run_all_script_batch.sh`**: `run_step` now forwards args;
  `RESUME_ROW=520000` (the crashed run's last offset) is a visible knob — set
  0 for a fresh corpus. `--n-pairs`/`--seed` must not change across a resume.

**Scale data point**: ~8.4k muon-cap truncation warnings within the first ~22k
muon showers (counts up to 55k vs the 25088 cap) — roughly a third of muon
components clipped. Raised as a retraining question (higher cap) for the next
model round.

## High-E blob band = train/generate energy mismatch (2026-06-14)

Plotting the 1M-pair production corpus (`cashed_showers_dual_1000000.pt`), the
high-point-count muon showers render as diffuse isotropic **blobs** while the
rest are clean rods. **Not** a plot artifact and **not** the post-LN/pre-LN
checkpoint bug (that blobs *every* shower; here only a tail degrades). Root
cause is a sampling-vs-training **energy** mismatch:

- **Diagnostic** (300 muon showers, elongation = std along travel ÷ std across,
  in x-y): elongation is 3.4–4.6 for E < 1e7 (rods) and collapses to **~1.0 for
  E ∈ [1e7,1e8]** (isotropic blobs). `corr(log10 E, elong) = −0.62`,
  `corr(n_pts, elong) = −0.53`. A near-horizontal shower must stay elongated
  along its axis at any energy, so elongation 1.0 is unphysical — the generator
  is extrapolating.
- **Training spectrum** (`combined_{electrons,muons}.h5`, the actual May-ckpt
  training files): median E **2.4e5**, p99.9 **1.8e6**, MAX **4.9e7** GeV;
  **0.02 %** of training is above 1e7 and **nothing** above ~5e7. `num_points`
  caps are already saturated in training (muon p99 = max = 25088).
- **Generation**: `sample_primary_particles` (TAMBO-opt
  `allshowers/generate_showers.py`) draws energy **log-uniform in [E_MIN=1e5,
  E_MAX=1e8]** — the dual `00` script passes no `e_max`, so this default rules.
  That dumps **~⅓** of showers into [1e7,1e8] (≈0 % training coverage) and 2×
  past the highest energy ever simulated. PointCountFM then extrapolates
  `num_points` straight to the cap and AllShowers can't lay ~23k points out as a
  rod → cloud. Zenith/azimuth are fine (training `cosθ ∈ [−0.174,0.500]` =
  zenith [60°,100°], matches the U[60,100] sampling exactly).

**Where the training data is generated** (traced today): a CORSIKA-8 binary
**`c8_air_shower`** run as SLURM batches under
`…/hhanif/tambo_simulations_for_training/` (driver `scripts/run_tambo_batch_*.sh`
+ `args_batches/args_*.txt`, auto-generated with a power-law **`gamma=1.5`**).
Primaries are tau-decay daughters (pdg ±11, 111, ±211; e±/π, no µ or τ),
injection-height 2500, per-run output split into `*_{muons,electrons,photons}.h5`
under `pdg_*/energy_*/`. TAMBO-opt `util/` (`combine_h5_files.py`,
`add_num_points_per_layer.py`, …) stitches these into the `h5_files_v3/combined_*`
training files. The `gamma=1.5` power law is exactly the steep falling spectrum
above — the surrogate never saw a flat log-uniform high-E tail.

**Fix**: cap the generation default `E_MAX` **1e8 → 2e6** in
`TAMBO-opt/allshowers/generate_showers.py`, keeping sampling inside training
support (2e6 ≈ p99.9). Log-uniform [1e5,2e6] still doesn't match the spectrum
*shape*, but it stops the extrapolation blobs; matching the physical `gamma=1.5`
spectrum is the more-correct follow-up.

Side fix from the same session: `plots/plot_cached_showers.py` was calling
`showerdata.load(ckpt)` with no range — fine at 50 showers, OOM-killed on the
**161 GB** dual cache. Now it reads only each species' leading N rows via
`showerdata.load(start, stop)` (peak RSS 731 MB regardless of corpus size).

## Anti-clip PointCountFM re-roll (2026-06-14)

Follow-up to the blob band: even *within* training-support energies, blobs still
appear for the occasional muon shower because the blob tracks **point
multiplicity**, not energy per se — a shower whose predicted total exceeds the
25088 cap is truncated by `generate()`, and losing the tail collapses the rod
into a cloud. `corr(n_pts, elong) = −0.53` in the corpus; at a *fixed* E=1e7 the
multiplicity (hence which cells blob) varies cell-to-cell with direction. So
energy capping reduces but never fully removes muon blobs.

**Key enabling fact: PointCountFM is stochastic.** Its TorchScript `sample()`
draws fresh `z = torch.randn(...)` and decodes through the flow ODE each call, so
re-running it on the *same* primary yields a different total count. And the clip
is decided from `num_points` **before** the expensive GPU `generate()` — so we
can re-roll just the counts on CPU and run the generate once.

**`resample_overclip(...)`** (new helper in `00_generate_data_dual_species.py`):
re-rolls PointCountFM for only the showers whose clip fraction
`(total − cap)/total` exceeds `MAX_CLIP_FRAC` (0.10). Each retry replaces the
previous draw (several retries → keep the **last**), re-rolls **only the still-
failing subset**, up to `MAX_PCFM_RETRIES` (set to **10**). Showers that stay
over threshold after the budget keep their last draw and truncate as before.
Shared by `_gen_chunk` (generation) and `plot_angle_grid_3d_dual_species.py`
(plots) so both apply the identical policy.

**Verified on an A100** (muon angle grid, the plot now calls the same helper):

| run | mean hits | over-cap | morphology |
|-----|-----------|----------|------------|
| E=1e6 (in support) | 6546 | 0 / 25 | all rods (loop no-op) |
| E=1e7, no re-roll | 20413 | 6 / 25 | blobs at the over-cap cells |
| E=1e7, 4 retries | — | 5→3→3→2→1 | 1 residual blob |
| E=1e7, 10 retries | — | 4→3→2→0 | all rods |

Generator smoke run (`--n-pairs 120`, full [1e5,1e8] energy) fired it on muons:
`[anti-clip 1/4] … 4/120 … >10%` → converged to 1 residual, re-rolling only the
failed subset each attempt — confirming the loop touches only the cheap CPU
stage and leaves the single GPU generate untouched. **Log bug fixed in the same
pass**: the summary keyed on `sum > cap` (any clip) and said "after N retries"
even for sub-threshold showers never re-rolled (e.g. electrons at 4097–4192,
~2% clip); now it keys on `clip_frac > MAX_CLIP_FRAC`, so only genuinely-failed
showers are reported. (Electrons are pinned ~4096 by training, so they sit just
over their cap by <3% and are correctly left alone.)

## pdg = EM/hadronic primary class, not species — generation discarded the label (2026-06-15)

Root-caused a conflation in the dual-species pipeline and corrected it
end-to-end. The generator's conditioning `label` is the primary's EM-vs-hadronic
shower class, NOT a constant: `allshowers/generate_showers.py` defines
`NUM_CLASSES = 2`, `sample_primary_particles` draws `labels` uniformly in {0, 1},
and BOTH PointCountFM (`run_point_count_fm`) and AllShowers one-hot encode that
label into their conditioning tensor. Both per-species checkpoints were trained
on BOTH classes. **This supersedes the 2026-06-10/06-11 "label always 0 / label 1
hits an untrained embedding" claim — that finding was the error.**

The bug: `00_generate_data_dual_species.py` read only `prim["energies"]` /
`prim["directions"]` and hard-coded `labels = torch.zeros(...)`, so the WHOLE
corpus was generated as a single primary class. It then stored the e/µ SPECIES
id (electron=0, muon=1) in the corpus `pdg` field, which flows into the primary
5-vector's 5th feature (`encode_primary`) — constant within each species subset,
so no signal — and was used as a species ROUTER in two places (`02`'s split and
`dual_surrogate._with_pdg`'s per-branch clobber).

Two distinct axes had been merged into one `pdg`:
- **EM (e±) vs hadronic (π) primary class** — the generator's conditioning
  label, a real physics feature. Now randomly sampled and stored as `pdg`.
- **e/µ secondary component** ("species") — which model generated a row. Now a
  Step-0 sidecar, no longer the corpus `pdg`.

Fix (all within `detector_optimization_v6/`):
- **00**: feed the random per-event `prim["labels"]` to PointCountFM + AllShowers
  for BOTH species blocks (paired rows `i` and `N+i` share the class); store it
  as the corpus `pdg`. Write the e/µ species to a row-aligned sidecar
  `cashed_showers_dual_<2N>_species.pt` (`DUAL_SPECIES_IDS_PATH`) — `showerdata`
  has no species field, so it cannot live in the corpus.
- **01** (NU + NE builders): load the species sidecar (indexed by the same
  `keep_idx` as the metadata), carry it strategy-major, save `species_ids.pt`
  beside `strategy_ids.pt`. The primary's 5th feature is now the EM/hadronic
  class — a real input the surrogate learns.
- **02**: split electron/muon on `species_ids.pt`, NOT on the pdg feature.
- **`dual_surrogate`**: delete the `_with_pdg` clobber — both models now receive
  the same primary (with its true EM/hadronic class); routing is by model
  identity (electron vs muon).
- **plots**: `02_plot` splits on `species_ids`; the 2D/3D angle-grid plots'
  `--label`/titles now read the EM/hadronic class (the `cfg["pdg"]` key, removed
  from the Step-0 `SPECIES` config, would otherwise `KeyError`).

Coherent only end-to-end: the de-clobbered surrogate + varying 5th feature need
RETRAINED checkpoints, so regenerate 00 → rebuild 01 → retrain 02/03 → re-run 04;
existing `cashed_showers_dual_*.pt` / `fnn_*.pt` / `recon.pt` are stale. Static
checks (`py_compile`, grep) pass; the full smoke chain is a cluster step.
