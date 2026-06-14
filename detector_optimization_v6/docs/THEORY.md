# Detector Array Optimization via Differentiable Surrogate Models (v6)

## 1. Problem Statement

TAMBO deploys ~100 particle detectors on the slopes of Colca Valley, Peru, to observe Earth-skimming tau neutrinos. From the spatiotemporal pattern of secondary-particle "showers" across the array, we reconstruct each primary cosmic-ray particle's **energy** $E$, **zenith** $\theta$, and **azimuth** $\phi$.

**The optimization problem:** given a fixed budget of $N_\text{det} = 100$ detectors and a mountainside with 2161 candidate placement regions, find the arrangement $(x_i, y_i)_{i=1}^{100}$ that **maximizes reconstruction quality** over the expected shower population.

This is a 200-D (two coordinates per detector), non-convex problem. Scoring one layout naively means running the full chain — shower generation → detector response → reconstruction — which is far too expensive for gradient-based search. v6 replaces the physics simulation with **differentiable neural surrogates**, so gradients flow end-to-end from reconstruction loss back to detector positions.


## 2. High-Level Pipeline Architecture

The pipeline comprises five sequential stages:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 00_generate  │    │ 01_build     │    │ 02_train     │    │ 03_train     │    │ 04_optimize  │
│ _data.py     │───▶│ _dataset.py  │───▶│ _fnn.py      │───▶│ _recon.py    │───▶│ .py          │
│              │    │              │    │              │    │              │    │              │
│ Flow-match   │    │ Layout ×     │    │ Surrogate    │    │ Reconstruc-  │    │ Gradient     │
│ shower gen.  │    │ kernel       │    │ FNN training │    │ tion NN      │    │ descent on   │
│              │    │ labelling    │    │              │    │ training     │    │ layout       │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

| Stage | Input | Output | Role |
|-------|-------|--------|------|
| **Step 0** | Energy/angle ranges | Point-cloud showers $(\mathbf{r}, E, t)$ | Generate the **paired dual-species** library: primaries sampled once, electron + muon component per event (§3.6) |
| **Step 1** | Shower library + geometry | $(primary, xy, E, T)$ tensors | Pair each row with 7 diverse layouts, recenter onto mountain, compute detector responses |
| **Step 2** | Training tensors | `fnn_electron.pt` + `fnn_muon.pt` | Train two per-species DeepSets surrogates $(primary, layout) \to (E_\text{det}, T_\text{det})$. Adam(OneCycle) + L-BFGS |
| **Step 3** | Tensors + frozen dual surrogate | `recon.pt` | Train reconstruction on the **combined** response: $(x, y, E_\text{comb}, T_\text{comb}) \to (\hat n_x, \hat n_y, \hat n_z, \widetilde{\log E})$ |
| **Step 4** | Frozen surrogate + recon + primaries | Optimized layout $(\mathbf{x}^*, \mathbf{y}^*)$ + uncertainty | Maximize composite utility by backprop (or gradient-free DE) through recon + **both** species surrogates summed (§4.5.4) |

> **Dual-species lineage (2026-06-11, current default).** Steps 0–4 use the per-species May checkpoints: Step 0 = `00_generate_data_dual_species.py` (paired corpus), Step 2 = `02_train_fnn_deepsets.py` (two models), Steps 3–4 evaluate both per event and combine physically via `modules_v6/dual_surrogate.py` (§3.6). Legacy single-model scripts (`00_generate_data.py`, `02_train_fnn.py`) remain for the old all-showers checkpoint.

> **(North, East) lineage.** Steps 1–4 also have a North–East branch (§3.5): `01_build_dataset_northeast.py` → `test_v6_run_01_northeast/`, retrained Steps 2–3, then `04_optimize_differential_evolution.py`. It is a *separate dataset + model lineage* — the `xy` feature means (North, East) and the labels differ — so the (North, Up) tables below describe the default branch only. The NE builder now reads the **dual** corpus (`DUAL_SHOWER_CACHE_PATH`, §4.2). **Caveat:** Stage-4 NE scripts are only physically meaningful once Steps 2–3 are retrained on the NE dataset; do **not** score the shared (North, Up) models — whose second `xy` feature was trained as Up ∈ [2442, 3886] m — on East ∈ [−2019, 1182] m inputs.

> **Run tree.** Production uses the *recentered* corpus (`RECENTER_TO_MOUNTAIN=True`); folders are `test_v6_run_0X_recentered` under `RUN_LOCATION` on holylfs05. Paths live in `modules_v6/constants.py`.


## 3. Physical Setup

### 3.1 Mountain Geometry

The array sits on a real mountainside encoded in an HDF5 file (`basic_geometry.h5`), discretised into 2161 triangular regions. Their centroids, converted from ECEF to local East-North-Up (ENU), define the feasible surface.

Each detector position $(N_i, U_i)$ in the North–Up plane maps to a unique **East** coordinate via a differentiable surface function $E_i = f_\text{surface}(N_i, U_i)$, implemented as bilinear interpolation over a $256 \times 256$ grid (fitted by `scipy.interpolate.LinearNDInterpolator`, made differentiable with `grid_sample`). Border-clamping prevents NaN gradients outside the convex hull.

The East coordinate encodes the detector's **depth into the shower** via a continuous layer index:

$$z_{\text{cont},i} = \frac{E_\text{entry} - E_i}{\Delta E_\text{layer}}$$

with $E_\text{entry} = 1500$ m (East at AllShowers layer 0) and $\Delta E_\text{layer} = 150$ m (layer spacing). Only detectors with $E_i < E_\text{entry}$ ($z_\text{cont} > 0$) observe particles. Both constants are hand-picked for this version so the 24 layer outputs span the chosen mountain slope — they are development coordinates, not real ones.

> **Two parameterizations (see §3.5).** The above is the *original* convention — free $(N, U)$, East extrapolated. The **(North, East)** convention instead places detectors by horizontal map coordinates and extrapolates *height* $U = g(N, E)$; with this calibration the full East span $[-2019, +1182]$ m gives $z_\text{cont} \in [\approx 2, 23.5]$, so every surface point sees showers.

### 3.2 Shower Point Clouds

Each shower is a point cloud of $P$ secondaries, each carrying five features:

$$\mathbf{p}_j = (x_j, y_j, l_j, e_j, t_j)$$

$(x_j, y_j)$ are transverse positions (m), $l_j$ the discrete AllShowers layer index (0–23), $e_j$ the particle energy, $t_j$ the arrival time (s, $\sim 10^{-12}$–$10^{-6}$; see §6 on the log-T rescale). Showers come from a pre-trained **flow-matching generative model** (AllShowers), conditioned on the primary's energy $E \in [10^5, 10^8]$ GeV, zenith $\theta \in [60^\circ, 100^\circ]$, and azimuth $\phi \in [0^\circ, 360^\circ]$.

**Per-shower recentering.** The cached showers' transverse extents overlap the mountain bbox for only ~23% of showers. With `RECENTER_TO_MOUNTAIN=True` (default), Step 1 translates each shower's energy-weighted $(x,y)$ centroid onto the mountain bbox-centre before kernel evaluation, raising the trigger-producing fraction from ~23% to ~100%.

### 3.3 Primary Particle Encoding

Each primary is a 5-vector:

$$\mathbf{q} = \bigl(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta,\;\tilde E,\;\text{pdg}\bigr)$$

The first three components are the direction unit vector $(\hat n_x, \hat n_y, \hat n_z)$; $\tilde E = (\log_{10} E - 5)/3$ normalises log-energy to $[0,1]$; pdg is a particle-type id.

### 3.4 Detector Response Kernel

The physics kernel that generates Step-1 labels combines a **spatial Gaussian** with a **triangular plane weight**:

$$K_{ij} = \exp\!\Bigl(-\frac{(x_j - N_i)^2 + (y_j - U_i)^2}{2\sigma^2}\Bigr) \;\cdot\; \max\!\bigl(0,\; 1 - |l_j - z_{\text{cont},i}|\bigr)$$

The Gaussian ($\sigma = 200$ m) models lateral spread; the triangular weight selects particles near the detector's depth (weight 1 at exact layer match, linearly → 0 at $\pm1$ layer). Unlike v3's hard plane filter, this is differentiable in $z_\text{cont}$ (hence in detector position). Per-detector observables:

$$E_{\text{det},i} = \sum_{j} e_j \cdot K_{ij}, \qquad T_{\text{det},i} = \frac{\sum_j t_j \cdot K_{ij}}{\sum_j K_{ij}}$$

i.e. total kernel-weighted energy and kernel-weighted mean arrival time.

### 3.5 Detector Parameterization: (North, Up) vs (North, East)

A detector lives on a 2-D manifold in ENU, so it needs **two free coordinates plus one surface-extrapolated coordinate**. The kernel (§3.4) always needs all three: the Gaussian runs in the **(North, Up)** transverse plane, and **East → $z_\text{cont}$** is the depth axis. Two equivalent parameterizations:

| | Free coords | Surface extrapolates | $z_\text{cont}$ from | Stored `xy` |
|---|---|---|---|---|
| **Original** (`modules_v4`) | $(N, U)$ | $E = f(N, U)$ (`SurfaceEastMap`) | extrapolated $E$ | $(N, U)$ |
| **North–East** (`modules_v6/*_ne.py`) | $(N, E)$ | $U = g(N, E)$ (`SurfaceUpMap`) | **defined** $E$ | $(N, E)$ |

The North–East convention is geographically natural — place detectors at map coordinates $(N, E)$ and the terrain sets elevation $U$. It is implemented as **per-original mirror files** (each diffs cleanly against its source), leaving `modules_v4` untouched:

| NE mirror | mirrors | change |
|-----------|---------|--------|
| `tr_surface_map_ne.py` (`SurfaceUpMap`) | `modules_v4/tr_surface_map.py` (`SurfaceEastMap`) | (N,Up)→East ⇒ (N,East)→Up |
| `tr_geometry_ne.py` (`project_to_mountain_ne`, `sample_initial_layout_ne`) | `MountainData` in `modules_v4/tr_geometry.py` | Up axis → East |
| `detector_strategies_ne.py` | `detector_strategies.py` | layouts in (North, East) |
| `fnn_surrogate_ne.py` (`compute_labels_batch`, `build_training_pairs`) | same names in `fnn_surrogate.py` | Up extrapolated, `z_cont` from defined East |

Because this changes both the **labels** and the **meaning of `xy`**, it is a *separate dataset + model lineage*: rebuild Step 1 with `01_build_dataset_northeast.py` (writes `test_v6_run_01_northeast/`, `xy = (North, East)`) and retrain Steps 2–3. Recentering is unchanged (the shower transverse plane is still (North, Up)).

> **Init-vs-bounds subtlety (fixed 2026-06-11).** `project_to_mountain_ne` is a *tolerance test*, not a box clamp: a point within `max_gap` (≈2× mean centroid spacing, ~170 m) of any centroid is left untouched, so valid layouts can sit up to ~`max_gap` **outside** the tight centroid bbox. SciPy's `differential_evolution` requires `x0` strictly inside `bounds`, so σ=1000 m perturbed starts crashed it. The DE bounds are now widened by `_ne_max_gap(mountain)` on both axes; candidates are still mountain-projected before scoring, so the optimum cannot leave the mountain. (`project_to_mountain_ne` is imported from `modules_v6.tr_geometry_ne` — both the DE and L-BFGS Stage-4 scripts.)

### 3.6 Dual-Species Event Model (electron + muon components)

**Provenance (verified 2026-06-10 against the training h5 files + `TAMBO-opt/util/combine_h5_files.py`).** The May per-species AllShowers/PointCountFM checkpoints were trained on the **same** 130k showers, matched row-for-row (identical energies, directions, shower ids in `combined_electrons.h5` / `combined_muons.h5`). Primaries are tau decay daughters (`actual_pdg` ∈ {±11, 111, ±211} — e± and pions; **no muon or tau primaries**). The simulation splits each shower's secondary hits by species, so **"species" means secondary COMPONENT of one event, not shower type**. Muons survive the through-rock geometry while the EM component is absorbed — hence the asymmetric caps (electron 4096, muon 25088 points).

Consequences baked into the pipeline:

1. **Paired corpus.** `00_generate_data_dual_species.py` samples $N$ primaries once and generates both components: electron rows $0..N{-}1$ and muon rows $N..2N{-}1$ share the same $(E, \hat{\mathbf n})$ — row $i$ and row $N{+}i$ are one physical event. Stored `pdg` is a species id (e=0, µ=1), a downstream feature only.
2. **Conditioning label is always 0.** Both models were trained with conditioning label 0 only (their training `pdg` is all-zero); label 1 hits an untrained embedding. Species identity never enters the generators as a label.
3. **Two surrogates, physically combined.** Step 2 trains one DeepSets surrogate per component; Steps 3–4 evaluate both with the same $(\mathbf q, \mathbf{xy})$ and combine in *physical* space (`modules_v6/dual_surrogate.py`), not by adding log-channels:

$$N_\text{tot} = N_e + N_\mu, \qquad t_\text{tot} = \frac{N_e t_e + N_\mu t_\mu}{N_e + N_\mu}$$

   with $N_s = \mathrm{expm1}(\hat E_s)$, $t_s = \mathrm{expm1}(\hat T_s)/10^8$ inverting the per-channel log transforms (§6), then re-encoded as $\hat E_\text{comb} = \log(1{+}N_\text{tot})$, $\hat T_\text{comb} = \log(1{+}10^8\, t_\text{tot})$. Counts add; times average count-weighted, matching the kernel's $T$ (§3.4). The combination is differentiable, so Stage-4 gradients flow into the layout through **both** models.

> **Checkpoint ↔ architecture pairing (the silent-blob trap, root-caused 2026-06-10).** TAMBO-opt's two checkpoint generations use different transformer encoder blocks with identical state-dict keys: the old `all_showers` ckpt (Apr 3) is **post-LayerNorm** ($x \leftarrow \mathrm{LN}(x + \mathrm{attn}(x))$), the May per-species ckpts are **pre-LayerNorm** ($x \leftarrow x + \mathrm{attn}(\mathrm{LN}(x))$). The wrong pairing loads without error and emits diffuse blobs instead of rod showers. Fix: `allshowers/transformer.py` takes `pre_ln: bool = False`, and `stage_run_dir` injects `pre_ln: true` into the per-species conf.yaml. Two companions from the same pass: the generator must keep `with_time` support (all current ckpts are time models, `dim_inputs[0] == 4`), and `generate()` must run under `torch.no_grad()` (else each batch retains its ODE autograd graph — 39 GB OOM at the 4096-point cap, hopeless at 25088).


## 4. Stage-by-Stage Theory

### 4.1 Step 0 — Shower Corpus Generation (`00_generate_data_dual_species.py`)

Builds the **paired dual-species corpus** (§3.6): `--n-pairs` primaries (default `NUM_SHOWERS`, seeded) are sampled once, then each species' staged model pair generates its component — electron block first, muon second, same primaries. Per species the chain runs PointCountFM on CPU (its TorchScript bakes device constants) and AllShowers on GPU with $T = 16$ midpoint steps; caps are explicit (electron 4096, muon 25088). `stage_run_dir` rebuilds Generator-loadable run-dirs from the raw checkpoints and injects `pre_ln: true` (§3.6).

Generation is **streamed in chunks**: the HDF5 file is preallocated once and each chunk written at its row offset, so peak RAM is one chunk regardless of corpus size. Output: `cashed_showers_dual_{2N}.pt` at `DUAL_SHOWER_CACHE_PATH`.

**Crash recovery (`--resume-at-row`, added after run 21376182).** A crashed run continues into the existing preallocated file: pass the last logged "file offset" and the script skips completed rows — finished species blocks are skipped outright (their models never load), a partial block resumes at its offset. Primaries are seeded, so regenerated slices pair exactly with rows on disk (`--n-pairs`/`--seed` must match). `run_all_script_batch.sh` exposes this as `RESUME_ROW` (set 0 for a fresh corpus). Guards reject an out-of-range row or missing file.

> **Energy-underflow guard (root cause of the 21376182 crash).** The inverse energy transform is an `exp` of a flow latent — mathematically positive, but float32 `exp` **underflows to exactly 0.0** for extreme negative latents (~1-in-10⁸ per point; guaranteed at production scale). `showerdata` requires real points contiguous at the front — its ragged save slices `[:num_points]` with `num_points = count_nonzero(e)` — so an *interior* zero silently drops the shower's last real point, and a zero at **slot 0** raises `"Padding should be in the end of the shower points."`, which killed the run 2.5 h in (electron block complete, muon at 20k/500k). Fix: `_gen_chunk` stable-partitions every shower (key on `e ≤ 0`, `argsort(stable=True)` + `gather` moving whole 5-feature points), putting real points first in original order, zeros at the end. Verified against the real showerdata validator.

> **Energy-overflow → Inf (handled in Step 1, found 2026-06-14).** The *opposite* float32 limit also bites: the muon generator's energy de-transform **overflows to +Inf** in the energy column of ~0.7–1.0% of muon showers (the electron block is clean; only the energy column is affected, no NaN in the raw corpus). On disk this is harmless, but in the kernel an Inf energy meets a far-away spatial weight of ~0, and `Inf·0 = NaN` poisons $E$ (and the energy-weighted $T$) — which silently NaN-trained the muon surrogate. Because the generator lives in TAMBO-opt (not modified), the guard lives in the Step-1 label builder (§4.2): any point with a non-finite component is zeroed (treated as padding), plus a `nan_to_num` on the kernel's $E$/$T$ outputs. **Regenerating the corpus is not needed.**

> **Muon point-budget saturation (quantified at production scale).** The muon PointCountFM routinely predicts totals above the 25088 cap — run 21376182 logged ~8.4k truncation warnings in its first ~22k muon showers (counts up to 55k), i.e. roughly a third of muon components clipped. Flagged for revisiting with retrained higher-cap models.

> **Anti-clip re-roll (`resample_overclip`, added 2026-06-14).** When a shower's predicted total exceeds the species cap it is truncated by `generate()`, and losing the tail collapses a rod into a diffuse **blob** — the blob morphology tracks point *multiplicity*, not energy per se (`corr(n_pts, elong) ≈ −0.53`), so it persists for the occasional high-multiplicity muon even inside training-support energies. PointCountFM is **stochastic** (its `sample()` draws fresh Gaussian noise and decodes the flow ODE each call), and the clip is decided from `num_points` *before* the expensive GPU `generate()`. So Step 0 re-rolls the **counts only** (cheap CPU stage) for the showers whose clip fraction `(total − cap)/total` exceeds `MAX_CLIP_FRAC` (0.10), up to `MAX_PCFM_RETRIES` (10) attempts; each retry replaces the previous draw and re-rolls **only the still-failing subset**, so the single GPU generate runs once with the accepted counts. A shower still over threshold after the budget keeps its last draw and truncates as before. Electrons are pinned at ~4096 by training and sit <3 % over their cap, below threshold — correctly left alone. The same helper is reused by the angle-grid plots (`plot_angle_grid_*_dual_species.py`) so plotted and generated showers share one truncation policy. Verified on an A100: a muon angle grid at E=1e7 went from 6/25 over-cap blobs (no re-roll) to all rods within ~3 retries.

> **Legacy script.** `00_generate_data.py` (single dual-class model, in-RAM save) remains for the old Apr-3 `all_showers` checkpoint; its peak-memory-at-save caveat applies to that path only.

### 4.2 Step 1 — Dataset Construction (`01_build_dataset.py`, `01_build_dataset_northeast.py`)

Each shower is paired with **7** detector layouts from diverse **placement strategies** (`modules_v6/detector_strategies.py`, all mountain-projected after construction):

| id | Name | Description | Purpose |
|----|----------|-------------|---------|
| 0 | `grid_jit20` | Grid + jitter ($\sigma = 20$ m) | Uniform coverage (tight) |
| 1 | `grid_jit200` | Grid + jitter ($\sigma = 200$ m) | Uniform coverage (loose) |
| 2 | `center_gauss200` | Cluster at bbox-centre ($\sigma = 200$ m) | Concentrated |
| 3 | `center_gauss400` | Cluster at bbox-centre ($\sigma = 400$ m) | Moderately concentrated |
| 4 | `rings_R300` | 5 rings, $R = 300$ m, jitter 200 m | Tight rings |
| 5 | `rings_R800` | 6 rings, $R = 800$ m, jitter 200 m | Medium rings |
| 6 | `rings_R1800` | 8 rings, $R = 1800$ m, jitter 200 m | Wide rings |

Ring layouts use v3's `Layouts` with a random per-sample rotation; all strategies anchor at the centroid nearest the mountain $(N, U)$ bbox-centre.

For each (shower, layout) pair, the kernel (§3.4) computes ground-truth $(E_{\text{det}}, T_{\text{det}})$. Energy is log-transformed via $\log(1+E)$ here; **$T$ is stored raw** (the log-T rescale happens in Step 2, §6). Each pair is:
- **Input**: primary $\mathbf{q} \in \mathbb{R}^5$ + layout $\mathbf{xy} \in \mathbb{R}^{100 \times 2}$
- **Label**: $\mathbf{E} \in \mathbb{R}^{100}$ (log1p energy), $\mathbf{T} \in \mathbb{R}^{100}$ (raw seconds)

Multiple strategies are what teach the surrogate the dependence on detector positions (not just the primary) — the point of using it for layout optimisation. Pairs are laid out **strategy-major** (shower $i$ under strategy $s$ at index $s \cdot N_\text{showers} + i$), so the shower-level train/val split keeps all 7 variants of a shower together.

With the paired dual-species corpus the input has $2N$ rows ($N$ electron then $N$ muon, same primaries; §3.6), giving $2N \times 7$ pairs whose `pdg` feature carries the species id. Layout draws are independent per batch — the two components of one event do **not** share a layout (each per-species surrogate only needs $(\mathbf q, \mathbf{xy})$ coverage).

> **Memory: bounded per-species streaming load (`DATASET_FRACTION`).** A 1M-row dual corpus is ~151 GB on disk (ragged), but loading it dense (`(N, 25088, 5)` float32) is ~501 GB — far past a 100 GB job. The NE builder (`fnn_surrogate_ne.build_training_pairs`) therefore reads metadata only (dir/energy/pdg) up front, then loads a **prefix of each species block** via `showerdata.load(start, stop)` into a preallocated tensor — never the whole corpus. `DATASET_FRACTION` (in `constants.py`, default 0.10) sets the kept fraction, split evenly across the two blocks so both species stay represented (Step 2 splits by pdg). At 10% peak RAM is ~50 GB.

> **Inf-energy sanitization (the §4.1 overflow, handled here).** After loading, any point with a non-finite component is zeroed (so `energy ≤ 0` drops it from both the recenter mask and the kernel sums), and the kernel's $E$/$T$ outputs pass through `nan_to_num` (a no-signal detector is naturally 0). Without this, the ~1% of muon showers with +Inf energy produce `Inf·0 = NaN` labels and NaN-train the surrogate. *Currently applied in the NE builder; the (North, Up) `fnn_surrogate.build_training_pairs` is not yet patched.*

### 4.3 Step 2 — Forward Surrogate Training (`02_train_fnn_deepsets.py`)

**Purpose**: learn a fast, differentiable approximation per component (§3.6):

$$f_s: (\mathbf{q}, \mathbf{xy}) \;\longmapsto\; (\hat{\mathbf{E}}_s, \hat{\mathbf{T}}_s) \in \mathbb{R}^{100 \times 2}, \qquad s \in \{e, \mu\}$$

**Current trainer (2026-06-11).** Splits dataset rows by species id and trains **two parallel DeepSets surrogates** (the architecture that broke the flat-MLP plateau — §10), saving `fnn_electron.pt` and `fnn_muon.pt`. Per-species z-score stats are computed on each subset (muon counts are ~10× electron; shared stats would crush the smaller component's loss) and shipped inside each checkpoint; the same split seed co-splits the two components of one event. Everything else (shower-level split, log-T, Adam(OneCycle) → chunked L-BFGS with best-val save) matches the recipe below. CLI: `--epochs`, `--lbfgs-iters`, `--species`.

**Legacy architecture** (single flat MLP, `02_train_fnn.py`): a feedforward net at **hidden = 1024 × 7 layers** (the module default is 512, but `02_train_fnn.py` instantiates 1024):

```
Input:  [q (5), xy_flat (200)] = 205 features
        ↓ z-score normalisation (frozen buffers)
Linear(205, 1024) → ReLU → Dropout(0.1)
6 × [Linear(1024, 1024) → ReLU → Dropout(0.1)]   (last hidden has no dropout)
Linear(1024, 200)
        ↓ z-score denormalisation
Output: [E (100), T (100)] reshaped to (100, 2)
```

Z-score normalisation is baked into the forward pass via registered buffers, so the model drops into the optimisation loop with no external preprocessing. Each checkpoint stores its exact `hidden`/`dropout` in `config`; Steps 3–4 read those rather than assuming a width.

**Loss**: MSE in z-scored output space, E and T weighted equally:

$$\mathcal{L}_\text{FNN} = \tfrac{1}{2}\bigl(\text{MSE}_E + \text{MSE}_T\bigr), \quad \text{MSE}_c = \frac{1}{BN}\sum_{b,i}\Bigl(\frac{\hat y_{b,i,c} - y_{b,i,c}}{\sigma_c}\Bigr)^2$$

**Permutation augmentation**: a flat MLP is order-dependent, so every batch applies an independent random permutation of the 100 detectors per sample, applied jointly to $\mathbf{xy}$ and $(\mathbf{E}, \mathbf{T})$ — teaching approximate equivariance by data augmentation. (DeepSets bakes this in by construction; see §10.)

**T target log-rescale**: mirroring $\log(1+E)$, Step 2 applies $T \leftarrow \log(1 + T\cdot 10^{8})$ in-memory (raw $T \in [10^{-12}, 2.4\times10^{-6}]$ s → a workable 0–5.5 range; scale = `T_LOG_SCALE`), recomputes the T-channel z-score per species, and ships the **modified** `norm_stats` *inside* each `fnn_*.pt`. The on-disk `norm_stats.pt` keeps raw-T; the §3.6 combination inverts the same transform, and Step 3 computes its own stats. After this, log-T is canonical (no eval-time inverse); `val_mse_T` is in z-scored log-T space.

**Train/Val split**: shower-level 90/10 (all 7 layout variants of a shower in the same split, no leakage). Optional `TRAIN_FRACTION < 1` subsamples the train split (val always full) for smoke tests.

**Optimiser — two phases:**

1. **Adam + OneCycleLR** (100 epochs, batch 256): warm up $10^{-5}\to10^{-4}$ over the first 10% of steps, cosine-anneal to $10^{-7}$ (Smith & Topin 2017). Grad clip 10.0. Best-val saved.
2. **L-BFGS fine-tune** (full-batch, `strong_wolfe`, ≤1500 iters, history 5) from the Phase-1 best. The 3.5M-pair full-batch forward would OOM, so the closure is **chunked** (`LBFGS_CHUNK_SIZE = 4096`): each chunk's sum-loss is backpropagated weighted by `chunk_size / N`, so the accumulated gradient equals the exact full-batch mean gradient at $O(\text{chunk})$ memory. The closure re-validates each iteration and **`fnn.pt` is overwritten whenever the val beats the running best** — so the final checkpoint is the global best across both phases.

### 4.4 Step 3 — Reconstruction Network Training (`03_train_recon.py`)

**Purpose**: invert detection directly into the primary encoding:

$$f_\text{recon}: (x_i, y_i, \hat E_i, \hat T_i)_{i=1}^{100} \;\longmapsto\; (\hat n_x, \hat n_y, \hat n_z, \widetilde{\log E}) \in \mathbb{R}^4$$

It predicts the same 4-D encoding as the first four columns of $\mathbf{q}$ (§3.3); $(E, \theta, \phi)$ are recovered analytically downstream (§4.5). Predicting a unit vector (not $(\theta, \phi)$) avoids the $\phi$ branch cut at $0/2\pi$ and stays well-behaved near the poles.

**Training data**: the frozen **dual surrogate** (`fnn_electron.pt` + `fnn_muon.pt` behind `DualSpeciesSurrogate`, §3.6) runs in eval mode on the whole corpus to produce the **combined** response $(\hat E_\text{comb}, \hat T_\text{comb})$ per row (both models on the same $(\mathbf q, \mathbf{xy})$, counts summed, times count-weight averaged; the row's pdg is ignored). Training on surrogate *predictions* (not kernel ground-truth) is critical: at optimisation time recon only ever sees surrogate outputs, so it must be robust to their patterns — and it learns to invert the *complete* event, matching how Step 4 scores layouts.

**Targets**: `primary[:, :4]` in raw units (pdg dropped — the combined response describes the whole event). Target z-score stats and recon-input per-detector stats are computed **directly from the data being trained on** (xy + combined predictions — no single species checkpoint describes the combined distribution) and baked into `recon.pt`, so Step 4 stays consistent.

**Architecture** (`modules_v6/reconstruction.py`, the narrower 3×512 form):

```
Input:  [x, y, E_pred, T_pred] × 100 = 400 features (raw units)
        ↓ z-score normalisation (frozen buffers)
Linear(400, 512) → ReLU → Dropout(0.1)
Linear(512, 512) → ReLU → Dropout(0.1)
Linear(512, 512) → ReLU
Linear(512,   4)
        ↓ z-score denormalisation (frozen buffers)
Output: (n_x, n_y, n_z, log_e_norm) in raw primary-encoding units
```

Both input z-score and output de-z-score are registered buffers (`set_normalization(...)`), so the caller feeds raw features and reads raw units — no Tanh squash. Each checkpoint's architecture is read from its `config` via `build_surrogate_from_ckpt`, so Steps 3–4 stay correct across surrogate generations.

**Loss**: per-axis MSE in raw primary-encoding space:

$$\mathcal{L}_\text{recon} = \text{MSE}_{n_x} + \text{MSE}_{n_y} + \text{MSE}_{n_z} + \text{MSE}_{\widetilde{\log E}}$$

**Permutation augmentation**: same scheme as Step 2, but the target is a 4-vector invariant to detector order — only inputs are permuted, teaching approximate **invariance**.

**Optimiser — two phases** (mirrors Step 2):

1. **Adam** at $3 \times 10^{-5}$, grad clip 10.0, 300 epochs, batch 256. Best-val saved.
2. **L-BFGS** (full-batch, `strong_wolfe`, ≤500 iters, history 20) from the Phase-1 best. Input precomputed once on-GPU; closure chunked (`LBFGS_CHUNK = 32768`) with the same mean-gradient weighting. **`recon.pt` is overwritten whenever the val improves** (per-iteration), so the saved checkpoint is the global best.

### 4.5 Step 4 — Layout Optimisation (`04_optimize.py`)

**Purpose**: find detector positions that maximise reconstruction quality by backpropagating through the frozen surrogate(s) and recon network. In the dual lineage the "FNN" slot is the `DualSpeciesSurrogate` wrapper (§3.6): both models are evaluated and combined, and the layout gradient flows through both branches; the rest of the graph is unchanged.

**Computational Graph** (per step):

```
                    ┌─────────────────────┐
                    │  LearnableXY module  │
                    │  x, y ∈ ℝ¹⁰⁰       │◄── gradient descent updates these
                    └──────────┬──────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Broadcast to batch  │
                    │  xy: (B, 100, 2)     │
                    └──────────┬───────────┘
              ┌────────────────┼────────────────┐
              ▼                                  ▼
    ┌──────────────────┐               ┌──────────────────┐
    │ primary_batch     │               │  FNN (frozen)    │
    │ (B, 5) sampled    │──────────────▶│  (q, xy) → (E,T) │
    │ from corpus       │               └────────┬─────────┘
    └──────────────────┘                        │ (B,100) E_pred, T_pred
                                                 ▼
                                    ┌──────────────────────┐
                                    │  Build recon input   │
                                    │  (x, y, E, T) flat   │
                                    └──────────┬───────────┘
                                               ▼
                                    ┌──────────────────────┐
                                    │  Recon NN (frozen)   │
                                    │  → (n̂_x,n̂_y,n̂_z,    │
                                    │     log_e_norm)     │
                                    │  decode → (Ê, θ̂, φ̂) │
                                    └──────────┬───────────┘
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                        U_E(Ê,E)         U_θ(θ̂,θ)         U_φ(φ̂,φ)
                              └────────────────┼────────────────┘
                                    ┌──────────▼──────────┐
                                    │ Reconstructability  │
                                    │ gate r + U_PR        │
                                    └──────────┬───────────┘
                                               ▼
                              ┌──────────────────────────┐
                              │  Composite Utility U     │
                              │  loss = -U; backward()   │
                              └──────────┬───────────────┘
                                         ▼
                              ┌──────────────────────────┐
                              │  Adam step + project     │
                              │  to mountain surface     │
                              └──────────────────────────┘
```

#### 4.5.1 Learnable Layout

Positions are wrapped in a `LearnableXY` module holding $(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^{100}$ `nn.Parameter`s. Init options: grid, centre-clustered, or random (default `"center"`). After each step, positions are **projected onto the mountain**: any detector drifting past the nearest-neighbour gap from a valid centroid snaps to it — projected gradient descent on a discrete feasible set.

#### 4.5.2 Utility Function

The base `04_optimize.py` uses the full four-term v4 composite:

$$U_\text{base} = \frac{1}{10^3}\Bigl(10^2 U_\theta + 10^2 U_\phi + 10^3 U_E + 5 \times 10^5 U_\text{PR}\Bigr)$$

> **The stage-4 *variants* drop $U_\text{PR}$.** `04_optimize_nuts.py`, `_hmc_chains.py`, `_lbfgs_ensemble.py` still *compute* $U_\text{PR}$ and the gate $r$, but optimise the **three-term** $U_\text{var} = \frac{1}{10^3}(10^2 U_\theta + 10^2 U_\phi + 10^3 U_E)$ ($r$ still weights $U_E, U_\theta, U_\phi$ internally). Utilities are on different scales across scripts.

**Reconstructability gate** $r$ — soft indicator of whether enough detectors triggered:

$$r_b = \sigma\!\Bigl(\tau_2 \Bigl[\sum_{i=1}^{100} \sigma\bigl(\tau_1 (E_{\text{det},i}^{(b)} - \epsilon_\text{layout})\bigr) - n_\text{thresh}\Bigr]\Bigr)$$

with $\tau_1 = \tau_2 = 5$, $\epsilon_\text{layout} = 0.05$ (min detector signal), $n_\text{thresh} = 10$ (min triggered count). $r_b \approx 1$ when many fire, $\approx 0$ when too few.

**Participation rate** $U_\text{PR} = \sqrt{\sum_b r_b + 10^{-6}}$ — rewards layouts that make more events reconstructable.

**Energy** $U_E = \frac{1}{B}\sum_b \frac{r_b}{(\log_{10}\hat E_b - \log_{10} E_b)^2 + 0.01}$ — rewards accurate log-energy; the $r_b$ weight restricts to reconstructable events, the floor bounds per-event reward.

**Angular** $U_\theta, U_\phi = \frac{1}{B}\sum_b \frac{r_b}{(\hat\alpha_b - \alpha_b)^2 + 0.001}$ for $\alpha \in \{\theta, \phi\}$ — same structure, tighter floor (radian residuals are smaller).

The **loss** is $\mathcal{L} = -U$, so descent maximises utility.

#### 4.5.3 Optimisation Loop (base `04_optimize.py`)

- **Optimiser**: Adam, $\text{lr} = 1.0$, grad clip 100.0
- **Batch**: 256 random primaries per epoch
- **Epochs**: `N_OPT_EPOCHS = 10{,}000`
- **Init schemes**: one run per entry in `INIT_SCHEMES = ("grid", "center")`, each to its own `_{scheme}` folder
- **Projection**: after each step; **logging**: utility components, grad norms, layout snapshots

#### 4.5.4 Stage-4 Variants (uncertainty + multi-start)

Four sibling scripts wrap the same frozen FNN+recon objective ($U_\text{var}$, §4.5.2) with richer search/uncertainty machinery. Common front end: K Gaussian-perturbed restarts of the init scheme, then a per-restart optimiser. A `"combined"` run pools grid + centre restarts.

> **State of the art.** `04_optimize_lbfgs_ensemble.py` is the recommended Stage-4 entry. The L-BFGS ensemble gives a deterministic local optimum per restart plus a network-input-invariant mean ± std uncertainty map (via position alignment) — more useful and far cheaper than the NUTS posterior, which concentrates on the typical set rather than the mode (lower best-$U$, ~6–7 h per combined run). Treat NUTS/HMC as exploratory.

- **`04_optimize_nuts.py`** — Adam warm-start, then one **Pyro NUTS** chain samples $\log p(\mathbf{xy}) = U(\mathbf{xy})/T + \log\mathcal{N}(\mathbf{xy}\mid\mathbf{xy}_\text{Adam}, \sigma_\text{prior}^2)$ on a fixed primary batch, anchored near the Adam optimum. Reports best-$U$ draw + per-detector 1σ ellipses.

- **`04_optimize_hmc_chains.py`** — Gelman–Rubin variant. K NUTS chains from **overdispersed** perturbed Adam optima run **sequentially in-process** (Pyro multi-process can't pickle the CUDA `potential_fn`; sequential is same wall-time on one GPU). R̂/ESS via ArviZ. The prior anchors at a **single real Adam-best layout** (not the per-index mean, which would collapse detectors centrally). Defaults: 4 chains × 1500 warmup × 1500 samples, $T = 3$, $\sigma_\text{prior} = 100$ m.

- **`04_optimize_lbfgs_ensemble.py`** — frequentist ensemble. Each of K perturbed Adam optima is refined by **L-BFGS** on a fixed batch, then the K layouts are **aligned by physical position** — a Hungarian assignment (`linear_sum_assignment`, with a greedy fallback) matches detectors by closest $(x,y)$, since permutation-equivariance makes detector *index* meaningless across runs. Per aligned group it reports **mean and std** (a network-input-invariant uncertainty map), and logs a **per-run consecutive-step gradient cosine distance** ($W$-step vector-averaged to cancel minibatch noise) as a convergence diagnostic.

- **`04_optimize_differential_evolution.py`** — global, **gradient-free** `scipy.optimize.differential_evolution` over the 200-D layout (100 North + 100 East, **North–East convention** §3.5), bounded by the North bbox and East span `[east_lo, east_hi]` **widened by `max_gap`** (§3.5 init-vs-bounds note). Each candidate is mountain-projected (`project_to_mountain_ne`) and scored by the same composite $U$ on a fixed batch; reports the best layout — a global baseline to check whether the gradient optimisers sit in a local optimum. Expensive in 200-D (population = `popsize` × 200) — keep `popsize`/`maxiter` modest. **Requires NE-retrained Steps 2–3** (§2 caveat) to be physically meaningful.


## 5. Key Design Decisions

### 5.1 Why Two Surrogate Networks?

The physics factorises naturally:
1. **Detection** (FNN): how does a layout respond to a shower? Depends on primary *and* positions.
2. **Reconstruction** (Recon): given what detectors saw, what was the primary? The inverse problem.

Training them separately lets the FNN focus on the spatial/temporal kernel without confounding reconstruction error, lets recon train on FNN *predictions* (robust to surrogate bias), and lets either be retrained independently. Both share the same **normalisation contract** (per-feature z-score in registered buffers) and the same Adam→L-BFGS recipe, but no longer share width (FNN 7×1024, recon 3×512, both dropout 0.1); width is stored in each `config`. Recon reuses the FNN checkpoint's `norm_stats` so train/val/optimisation see one z-score.

### 5.2 Why Permutation Augmentation?

Flat MLPs are order-dependent, so training applies random per-sample detector permutations to *approximate* the symmetry: equivariance for the FNN, invariance for recon. Chosen for simplicity, but the architecture search (§10) found it the dominant bottleneck — a set-equivariant model (DeepSets) that bakes the symmetry in by construction is the replacement. Augmentation is the expedient, not the endpoint.

### 5.3 Why Train Recon on FNN Predictions?

Trained on kernel ground-truth, recon would exploit exact-kernel features the FNN can't reproduce. At optimisation time recon only sees FNN outputs, so training on predictions avoids a train/deploy domain gap.

### 5.4 Why a Composite Utility?

Pure reconstruction MSE ignores viability — a layout where nothing triggers has undefined quality. $U_\text{PR}$ rewards triggering enough detectors; $U_E, U_\theta, U_\phi$ reward accuracy but only for reconstructable events. The weighting was hand-tuned during testing to balance the fractions — further work needed.

### 5.5 Why Project to the Mountain?

The surface is a non-convex 2D manifold in 3D; unconstrained descent would push detectors into impossible positions. After each update, `project_to_mountain()` snaps drifted detectors to the nearest valid centroid — projected gradient descent on a discrete feasible set.

### 5.6 Why Two Per-Species Surrogates (and why their outputs sum)?

The May checkpoints are per-species because the *training data* is per-component: the simulation writes each shower's electron and muon hits to separate files of the **same** events (§3.6). A single surrogate with a species flag would model "an electron-component shower" or "a muon-component shower" — but no physical event is only one.

1. **Each surrogate learns one component's response** $f_s(\mathbf q, \mathbf{xy})$, on its own scale (per-species stats; muon counts dominate electron by ~10× in the through-rock geometry).
2. **A complete event is the superposition of both components for one primary.** Counts are extensive (they add) and the kernel's $T$ is count-weighted, so the correct combination is $N_e + N_\mu$ and $(N_e t_e + N_\mu t_\mu)/(N_e + N_\mu)$ in *physical* space (adding log-channels would be meaningless). It is differentiable, so Stage-4 gradients flow through both models.

The corpus is generated **paired** (same primaries for both blocks) so this superposition is faithful to the simulation, and recon/optimizer always see complete events, not half-showers.


## 6. Normalisation Strategy

All z-scoring is baked into the forward passes via registered buffers; each stage computes its stats from the data it trains on:

| Where | What | Why |
|-------|------|-----|
| **Surrogate in/out** | Z-score, **per-species stats** on each species' subset (Step 2; shipped inside each checkpoint) | Muon/electron count scales differ ~10×; shared stats mis-weight the smaller component. Per-model denorm keeps outputs in the same physical units for the §3.6 combination |
| **Recon input** | Z-score, per-detector stats from the actual recon inputs (xy + **combined** $\hat E/\hat T$), stored in `recon.pt` | Combined distributions aren't any single checkpoint's stats; computing from data is exact and keeps Step 4 consistent |
| **Recon output** | De-z-score from `primary[:, :4]` of the corpus | The 4-D encoding is in known raw units; baking de-z-score into `forward()` lets losses/decoding work in raw primary units, no Tanh squash |
| **Labels (E)** | $\log(1 + E)$ before z-score (Step 1, stored in `E.pt`) | Compresses the heavy energy tail |
| **Labels (T)** | $\log(1 + T\cdot 10^{8})$ in **Step 2** per species (`T_LOG_SCALE`; not stored on disk) | Raw $T$ spans $10^{-12}$–$10^{-6}$ s → ~0–5.5. Modified T stats ship inside each `fnn_*.pt`; the §3.6 combination inverts the same transform |


## 7. Data Layout and Storage

All intermediate data is stored as PyTorch tensors under `RUN_LOCATION` (holylfs05); production names are the `test_v6_run_0X_recentered` folders in `modules_v6/constants.py`. Numbers below are for the default 500k-pair (1M-row) / 7-strategy corpus.

```
v6_run_00/                       ← Step 0: cached showers (shared across runs)
    cashed_showers_dual_{2N}.pt  PAIRED dual-species corpus (§3.6): rows 0..N-1
                                 electron, N..2N-1 muon, same primaries; ragged
                                 HDF5 via showerdata, streamed chunked writes.
                                 1M rows ≈ 151 GB on disk (ragged); ~501 GB dense.
    cashed_showers_500000.pt     legacy single-model corpus (~20 GB at 500k)

test_v6_run_01_recentered/       ← Step 1: training dataset (North, Up convention)
    primary.pt          (2N·7, 5)      primary features (pdg = species id)
    xy.pt               (2N·7, 100, 2) detector layouts (recentered)
    E.pt                (2N·7, 100)    log1p(energy per detector)
    T.pt                (2N·7, 100)    RAW time per detector (seconds)
    strategy_ids.pt     (2N·7,)        layout strategy id [0–6], strategy-major
    norm_stats.pt       z-score tensors (raw-T; corpus-wide — Step 2 recomputes
                        per-species stats on its subsets)

test_v6_run_01_northeast/        ← Step 1 (North, East) variant — same tensors,
                                   xy = (North, East); 01_build_dataset_northeast.py
                                   (§3.5). Built from the dual corpus, bounded by
                                   DATASET_FRACTION (§4.2)

test_v6_run_02_recentered/       ← Step 2: per-species DeepSets checkpoints
    fnn_electron.pt     state_dict + per-species norm_stats (log-T) + config
    fnn_muon.pt         (model_type=deepsets, species tag)
    fnn_{species}_train_log.json / _train_curves.png / _target_vs_pred.png
    fnn.pt              legacy single-model checkpoint (untouched by the dual trainer)

test_v6_run_03_recentered/       ← Step 3: Recon checkpoint
    recon.pt            state_dict + input/target mean+std + config
    recon_train_log.json
    recon_train_curves.png
    recon_target_vs_pred.png      density heatmap, auto-rendered at end of Step 3

test_v6_run_04_optimize_{scheme}/         ← base 04, one per INIT_SCHEME (grid|center)
    layout_best.pt, layout_final.pt, xy_trajectory.pt,
    optimize_log.json, optimize_curves.png, layout_before_after.png

test_v6_run_04_optimize_hmc_chains_{grid|center|combined}/   ← NUTS multi-chain variant
    layout_best.pt, layout_adam.pt, layout_init.pt, nuts_samples.pt,
    nuts_diagnostics.csv, optimize_log.json, optimize_curves.png,
    layout_before_after.png, nuts_diagnostics.png

test_v6_run_04_optimize_lbfgs_ensemble_{grid|center|combined}/  ← L-BFGS ensemble variant
    layout_best.pt, layout_mean.pt (aligned per-position mean+std),
    layouts_all.pt (aligned ensemble + perms + utilities),
    optimize_log.json, optimize_curves.png, utility_components.png,
    layout_ensemble.png

test_v6_run_04_optimize_de_ensemble_{grid|center|combined}/  ← DE ensemble variant (North, East)
    layout_best.pt, layout_mean.pt, layouts_all.pt,
    optimize_log.json (incl. de_best_U_history), optimize_curves.png,
    utility_components.png, layout_ensemble.png, layout_density.png
```


## 8. Execution Environment

SLURM HPC cluster with A100 GPUs:

- **Step 0** (generation): GPU-intensive, streamed (peak RAM ≈ one chunk, not the corpus). A 1M-row dual corpus is ~15 h on one A100 (~9 muon rows/s steady-state); the muon block dominates. `gpu_requeue` preemption is survivable via `--resume-at-row` (§4.1).
- **Step 1**: I/O-bound kernel evaluation; the NE builder streams per-species (bounded by `DATASET_FRACTION`, §4.2) so it fits a 100 GB job.
- **Steps 2–4**: single A100. With the 3.5M-pair corpus these are multi-hour: Step 2 ≈ 5 h (100 Adam + 1500 L-BFGS), Step 3 ≈ 5 h; stage-4 variants are dominated by sampling/refinement (combined NUTS ≈ 6–7 h). L-BFGS phases use chunked closures to bound GPU memory.
- For long generation runs, prefer a non-preemptable partition or request a longer wall + adequate memory.


## 9. Mathematical Summary

The full objective through both frozen networks (base `04_optimize.py`; variants drop $w_\text{PR}U_\text{PR}$, §4.5.2):

$$\max_{\mathbf{x}, \mathbf{y}} \;\; \mathbb{E}_{\mathbf{q} \sim \mathcal{D}} \left[ \frac{w_\theta U_\theta + w_\phi U_\phi + w_E U_E + w_\text{PR} U_\text{PR}}{w_\text{div}} \right]$$

subject to $(x_i, y_i) \in \mathcal{M}$ (the mountain surface), approximated by mini-batches of 256 primaries per epoch. Gradients flow:

$$U \;\xleftarrow{\text{utility}}\; (\hat E, \hat\theta, \hat\phi, r) \;\xleftarrow{\text{decode}}\; (\hat n_x, \hat n_y, \hat n_z, \widetilde{\log E}) \;\xleftarrow{f_\text{recon}}\; (x_i, y_i, \hat E_i, \hat T_i) \;\xleftarrow{f_\text{FNN}}\; (\mathbf{q}, x_i, y_i)$$

back to $(x_i, y_i)$. Both networks are frozen; only the positions receive updates.


## 10. FNN Surrogate: Development Log and Known Limitation

The Step-2 surrogate has had an extensive architecture/hyperparameter search (full chronology in `diary.md`). The findings below are load-bearing.

### 10.1 What was tried, and the verdict

Search ran on a 10% subset; numbers are relative z-scored val-MSE within the search, **not** comparable to full-corpus production.

- **Flat-MLP family plateaus at val ≈ 0.69–0.71.** OneCycleLR, GELU+LayerNorm, width to 1024 — all marginal, none broke it.
- **DeepSets was the one decisive lever:** a per-detector shared MLP with pooled context reached **0.546 (−23%)**, E dropping 40%. The permutation-equivariance bias — which the flat MLP only *approximates*, expensively, via augmentation — is what mattered.
- **Capacity is not the bottleneck.** A 4× wider DeepSets did not improve.
- **A hard cross-method floor at ≈ 0.546.** Set Transformer (SAB/ISAB), a k-NN GNN, primary→detector cross-attention, learned uncertainty weighting, AdamW+3× LR all landed within ±0.3% of DeepSets. Such tightness across radically different models points to a **data/label ceiling**, confirmed aleatoric in §10.4.
- **Zero-inflation handling backfires.** Soft reweighting, BCE hit-gates, and hard masks all *regressed* pure-MSE — the ~96% zero positions dominate the MSE and masking starves their gradient. Zero-inflation is **not** the bottleneck.
- **log-T target** (§6) and the **L-BFGS best-iter save fix** (§4.3) were the two changes that stuck and are in production.

### 10.2 Path-c schedule de-risk (2026-06-04) — failed, reverted

A controlled attempt to lift the *production flat MLP* by schedule/optimizer alone (LR-range test → AdamW + weight decay, dropout off, raised LR floor, L-BFGS capped, OneCycle `final_div_factor` fix) **regressed** (val 0.60 vs 0.40). The conditional metrics exposed the real failure mode:

| metric | value | meaning |
|--------|-------|---------|
| E R² (all detectors) | 0.45 | flattering — dominated by empty detectors |
| **E R² (fired only)** | **−0.14** | worse than predicting the fired-channel mean |
| fire precision / recall | 0.42 / 0.99 | **over-fires**, leaking energy onto empty detectors |
| fired pred/target std | 0.69 | magnitude compression — predict-the-mean |

All path-c edits were reverted; the working tree matches §4.3.

### 10.3 Standing recommendation

Track **conditional-on-fired E/T R² and fire precision/recall**, not total val-MSE (flattering because ~68% of detector-samples are near-zero). The next high-leverage change is **path (a): re-architect the FNN as a pointwise DeepSets** `φ(q, xᵢ, yᵢ) → (Eᵢ, Tᵢ)` with weights shared across detectors — permutation-equivariant by construction (matching the per-detector-local kernel of §3.4), removing augmentation, ~34× fewer params, each detector its own training example (~100× more effective samples). It preserves the `forward(primary, xy) → (B,100,2)` contract, so Steps 3–4 are unaffected. Recon (§4.4) shares the same mismatch (its target is permutation-*invariant*) and is the natural follow-on.

> **Status (2026-06-11): path (a) is implemented.** `modules_v6/deepsets_surrogate.py` + `02_train_fnn_deepsets.py` are the production Step-2 trainer, now in dual-species form (§3.6/§4.3). The set-equivariant recon follow-on remains open.

### 10.4 The cross-method floor is largely aleatoric (not an architecture limit)

The §10.1 floor has a kernel-level explanation that bounds what *any* Step-2 surrogate can achieve. The ground truth (§3.4) is a function of the **full stochastic shower point cloud** `samples (B, P, 5)`:

$$E_{\text{det},i} = \sum_j e_j K_{ij}, \qquad T_{\text{det},i} = \frac{\sum_j t_j K_{ij}}{\sum_j K_{ij}}$$

But the surrogate sees only the **primary summary** $\mathbf{q}$ and layout $\mathbf{xy}$ — never the secondaries. The generator draws a *different* cloud per shower, so two showers with the same primary yield different $(E_\text{det}, T_\text{det})$. The best any model can do is the **conditional mean** $\mathbb{E}[(E,T)\mid \mathbf q, \mathbf{xy}]$; the shower-to-shower variance is **irreducible aleatoric noise** — a hard floor independent of capacity or optimiser. (This supersedes the earlier "feature gap" guess: the missing information is the entire random realisation, not one engineered feature.)

**Why T floors higher than E.** $E_\text{det}$ is a *sum* over many particles and self-averages → small conditional variance. $T_\text{det}$ is a kernel-weighted *mean arrival time*, dominated by where the few near-detector particles land per realisation → fluctuates more. Aleatoric noise ranks the channels as observed.

**Measured floor (2026-06-05, `compute_aleatoric_floor.py`):** 128 primaries × 64 *independent* showers each, through the exact kernel + recentering + log-transforms, within-primary label variance / full-corpus variance (z-MSE units):

| channel | aleatoric floor | DeepSets val (1% data) | gap | max R² |
|---------|-----------------|------------------------|-----|--------|
| E (all) | **0.306** | 0.320 | 0.014 | 0.69 |
| T (all) | **0.414** | 0.425 | 0.011 | 0.59 |
| total   | **0.360** | 0.373 | 0.013 | — |

**The DeepSets surrogate is essentially Bayes-optimal** — within ~0.01–0.015 of the floor on every channel, on just 1% of the data. Tuning past this point has ~zero headroom; full-data training only closes the residual ~0.013. The measured floor T/E ratio is ~1.35 (not the ≈2.05 from the old 10%-subset search). For *fired-only* detectors the floor jumps to E≈0.75, T≈1.06 — a fired detector's precise arrival time is almost pure shower noise, so T signal must be aggregated across detectors, not read per-detector. The DeepSets rewrite (§10.3) is still worth doing — it reaches the floor with ~34× fewer params and no augmentation — but it lowers MSE by removing *approximation* error, not by beating the aleatoric floor.

**Downstream note.** Stage 4 is the deterministic **L-BFGS ensemble** (§4.5.4), not a sampler. It uses the FNN as a point forward map; its uncertainty map is the spread of K perturbed-init optima (position-aligned mean ± std), reading **no** FNN predictive variance. So a heteroscedastic / Gaussian-NLL FNN head would improve label-noise calibration but **not** propagate into stage-4 uncertainty — the only FNN property that matters downstream is the accuracy of its conditional-mean $(E, T)$. Reserve a predictive-variance head for a future Bayesian Stage 4.
