# Detector Array Optimization via Differentiable Surrogate Models (v6)

## 1. Problem Statement

The TAMBO experiment deploys an array of particle detectors on the slopes of Colca Valley, Peru, to observe Earth-skimming tau neutrinos. The scientific goal is to reconstruct the properties of incoming primary cosmic-ray particles — their **energy** $E$, **zenith angle** $\theta$, and **azimuth angle** $\phi$ — from the spatiotemporal pattern of secondary particles ("showers") detected across the array.

The **optimization problem** is: given a fixed budget of $N_\text{det} = 100$ detectors and a mountainside geometry with 2161 candidate placement regions, find the spatial arrangement $(x_i, y_i)_{i=1}^{100}$ on the mountain surface that **maximizes the quality of primary-particle reconstruction** across the full range of expected shower types.

This is a high-dimensional, non-convex optimization problem. Each detector is parameterised by two coordinates (North, Up) on the mountain surface, giving 200 continuous degrees of freedom. Evaluating the quality of a layout requires simulating the full chain: shower generation, detector response, and reconstruction — a process far too expensive for direct gradient-based optimization. The v6 pipeline solves this by replacing the expensive physics simulation with **differentiable neural-network surrogates**, enabling end-to-end gradient flow from reconstruction loss back to detector positions.


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

### Data Flow Summary

| Stage | Input | Output | Role |
|-------|-------|--------|------|
| **Step 0** | Energy/angle sampling ranges | Point-cloud showers $(\mathbf{r}, E, t)$ | Generate synthetic shower library (default 500k showers) |
| **Step 1** | Shower library + mountain geometry | $(primary, xy, E, T)$ training tensors | Pair each shower with **7** diverse layouts; per-shower recenter onto mountain; compute detector responses (3.5M pairs) |
| **Step 2** | Training tensors | Frozen `fnn.pt` checkpoint | Train surrogate: $(primary, layout) \to (E_\text{det}, T_\text{det})$. Adam (OneCycle) + L-BFGS fine-tune |
| **Step 3** | Training tensors + frozen FNN | Frozen `recon.pt` checkpoint | Train reconstruction: $(x, y, E_\text{det}, T_\text{det}) \to (\hat n_x, \hat n_y, \hat n_z, \widetilde{\log E})$. Adam + L-BFGS fine-tune |
| **Step 4** | Frozen FNN + frozen recon + primaries | Optimized layout $(\mathbf{x}^*, \mathbf{y}^*)$ + uncertainty | Maximise composite utility via backpropagation through both NNs. Base + 3 sampler/ensemble variants |

> **Note on the run tree.** The current production runs use the *recentered* corpus (`RECENTER_TO_MOUNTAIN=True`); folders are named `test_v6_run_0X_recentered` under `RUN_LOCATION` on holylfs05 rather than the historical `v6_run_0X`. Paths are centralised in `modules_v6/constants.py`.


## 3. Physical Setup

### 3.1 Mountain Geometry

The detector array is sited on a real mountainside whose geometry is encoded in an HDF5 file (`basic_geometry.h5`). The mountain surface is discretised into 2161 triangular regions; the centroids of these triangles, converted from Earth-Centred Earth-Fixed (ECEF) to local East-North-Up (ENU) coordinates, define the feasible surface.

Each detector position $(N_i, U_i)$ in the North–Up plane maps to a unique **East** coordinate via a differentiable surface function:

$$E_i = f_\text{surface}(N_i, U_i)$$

The surface map $f_\text{surface}$ is implemented as bilinear interpolation over a $256 \times 256$ regular grid fitted by `scipy.interpolate.LinearNDInterpolator` to the 2161 centroid scatter, with `torch.nn.functional.grid_sample` providing differentiability. Border-clamping ensures no NaN gradients for detectors that wander outside the mountain convex hull.

The East coordinate encodes the detector's **depth into the shower** via a continuous layer index:

$$z_{\text{cont},i} = \frac{E_\text{entry} - E_i}{\Delta E_\text{layer}}$$

where $E_\text{entry} = 1500$ m is the East coordinate at AllShowers layer 0 and $\Delta E_\text{layer} = 150$ m is the layer spacing. Only detectors with $E_i < E_\text{entry}$ (i.e., $z_\text{cont} > 0$) can observe shower particles. $E_\text{entry}$ and $\Delta E_\text{layer}$ are manually selected for this version, such that the predefined 24 z output values span within the preselected mountain slope. Those are not real coordinates, rather adapted for development purposes.

### 3.2 Shower Point Clouds

Each particle shower is represented as a point cloud of $P$ secondary particles, with each point carrying five features:

$$\mathbf{p}_j = (x_j, y_j, l_j, e_j, t_j)$$

where $(x_j, y_j)$ are transverse positions (metres), $l_j$ is the discrete AllShowers layer index (0–23), $e_j$ is the particle energy, and $t_j$ is the arrival time (seconds, $\sim 10^{-12}$–$10^{-6}$ s; see Section 6 on the log-T rescale). Showers are generated by a pre-trained **flow-matching generative model** (AllShowers), which is conditioned on the primary particle's energy $E \in [10^5, 10^8]$ GeV, zenith $\theta \in [60^\circ, 100^\circ]$, and azimuth $\phi \in [0^\circ, 360^\circ]$.

**Per-shower recentering.** The cached showers' transverse $(x, y)$ extents only overlap the mountain bounding box for ~23% of showers. When `RECENTER_TO_MOUNTAIN=True` (the current default), Step 1 translates each shower's energy-weighted $(x,y)$ centroid onto the mountain bbox-centre before kernel evaluation, so every shower lands on the array and contributes useful (non-zero) detector responses. This raised the fraction of trigger-producing showers from ~23% to ~100%.

### 3.3 Primary Particle Encoding

Each primary particle is encoded as a 5-dimensional vector:

$$\mathbf{q} = \bigl(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta,\;\tilde E,\;\text{pdg}\bigr)$$

where $\tilde E = (\log_{10} E - 5) / 3$ normalises the log-energy to $[0, 1]$ and pdg is a particle-type identifier. $\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta$ are the normalized unit vectors $\hat{n}_x, \hat{n}_y, \hat{n}_z$, respectively.

### 3.4 Detector Response Kernel

The physics-based detector response (used to generate training labels in Step 1) combines a **spatial Gaussian kernel** with a **triangular plane weight**:

$$K_{ij} = \exp\!\Bigl(-\frac{(x_j - N_i)^2 + (y_j - U_i)^2}{2\sigma^2}\Bigr) \;\cdot\; \max\!\bigl(0,\; 1 - |l_j - z_{\text{cont},i}|\bigr)$$

The spatial kernel (width $\sigma = 200$ m) models lateral particle spread. The triangular weight smoothly selects particles near the detector's longitudinal depth, giving weight 1 for an exact layer match and linearly decaying to 0 at $\pm 1$ layer. This is differentiable in $z_\text{cont}$ (hence in detector position), a key improvement over v3's hard plane filter.

Per-detector observables are then:

$$E_{\text{det},i} = \sum_{j} e_j \cdot K_{ij}, \qquad T_{\text{det},i} = \frac{\sum_j t_j \cdot K_{ij}}{\sum_j K_{ij}}$$

i.e., total kernel-weighted energy and kernel-weighted mean arrival time.


## 4. Stage-by-Stage Theory

### 4.1 Step 0 — Shower Corpus Generation (`00_generate_data.py`)

A library of shower point clouds (default `NUM_SHOWERS = 500{,}000`, configurable in `modules_v6/constants.py`) is generated by the AllShowers flow-matching model and cached to disk. Primary particles are sampled uniformly in $(\theta, \phi, \log E)$ and the flow-matching solver produces stochastic point clouds via $T = 16$ midpoint integration steps. This step is GPU-intensive and runs once; all downstream stages read the cached corpus.

> **Memory caveat.** The generator accumulates the entire `(N, max_points, 5)` sample tensor in RAM and writes a single HDF5 file only at the very end (`save_output`), with a transient numpy copy during the save. There is no incremental checkpointing, so for large `NUM_SHOWERS` the save step is the peak-memory moment and an OOM there loses the whole run. Size `--mem` for the *save* peak, not the steady state.

### 4.2 Step 1 — Dataset Construction (`01_build_dataset.py`)

Each shower is paired with **7** different detector layouts drawn from diverse **placement strategies** (defined in `modules_v6/detector_strategies.py`, all projected onto the mountain after construction):

| Strategy id | Name | Description | Purpose |
|----|----------|-------------|---------|
| 0 | `grid_jit20` | Regular grid + Gaussian jitter ($\sigma = 20$ m) | Covers mountain uniformly (tight) |
| 1 | `grid_jit200` | Regular grid + Gaussian jitter ($\sigma = 200$ m) | Covers mountain uniformly (loose) |
| 2 | `center_gauss200` | Cluster at mountain bbox-centre anchor ($\sigma = 200$ m) | Concentrated layout |
| 3 | `center_gauss400` | Cluster at mountain bbox-centre anchor ($\sigma = 400$ m) | Moderately concentrated layout |
| 4 | `rings_R300` | 5 concentric rings, $R = 300$ m, jitter 200 m | Tight ring pattern |
| 5 | `rings_R800` | 6 concentric rings, $R = 800$ m, jitter 200 m | Medium ring pattern |
| 6 | `rings_R1800` | 8 concentric rings, $R = 1800$ m, jitter 200 m | Wide ring spanning mountain |

Ring layouts are built with v3's `Layouts` and given a random rotation per sample; all strategies are anchored at the centroid nearest the mountain $(N, U)$ bounding-box centre.

For each (shower, layout) pair, the physics-based kernel (Section 3.4) computes ground-truth detector responses $(E_{\text{det}}, T_{\text{det}})$. The energy is log-transformed via $\log(1 + E)$ in Step 1 to compress the heavy right tail; **$T$ is stored raw** (the log-T rescale happens later, in Step 2 — see Section 6). Z-score normalisation statistics are computed over the full corpus.

This produces a training corpus of $500{,}000 \times 7 = 3{,}500{,}000$ training pairs, each consisting of:
- **Input**: primary vector $\mathbf{q} \in \mathbb{R}^5$ + detector layout $\mathbf{xy} \in \mathbb{R}^{100 \times 2}$
- **Label**: detector responses $\mathbf{E} \in \mathbb{R}^{100}$ (log1p energy), $\mathbf{T} \in \mathbb{R}^{100}$ (raw seconds)

The use of multiple layout strategies ensures the surrogate learns the dependence on detector positions, not just on the primary particle — this is what makes the surrogate useful for layout optimisation. The pairs are laid out **strategy-major** (entry for shower $i$ under strategy $s$ sits at index $s \cdot N_\text{showers} + i$), which the shower-level train/val split exploits to keep all 7 variants of one shower in the same split.

### 4.3 Step 2 — Forward Surrogate Training (`02_train_fnn.py`)

**Purpose**: Learn a fast, differentiable approximation:

$$f_\text{FNN}: (\mathbf{q}, \mathbf{xy}) \;\longmapsto\; (\hat{\mathbf{E}}, \hat{\mathbf{T}}) \in \mathbb{R}^{100 \times 2}$$

**Architecture**: A feedforward network (FNNSurrogate). The current production width is **hidden = 1024 across 7 hidden layers** (the `hidden` kwarg defaults to 512 in the module, but `02_train_fnn.py` instantiates it at 1024):

```
Input:  [q (5), xy_flat (200)] = 205 features
        ↓ z-score normalisation (frozen buffers)
Linear(205, 1024) → ReLU → Dropout(0.1)
Linear(1024, 1024) → ReLU → Dropout(0.1)   ┐
Linear(1024, 1024) → ReLU → Dropout(0.1)   │  6 hidden blocks total
Linear(1024, 1024) → ReLU → Dropout(0.1)   │  (5 with dropout, then …)
Linear(1024, 1024) → ReLU → Dropout(0.1)   │
Linear(1024, 1024) → ReLU → Dropout(0.1)   ┘
Linear(1024, 1024) → ReLU                       ← final hidden, no dropout
Linear(1024, 200)
        ↓ z-score denormalisation
Output: [E (100), T (100)] reshaped to (100, 2)
```

Z-score normalisation is baked into the forward pass via registered buffers, so the model can be dropped into the optimisation loop without external preprocessing. The checkpoint stores the exact `hidden`/`dropout` in its `config`, and Steps 3–4 read those rather than assuming a fixed width.

**Loss**: Mean squared error in z-scored output space, with E and T channels weighted equally:

$$\mathcal{L}_\text{FNN} = \frac{1}{2}\bigl(\text{MSE}_E + \text{MSE}_T\bigr) \quad\text{where}\quad \text{MSE}_c = \frac{1}{BN}\sum_{b,i}\Bigl(\frac{\hat y_{b,i,c} - y_{b,i,c}}{\sigma_c}\Bigr)^2$$

**Permutation Augmentation**: Because the FNN is a flat MLP (not a set-equivariant architecture), the detector ordering is arbitrary. To teach approximate permutation equivariance by data augmentation, every training batch applies an **independent random permutation** of the 100 detectors per sample. The same permutation is applied to the input layout $\mathbf{xy}$ and the target $(\mathbf{E}, \mathbf{T})$ jointly, preserving the detector-wise correspondence while exposing the network to all orderings.

**T target is log-rescaled at training time.** Mirroring the $\log(1+E)$ treatment, Step 2 applies $T \leftarrow \log(1 + T\cdot 10^{8})$ in-memory (raw $T \in [10^{-12}, 2.4\times10^{-6}]$ s maps to a workable $0$–$5.5$ range), recomputes the T-channel z-score stats, and ships the **modified** `norm_stats` *inside* `fnn.pt`. The on-disk `norm_stats.pt` keeps the raw-T values, so Steps 3–4 deliberately read the FNN checkpoint's own `norm_stats` to stay consistent. After this, log-T is the canonical target — there is no inverse at eval, and `val_mse_T` is reported in z-scored log-T space.

**Train/Val Split**: Shower-level 90/10 split — all 7 layout variants of the same shower go into the same split, preventing information leakage. An optional `TRAIN_FRACTION < 1` subsamples the train split (val always full) for smoke tests.

**Optimiser — two phases:**

1. **Phase 1, Adam + OneCycleLR** (100 epochs, batch 256). The LR follows a one-cycle schedule (Smith & Topin 2017): warm up from $10^{-5}$ to a peak of $10^{-4}$ over the first 10% of steps, then cosine-anneal to $10^{-7}$. Gradient clipping at 10.0. The best-val checkpoint is saved.
2. **Phase 2, L-BFGS fine-tuning** (full-batch, `strong_wolfe` line search, up to 1500 iterations, history 5). Starts from the Phase-1 best. Because the 3.5M-pair full-batch forward would OOM a wide model, the closure is **chunked** (`LBFGS_CHUNK_SIZE = 4096`): each chunk's sum-reduced loss is back-propagated weighted by `chunk_size / N`, so the accumulated gradient equals the exact full-batch mean gradient at $O(\text{chunk})$ peak memory. The closure re-validates each iteration and **`fnn.pt` is overwritten whenever the L-BFGS val beats the running best** (not just at the last iter), so the final checkpoint is the global best across both phases.

### 4.4 Step 3 — Reconstruction Network Training (`03_train_recon.py`)

**Purpose**: Learn to invert the detection process directly into the v6 primary encoding:

$$f_\text{recon}: (x_i, y_i, \hat E_i, \hat T_i)_{i=1}^{100} \;\longmapsto\; (\hat n_x, \hat n_y, \hat n_z, \widetilde{\log E}) \in \mathbb{R}^4$$

Given the per-detector positions and the FNN-predicted responses, reconstruct the primary's **direction unit vector** $\hat{\mathbf n} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ and **normalised log-energy** $\widetilde{\log E} = (\log_{10} E - 5)/3$. The network predicts the same 4-D encoding as the first four columns of the primary vector $\mathbf{q}$ (Section 3.3); $(E, \theta, \phi)$ are recovered analytically downstream (Section 4.5). Predicting a unit vector instead of $(\theta, \phi)$ avoids the $\phi$ branch cut at $0/2\pi$ and keeps the loss geometrically well-behaved near the poles.

**Training data generation**: The frozen FNN from Step 2 is run in eval mode on the entire training corpus to produce predicted $(\hat E_i, \hat T_i)$. The reconstruction network is trained on FNN *predictions* (not ground-truth kernel outputs) — this is critical because at optimisation time the reconstruction network will only ever see FNN outputs, so it must be robust to FNN-specific prediction patterns and artefacts.

**Targets**: The recon target is `primary[:, :4]` — i.e. the raw 4-D primary encoding $(n_x, n_y, n_z, \widetilde{\log E})$ — used in raw units. Per-channel z-score stats are taken from the FNN's `norm_stats.pt` (slots 0–3 of the primary input stats) and baked into the model as output denormalisation buffers so that both training loss and inference live in the same raw units.

**Architecture**: The v6 `Reconstruction` module (`modules_v6/reconstruction.py`) mirrors `FNNSurrogate` one-for-one:

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

Both the input z-score and the output de-z-score are registered buffers populated via `set_normalization(...)`, so the caller feeds raw detector features and reads raw primary-encoding units — no external pre-/post-processing, and no Tanh squashing.

**Shared normalisation with the FNN**: The recon input stats are reused directly from `norm_stats.pt` (rather than recomputed over the train subset). The per-detector stats `(x, y, E, T)` are broadcast to all 100 slots so that a permutation of the detectors leaves the normalisation invariant. This guarantees that training, validation, and `04_optimize.py` all see one identical z-score.

**Loss**: Sum of per-axis MSE in the raw primary-encoding space:

$$\mathcal{L}_\text{recon} = \text{MSE}_{n_x} + \text{MSE}_{n_y} + \text{MSE}_{n_z} + \text{MSE}_{\widetilde{\log E}}$$

**Permutation Augmentation**: Same random-permutation scheme as Step 2. The target is a scalar 4-vector (primary encoding) that is invariant to detector ordering — only the input features are permuted. This teaches the MLP to be approximately **permutation-invariant** in its output.

**Architecture note**: the recon MLP is the narrower **3×512-hidden** form (vs the FNN's 7×1024). The FNN's width is read from `fnn.pt`'s `config` at load time, so Step 3 stays correct even when the FNN width changes.

**Optimiser — two phases** (mirrors Step 2):

1. **Phase 1, Adam** at $3 \times 10^{-5}$, gradient clipping at 10.0, 300 epochs, batch 256. Best-val checkpoint saved.
2. **Phase 2, L-BFGS fine-tuning** (full-batch, `strong_wolfe`, up to 500 iterations, history 20) from the Phase-1 best. The full training input is precomputed once on-GPU; the closure is chunked (`LBFGS_CHUNK = 32768`) with the same mean-gradient-preserving weighting as Step 2. **`recon.pt` is overwritten whenever the L-BFGS val improves on the running best** (per-iteration, not just last iter), so the saved checkpoint is the global best across both phases. (Earlier revisions only checked the last L-BFGS iter and could discard a better mid-run minimum — now fixed to match Step 2.)

### 4.5 Step 4 — Layout Optimisation (`04_optimize.py`)

**Purpose**: Find detector positions that maximise reconstruction quality by backpropagating through the frozen FNN and reconstruction networks.

**Computational Graph** (per optimisation step):

```
                    ┌─────────────────────┐
                    │  LearnableXY module  │
                    │  x, y ∈ ℝ¹⁰⁰       │◄── gradient descent updates these
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Broadcast to batch  │
                    │  xy: (B, 100, 2)     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                  ▼
    ┌──────────────────┐               ┌──────────────────┐
    │ primary_batch     │               │                  │
    │ (B, 5)           │               │  FNN (frozen)    │
    │ sampled from     │──────────────▶│  (q, xy) → (E,T) │
    │ training corpus  │               │                  │
    └──────────────────┘               └────────┬─────────┘
                                                │
                                    (B, 100) E_pred, T_pred
                                                │
                                                ▼
                                    ┌──────────────────────┐
                                    │  Build recon input   │
                                    │  (x, y, E, T) flat   │
                                    │  (raw units)         │
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  Recon NN (frozen)   │
                                    │  z-score in/out baked │
                                    │  → (n̂_x, n̂_y, n̂_z,  │
                                    │     log_e_norm)     │
                                    │  decode → (Ê, θ̂, φ̂) │
                                    └──────────┬───────────┘
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                ▼                ▼
                    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                    │ U_E(Ê, E)    │  │ U_θ(θ̂, θ)    │  │ U_φ(φ̂, φ)    │
                    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                           │                 │                 │
                           └─────────────────┼─────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Reconstructa-   │
                                    │  bility gate r  │
                                    │  + U_PR          │
                                    └────────┬────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │  Composite Utility U     │
                              │  loss = -U               │
                              │  loss.backward()         │
                              └──────────┬───────────────┘
                                         │
                              ┌──────────▼───────────────┐
                              │  Adam step + project     │
                              │  to mountain surface     │
                              └──────────────────────────┘
```

#### 4.5.1 Learnable Layout

Detector positions are wrapped in a `LearnableXY` module holding two `nn.Parameter` tensors $(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^{100}$. Initialisation options: grid, centre-clustered, or random (default: `"center"`).

After each gradient step, positions are **projected back onto the mountain surface**: any detector that has drifted further than the nearest-neighbour gap from a valid centroid is snapped to that centroid. This constrains optimisation to the physically feasible region.

#### 4.5.2 Utility Function

The composite utility combines differentiable terms. The base `04_optimize.py` uses the full four-term v4 composite:

$$U_\text{base} = \frac{1}{10^3}\Bigl(10^2 \cdot U_\theta + 10^2 \cdot U_\phi + 10^3 \cdot U_E + 5 \times 10^5 \cdot U_\text{PR}\Bigr)$$

> **Important — the stage-4 *variants* drop $U_\text{PR}$.** All three newer scripts (`04_optimize_nuts.py`, `04_optimize_hmc_chains.py`, `04_optimize_lbfgs_ensemble.py`) still *compute* $U_\text{PR}$ and the gate $r$, but their optimised objective is the **three-term** composite
> $$U_\text{var} = \frac{1}{10^3}\bigl(10^2\,U_\theta + 10^2\,U_\phi + 10^3\,U_E\bigr),$$
> with the $5\times10^5\,U_\text{PR}$ term deliberately omitted (the gate $r$ still weights $U_E, U_\theta, U_\phi$ internally). When comparing utilities across scripts, note they are on different scales.

Each term is defined as follows:

**Reconstructability Gate** $r$:

A soft, per-event indicator of whether enough detectors were triggered:

$$r_b = \sigma\!\Bigl(\tau_2 \Bigl[\sum_{i=1}^{100} \sigma\bigl(\tau_1 (E_{\text{det},i}^{(b)} - \epsilon_\text{layout})\bigr) - n_\text{thresh}\Bigr]\Bigr)$$

where $\sigma$ is the sigmoid, $\tau_1 = 5$, $\epsilon_\text{layout} = 0.05$ is the minimum detector signal threshold, $\tau_2 = 5$, and $n_\text{thresh} = 10$ is the minimum triggered-detector count. This gives $r_b \approx 1$ when many detectors fire and $r_b \approx 0$ when too few do.

**Participation Rate Utility** $U_\text{PR}$:

$$U_\text{PR} = \sqrt{\sum_b r_b + 10^{-6}}$$

Rewards layouts that make more events reconstructable (i.e., that trigger enough detectors).

**Energy Utility** $U_E$:

$$U_E = \frac{1}{B}\sum_b \frac{r_b}{(\log_{10}\hat E_b - \log_{10} E_b)^2 + 0.01}$$

Rewards accurate energy reconstruction in log-space. The $r_b$ weighting means only reconstructable events contribute; the $0.01$ floor prevents division by zero and bounds the maximum reward per event.

**Angular Utility** $U_\theta$, $U_\phi$:

$$U_\text{angle} = \frac{1}{B}\sum_b \frac{r_b}{(\hat\alpha_b - \alpha_b)^2 + 0.001}$$

for $\alpha \in \{\theta, \phi\}$. Same structure as $U_E$ but with a tighter floor (angles are in radians, so residuals are inherently smaller).

The **loss** is $\mathcal{L} = -U$, so gradient descent maximises utility.

#### 4.5.3 Optimisation Loop (base `04_optimize.py`)

- **Optimiser**: Adam with $\text{lr} = 1.0$, gradient clipping at 100.0
- **Batch size**: 256 random primaries sampled each epoch from the training corpus
- **Epochs**: `N_OPT_EPOCHS = 10{,}000`
- **Init schemes**: runs once per entry in `INIT_SCHEMES = ("grid", "center")`; each writes to its own `_{scheme}` folder so trajectories can be compared
- **Mountain projection**: After each Adam step, detector positions are projected back to the mountain surface
- **Logging**: Utility components, gradient norms, and layout snapshots saved periodically

#### 4.5.4 Stage-4 Variants (uncertainty + multi-start)

Three sibling scripts wrap the same frozen FNN+recon objective ($U_\text{var}$, Section 4.5.2) with richer search/uncertainty machinery. All share a common front end: K Gaussian-perturbed restarts of the chosen init scheme, each Adam-warm-started, then a second stage. A `"combined"` run pools the per-scheme restarts (grid + centre) into one analysis.

> **State of the art.** `04_optimize_lbfgs_ensemble.py` is the current recommended/most-developed Stage-4 entry point. The L-BFGS ensemble gives a deterministic local optimum per restart plus a network-input-invariant mean ± std uncertainty map (via position alignment), which proved more useful and far cheaper than the NUTS posterior — the samplers concentrate on the typical set rather than the mode, so they report a *lower* best-$U$ than the optimisers and cost ~6–7 h per combined run. Prefer the L-BFGS ensemble; treat the NUTS/HMC scripts as exploratory.

- **`04_optimize_nuts.py`** — Adam warm-start, then a single **Pyro NUTS** chain samples the $U$-weighted posterior $\log p(\mathbf{xy}) = U(\mathbf{xy})/T + \log\mathcal{N}(\mathbf{xy}\mid\mathbf{xy}_\text{Adam}, \sigma_\text{prior}^2)$ on a fixed primary batch, anchored near the Adam optimum. Reports the best-$U$ draw plus per-detector 1σ ellipses.

- **`04_optimize_hmc_chains.py`** — multi-sequence Gelman–Rubin variant. K NUTS chains start from **overdispersed** perturbed Adam optima (init spread > prior σ) and run **sequentially in-process** (Pyro's multi-process mode can't pickle the CUDA + closure `potential_fn`; sequential is the same wall-time on one GPU). R̂ and ESS are computed from the stacked `(chains, draws, dim)` array via ArviZ. The prior is anchored at a **single real Adam-best layout** (not the per-index mean across layouts, which would collapse detectors toward the centroid and bias the result central). Defaults: 4 chains × 1500 warmup × 1500 samples, $T = 3$, $\sigma_\text{prior} = 100$ m.

- **`04_optimize_lbfgs_ensemble.py`** — frequentist ensemble. Each of K perturbed Adam optima is refined to a local optimum by **L-BFGS** on a fixed batch, then the K layouts are **aligned by physical position** — a Hungarian assignment (`scipy.optimize.linear_sum_assignment`, with a dependency-free greedy fallback) matches detectors across runs by closest $(x,y)$, since the permutation-equivariant networks make detector *index* meaningless across runs. Per aligned position-group it reports **mean and std**, giving a network-input-invariant uncertainty map. It also logs a **per-run consecutive-step gradient cosine distance** (with $W$-step vector-averaging to cancel minibatch-noise inflation) as a convergence diagnostic.


## 5. Key Design Decisions

### 5.1 Why Two Surrogate Networks?

The pipeline decomposes the problem into two stages because the physics has a natural factorisation:

1. **Detection** (FNN): How does a given layout respond to a given shower? This depends on both the primary particle and the detector positions.
2. **Reconstruction** (Recon): Given what the detectors saw, what was the primary particle? This is the inverse problem.

Training them separately allows:
- The FNN to focus on learning the spatial/temporal kernel without confounding reconstruction errors.
- The recon network to be trained on FNN *predictions* (not ground truth), making it robust to the surrogate's systematic biases.
- Either network to be retrained independently if the architecture or training data changes.

Both networks share the same **normalisation contract** (per-feature z-score baked into the forward pass via registered buffers) and the same two-phase Adam→L-BFGS training recipe. They no longer share width: the FNN is currently 7×1024 hidden while the recon MLP is 3×512 (both dropout 0.1). Width is stored in each checkpoint's `config` and read back by downstream stages, so the surrogates can be resized independently. The recon network's stats are reused from the FNN checkpoint's `norm_stats` so train / val / optimisation all see one identical z-score.

### 5.2 Why Permutation Augmentation?

Both networks use flat MLPs, which are inherently order-dependent, so training applies random per-sample detector permutations to *approximate* the symmetry: equivariance for the FNN (permuting the layout permutes the responses), invariance for the recon (reordering detectors must not change the inferred primary). This was chosen for simplicity, but the architecture search (§10) found it to be the dominant bottleneck — a set-equivariant model that bakes the symmetry in *by construction* (DeepSets) is the recommended replacement. Augmentation is the current expedient, not the endpoint.

### 5.3 Why Train Recon on FNN Predictions?

If the recon network were trained on ground-truth kernel outputs, it would learn to exploit features of the exact kernel that the FNN cannot reproduce. At optimisation time, the recon network only sees FNN outputs; training on FNN predictions ensures consistency and avoids a domain gap between training and deployment.

### 5.4 Why a Composite Utility (Not Just Reconstruction Loss)?

Pure reconstruction MSE would ignore whether a layout is even viable — a layout where no detectors trigger would have undefined reconstruction quality. The composite utility:
- $U_\text{PR}$ directly rewards layouts that trigger enough detectors.
- $U_E$, $U_\theta$, $U_\phi$ reward reconstruction accuracy but only for reconstructable events.
- The weighting has been manually selected during the testing runns to balance the separate utility fractions. Further work is needed.

### 5.5 Why Project to the Mountain?

The mountain surface is a non-convex 2D manifold embedded in 3D. Unconstrained gradient descent would push detectors off the mountain into physically impossible positions. After each Adam update, the `project_to_mountain()` method snaps any drifted detector back to the nearest valid centroid. This is a projected gradient descent approach on a discrete feasible set.


## 6. Normalisation Strategy

The pipeline normalises both surrogates through the **same z-score pipeline**, sourced from a single `norm_stats.pt`:

| Where | What | Why |
|-------|------|-----|
| **FNN input/output** | Z-score (mean/std from training corpus, baked into forward pass buffers) | Ensures all features contribute equally to MSE; mountain-scale coordinates and small energy counts are on comparable scales |
| **Recon input** | Z-score reusing the FNN's per-feature primary/xy/E/T stats (broadcast per-detector, baked into forward pass buffers) | Mountain-scale $(x, y)$ values and FNN output scales need normalisation; sharing stats with the FNN keeps train / val / optimisation on one identical scale |
| **Recon output** | Z-score denormalisation with stats from FNN primary slots 0–3 (baked into forward pass buffers) | The 4-D primary encoding lives in known raw units; baking de-z-score into `forward()` lets losses and downstream decoding work directly in raw primary units, with no min-max/Tanh squash |
| **FNN training labels (E)** | $\log(1 + E)$ before z-score (applied in Step 1, stored in `E.pt`) | Compresses heavy right tail of the energy distribution |
| **FNN training labels (T)** | $\log(1 + T\cdot 10^{8})$ applied **in Step 2** (not stored on disk) | Raw $T$ spans $10^{-12}$–$10^{-6}$ s; the rescale maps it to $\sim 0$–$5.5$. The modified T-channel stats are shipped **inside `fnn.pt`**; disk `norm_stats.pt` keeps raw-T, so Steps 3–4 read the checkpoint's `norm_stats` for consistency |


## 7. Data Layout and Storage

All intermediate data is stored as PyTorch tensors under `RUN_LOCATION` (holylfs05). The current run tree uses the recentered corpus; the production names below are the `test_v6_run_0X_recentered` folders set in `modules_v6/constants.py`. Numbers shown are for the default 500k-shower / 7-strategy corpus.

```
v6_run_00/                       ← Step 0: cached showers (shared across runs)
    cashed_showers_500000.pt     (HDF5 via showerdata; ~20 GB at 500k)

test_v6_run_01_recentered/       ← Step 1: training dataset
    primary.pt          (3500000, 5)      primary features
    xy.pt               (3500000, 100, 2) detector layouts (recentered)
    E.pt                (3500000, 100)    log1p(energy per detector)
    T.pt                (3500000, 100)    RAW time per detector (seconds)
    strategy_ids.pt     (3500000,)        layout strategy id [0–6], strategy-major
    norm_stats.pt       dict of z-score tensors (raw-T)

test_v6_run_02_recentered/       ← Step 2: FNN checkpoint
    fnn.pt              state_dict + norm_stats (log-T) + config(hidden,dropout)
    fnn_train_log.json  Adam per-epoch + L-BFGS iter log
    fnn_train_curves.png
    fnn_target_vs_pred.png        density heatmap, auto-rendered at end of Step 2

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
```


## 8. Execution Environment

The pipeline runs on a SLURM-managed HPC cluster with A100 GPUs:

- **Step 0** (shower generation): GPU-intensive; size `--mem` for the end-of-run save peak (Section 4.1). At 500k showers, ~2.5 h; at 5M, ~20 h and 200 GB+.
- **Steps 1–4**: single A100. With the 3.5M-pair corpus these are no longer "1-hour" jobs — Step 2 ≈ 5 h (100 Adam epochs + 1500 L-BFGS iters), Step 3 ≈ 5 h, and the stage-4 variants are dominated by sampling/refinement (e.g. the combined NUTS run ≈ 6–7 h for 8 sequential chains × 3000 steps).
- Step 1 is CPU-bound (I/O-dominated kernel evaluation).
- Steps 2–4 use GPU for neural-network operations; L-BFGS phases use chunked closures to bound peak GPU memory.
- The `gpu_requeue` partition preempts jobs (no incremental Step-0 checkpointing → a preemption restarts generation from scratch); request a longer wall and adequate memory, or use a non-preemptable partition for long generation runs.


## 9. Mathematical Summary

The full optimisation objective, written as a single differentiable expression through both frozen networks (base `04_optimize.py`; the variants drop the $w_\text{PR}U_\text{PR}$ term — see Section 4.5.2):

$$\max_{\mathbf{x}, \mathbf{y}} \;\; \mathbb{E}_{\mathbf{q} \sim \mathcal{D}} \left[ \frac{w_\theta \cdot U_\theta + w_\phi \cdot U_\phi + w_E \cdot U_E + w_\text{PR} \cdot U_\text{PR}}{w_\text{div}} \right]$$

subject to $\;(x_i, y_i) \in \mathcal{M}\;$ for all $i$, where $\mathcal{M}$ is the mountain surface.

The expectation is approximated by mini-batches of 256 primaries sampled uniformly from the training corpus each epoch. Gradients flow from the scalar utility $U$ through:

$$U \;\xleftarrow{\text{utility}}\; (\hat E, \hat\theta, \hat\phi, r) \;\xleftarrow{\text{decode}}\; (\hat n_x, \hat n_y, \hat n_z, \widetilde{\log E}) \;\xleftarrow{f_\text{recon}}\; (x_i, y_i, \hat E_i, \hat T_i) \;\xleftarrow{f_\text{FNN}}\; (\mathbf{q}, x_i, y_i)$$

back to the learnable parameters $(x_i, y_i)$. Both $f_\text{FNN}$ and $f_\text{recon}$ are frozen (no weight updates); only the detector positions receive gradient updates.


## 10. FNN Surrogate: Development Log and Known Limitation

The Step-2 surrogate has been the subject of an extensive architecture/hyperparameter search (full chronology in `diary.md`). The findings below are load-bearing for anyone continuing the work.

### 10.1 What was tried, and the verdict

Search ran on a 10% data subset; numbers are relative z-scored val-MSE within the search, **not** comparable to full-corpus production runs.

- **Flat-MLP family plateaus at val ≈ 0.69–0.71.** OneCycleLR, GELU+LayerNorm, and width up to 1024 each gave only marginal gains; none broke the plateau.
- **Deep Sets was the one decisive lever:** a per-detector shared MLP with pooled context reached **0.546 (−23%)**, with E dropping 40%. The permutation-equivariance inductive bias — which the current flat MLP only *approximates*, expensively, via augmentation — is what mattered.
- **Capacity is not the bottleneck.** A 4× wider Deep Sets did not improve; it converged faster to a no-better minimum.
- **A hard cross-method floor at ≈ 0.546.** Set Transformer (SAB/ISAB), a k-NN GNN, primary→detector cross-attention, learned uncertainty weighting, and AdamW+3× LR **all landed within ±0.3%** of Deep Sets. Such tightness across radically different models points to a **data/label ceiling**, not an architecture limit — most likely irreducible label noise from the upstream shower simulation, or a feature gap for the T channel (T stays ~2× harder than E throughout).
- **Zero-inflation handling backfires.** Soft reweighting, BCE hit-gates, and hard masks to focus the loss on the rare non-zero T positions all *regressed* the pure-MSE metric — the ~96% zero-target positions dominate the MSE and masking starves their gradient. Zero-inflation is **not** the fitting bottleneck.
- **log-T target** (`T ← log1p(T·10⁸)`, §6) and the **L-BFGS best-iter save fix** (§4.3) were the two changes that did stick and are now in production.

### 10.2 Path-c schedule de-risk (2026-06-04) — failed, reverted

A controlled attempt to lift the *production flat MLP* by schedule/optimizer alone (LR-range test → AdamW + weight decay, dropout off, raised LR floor, L-BFGS capped, and a fix to a latent OneCycle `final_div_factor` bug) **regressed** (val 0.60 vs the prior 0.40). The honest conditional metrics exposed the real failure mode the total-MSE hides:

| metric | value | meaning |
|--------|-------|---------|
| E R² (all detectors) | 0.45 | flattering — dominated by trivially-correct empty detectors |
| **E R² (fired only)** | **−0.14** | magnitude is *worse* than predicting the fired-channel mean |
| fire precision / recall | 0.42 / 0.99 | the model **over-fires**, leaking energy onto empty detectors |
| fired pred/target std | 0.69 | magnitude compression — classic predict-the-mean |

All path-c edits were reverted; the working tree matches the recipe documented in §4.3.

### 10.3 Standing recommendation

Track **conditional-on-fired E/T R² and fire precision/recall**, not total val-MSE — the latter is flattering because ~68% of detector-samples are near-zero. The next high-leverage change is **path (a): re-architect the FNN as a pointwise DeepSets** `φ(q, xᵢ, yᵢ) → (Eᵢ, Tᵢ)` with weights shared across detectors. This is permutation-equivariant by construction (matching the provably per-detector-local kernel of §3.4), removes the augmentation, uses ~34× fewer parameters, and turns each detector into its own training example (~100× more effective samples). It preserves the `forward(primary, xy) → (B,100,2)` contract, so Steps 3–4 are unaffected. The recon network (§4.4) shares the same flat-MLP mismatch (its target is permutation-*invariant*) and is the natural follow-on.
