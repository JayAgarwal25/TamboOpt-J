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
| **Step 0** | Energy/angle sampling ranges | Point-cloud showers $(\mathbf{r}, E, t)$ | Generate synthetic shower library |
| **Step 1** | Shower library + mountain geometry | $(primary, xy, E, T)$ training tensors | Pair showers with diverse layouts; compute detector responses |
| **Step 2** | Training tensors | Frozen `fnn.pt` checkpoint | Train surrogate: $(primary, layout) \to (E_\text{det}, T_\text{det})$ |
| **Step 3** | Training tensors + frozen FNN | Frozen `recon.pt` checkpoint | Train reconstruction: $(x, y, E_\text{det}, T_\text{det}) \to (\hat E, \hat\theta, \hat\phi)$ |
| **Step 4** | Frozen FNN + frozen recon + primaries | Optimized layout $(\mathbf{x}^*, \mathbf{y}^*)$ | Maximise composite utility via backpropagation through both NNs |


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

where $(x_j, y_j)$ are transverse positions (metres), $l_j$ is the discrete AllShowers layer index (0–23), $e_j$ is the particle energy, and $t_j$ is the arrival time (nanoseconds). Showers are generated by a pre-trained **flow-matching generative model** (AllShowers), which is conditioned on the primary particle's energy $E \in [10^5, 10^8]$ GeV, zenith $\theta \in [60^\circ, 100^\circ]$, and azimuth $\phi \in [0^\circ, 360^\circ]$.

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

A library of 100,000 shower point clouds is generated by the AllShowers flow-matching model and cached to disk. Primary particles are sampled uniformly in $(\theta, \phi, \log E)$ and the flow-matching solver produces stochastic point clouds via $T = 16$ midpoint integration steps. This step is GPU-intensive and runs once; all downstream stages read the cached corpus.

### 4.2 Step 1 — Dataset Construction (`01_build_dataset.py`)

Each shower is paired with 5 different detector layouts drawn from diverse **placement strategies**:

| Strategy | Description | Purpose |
|----------|-------------|---------|
| `grid_jit20` | Regular grid + Gaussian jitter ($\sigma = 20$ m) | Covers mountain uniformly |
| `center_gauss200` | Cluster at mountain centre ($\sigma = 200$ m) | Tests concentrated layouts |
| `rings_R300` | 5 concentric rings, $R = 300$ m | Tight ring pattern |
| `rings_R800` | 6 concentric rings, $R = 800$ m | Medium ring pattern |
| `rings_R1800` | 6 concentric rings, $R = 1800$ m | Wide ring spanning mountain |

For each (shower, layout) pair, the physics-based kernel (Section 3.4) computes ground-truth detector responses $(E_{\text{det}}, T_{\text{det}})$. The E and T values are log-transformed via $\log(1 + x)$ to compress the heavy right tail, and z-score normalisation statistics are computed over the full corpus.

This produces a training corpus of $100{,}000 \times 5 = 500{,}000$ training pairs, each consisting of:
- **Input**: primary vector $\mathbf{q} \in \mathbb{R}^5$ + detector layout $\mathbf{xy} \in \mathbb{R}^{100 \times 2}$
- **Label**: detector responses $\mathbf{E} \in \mathbb{R}^{100}$, $\mathbf{T} \in \mathbb{R}^{100}$

The use of multiple layout strategies ensures the surrogate learns the dependence on detector positions, not just on the primary particle — this is what makes the surrogate useful for layout optimisation.

### 4.3 Step 2 — Forward Surrogate Training (`02_train_fnn.py`)

**Purpose**: Learn a fast, differentiable approximation:

$$f_\text{FNN}: (\mathbf{q}, \mathbf{xy}) \;\longmapsto\; (\hat{\mathbf{E}}, \hat{\mathbf{T}}) \in \mathbb{R}^{100 \times 2}$$

**Architecture**: A feedforward network (FNNSurrogate):

```
Input:  [q (5), xy_flat (200)] = 205 features
        ↓ z-score normalisation (frozen buffers)
Linear(205, 512) → ReLU → Dropout(0.1)
Linear(512, 512) → ReLU → Dropout(0.1)
Linear(512, 512) → ReLU
Linear(512, 200)
        ↓ z-score denormalisation
Output: [E (100), T (100)] reshaped to (100, 2)
```

Z-score normalisation is baked into the forward pass via registered buffers, so the model can be dropped into the optimisation loop without external preprocessing.

**Loss**: Mean squared error in z-scored output space, with E and T channels weighted equally:

$$\mathcal{L}_\text{FNN} = \frac{1}{2}\bigl(\text{MSE}_E + \text{MSE}_T\bigr) \quad\text{where}\quad \text{MSE}_c = \frac{1}{BN}\sum_{b,i}\Bigl(\frac{\hat y_{b,i,c} - y_{b,i,c}}{\sigma_c}\Bigr)^2$$

**Permutation Augmentation**: Because the FNN is a flat MLP (not a set-equivariant architecture), the detector ordering is arbitrary. To teach approximate permutation equivariance by data augmentation, every training batch applies an **independent random permutation** of the 100 detectors per sample. The same permutation is applied to the input layout $\mathbf{xy}$ and the target $(\mathbf{E}, \mathbf{T})$ jointly, preserving the detector-wise correspondence while exposing the network to all orderings.

**Train/Val Split**: Shower-level 90/10 split — all 5 layout variants of the same shower go into the same split, preventing information leakage.

**Optimiser**: Adam with learning rate $10^{-5}$, cosine annealing to $10^{-7}$, gradient clipping at 10.0.

### 4.4 Step 3 — Reconstruction Network Training (`03_train_recon.py`)

**Purpose**: Learn to invert the detection process:

$$f_\text{recon}: (x_i, y_i, \hat E_i, \hat T_i)_{i=1}^{100} \;\longmapsto\; (\hat E, \hat\theta, \hat\phi)$$

Given the per-detector positions and the FNN-predicted responses, reconstruct the primary particle's physical properties.

**Training data generation**: The frozen FNN from Step 2 is run in eval mode on the entire training corpus to produce predicted $(\hat E_i, \hat T_i)$. The reconstruction network is trained on FNN *predictions* (not ground-truth kernel outputs) — this is critical because at optimisation time the reconstruction network will only ever see FNN outputs, so it must be robust to FNN-specific prediction patterns and artefacts.

**Targets**: Physical labels $(E_\text{GeV}, \theta_\text{rad}, \phi_\text{rad})$ are extracted from the primary encoding and normalised to $[0, 1]$ via:

$$\tilde E = \frac{E - E_\text{min}}{E_\text{max} - E_\text{min}}, \quad \tilde\theta = \frac{\theta - \theta_\text{min}}{\theta_\text{max} - \theta_\text{min}}, \quad \tilde\phi = \frac{\phi}{2\pi}$$

**Architecture**: A 3-layer MLP (v3 Reconstruction):

```
Input:  [x, y, E_pred, T_pred] × 100 = 400 features
        ↓ z-score normalisation (frozen training-time stats)
Linear(400, 256) → ReLU → Dropout(0.1)
Linear(256, 128) → ReLU
Linear(128,  32) → ReLU
Linear( 32,   3) → Tanh
Output: (E_norm, θ_norm, φ_norm) ∈ [-1, 1]³
```

The frozen z-score normalisation of the input is essential: raw mountain-scale coordinates (thousands of metres) would saturate the Tanh output without it.

**Loss**: Sum of per-axis MSE in the normalised label space:

$$\mathcal{L}_\text{recon} = \text{MSE}_E + \text{MSE}_\theta + \text{MSE}_\phi$$

**Permutation Augmentation**: Same random-permutation scheme as Step 2, but now the *target* is a scalar 3-vector (primary properties) that is invariant to detector ordering — only the input features are permuted. This teaches the MLP to be approximately **permutation-invariant** in its output.

**Optimiser**: Adam at $3 \times 10^{-5}$, gradient clipping at 10.0.

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
                                    │  → z-score norm      │
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  Recon NN (frozen)   │
                                    │  → (Ê, θ̂, φ̂)        │
                                    │  → Denormalize       │
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

The composite utility combines four differentiable terms:

$$U = \frac{1}{10^3}\Bigl(10^2 \cdot U_\theta + 10^2 \cdot U_\phi + 10^3 \cdot U_E + 5 \times 10^5 \cdot U_\text{PR}\Bigr)$$

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

#### 4.5.3 Optimisation Loop

- **Optimiser**: Adam with $\text{lr} = 1.0$, gradient clipping at 100.0
- **Batch size**: 256 random primaries sampled each epoch from the training corpus
- **Epochs**: 10,000
- **Mountain projection**: After each Adam step, detector positions are projected back to the mountain surface
- **Logging**: Utility components, gradient norms, and layout snapshots saved periodically


## 5. Key Design Decisions

### 5.1 Why Two Surrogate Networks?

The pipeline decomposes the problem into two stages because the physics has a natural factorisation:

1. **Detection** (FNN): How does a given layout respond to a given shower? This depends on both the primary particle and the detector positions.
2. **Reconstruction** (Recon): Given what the detectors saw, what was the primary particle? This is the inverse problem.

Training them separately allows:
- The FNN to focus on learning the spatial/temporal kernel without confounding reconstruction errors.
- The recon network to be trained on FNN *predictions* (not ground truth), making it robust to the surrogate's systematic biases.
- Either network to be retrained independently if the architecture or training data changes.

### 5.2 Why Permutation Augmentation?

Both networks use flat MLPs, which are inherently order-dependent. A principled solution would be a set-equivariant architecture (e.g., DeepSets, Set Transformer), but MLPs are simpler, faster, and sufficient when combined with permutation augmentation during training. The augmentation ensures:
- The FNN learns that permuting the input layout and the output responses together produces the same physical situation (equivariance).
- The recon network learns that the scalar primary properties are invariant to detector ordering (invariance).

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

The pipeline uses three distinct normalisation schemes, each for a specific purpose:

| Where | What | Why |
|-------|------|-----|
| **FNN input/output** | Z-score (mean/std from training corpus, baked into forward pass buffers) | Ensures all features contribute equally to MSE; mountain-scale coordinates and small energy counts are on comparable scales |
| **Recon targets** | Min-max to $[0, 1]$ via NormalizeLabels | Tanh output head naturally matches $[-1, 1]$; physical range is known a priori |
| **Recon input** | Z-score (frozen training-time stats) | Mountain-scale $(x, y)$ values (thousands of metres) would saturate Tanh without normalisation |
| **FNN training labels** | $\log(1 + x)$ before z-score | Compresses heavy right tail of energy/time distributions |


## 7. Data Layout and Storage

All intermediate data is stored as PyTorch tensors in the `outputs/` directory structure:

```
v6_run_00/          ← Step 0: cached showers
    cashed_showers_100000.pt

v6_run_01/          ← Step 1: training dataset
    primary.pt          (500000, 5)     primary features
    xy.pt               (500000, 100, 2) detector layouts
    E.pt                (500000, 100)   log1p(energy per detector)
    T.pt                (500000, 100)   log1p(time per detector)
    strategy_ids.pt     (500000,)       which layout strategy [0–4]
    norm_stats.pt       dict of z-score tensors

v6_run_02/          ← Step 2: FNN checkpoint
    fnn.pt              state_dict + norm_stats + training config
    fnn_train_log.json
    fnn_train_curves.png

v6_run_03/          ← Step 3: Recon checkpoint
    recon.pt            state_dict + input_mean/std + config
    recon_train_log.json
    recon_train_curves.png

v6_run_04_optimize/ ← Step 4: Optimisation results
    layout_best.pt      best-utility (x, y) snapshot
    layout_final.pt     last-epoch layout
    xy_trajectory.pt    periodic snapshots for trajectory analysis
    optimize_log.json   per-epoch utility breakdown
    optimize_curves.png
    layout_before_after.png
```


## 8. Execution Environment

The pipeline runs on a SLURM-managed HPC cluster with A100 GPUs:

- **Step 0** (shower generation): GPU-intensive, 1-day walltime, 100 GB memory
- **Steps 1–4** (dataset + training + optimisation): 1-hour walltime, single A100
- Step 1 is CPU-bound (I/O-dominated kernel evaluation)
- Steps 2–4 use GPU for neural network operations


## 9. Mathematical Summary

The full optimisation objective, written as a single differentiable expression through both frozen networks:

$$\max_{\mathbf{x}, \mathbf{y}} \;\; \mathbb{E}_{\mathbf{q} \sim \mathcal{D}} \left[ \frac{w_\theta \cdot U_\theta + w_\phi \cdot U_\phi + w_E \cdot U_E + w_\text{PR} \cdot U_\text{PR}}{w_\text{div}} \right]$$

subject to $\;(x_i, y_i) \in \mathcal{M}\;$ for all $i$, where $\mathcal{M}$ is the mountain surface.

The expectation is approximated by mini-batches of 256 primaries sampled uniformly from the training corpus each epoch. Gradients flow from the scalar utility $U$ through:

$$U \;\xleftarrow{\text{utility}}\; (\hat E, \hat\theta, \hat\phi, r) \;\xleftarrow{f_\text{recon}}\; (x_i, y_i, \hat E_i, \hat T_i) \;\xleftarrow{f_\text{FNN}}\; (\mathbf{q}, x_i, y_i)$$

back to the learnable parameters $(x_i, y_i)$. Both $f_\text{FNN}$ and $f_\text{recon}$ are frozen (no weight updates); only the detector positions receive gradient updates.
