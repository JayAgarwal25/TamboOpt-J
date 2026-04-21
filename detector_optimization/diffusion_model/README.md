# diffusion_model (v1)

First-generation diffusion-based shower generators for the TAMBO/SWGO detector optimization pipeline. Archival — superseded by `detector_optimization_v2/diffusion_model/`.

Three class-based generators live here, each wrapping a trained checkpoint behind a reusable interface:

| Class | File | Output | Sampler |
|-------|------|--------|---------|
| `TamboDiffusionGenerator` | `tambo_diffusion_generator.py` | 2D shower images `(3, 32, 32)` | DDIM |
| `PlaneDiffusionEvaluator` | `tambo_3D_diffusion_generator.py` | 3D multi-plane shower volumes | DDIM (planes) |
| `PlaneFNNGenerator` | `tambo_3D_fnn_scaler.py` | Per-plane bounding boxes (direct regression) | FNN (no sampling) |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_validate.ipynb` | Validation of `TamboDiffusionGenerator` against ground truth |
| `02_reduce_runtime.ipynb` | Runtime profiling / tuning of `TamboDiffusionGenerator` |
| `03_3D_generator_update.ipynb` | Validation of `PlaneDiffusionEvaluator` |
| `04_3D_generator_scaled.ipynb` | Scaled outputs of `PlaneDiffusionEvaluator` |

`example_usage.py` — copy-pasteable snippets for `TamboDiffusionGenerator`.

---

## TamboDiffusionGenerator

2D DDIM sampler over shower images, conditioned on 5-element feature vectors.

### Quick start

```python
from tambo_diffusion_generator import TamboDiffusionGenerator

generator = TamboDiffusionGenerator(
    checkpoint_path="/path/to/ckpt_epoch=1999.ckpt",
    output_dir="output/run_1",
    tambo_optimization_path="/path/to/tambo_optimization",
)

generator.run_full_pipeline(
    num_samples=1000,   # per condition
    num_conditions=10,  # number of test conditions
    chunk_size=200,     # batch size for generation (OOM control)
)
```

### Step-by-step

```python
generator.load_model()
generator.setup_data()
generator.extract_test_samples(num_conditions=20)
generated_sets = generator.generate_samples(num_samples=500, num_conditions=10, chunk_size=100)
generator.save_results()
generator.plot_results(num_conditions=10, dpi=300)
```

### `generate_samples` return value

Returns (and caches on `generator.generated_sets`) a list of per-condition dicts. Each entry:

```python
{
    "condition": torch.Tensor,  # shape (5,),                  on CPU
    "images":    torch.Tensor,  # shape (num_samples, 3, 32, 32), on CPU, float
}
```

Length of the list equals `num_conditions` (or all extracted conditions if `num_conditions=None`). Tensors are moved to CPU inside the loop so the GPU is free between conditions — move to `.to(device)` before downstream use.

**Custom conditions** — `extract_test_samples()` is optional. You can skip `setup_data()` / `extract_test_samples()` entirely and assign `test_conditions` directly (e.g. for targeted energy/angle sweeps or out-of-distribution probes):

```python
generator.load_model()
generator.test_conditions = [
    torch.tensor([energy, sin_z, cos_z, sin_a, cos_a]),
    # ... one (5,) tensor per condition
]
generator.generate_samples(num_samples=500, chunk_size=100)
```

`generate_samples` only requires that `self.test_conditions` be a non-empty iterable of `(5,)` tensors.

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | str | required | Path to trained model checkpoint |
| `output_dir` | str | required | Output directory (auto-created) |
| `device` | str | auto | `"cuda:0"`, `"cpu"`, etc. |
| `ddim_steps` | int | 100 | DDIM sampling steps |
| `ddim_eta` | float | 0.0 | DDIM eta (0 = deterministic) |
| `batch_size` | int | 64 | Dataloader batch size |
| `train_ratio` / `val_ratio` / `test_ratio` | float | 0.85 / 0.10 / 0.05 | Data split |
| `num_workers` | int | 4 | Dataloader workers |
| `seed` | int | 42 | Random seed |
| `tambo_optimization_path` | str | None | Path appended to `sys.path` to locate `diffusion_train` / `models.DiffusionCondition` |

### Output layout from save_results

```
output_dir/
├── condition_1.npz       # bundle per condition
│   ├── input             #   (5,) condition vector
│   ├── target            #   (3, 32, 32) ground-truth image
│   ├── output            #   (N, 3, 32, 32) generated images
│   └── meta              #   metadata dict
├── condition_2.npz
├── ...
├── summary.npz           # { all_conditions, total_images, num_conditions }
├── condition_1.png       # GT vs generated comparison plot
└── ...
```

### Loading saved results

```python
import numpy as np

data = np.load("output_dir/condition_1.npz", allow_pickle=True)
bundle = data["bundle"].item()

condition    = bundle["input"]    # (5,)
ground_truth = bundle["target"]   # (3, 32, 32)
generated    = bundle["output"]   # (N, 3, 32, 32)
metadata     = bundle["meta"]     # dict
```

### Chunk-size guidance

`chunk_size` controls how many samples are diffused in parallel. Tune to GPU memory:

- 4 GB GPU → 50–100
- 8 GB GPU → 100–200
- 16 GB+ GPU → 200–500

---

## PlaneDiffusionEvaluator (3D)

DDIM sampler over stacked detector planes. Generates 24-plane shower volumes **autoregressively** — each plane is denoised conditioned on the previously generated plane. Supports classifier-free guidance.

```python
from tambo_3D_diffusion_generator import PlaneDiffusionEvaluator

evaluator = PlaneDiffusionEvaluator(
    data_dir="/path/to/data",
    checkpoint_path="/path/to/ckpt.ckpt",
    device="cuda:0",
    ddim_steps=50,
    eta=0.0,
    guidance_w=0.0,      # classifier-free guidance weight
    imports_path="/path/to/tambo_optimization",
)
evaluator.run_full_pipeline(num_samples=10, ddim_steps=50)
```

Depends on `lightning_training.PlaneDataset`, `lightning_training.PlaneDiffusionModule`, and `diffusion.DDIMSamplerPlanes` from the parent project.

### `load_model`

Loads the checkpoint as a `PlaneDiffusionModule` (via `load_from_checkpoint`), puts it in `eval()` on the configured device, extracts the underlying net, and builds a `DDIMSamplerPlanes` using the constructor values for `beta_1`, `beta_T`, `T`, `eta`, `ddim_steps`, and `guidance_w`. After this call the evaluator is ready to sample — `self.net` and `self.sampler` are populated.

```python
evaluator.load_model()
```

### `generate_samples`

Runs the autoregressive sampler across all 24 planes for each test condition, chunked by `batch_size` to control memory. Requires `load_model()` and `extract_test_samples()` to have run first.

```python
gen = evaluator.generate_samples(
    num_conditions=None,  # None = all extracted conditions
    batch_size=100,       # conditions processed per chunk
)
```

Returns (and caches on `evaluator.generated_sets`) a single dict containing tensors for **all** conditions (not a per-condition list like `TamboDiffusionGenerator`):

```python
{
    "conditions": torch.Tensor,  # (N, 6), on CPU — [energy, class_id, sin_z, cos_z, sin_a, cos_a]
    "images":     torch.Tensor,  # (N, 24, 3, 32, 32), on CPU, float — 24 stacked RGB planes per shower
}
```

where `N = num_conditions` (or the full set if `None`).

**Custom conditions** — as with `TamboDiffusionGenerator`, `extract_test_samples()` is optional. Assign `test_conditions` directly to sample at arbitrary points in condition space:

```python
evaluator.load_model()
evaluator.test_conditions = torch.tensor([
    [energy, class_id, sin_z, cos_z, sin_a, cos_a],
    # ... one row per condition
])
evaluator.generate_samples(batch_size=100)
```

`generate_samples` only requires that `self.test_conditions` be a non-empty `(N, 6)` tensor.

---

## PlaneFNNGenerator

Feedforward bbox regressor — no diffusion sampling, just a direct forward pass conditioned on `[energy, class_id, sin_z, cos_z, sin_a, cos_a]`. Used as a fast per-plane bounding-box predictor in the 3D pipeline.

```python
from tambo_3D_fnn_scaler import PlaneFNNGenerator

generator = PlaneFNNGenerator(
    data_dir="/path/to/data",
    checkpoint_path="/path/to/fnn_ckpt.ckpt",
    device="cuda:0",
    imports_path="/path/to/tambo_optimization",
)
generator.load_model()
generator.test_conditions = torch.tensor([[energy, class_id, sin_z, cos_z, sin_a, cos_a]])
outputs = generator.generate_samples(num_samples=100)
```

Depends on `lightning_training_fnn.PlaneDataset` and `lightning_training_fnn.PlaneFNNModule`. Bbox standardization stats are loaded from `data_dir`.

---

## Requirements

- PyTorch, PyTorch Lightning
- NumPy, Matplotlib
- Project-local modules: `diffusion_train`, `models.DiffusionCondition`, `lightning_training(_fnn)`, `diffusion.DDIMSamplerPlanes`
  (add their parent dir via `tambo_optimization_path` / `imports_path`)

---

## Status

Archival v1 generators — kept for reproducibility of the earlier SWGOLO7 runs. Current work uses the refactored generators under `detector_optimization_v2/diffusion_model/` (`PlaneDiffusionEvaluator`, `PlaneFNNGenerator` there have been integrated into the differentiable shower-generation step).
