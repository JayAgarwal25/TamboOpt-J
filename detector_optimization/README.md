# detector_optimization (v1)

Original, exploratory implementation of the SWGO/TAMBO detector layout optimization pipeline. Superseded by `detector_optimization_v2/` (and later versions) but kept for reference: earliest end-to-end notebooks, experimental simulator, and the first diffusion-based shower generator.

End-to-end flow (same surrogate as v2, but everything lives inside `SWGOLO7_optimization.ipynb` rather than extracted modules):

```
sample primary particles  ──▶  24-plane diffusion + FNN bbox  ──▶  Bilinear grid_sample → counts (N, T)
 (E, class, zenith, azimuth)     images: (N, 24, 3, 32, 32)            (differentiable in x_det, y_det)
                                 bboxes: (N, 24, 4)
                                         │
                                         ▼
                                 take plane 20, map image coords → world coords via bbox
                                 (X0 = ΣᵢⱼᵢWᵢⱼ/ΣWᵢⱼ, then rescaled through the plane-20 bbox)
                                                                           │
                                                                           ▼
                                                         Reconstruction MLP → [X0, Y0, E, θ, φ]
                                                                           │
                                                                           ▼
                                                Utility U = α·U_PR + β·U_E + γ·U_angle
                                                          backprop → LearnableXY → push_apart
```

The earliest `SWGOLO7.ipynb` also contains the 2D `TamboDiffusionGenerator` `(3, 32, 32)` pathway, which predates the 24-plane + bbox architecture.

---

## Contents

```
detector_optimization/
├── SWGOLO7.ipynb                    # First end-to-end notebook for SWGO project
├── SWGOLO7_optimization.ipynb       # End-to-end notebook for 2D TAMBO project
├── SWGOLO7_opt_only.ipynb           # Notebook conatining only the optimization part of the pipeline
├── SWGOLO7_plots.ipynb              # Results visualization
│
├── auto_run_notebook.py             # Papermill-based notebook executor with timestamped outputs
├── auto_run_notebook_batch.sh       # SLURM batch script (A100 80GB, arguelles_delgado partition)
├── common_gpu_auto_run_notebook_batch.sh  # SLURM batch (gpu / gpu_h200 partitions)
│
├── diffusion_model/                 # First-generation diffusion shower generator
│   ├── tambo_diffusion_generator.py #   TamboDiffusionGenerator class (DDIM sampler wrapper)
│   ├── tambo_3D_diffusion_generator.py #   PlaneDiffusionEvaluator class with 3D output
│   ├── tambo_3D_fnn_scaler.py       #   FNN bbox scaler for 3D generation
│   ├── example_usage.py
│   ├── 01_validate.ipynb            #   Validation of TamboDiffusionGenerator
│   ├── 02_reduce_runtime.ipynb      #   Runtime profiling / reduction of TamboDiffusionGenerator
│   ├── 03_3D_generator_update.ipynb # Validation of PlaneDiffusionEvaluator
│   ├── 04_3D_generator_scaled.ipynb # Scaled outputs of the PlaneDiffusionEvaluator
│   └── README.md                    #   Detailed docs for the generator class
│
├── simple_simulator_class/          # Experimental standalone simulator
│   ├── simulator.py                 #   Simulator class (mountain plane + circular detectors)
│   ├── simulator_test_swgo.ipynb    # initial test of emualting detector positions in the swgo data
│   ├── simulator_test_tambo.ipynb.  # initial test of emualting detector positions in the tambo data
│   ├── requirements.txt
│   └── test.mp4                     #   Animation of a simulated shower
│
├── data_exploration/                # EDA on raw shower data
│   ├── tambo_data_exploration.ipynb
│   └── first_10k_rows.csv
│
├── flow_chart/
│   └── pipeline_flowchart.html      # Visual pipeline diagram
│
└── outputs_notebooks/               # Timestamped papermill outputs from prior runs
                                     # (filenames encode run notes: kernel_died, timeout,
                                     #  first_successful_optimization, etc.)
```

---

## Running a Notebook on the Cluster

The batch scripts call `auto_run_notebook.py`, which executes a notebook with papermill and writes a timestamped copy to `outputs_notebooks/`.

```bash
# Edit the target notebook in the .sh file, then submit:
sbatch common_gpu_auto_run_notebook_batch.sh
```

Both batch scripts assume a `multiproc_env` conda environment with PyTorch + papermill available.

---

## Subcomponent Notes

- **`diffusion_model/`** — Class-based wrapper around the original DDIM diffusion checkpoint. See its own README for constructor parameters, output bundle layout (`condition_*.npz`, `summary.npz`), and chunk-size guidance for GPU memory.
- **`simple_simulator_class/`** — Models detectors as circles on a 2D mountain plane and assigns shower energy via geometric overlap. Used to prototype the optimization target before the differentiable response model in v2 took over.
- **`outputs_notebooks/`** — Historical run log; filenames carry annotations (`first_successful_optimization`, `kernel_died`, `larger_array`, etc.) describing what happened in each run.

---

## Status

This directory is **archival**. New work should go into `detector_optimization_v2/` (or later `_v3`–`_v6` iterations), which split the pipeline into reusable modules.
