#!/usr/bin/env python3
import sys
import time
import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .diffusion_loc_zlt import DDIMSamplerPlanes


class PlaneDiffusionEvaluator:
    """
    A class for evaluating plane diffusion models with DDIM sampling.

    Example usage:
        evaluator = PlaneDiffusionEvaluator(
            data_dir="/path/to/data",
            checkpoint_path="/path/to/ckpt.ckpt",
            device="cuda:0"
        )
        evaluator.run_full_pipeline(num_samples=10, ddim_steps=50)
    """

    def __init__(
        self,
        data_dir: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        batch_size: int = 2,
        num_workers: int = 2,
        ddim_steps: int = 50,
        eta: float = 0.0,
        guidance_w: float = 0.0,
        T: int = 1000,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        seed: int = 24,
        imports_path: Optional[str] = None
    ):
        """
        Initialize the PlaneDiffusionEvaluator.

        Args:
            data_dir: Directory containing test data
            checkpoint_path: Path to the trained model checkpoint
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None
            batch_size: Batch size for data loading
            num_workers: Number of data loading workers
            ddim_steps: Number of DDIM sampling steps
            eta: DDIM eta parameter (0.0 = deterministic)
            guidance_w: Guidance weight for classifier-free guidance
            T: Total diffusion timesteps
            beta_1: Starting beta value
            beta_T: Ending beta value
            seed: Random seed for reproducibility
            imports_path: Path to add to sys.path for imports
        """
        # Add path if provided
        if imports_path:
            sys.path.append(imports_path)

        # Import dependencies (after path is set)
        from lightning_training import PlaneDataset, PlaneDiffusionModule

        self.PlaneDataset = PlaneDataset
        self.PlaneDiffusionModule = PlaneDiffusionModule
        self.DDIMSamplerPlanes = DDIMSamplerPlanes

        # Store parameters
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.guidance_w = guidance_w
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.seed = seed

        # Initialize containers
        self.pl_module = None
        self.net = None
        self.sampler = None
        self.test_dataset = None
        self.test_loader = None

        # Set random seeds
        self._seed_all(self.seed)
        
        self.test_images = []
        self.test_conditions = []
        self.generated_sets = []

        print(f"Initialized PlaneDiffusionEvaluator")
        print(f"Device: {self.device}")

    def _seed_all(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self):
        """Load the trained model checkpoint and create DDIM sampler."""
        start_time = time.time()

        # Load checkpoint
        self.pl_module = self.PlaneDiffusionModule.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device
        )
        self.pl_module.eval()
        self.pl_module.to(self.device)

        print(f"Loaded checkpoint: {self.checkpoint_path}")

        # Extract model
        self.net = self.pl_module.model

        # Build DDIM Sampler
        self.sampler = self.DDIMSamplerPlanes(
            self.net,
            beta_1=self.beta_1,
            beta_T=self.beta_T,
            T=self.T,
            eta=self.eta,
            ddim_steps=self.ddim_steps,
            w=self.guidance_w,
        ).to(self.device)
        self.sampler.eval()

        print(f"DDIM sampler built (ddim_steps={self.ddim_steps}, eta={self.eta}, w={self.guidance_w})")
        print(f"Model loading time: {time.time() - start_time:.2f}s")


    def generate_samples(
        self, 
        num_conditions: Optional[int] = None,
        batch_size : int = 100,
        logs=False
    ):
        """
        Generate samples for each test condition using autoregressive plane generation.

        Args:
            num_samples: Number of samples to generate per condition
            num_conditions: Number of conditions to use (None = use all extracted)
        """
        if self.sampler is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if len(self.test_conditions) == 0:
            raise RuntimeError("No test samples extracted. Call extract_test_samples() first.")

        # Determine how many conditions to process
        conditions_to_process = self.test_conditions
        if num_conditions is not None:
            conditions_to_process = self.test_conditions[:num_conditions]
        
        num_total = conditions_to_process.shape[0]
        
        all_chunks = []
        start_time = time.time()
        
        conditions_to_process = conditions_to_process.to(self.device)
        # Get the ground truth planes shape from test_images
        P, C, H, W = (24, 3, 32, 32)
        
        # with torch.no_grad():
        for chunk_start in range(0, num_total, batch_size):
            print(f"Processing conditions {chunk_start} to {min(chunk_start + batch_size, num_total)} / {num_total}...")
            chunk = conditions_to_process[chunk_start:chunk_start + batch_size]
            bs = chunk.shape[0]
            
            p_energy = chunk[:, 0]
            class_id = chunk[:, 1].long()
            sin_zenith = chunk[:, 2]
            cos_zenith = chunk[:, 3]
            sin_azimuth = chunk[:, 4]
            cos_azimuth = chunk[:, 5]
            
            # Generate all 24 planes autoregressively
            pred_all = torch.zeros((bs, P, C, H, W), device=self.device)
            past = torch.zeros((bs, C, H, W), device=self.device)
            
            for plane_idx in range(P):
                plane_idx_tensor = torch.full((bs,), plane_idx, device=self.device, dtype=torch.long)
                noise = torch.randn((bs, C, H, W), device=self.device)

                pred = self.sampler(
                    noise,
                    p_energy,
                    class_id,
                    sin_zenith,
                    cos_zenith,
                    sin_azimuth,
                    cos_azimuth,
                    plane_idx_tensor,
                    past,
                )
                pred_all[:, plane_idx] = pred
                past = pred

            all_chunks.append(pred_all.cpu())

            del noise, pred, pred_all, past
            torch.cuda.empty_cache()
        
        all_samples = torch.cat(all_chunks, dim=0)

                        
        # Concatenate all chunks -> (num_samples, 24, 3, H, W)
        self.generated_sets = {
            "conditions": conditions_to_process.cpu(),
            "images": all_samples,
        }

        # Extra safety between conditions
        torch.cuda.empty_cache()

        total_images = self.generated_sets["images"].shape[0]
        total_conditions = self.generated_sets["conditions"].shape[0]
        if logs:
            print(f"✔ Done: generated {total_images} showers across {total_conditions} conditions.")
            print(f"Total generation time: {time.time() - start_time:.2f}s")
        return self.generated_sets


    def plot_xy_profiles(self, num_conditions: Optional[int] = None, save_dir: Optional[str] = None):
        """
        Plot average X and Y profiles across all generated samples.

        Args:
            num_conditions: Number of conditions to plot (None = all)
            save_dir: Directory to save plots (None = display only)
        """
        if len(self.generated_sets) == 0:
            print("No generated sets to plot.")
            return

        condition_names = ['energy', 'class_id', 'sin_zenith', 'cos_zenith', 'sin_azimuth', 'cos_azimuth']
        num_to_plot = num_conditions if num_conditions else len(self.generated_sets)
        num_to_plot = min(num_to_plot, len(self.generated_sets))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for idx in range(num_to_plot):
            output = self.generated_sets[idx]
            
            # Print conditions
            print(f"\nCondition {idx + 1}:")
            for icondition, condition_val in enumerate(output['condition']):
                print(f"  {condition_names[icondition]}: {condition_val.item():.4f}")

            images = output['images']  # Shape: (N, 24, 3, H, W)
            images_np = images.cpu().numpy()

            # Average over samples (N) and channels (3) -> (24, H, W)
            avg_profiles = images_np.mean(axis=(0, 2))  # Shape: (24, H, W)

            # X profile: average over Y (axis=1) -> (24, W)
            avg_x_profile = avg_profiles.mean(axis=1)
            
            # Y profile: average over X (axis=2) -> (24, H)
            avg_y_profile = avg_profiles.mean(axis=2)

            # Plot
            fig, axes = plt.subplots(4, 6, figsize=(24, 16))
            fig.suptitle(f"Condition {idx + 1}: X and Y Profiles", fontsize=14)

            for plane_idx in range(24):
                ax = axes[plane_idx // 6, plane_idx % 6]
                
                ax.plot(avg_x_profile[plane_idx], label='X Profile', linewidth=1.5)
                ax.plot(avg_y_profile[plane_idx], label='Y Profile', linewidth=1.5, linestyle='--')
                
                ax.set_title(f"Plane {plane_idx}", fontsize=9)
                ax.set_xlabel('Position', fontsize=8)
                ax.set_ylabel('Average Intensity', fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])

            if save_dir:
                save_path = os.path.join(save_dir, f"xy_profiles_condition_{idx + 1}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {save_path}")
                plt.close()
            else:
                plt.show()