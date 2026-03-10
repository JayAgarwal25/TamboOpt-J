#!/usr/bin/env python3
"""
FNN-based bounding box generator for Tambo 3D detector optimization.
Similar interface to PlaneDiffusionEvaluator but uses FNN for direct bbox prediction.
"""
import sys
import time
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt


class PlaneFNNGenerator:
    """
    A class for generating bounding box predictions using FNN models.

    Unlike the diffusion model which generates images, this predicts
    bounding boxes directly from conditioning features.

    Example usage:
        generator = PlaneFNNGenerator(
            data_dir="/path/to/data",
            checkpoint_path="/path/to/ckpt.ckpt",
            device="cuda:0"
        )
        generator.load_model()
        generator.test_conditions = torch.tensor([[energy, class_id, sin_z, cos_z, sin_a, cos_a]])
        outputs = generator.generate_samples(num_samples=100)
    """

    def __init__(
        self,
        data_dir: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        output_dir: Optional[str] = None,
        seed: int = 42,
        imports_path: Optional[str] = None
    ):
        """
        Initialize the PlaneFNNGenerator.

        Args:
            data_dir: Directory containing data (used to load bbox standardization stats)
            checkpoint_path: Path to the trained FNN model checkpoint
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None
            output_dir: Directory for saving outputs (optional)
            seed: Random seed for reproducibility
            imports_path: Path to add to sys.path for imports
        """
        # Add path if provided
        if imports_path:
            sys.path.append(imports_path)

        # Import dependencies (after path is set)
        from lightning_training_fnn import PlaneDataset, PlaneFNNModule

        self.PlaneDataset = PlaneDataset
        self.PlaneFNNModule = PlaneFNNModule

        # Store parameters
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.output_dir = output_dir
        self.seed = seed

        # Initialize containers
        self.pl_module = None
        self.model = None
        self.bbox_mean = None
        self.bbox_std = None

        # Storage for conditions and generated outputs
        self.test_conditions = []
        self.generated_sets = []

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Initialized PlaneFNNGenerator")
        print(f"Device: {self.device}")
        if output_dir:
            print(f"Output directory: {output_dir}")


    def _load_standardization_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load bbox standardization statistics from data directory."""
        stats_path = os.path.join(self.data_dir, "global_bbox_stats.pt")

        if not os.path.exists(stats_path):
            # Try alternative path
            stats_path = os.path.join(self.data_dir, "..", "global_bbox_stats.pt")

        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Standardization stats not found at {self.data_dir}/global_bbox_stats.pt"
            )

        stats = torch.load(stats_path, map_location="cpu")

        # Handle different stat formats
        if "mean" in stats and "std" in stats:
            mean = stats["mean"]
            std = stats["std"]
        elif "bbox_mean" in stats and "bbox_std" in stats:
            mean = stats["bbox_mean"]
            std = stats["bbox_std"]
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
                std = torch.tensor(std)
        else:
            raise ValueError(f"Unrecognized stats format in {stats_path}")

        print(f"Loaded standardization stats from: {stats_path}")
        print(f"  Mean: {mean}")
        print(f"  Std: {std}")

        return mean, std

    def load_model(self):
        """Load the trained FNN model checkpoint."""
        start_time = time.time()

        # Load standardization stats
        self.bbox_mean, self.bbox_std = self._load_standardization_stats()
        self.bbox_mean = self.bbox_mean.to(self.device)
        self.bbox_std = self.bbox_std.to(self.device)

        # Load checkpoint
        self.pl_module = self.PlaneFNNModule.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device
        )
        self.pl_module.eval()
        self.pl_module.to(self.device)

        print(f"Loaded checkpoint: {self.checkpoint_path}")

        # Extract model
        self.model = self.pl_module.model

        # Apply EMA weights if available
        if hasattr(self.pl_module, 'model_ema') and self.pl_module.model_ema is not None:
            print("Using EMA weights for inference")
            for name, param in self.model.named_parameters():
                if name in self.pl_module.model_ema:
                    param.data = self.pl_module.model_ema[name].to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        print(f"Model loading time: {time.time() - start_time:.2f}s")

    def _denormalize_bbox(self, bbox_std: torch.Tensor) -> torch.Tensor:
        """
        Convert from standardized (zero mean, unit variance) back to original scale.

        Args:
            bbox_std: (B, 24, 4) or (B, 96) - standardized bounding boxes

        Returns:
            denormalized: same shape as input, in original scale
        """
        # Handle scalar stats (global normalization)
        if self.bbox_mean.ndim == 0 or (isinstance(self.bbox_mean, torch.Tensor) and self.bbox_mean.numel() == 1):
            return bbox_std * self.bbox_std + self.bbox_mean

        # Handle per-plane, per-coordinate stats
        if bbox_std.dim() == 2:
            # (B, 96) -> reshape to (B, 24, 4) for denorm
            bbox_reshaped = bbox_std.view(-1, 24, 4)
            return (bbox_reshaped * self.bbox_std[None, :, :] + self.bbox_mean[None, :, :]).view(-1, 96)
        elif bbox_std.dim() == 3:
            # (B, 24, 4)
            return bbox_std * self.bbox_std[None, :, :] + self.bbox_mean[None, :, :]
        else:
            raise ValueError(f"Unexpected dimension: {bbox_std.dim()}")

    @torch.no_grad()
    def generate_samples(
        self,
        num_conditions: Optional[int] = None,
    ) -> Dict:
        """
        Generate bbox predictions for each test condition.

        Args:
            num_samples: Number of samples to generate per condition
                        (Note: FNN is deterministic, so multiple samples will be identical
                         unless noise/dropout is added)
            num_conditions: Number of conditions to use (None = use all)
            chunk_size: Batch size for generation (to avoid OOM)

        Returns:
            List of dicts with 'condition' and 'bboxes' keys
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if len(self.test_conditions) == 0:
            raise RuntimeError("No test conditions set. Set test_conditions attribute first.")

        # Determine how many conditions to process
        conditions_to_process = self.test_conditions
        if num_conditions is not None:
            conditions_to_process = self.test_conditions[:num_conditions]

        self.generated_sets = []
        start_time = time.time()
        
        conditions_to_process = conditions_to_process.to(self.device)
        
        bs = conditions_to_process.shape[0]

        
        all_bboxes = []
        samples_done = 0

        # Prepare condition batch - expand to batch size
        p_energy = conditions_to_process[:, 0]
        class_id = conditions_to_process[:, 1].long()
        sin_zenith = conditions_to_process[:, 2]
        cos_zenith = conditions_to_process[:, 3]
        sin_azimuth = conditions_to_process[:, 4]
        cos_azimuth = conditions_to_process[:, 5]
        
        # Forward pass - direct prediction
        pred_bbox_flat_std = self.model(
            p_energy,
            class_id,
            sin_zenith,
            cos_zenith,
            sin_azimuth,
            cos_azimuth
        )  # (B, 96) - standardized

        # Reshape to (B, 24, 4)
        pred_bbox_std = pred_bbox_flat_std.reshape(bs, 24, 4)

        # Denormalize
        pred_bbox = self._denormalize_bbox(pred_bbox_std)
        
        self.generated_sets = {
            "condition": conditions_to_process.cpu(),
            "bboxes": pred_bbox.cpu(),
        }

        total_predictions = self.generated_sets['bboxes'].shape[0]
        print(f"Done: generated {total_predictions} bbox predictions across {num_conditions} conditions.")
        print(f"Total generation time: {time.time() - start_time:.2f}s")
        
        return self.generated_sets


if __name__ == "__main__":
    # Example usage
    print("PlaneFNNGenerator - Example Usage")
    print("="*60)
    print("""
    from tambo_3D_fnn_generator import PlaneFNNGenerator
    import torch

    # Initialize
    generator = PlaneFNNGenerator(
        data_dir="/path/to/data",
        checkpoint_path="/path/to/checkpoint.ckpt",
        output_dir="fnn_outputs",
        imports_path="/path/to/FNN/module"
    )

    # Load model
    generator.load_model()

    # Set conditions (energy, class_id, sin_z, cos_z, sin_a, cos_a)
    generator.test_conditions = torch.tensor([
        [0.2398, 1.0, 0.6737, -0.7390, -0.3746, -0.9272]
    ])

    # Generate bbox predictions
    outputs = generator.generate_samples(num_samples=1, num_conditions=1)

    # outputs[0]['bboxes'] has shape (1, 24, 4)
    # Each plane has: [xmin, xmax, ymin, ymax]

    # Visualize
    generator.plot_bbox_predictions()
    generator.plot_bbox_ranges()
    """)
