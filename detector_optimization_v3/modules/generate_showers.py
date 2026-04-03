import sys
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TAMBO-opt")
from allshowers.generate_showers import (
    sample_primary_particles, 
    run_point_count_fm, 
    run_allshowers, 
    save_output
)

class GenerateShowers:
    """
    GenerateShowers class with the specified parameters.
    Args:
        output_dir (str): Directory to save the generated shower samples.
        num_timesteps (int, optional): Number of timesteps for the shower simulation. Defaults to 16.
        device (str, optional): Device to run the simulations on. Defaults to "cuda:0".
        solver (str, optional): Numerical solver to use for the shower simulation. Defaults to "midpoint".
        batch_size (int, optional): Batch size for running the simulations. Defaults to 30.
        point_count_model (str, optional): Path to the model used for predicting the number of points in the shower.
            Defaults to ".../num_of_point_clouds_dequantize_compiled.pt".
        allshowers_run_dir (str, optional): Directory where the AllShowers simulations are run and stored.
            Defaults to ".../checkpoints/all_showers".
        e_min (float, optional): Minimum energy of the primary particles. Defaults to 1e5.
        e_max (float, optional): Maximum energy of the primary particles. Defaults to 1e8.
        zenith_min (float, optional): Minimum zenith angle of the primary particles. Defaults to 60.0.
        zenith_max (float, optional): Maximum zenith angle of the primary particles. Defaults to 100.0.
        azimuth_min (float, optional): Minimum azimuth angle of the primary particles. Defaults to 0.0.
        azimuth_max (float, optional): Maximum azimuth angle of the primary particles. Defaults to 360.0.
    """
    def __init__(
            self,
            output_dir,
            num_timesteps      = 16,
            device             = "cuda:0",
            solver             = "midpoint",
            batch_size         = 30,
            point_count_model  = "/n/home05/zdimitrov/tambo/TAMBO-opt/allshowers/checkpoints/num_of_point_clouds_dequantize_compiled.pt",
            allshowers_run_dir = "/n/home05/zdimitrov/tambo/TAMBO-opt/allshowers/checkpoints/all_showers",
            e_min              = 1e5,
            e_max              = 1e8,
            zenith_min         = 60.0,
            zenith_max         = 100.0,
            azimuth_min        = 0.0,
            azimuth_max        = 360.0,
    ):
        self.num_timesteps      = num_timesteps
        self.device             = device
        self.solver             = solver
        self.batch_size         = batch_size
        self.point_count_model  = point_count_model
        self.allshowers_run_dir = allshowers_run_dir
        self.output_dir         = output_dir
        self.e_min              = e_min
        self.e_max              = e_max
        self.zenith_min         = zenith_min
        self.zenith_max         = zenith_max
        self.azimuth_min        = azimuth_min
        self.azimuth_max        = azimuth_max

        
    def __call__(self, num_samples=2000, save=True):
        """
        Generates shower samples using the AllShowers framework.
        Args:
            num_samples (int, optional): Number of shower samples to generate. Defaults to 2000.
            save (bool, optional): Whether to save the generated samples to disk. Defaults to True.
        Returns:
            samples (torch.Tensor): Generated shower samples of shape (N, max_points, 5).
            energies (torch.Tensor): Generated primary particle energies.
            directions (torch.Tensor): Generated primary particle directions.
            labels (torch.Tensor): Generated primary particle labels.
        """
        primary = sample_primary_particles(
            n=num_samples, e_min=self.e_min, e_max=self.e_max,
            zenith_min=self.zenith_min, zenith_max=self.zenith_max, 
            azimuth_min=self.azimuth_min, azimuth_max=self.azimuth_max,)

        num_points = run_point_count_fm(
            model_path=self.point_count_model,
            energies=primary["energies"],
            directions=primary["directions"],
            labels=primary["labels"],
        )

        samples = run_allshowers(
            run_dir=self.allshowers_run_dir,
            energies=primary["energies"],
            directions=primary["directions"],
            labels=primary["labels"],
            num_points=num_points,
            num_timesteps=self.num_timesteps,
            batch_size=self.batch_size,
            solver=self.solver,
            device=str(self.device),
        )

        # samples: (N, max_points, 5) float32
        # columns: x, y, layer_index, energy, time
        print("samples shape:", samples.shape)
        if save:
            save_output(
                path=f'{self.output_dir}/cashed_showers_{num_samples}.pt',
                samples=samples,
                energies=primary["energies"],
                directions=primary["directions"],
                labels=primary["labels"],
            )
        return samples, primary["energies"], primary["directions"], primary["labels"]
