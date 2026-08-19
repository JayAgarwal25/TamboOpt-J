import sys, os
sys.path.insert(0, os.getcwd())
import torch
import modules_v6  # noqa: F401
from modules_v6.constants import LOG_E_MIN, LOG_E_MAX

path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/07_750k_primaires_meanvar/test_v6_run_01_northeast/primary.pt"
p = torch.load(path, map_location="cpu")
print("shape:", tuple(p.shape), "dtype:", p.dtype)

log_e_norm = p[:, 3]
print(f"log_e_norm range: [{log_e_norm.min().item():.6f}, {log_e_norm.max().item():.6f}]")
print(f"LOG_E_MIN={LOG_E_MIN}  LOG_E_MAX={LOG_E_MAX}")

log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
E_gev = torch.pow(10.0, log_e)

print(f"log10(E/GeV) range: [{log_e.min().item():.4f}, {log_e.max().item():.4f}]")
print(f"E range: [{E_gev.min().item():.4e}, {E_gev.max().item():.4e}] GeV")
print(f"E range: [{(E_gev.min()*1e9).item():.4e}, {(E_gev.max()*1e9).item():.4e}] eV")
print(f"log10(E/eV) range: [{(log_e.min()+9).item():.4f}, {(log_e.max()+9).item():.4f}]")

# percentile sanity
import numpy as np
e = E_gev.numpy()
for q in (0, 1, 5, 50, 95, 99, 100):
    print(f"  p{q:3d}: {np.percentile(e, q):.4e} GeV")
