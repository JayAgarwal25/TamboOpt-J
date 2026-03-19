import sys
# sys.path.append('/n/home04/hhanif/Diffusion-Surrogate-Detector-Design')
sys.path.append('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_optimization')


import torch
import numpy as np
import matplotlib.pyplot as plt

from flow_train_loc import TamboDataModule, df_cleaned
from flow_train_loc import TamboFlowLitModel
from models.FlowCondition import OTCondFlowMatcher
from models.odesolve import (
    FixedStepConfig, AdaptiveStepConfig, ODESolver
)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Load model + data
# ------------------------------------------------------------
ckpt_path = "/n/home04/hhanif/tambo_ckpts/flow/ckpt_epoch=1999.ckpt"
print(f"Loading checkpoint from: {ckpt_path}")

model = TamboFlowLitModel.load_from_checkpoint(ckpt_path, strict=False)
model.eval()
model.to(device)

# Data module & test loader
data_module = TamboDataModule(
    df=df_cleaned,
    batch_size=64,
    train_ratio=0.85,
    val_ratio=0.10,
    test_ratio=0.05,
    num_workers=4,
    seed=42,
)

data_module.setup("test")
test_loader = data_module.test_dataloader()


# ------------------------------------------------------------
# FlowMatching SAMPLER
# ------------------------------------------------------------
@torch.no_grad()
def sample_flowmatching(model, cond_vector, solver_config,
                        num_sample=1000, batch_size=25):
    """
    model: TamboFlowLitModel
    cond_vector: tensor/ndarray of shape (5,) -> (energy, sin_z, cos_z, sin_a, cos_a)
    solver_config: ODE solver config (FixedStepConfig or AdaptiveStepConfig)
    """
    model.eval()
    model.to(device)

    flow_matcher = OTCondFlowMatcher()
    preds = []

    # Ensure tensor on device
    if isinstance(cond_vector, np.ndarray):
        cond_vector = torch.tensor(cond_vector, dtype=torch.float32)
    cond_vector = cond_vector.to(device)

    # Unpack condition scalars
    energy, sin_z, cos_z, sin_a, cos_a = cond_vector

    remaining = num_sample
    img_shape = (3, 32, 32)  # based on rgb_32 dataset

    while remaining > 0:
        b = min(batch_size, remaining)
        remaining -= b

        # Expand condition scalars into batch
        E  = energy.repeat(b)
        SZ = sin_z.repeat(b)
        CZ = cos_z.repeat(b)
        SA = sin_a.repeat(b)
        CA = cos_a.repeat(b)

        # Starting noise x0
        noise = torch.randn((b, *img_shape), device=device)

        # FlowMatching sampler call
        x_hat = flow_matcher.sample(
            x_0=noise,
            network=model.model,   # UNet inside Lightning module
            solver_config=solver_config,
            energy=E,
            sin_zenith=SZ,
            cos_zenith=CZ,
            sin_azimuth=SA,
            cos_azimuth=CA,
        )

        preds.append(x_hat.detach().cpu().numpy())

    return np.concatenate(preds, axis=0)   # (num_sample, 3, 32, 32)


# ------------------------------------------------------------
# Get first batch and select first N conditions
# ------------------------------------------------------------
batch = next(iter(test_loader))
test_imgs, test_cond = batch   # test_cond is a tuple/list of 5 tensors

# How many conditions to use
desired_num_conds = 5
num_conds = min(desired_num_conds, test_imgs.shape[0])

# First N real images
firstN_imgs = test_imgs[:num_conds]   # (N, 3, 32, 32)

# Extract first N from each condition tensor
# test_cond is (energy_batch, sin_z_batch, cos_z_batch, sin_a_batch, cos_a_batch)
firstN_cond = [
    test_cond[0][:num_conds],   # energy
    test_cond[1][:num_conds],   # sin_z
    test_cond[2][:num_conds],   # cos_z
    test_cond[3][:num_conds],   # sin_a
    test_cond[4][:num_conds],   # cos_a
]


# ------------------------------------------------------------
# For each condition, sample 1000 images
# ------------------------------------------------------------
generated_sets = []

solver_config = FixedStepConfig(
    dt=1 / 100,
    solver=ODESolver.EULER,
)

print(f"Generating samples for {num_conds} conditions...\n")

for i in range(num_conds):
    # Build condition vector (5,) for condition i
    cond_vec = torch.stack([
        firstN_cond[0][i],  # energy
        firstN_cond[1][i],  # sin_z
        firstN_cond[2][i],  # cos_z
        firstN_cond[3][i],  # sin_a
        firstN_cond[4][i],  # cos_a
    ])

    print(f"  → Sampling 1000 images for condition {i+1}/{num_conds} ...")

    gen_imgs = (
        model=model,
        cond_vector=cond_vec,
        solver_config=solver_config,
        num_sample=500,
        batch_size=100,
    )  # (1000, 3, 32, 32)

    generated_sets.append({
        "images": torch.tensor(gen_imgs),  # convert back to tensor for convenience
        "condition": cond_vec,
    })

print("\nSampling complete.\n")


# ------------------------------------------------------------
# PLOT MARGINAL COMPARISONS FOR EACH CONDITION
# ------------------------------------------------------------
for idx in range(num_conds):
    # ----------------------------------------------------
    # 1. Ground truth image for this condition
    # ----------------------------------------------------
    gt_img = firstN_imgs[idx].cpu().numpy()   # (3, H, W)

    gt_R = gt_img[0]
    gt_G = gt_img[1]
    gt_B = gt_img[2]

    gt_R_x, gt_R_y = gt_R.mean(axis=0), gt_R.mean(axis=1)
    gt_G_x, gt_G_y = gt_G.mean(axis=0), gt_G.mean(axis=1)
    gt_B_x, gt_B_y = gt_B.mean(axis=0), gt_B.mean(axis=1)

    # ----------------------------------------------------
    # 2. Generated images for this condition
    # ----------------------------------------------------
    gen_imgs = generated_sets[idx]["images"].cpu().numpy()  # (1000, 3, H, W)

    R_batch = gen_imgs[:, 0]   # (1000, H, W)
    G_batch = gen_imgs[:, 1]
    B_batch = gen_imgs[:, 2]

    # Average over samples and y/x dims to match GT marginals
    R_x_mean = R_batch.mean(axis=1).mean(axis=0)  # (W,)
    R_y_mean = R_batch.mean(axis=2).mean(axis=0)  # (H,)

    G_x_mean = G_batch.mean(axis=1).mean(axis=0)
    G_y_mean = G_batch.mean(axis=2).mean(axis=0)

    B_x_mean = B_batch.mean(axis=1).mean(axis=0)
    B_y_mean = B_batch.mean(axis=2).mean(axis=0)

    # ----------------------------------------------------
    # 3. Condition values (for title)
    # ----------------------------------------------------
    cond_vec = generated_sets[idx]["condition"].cpu().numpy()  # (5,)
    energy, sin_z, cos_z, sin_a, cos_a = cond_vec.tolist()

    # ----------------------------------------------------
    # 4. Plot: 1 row, 6 columns
    # ----------------------------------------------------
    fig, axs = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle(
        f"Condition {idx+1}: "
        f"E={energy:.3f}, sin_z={sin_z:.3f}, cos_z={cos_z:.3f}, "
        f"sin_a={sin_a:.3f}, cos_a={cos_a:.3f}",
        y=1.05,
    )

    # --- R-X ---
    axs[0].plot(gt_R_x, label="GT")
    axs[0].plot(R_x_mean, label="Gen Mean", linestyle="--")
    axs[0].set_title("Particle Density vs x")

    # --- R-Y ---
    axs[1].plot(gt_R_y, label="GT")
    axs[1].plot(R_y_mean, label="Gen Mean", linestyle="--")
    axs[1].set_title("Particle Density vs y")

    # --- G-X ---
    axs[2].plot(gt_G_x, label="GT")
    axs[2].plot(G_x_mean, label="Gen Mean", linestyle="--")
    axs[2].set_title("Average arrival time vs x")

    # --- G-Y ---
    axs[3].plot(gt_G_y, label="GT")
    axs[3].plot(G_y_mean, label="Gen Mean", linestyle="--")
    axs[3].set_title("Average arrival time vs y")

    # --- B-X ---
    axs[4].plot(gt_B_x, label="GT")
    axs[4].plot(B_x_mean, label="Gen Mean", linestyle="--")
    axs[4].set_title("Average Kinetic Energy vs x")

    # --- B-Y ---
    axs[5].plot(gt_B_y, label="GT")
    axs[5].plot(B_y_mean, label="Gen Mean", linestyle="--")
    axs[5].set_title("Average Kinetic Energy vs y")

    axs[0].legend(loc="upper right")

    plt.tight_layout()
    out_name = f"condition_{idx+1}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_name}")

print(f"\nDone. Saved {num_conds} plots: condition_1.png ... condition_{num_conds}.png")
