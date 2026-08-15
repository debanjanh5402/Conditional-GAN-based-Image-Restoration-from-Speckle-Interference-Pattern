from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from torchvision import io


# Device Selection (MPS for Mac Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device : Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device : NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Device : CPU")

# Directories
TRAIN_DIR = Path("./dataset/train")
RESULTS_DIR = Path("./_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("./_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sample_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
print(f"Training Samples : {len(sample_dirs)}")

if not sample_dirs:
    raise ValueError(f"No training sample directories found in {TRAIN_DIR.resolve()}")

# Gamma configuration on device
GAMMAS = torch.linspace(0.01, 1.00, 100, device=device)

# SSIM Metric placed on device
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def __read_img(path: Path) -> torch.Tensor:
    return io.read_image(str(path), io.ImageReadMode.GRAY)


def transform_speckle(x: torch.Tensor) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()

    if torch.isclose(x_max, x_min):
        return torch.zeros_like(x)

    return (x - x_min) / (x_max - x_min)


def transform_clean(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


# Storage for sample scores per gamma: shape -> [100 gammas, N samples]
gamma_sample_scores = [[] for _ in range(len(GAMMAS))]

# Fast Loop: Load files ONCE per sample, evaluate across all GAMMAS on GPU/MPS
for sample_dir in tqdm(sample_dirs, desc="Processing Samples", unit="sample"):
    clean_path = next(sample_dir.glob("*_clean.png"))
    speckle_path = next(sample_dir.glob("*_speckle.pt"))

    # Load clean & speckles ONCE and move to device as float32
    clean = transform_clean(__read_img(clean_path))
    if clean.ndim == 2:
        clean = clean.unsqueeze(0)
    clean_batch = clean.unsqueeze(0).to(device)  # Shape: (1, 1, H, W)

    speckles = torch.load(speckle_path)

    # Pre-normalize speckles into [0, 1] ONCE per sample and move to MPS
    norm_speckles = [transform_speckle(s).float().to(device) for s in speckles]

    # Evaluate all gammas on MPS
    for g_idx, gamma in enumerate(GAMMAS):
        ssim_metric.reset()
        for norm_speckle in norm_speckles:
            speckle_transformed = (norm_speckle ** gamma).view_as(clean_batch)
            ssim_metric.update(speckle_transformed, clean_batch)

        gamma_sample_scores[g_idx].append(ssim_metric.compute().item())


# Compute dataset mean SSIM per gamma
dataset_ssims = [sum(scores) / len(scores) for scores in gamma_sample_scores]

# Save Results
GAMMAS_cpu = GAMMAS.cpu()
result = {float(gamma): float(ssim) for gamma, ssim in zip(GAMMAS_cpu, dataset_ssims)}
result_file = RESULTS_DIR / "gamma_vs_ssim.txt"

with open(result_file, "w") as f:
    f.write("Gamma\tMean Dataset SSIM\n")
    f.write("-" * 35 + "\n")
    for gamma, ssim in result.items():
        f.write(f"{gamma:.4f}\t{ssim:.6f}\n")

# Best Gamma
dataset_ssims_tensor = torch.tensor(dataset_ssims)
best_index = torch.argmax(dataset_ssims_tensor).item()
best_gamma = GAMMAS_cpu[best_index].item()
best_ssim = dataset_ssims_tensor[best_index].item()

print("\n==========================================")
print(f"Optimal Gamma : {best_gamma:.4f}")
print(f"Maximum SSIM  : {best_ssim:.6f}")
print("==========================================\n")

with open(result_file, "a") as f:
    f.write("\n")
    f.write(f"Optimal Gamma : {best_gamma:.4f}\n")
    f.write(f"Maximum SSIM  : {best_ssim:.6f}\n")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(GAMMAS_cpu.numpy(), dataset_ssims_tensor.numpy(), lw=2)
plt.scatter(best_gamma, best_ssim, color="red", label=f"γ={best_gamma:.4f}")
plt.xlabel("Gamma")
plt.ylabel("Mean Dataset SSIM")
plt.title("Training Dataset SSIM vs Gamma")
plt.grid(True)
plt.legend()

plt.tight_layout()
plot_file = PLOTS_DIR / "gamma_vs_ssim.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nResults saved to : {result_file}")
print(f"Plot saved to    : {plot_file}")