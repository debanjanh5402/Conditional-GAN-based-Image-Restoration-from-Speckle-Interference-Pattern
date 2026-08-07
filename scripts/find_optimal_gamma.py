from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm



# Directories
TRAIN_DIR = Path("./dataset/train")
RESULTS_DIR = Path("./results")
PLOTS_DIR = Path("./plots")

sample_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
print(f"Training Samples : {len(sample_dirs)}")

# Gamma configuration
GAMMAS = torch.linspace(0.0, 1.0, 101)



# SSIM Metric
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)



# Normalization
def transform_speckle(x: torch.Tensor):
    x_min = x.min()
    x_max = x.max()

    if torch.isclose(x_max, x_min):
        return torch.zeros_like(x)

    return (x - x_min) / (x_max - x_min)

def transform_clean(x: torch.Tensor):
    return x / 255.0



# Gamma Search
dataset_ssims = []

for gamma in tqdm(GAMMAS, desc="Searching Gamma", leave=False):
    sample_scores = []
    for sample_dir in tqdm(sample_dirs, desc=f"gamma = {gamma:.4f}", leave=False):
        clean_path = next(sample_dir.glob("*_clean.pt"))
        speckle_path = next(sample_dir.glob("*_speckle.pt"))

        clean = torch.load(clean_path)
        speckles = torch.load(speckle_path)
        clean = transform_clean(clean)

        ssim_metric.reset()
        for speckle in speckles:
            speckle = transform_speckle(speckle ** gamma)
            ssim_metric.update(speckle.unsqueeze(0).unsqueeze(0), clean.unsqueeze(0))

        sample_scores.append(ssim_metric.compute().item())
    dataset_ssim = sum(sample_scores) / len(sample_scores)
    dataset_ssims.append(dataset_ssim)
    print(f"gamma: {gamma:.4f}, ssim: {dataset_ssim:.6f}")


# Dictionary: {gamma: mean_dataset_ssim}
result = {float(gamma): float(ssim) for gamma, ssim in zip(GAMMAS, dataset_ssims)}

# Save dictionary to text file
result_file = RESULTS_DIR / "gamma_vs_ssim.txt"

with open(result_file, "w") as f:
    f.write("Gamma\tMean Dataset SSIM\n")
    f.write("-" * 35 + "\n")
    for gamma, ssim in result.items():
        f.write(f"{gamma:.4f}\t{ssim:.6f}\n")



# Best Gamma
dataset_ssims = torch.tensor(dataset_ssims)
best_index = torch.argmax(dataset_ssims).item()
best_gamma = GAMMAS[best_index].item()
best_ssim = dataset_ssims[best_index].item()

print("\n==========================================")
print(f"Optimal Gamma : {best_gamma:.4f}")
print(f"Maximum SSIM  : {best_ssim:.6f}")
print("==========================================\n")

# Append best result to file
with open(result_file, "a") as f:
    f.write("\n")
    f.write(f"Optimal Gamma : {best_gamma:.4f}\n")
    f.write(f"Maximum SSIM  : {best_ssim:.6f}\n")



# Plot
plt.figure(figsize=(8, 5))

plt.plot(GAMMAS.numpy(), dataset_ssims.numpy(), lw=2)
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