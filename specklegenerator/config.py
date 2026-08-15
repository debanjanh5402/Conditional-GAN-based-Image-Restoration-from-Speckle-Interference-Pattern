from pathlib import Path



# Essential directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DATASET_DIR = Path("/Users/debanjan_5402/Desktop/Dataset/maps")
TRAIN_DIR = INPUT_DATASET_DIR / "train"
TEST_DIR = INPUT_DATASET_DIR / "test"

OUTPUT_DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_TRAIN_DIR = OUTPUT_DATASET_DIR / "train"
OUTPUT_TEST_DIR = OUTPUT_DATASET_DIR / "test"


# Final image size used by the network
IMAGE_SIZE = 256


# Number of independent speckle realizations generated per image
NUM_SPECKLES = 3


# File names
CLEAN_SUFFIX = "_clean"
SPECKLE_SUFFIX = "_speckle"
TENSOR_EXTENSION = ".pt"
IMG_EXTENSION = ".png" 


# Gaussian PSF
GAUSSIAN_SIGMA = 1.0
