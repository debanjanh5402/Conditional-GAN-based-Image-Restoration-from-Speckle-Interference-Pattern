from pathlib import Path


TRAIN_DIR = Path("./dataset/train")
VAL_DIR = Path("./dataset/test")
TEST_DIR = Path("./dataset/test")


DEVICE = "mps"
NUM_WORKERS = 0

BATCH_SIZE = 4

IN_CHANNELS = 3
OUT_CHANNELS = 1

# GAMMA for dataset
GAMMA = 0.39



# UNet Pretraining 
UNET_LEARNING_RATE = 1e-3
UNET_BETAS = (0.9, 0.999)

# Pix2Pix GAN Training
GENERATOR_LEARNING_RATE = 3e-4
GENERATOR_BETAS = (0.5, 0.999)

DISCRIMINATOR_LEARNING_RATE = 1e-4
DISCRIMINATOR_BETAS = (0.5, 0.999)

GD_UPDATE_RATIO = 1

