from pathlib import Path


TRAIN_DIR = Path("./dataset/train")
VAL_DIR = Path("./dataset/val")
TEST_DIR = Path("./dataset/test")
CHECKPOINT_DIR = Path("./_checkpoints")


DEVICE = "mps"
NUM_WORKERS = 0

BATCH_SIZE = 4

IN_CHANNELS = 3
OUT_CHANNELS = 1

# GAMMA for dataset
GAMMA = 0.39


# UNet Pretraining 
UNET_LEARNING_RATE = 5e-4
UNET_BETAS = (0.5, 0.999)
EPOCHS = 100

PRETRAINING_FNAME_IDENTIFIER = "unet_pretraining"

# Pix2Pix GAN Training
GENERATOR_LEARNING_RATE = 2e-4
GENERATOR_BETAS = (0.5, 0.999)
GENERATOR_CHECKPOINT = Path("./_checkpoints/best_checkpoint_unet_pretraining.pt")

DISCRIMINATOR_LEARNING_RATE = 1e-4
DISCRIMINATOR_BETAS = (0.5, 0.999)

TOTAL_STEPS = 50000
DECAY_START = 25000

Pix2Pix_FNAME_IDENTIFIER = "pix2pix_finetuning"
