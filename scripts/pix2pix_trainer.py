from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------------------------------------
from pathlib import Path

import torch
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.optim import Adam

from specklerestore.config import *
from specklerestore.data import SpeckleDataset
from specklerestore.pix2pix_engine import fit_pix2pix
from specklerestore.models import Pix2PixUNet, Discriminator
from specklerestore.losses import Pix2PixLoss

from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(step):
    if step < DECAY_START:
        return 1.0
    return 0.1 + 0.9 * (1.0 - (step - DECAY_START) / DECAY_START)



def main():

    train_dataset = SpeckleDataset(root_dir=TRAIN_DIR, gamma=GAMMA)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = SpeckleDataset(root_dir=VAL_DIR, gamma=GAMMA)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = SpeckleDataset(root_dir=TEST_DIR, gamma=GAMMA)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    G = Pix2PixUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    D = Discriminator(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)

    checkpoint = torch.load(GENERATOR_CHECKPOINT, map_location=DEVICE)
    G.load_state_dict(checkpoint["model"])

    opt_G = Adam(params=G.parameters(), lr=1e-5, betas=GENERATOR_BETAS)
    opt_D = Adam(params=D.parameters(), lr = DISCRIMINATOR_LEARNING_RATE, betas=DISCRIMINATOR_BETAS)

    scheduler_G = LambdaLR(opt_G, lr_lambda=lr_lambda)
    scheduler_D = LambdaLR(opt_D, lr_lambda=lr_lambda)

    history = fit_pix2pix(
        train_loader=train_loader,
        generator=G, discriminator=D,
        optimizer_g=opt_G, optimizer_d=opt_D, 
        loss_fn=Pix2PixLoss(recon_loss=L1Loss(), lambda_recon=100.0),
        total_steps=1000, #TOTAL_STEPS,
        val_interval=100,
        device=DEVICE,
        g_updates_per_step=1,
        d_updates_per_step=1,
        checkpoint_dir=CHECKPOINT_DIR,
        fname_identifier=Pix2Pix_FNAME_IDENTIFIER,
        val_loader=val_loader,
        test_loader=test_loader, 
        resume=False,
        scheduler_d=scheduler_D,
        scheduler_g=scheduler_G)


if __name__ == "__main__":
    main()

    