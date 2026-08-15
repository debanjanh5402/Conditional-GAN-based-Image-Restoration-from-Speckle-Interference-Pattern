from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------------------------------------
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam

from specklerestore.config import *
from specklerestore.pix2pix.data import SpeckleDataset
from specklerestore.pix2pix_engine import fit_pix2pix
from specklerestore.pix2pix.models import Generator, Discriminator
from specklerestore.pix2pix.losses import Pix2PixLoss



def main():

    train_dataset = SpeckleDataset(root_dir=TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=False)

    val_dataset = SpeckleDataset(root_dir=VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=False)

    G = Generator(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    D = Discriminator(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)

    opt_G = Adam(params=G.parameters(), lr=GENERATOR_LEARNING_RATE, betas=GENERATOR_BETAS)
    opt_D = Adam(params=D.parameters(), lr = DISCRIMINATOR_LEARNING_RATE, betas=DISCRIMINATOR_BETAS)

    history = fit_pix2pix(train_loader=train_loader, 
                  generator=G, discriminator=D, 
                  optimizer_g=opt_G, optimizer_d=opt_D,
                  loss_fn=Pix2PixLoss(),
                  steps=2500,
                  device=DEVICE,
                  gd_update_ratio=GD_UPDATE_RATIO, 
                  val_loader=val_loader,
                  validate_every=32,
                  latest_checkpoint_dir=Path("./_checkpoints"),
                  checkpoint_every=32,
                  best_model_dir=Path("./_checkpoints"),
                  fname_identifier="lr-3_1__gd-5")


if __name__ == "__main__":
    main()

    