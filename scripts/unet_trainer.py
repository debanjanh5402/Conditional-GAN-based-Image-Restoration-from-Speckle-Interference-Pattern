from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------------------------------------
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss, HuberLoss

from specklerestore.data import SpeckleDataset
from specklerestore.models import Pix2PixUNet
from specklerestore.unet_engine import fit_unet
from specklerestore.config import (TRAIN_DIR, VAL_DIR, TEST_DIR, 
                                   DEVICE, NUM_WORKERS, 
                                   BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS,
                                   GAMMA, UNET_LEARNING_RATE, UNET_BETAS)


def main():

    train_dataset = SpeckleDataset(root_dir=TRAIN_DIR, gamma=GAMMA)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dataset = SpeckleDataset(root_dir=VAL_DIR, gamma=GAMMA)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_dataset = SpeckleDataset(root_dir=TEST_DIR, gamma=GAMMA)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    unet_model = Pix2PixUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)

    opt = Adam(params=unet_model.parameters(), lr=UNET_LEARNING_RATE, betas=UNET_BETAS)

    history = fit_unet(train_loader=train_loader, 
                       model=unet_model, 
                       optimizer=opt, 
                       loss_fn=HuberLoss(delta=0.5), 
                       epochs=5, 
                       device=DEVICE,
                       checkpoint_dir=Path("./_checkpoints"),
                       val_loader=val_loader,
                       fname_identifier=f"unet_lr{UNET_LEARNING_RATE:0.3f}_batch{BATCH_SIZE:1d}_huber0.5_test",
                       test_loader=test_loader,
                       resume=False)


if __name__ == "__main__":
    main()

