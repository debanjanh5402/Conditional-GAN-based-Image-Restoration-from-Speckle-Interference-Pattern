from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------------------------------------
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from specklerestore.data import SpeckleDataset
from specklerestore.models import Pix2PixUNet, UNet
from specklerestore.unet_engine import fit_unet
from specklerestore.config import *


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
                       loss_fn=L1Loss(), 
                       epochs=EPOCHS, 
                       device=DEVICE,
                       checkpoint_dir=CHECKPOINT_DIR,
                       val_loader=val_loader,
                       fname_identifier=PRETRAINING_FNAME_IDENTIFIER,
                       test_loader=test_loader,
                       resume=False)


if __name__ == "__main__":
    main()

