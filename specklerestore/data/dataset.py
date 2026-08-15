from torch.utils.data import Dataset
from pathlib import Path
import torch

from torchvision import io

from .utils import _normalize_img, _normalize_speckle

class SpeckleDataset(Dataset):
    def __init__(self, root_dir:str|Path, gamma:float):
        super().__init__()

        self.gamma = gamma

        self.root_dir = Path(root_dir)
        self.samples = sorted([folder for folder in self.root_dir.iterdir() if folder.is_dir()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        folder = self.samples[index]
        foldername = folder.name

        clean_path = folder / f"{foldername}_clean.png"
        clean = io.read_image(clean_path, mode=io.ImageReadMode.GRAY) # (1, 256, 256)
        clean = _normalize_img(clean)

        speckle_path = folder / f"{foldername}_speckle.pt"
        speckle_tensor = torch.load(speckle_path)
        speckle_tensor = _normalize_speckle(speckle_tensor, self.gamma)

        return {"input": speckle_tensor,
                "target": clean,
                "name": str(foldername)}