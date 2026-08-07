from __future__ import annotations

from pathlib import Path

import torch
from torchvision import io
from torchvision import transforms as T
from torchvision.transforms.functional import resize

from .config import IMAGE_SIZE, TENSOR_EXTENSION, IMG_EXTENSION

def _read_img(img_path:str|Path) -> torch.Tensor:
    img = io.read_image(path=str(img_path), mode=io.ImageReadMode.GRAY) # (1, 600, 1200), uint8, [0, 255]
    img = img[:, :, :600] # crop the left part
    img = resize(img, size=[IMAGE_SIZE, IMAGE_SIZE], interpolation=T.InterpolationMode.BICUBIC) # (1, 256, 256)
    img = torch.clamp(img, max=255.0, min=0.0).to(torch.uint8)
    return img


def _get_imgname(img_path:str|Path) -> str:
    return Path(img_path).stem


def _create_sample_dir(dir:str|Path, name:str) -> Path:
    sample_dir = Path(dir) / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def _save_as_tensor(sample_dir:str|Path, name:str, tensor:torch.Tensor) -> Path:
    sample_dir = Path(sample_dir)
    file_path = sample_dir / f"{name}{TENSOR_EXTENSION}"
    torch.save(tensor, file_path)
    return file_path

def _save_as_image(sample_dir:str|Path, name:str, tensor:torch.Tensor) -> Path:
    sample_dir = Path(sample_dir)
    file_path = sample_dir / f"{name}{IMG_EXTENSION}"
    tensor = (((tensor - tensor.min())/(tensor.max() - tensor.min())) * 255).to(torch.uint8)
    io.write_jpeg(tensor, str(file_path), quality=100)
    return file_path