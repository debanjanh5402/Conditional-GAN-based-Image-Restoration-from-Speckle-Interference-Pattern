from __future__ import annotations

from pathlib import Path

from tqdm import tqdm
import torch

from .config import (NUM_SPECKLES, IMAGE_SIZE, GAUSSIAN_SIGMA,
                     TRAIN_DIR, TEST_DIR, OUTPUT_TRAIN_DIR, OUTPUT_TEST_DIR, 
                     CLEAN_SUFFIX, SPECKLE_SUFFIX)
from .utils import _read_img, _get_imgname, _create_sample_dir, _save_as_image, _save_as_tensor
from .speckle_physics import generate_speckle, gaussian_psf, _fft2c


class DatasetGenerator:

    def __init__(self) -> None:
        self.sigma = GAUSSIAN_SIGMA
        self.img_size = IMAGE_SIZE


    def generate(self) -> None:
        self._generate_split(TRAIN_DIR, OUTPUT_TRAIN_DIR)
        self._generate_split(TEST_DIR, OUTPUT_TEST_DIR)


    def _generate_split(self, input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        img_paths = sorted(input_dir.glob("*.jpg"))

        psf = gaussian_psf(size=self.img_size, sigma=self.sigma)
        H = _fft2c(psf)

        for img_path in tqdm(img_paths, desc=input_dir.name):
            imgname = _get_imgname(img_path)
            sample_dir = _create_sample_dir(output_dir, imgname)
            clean = _read_img(img_path)
            _save_as_image(sample_dir, name=f"{imgname}{CLEAN_SUFFIX}", tensor=clean)
            _save_as_tensor(sample_dir, name=f"{imgname}{CLEAN_SUFFIX}", tensor=clean)


            shots = []
            for i in range(1, NUM_SPECKLES + 1):
                speckle = generate_speckle(clean, H)
                _save_as_image(sample_dir, name=f"{imgname}{i}{SPECKLE_SUFFIX}", tensor=speckle)
                shots.append(speckle.squeeze())
            _save_as_tensor(sample_dir, name=f"{imgname}{SPECKLE_SUFFIX}", tensor=torch.stack(shots, dim=0))
            