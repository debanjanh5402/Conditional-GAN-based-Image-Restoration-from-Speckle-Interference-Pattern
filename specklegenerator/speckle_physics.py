from __future__ import annotations

import math

import torch
from torch.fft import fftshift, ifftshift, fft2, ifft2


def _fft2c(x: torch.Tensor) -> torch.Tensor:
    return fftshift(fft2(ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def _ifft2c(x: torch.Tensor) -> torch.Tensor:
    return fftshift(ifft2(ifftshift(x, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


def gaussian_psf(size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(-size//2, size//2, dtype=torch.float32)
    y, x = torch.meshgrid(x, x, indexing="ij")
    psf = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


def generate_speckle(clean_image: torch.Tensor, H:torch.Tensor) -> torch.Tensor:
    amplitude = (clean_image/255.0).to(dtype=torch.float32)
    random_phase = (- math.pi + 2 * math.pi * torch.rand_like(amplitude, dtype=torch.float32))
    g = (amplitude * torch.exp(1j * random_phase)).to(dtype=torch.complex64)
    G = _fft2c(g)

    propagated = _ifft2c(G * H)
    speckle = torch.abs(propagated) ** 2
    return speckle.to(torch.float32).cpu()