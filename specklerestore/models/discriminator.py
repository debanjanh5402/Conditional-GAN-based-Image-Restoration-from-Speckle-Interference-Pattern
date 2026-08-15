import torch
from torch import nn

from .blocks import DiscBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        blocks = [DiscBlock(in_channels+out_channels, 64, norm=False),
                  DiscBlock(64, 128),
                  DiscBlock(128, 256),
                  DiscBlock(256, 512, stride=1),
                  nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)]
        self.model = nn.Sequential(*blocks)
        

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        x = torch.cat([x, y], dim=1)
        return self.model(x)
    