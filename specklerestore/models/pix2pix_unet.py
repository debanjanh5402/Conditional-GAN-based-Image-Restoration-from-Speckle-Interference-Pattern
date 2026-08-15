import torch
from torch import nn
from .blocks import DownBlock, UpBlock, BottleNeck, FinalBlock


class Pix2PixUNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__() 

        self.down1 = DownBlock(in_channels, 64, False)
        self.down2 = DownBlock(64, 128, True)
        self.down3 = DownBlock(128, 256, True)
        self.down4 = DownBlock(256, 512, True)
        self.down5 = DownBlock(512, 512, True)
        self.down6 = DownBlock(512, 512, True)
        self.down7 = DownBlock(512, 512, True)

        self.bottleneck = BottleNeck(512, 512, False)

        self.up1 = UpBlock(512, 512, True)
        self.up2 = UpBlock(1024, 512, True)
        self.up3 = UpBlock(1024, 512, True)
        self.up4 = UpBlock(1024, 512, True)
        self.up5 = UpBlock(1024, 256, False)
        self.up6 = UpBlock(512, 128, False)
        self.up7 = UpBlock(256, 64, False)

        self.final = FinalBlock(128, out_channels)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Input                         # (bs, 3, 256, 256)
        e1 = self.down1(x)              # (bs, 64, 128, 128)
        e2 = self.down2(e1)             # (bs, 128, 64, 64)
        e3 = self.down3(e2)             # (bs, 256, 32, 32)
        e4 = self.down4(e3)             # (bs, 512, 16, 16)
        e5 = self.down5(e4)             # (bs, 512, 8, 8)
        e6 = self.down6(e5)             # (bs, 512, 4, 4)
        e7 = self.down7(e6)             # (bs, 512, 2, 2)

        b = self.bottleneck(e7)         # (bs, 512, 1, 1)

        d = self.up1(b)                 # (bs, 512, 2, 2)
        d = torch.cat([d, e7], dim=1)   # (bs, 1024, 2, 2)

        d = self.up2(d)                 # (bs, 512, 4, 4)
        d = torch.cat([d, e6], dim=1)   # (bs, 1024, 4, 4)

        d = self.up3(d)                 # (bs, 512, 8, 8)
        d = torch.cat([d, e5], dim=1)   # (bs, 1024, 8, 8)

        d = self.up4(d)                 # (bs, 512, 16, 16)
        d = torch.cat([d, e4], dim=1)   # (bs, 1024, 16, 16)

        d = self.up5(d)                 # (bs, 256, 32, 32)
        d = torch.cat([d, e3], dim=1)   # (bs, 512, 32, 32)

        d = self.up6(d)                 # (bs, 128, 64, 64)
        d = torch.cat([d, e2], dim=1)   # (bs, 256, 64, 64)

        d = self.up7(d)                 # (bs, 64, 128, 128)
        d = torch.cat([d, e1], dim=1)   # (bs, 128, 128, 128)

        return self.final(d)            # (bs, 1, 256, 256)