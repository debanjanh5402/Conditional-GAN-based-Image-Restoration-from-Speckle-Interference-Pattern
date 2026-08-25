import torch
from torch import nn

from .blocks import EncoderBlock, DecoderBlock



class UNet(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.encoder_block1 = EncoderBlock(64, 128, num_convs=2)
        self.encoder_block2 = EncoderBlock(128, 256, num_convs=2)
        self.encoder_block3 = EncoderBlock(256, 512, num_convs=2)
        self.encoder_block4 = EncoderBlock(512, 512, num_convs=2)
        self.encoder_block5 = EncoderBlock(512, 512, num_convs=2)
        self.encoder_block6 = EncoderBlock(512, 512, num_convs=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
        )

        self.decoder_block6 = DecoderBlock(512, 512, 512, num_convs=2, apply_drop=True)
        self.decoder_block5 = DecoderBlock(512, 512, 512, num_convs=2, apply_drop=True)
        self.decoder_block4 = DecoderBlock(512, 512, 512, num_convs=2, apply_drop = False)
        self.decoder_block3 = DecoderBlock(512, 512, 256, num_convs=2, apply_drop = False)
        self.decoder_block2 = DecoderBlock(256, 256, 128, num_convs=2, apply_drop = False)
        self.decoder_block1 = DecoderBlock(128, 128, 64, num_convs=2, apply_drop = False)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=True),
            nn.Tanh()
        )
        


    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x = self.initial(x)            # (  3, 256, 256) -> ( 64, 256, 256)

        e1, p = self.encoder_block1(x) # ( 64, 256, 256) -> (128, 256, 256), (128, 128, 128)
        e2, p = self.encoder_block2(p) # (128, 128, 128) -> (256, 128, 128), (256,  64,  64)
        e3, p = self.encoder_block3(p) # (256,  64,  64) -> (512,  64,  64), (512,  32,  32)
        e4, p = self.encoder_block4(p) # (512,  32,  32) -> (512,  32,  32), (512,  16,  16)
        e5, p = self.encoder_block5(p) # (512,  16,  16) -> (512,  16,  16), (512,   8,   8)
        e6, p = self.encoder_block6(p) # (512,   8,   8) -> (512,   8,   8), (512,   4,   4)

        bn = self.bottleneck(p) # (512, 4, 4) -> (512, 4, 4)

        d = self.decoder_block6(bn, e6) # (512,   4,   4) -> (512,   8,   8) + e6 -> (512,   8,   8)
        d = self.decoder_block5(d, e5)  # (512,   8,   8) -> (512,  16,  16) + e5 -> (512,  16,  16)
        d = self.decoder_block4(d, e4)  # (512,  16,  16) -> (512,  32,  32) + e4 -> (512,  32,  32)
        d = self.decoder_block3(d, e3)  # (512,  32,  32) -> (512,  64,  64) + e3 -> (256,  64,  64)
        d = self.decoder_block2(d, e2)  # (256,  64,  64) -> (256, 128, 128) + e2 -> (128, 128, 128)
        d = self.decoder_block1(d, e1)  # (128, 128, 128) -> (128, 256, 256) + e1 -> ( 64, 256, 256)

        out = self.final(d)

        return out