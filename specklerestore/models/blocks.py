import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, norm:bool):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)] 
        if norm: layers.append(nn.InstanceNorm2d(num_features=out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.block(x)



class UpBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, drop:bool):
        super().__init__()

        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(out_channels, affine=True),
                  nn.ReLU()]
        if drop: layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.block(x)



class BottleNeck(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, norm:bool):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)] 
        if norm: layers.append(nn.InstanceNorm2d(num_features=out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.block(x)



class FinalBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())

    def forward(self, x:torch.Tensor):
        return self.block(x)



class DiscBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int = 2, norm:bool = True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
        if norm: layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.block(x)



class EncoderBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_convs:int):
        super().__init__()

        conv_layers = []
        if num_convs > 1: 
            for _ in range(num_convs-1):
                conv_layers.extend([
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=False),
                    nn.InstanceNorm2d(in_channels, affine=True),
                    nn.LeakyReLU(negative_slope=0.2)
                ])
        conv_layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        ])
        self.conv = nn.Sequential(*conv_layers)

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        p = self.pool(x)
        return x, p



class DecoderBlock(nn.Module):
    def __init__(self, in_channels:int, skip_channels:int, out_channels:int, num_convs:int, apply_drop:bool=False):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(skip_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

        conv_layers = [
            nn.Conv2d(in_channels+skip_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        if num_convs > 1:
            for _ in range(num_convs-1):
                conv_layers.extend([
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=False),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(negative_slope=0.2)
                ])
        if apply_drop:
            conv_layers.append(nn.Dropout2d())
            
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x:torch.Tensor, skip_x:torch.Tensor) -> torch.Tensor: 
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)
        return x