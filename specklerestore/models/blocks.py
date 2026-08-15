import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, norm:bool):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)] 
        if norm: layers.append(nn.BatchNorm2d(num_features=out_channels, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.block(x)



class UpBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, drop:bool):
        super().__init__()

        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels, bias=False),
                  nn.ReLU(inplace=True)]
        if drop: layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.block(x)



class BottleNeck(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, norm:bool):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)] 
        if norm: layers.append(nn.BatchNorm2d(num_features=out_channels, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
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
        if norm: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.block(x)