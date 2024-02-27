""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from unet.lambdaNet import LambdaLayer

class InitConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=math.floor(kernel_size/2), stride=1)
        self.fullyconnect = nn.Linear(in_channels, out_channels)
        self.activation1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.activation2 = nn.SELU()

    def forward(self, x):
        mean_x = torch.mean(x, dim=(2,3), keepdim=True)
        x = self.activation1(self.conv(x))
        mean_x = torch.permute(mean_x, (0, 2, 3, 1))
        mean_x = self.activation2(self.fullyconnect(mean_x))
        mean_x = torch.permute(mean_x, (0, 3, 1, 2))
        return [x, mean_x]


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, lambdaflag=False, dropoutflag=False, n=9, norm_flag=True):
        super().__init__()

        self.dropoutflag = dropoutflag
        if not mid_channels:
            mid_channels = out_channels
            
        if lambdaflag:
            customlayer1 = LambdaLayer(dim=in_channels, dim_out=mid_channels, r=n, dim_k=16, heads=4, dim_u=1)
            customlayer2 = LambdaLayer(dim=mid_channels, dim_out=out_channels,r=n, dim_k=16, heads=4, dim_u=1)
        else:
            customlayer1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=math.floor(kernel_size/2), stride=1)
            customlayer2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=math.floor(kernel_size/2), stride=1, bias=False)

        self.fullyconnect1 = nn.Linear(in_channels, out_channels)
        self.fullyconnect2 = nn.Linear(in_channels+out_channels, out_channels)
        self.fullyconnect3 = nn.Linear(in_channels//2, out_channels)
        self.fullyconnect4 = nn.Linear(in_channels//2+out_channels, out_channels)
        self.conv = nn.Sequential(customlayer1,
                                  customlayer2)
        self.normlayer = nn.InstanceNorm2d(out_channels) if norm_flag else nn.Identity()
        self.activation1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.activation2 = nn.SELU()
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x, downflag=True):
        x0 = x[0]
        x1 = x[1]

        x0 = self.conv(x0)
        x0 = self.normlayer(x0)
        mean_x0 = torch.mean(x0, dim=(2,3), keepdim=True)

        if downflag:
            x2 = torch.permute(x1, (0, 2, 3, 1))
            x2 = self.fullyconnect1(x2)
            x2 = torch.permute(x2, (0, 3, 1, 2))
        else:
            x2 = torch.permute(x1, (0, 2, 3, 1))
            x2 = self.fullyconnect3(x2)
            x2 = torch.permute(x2, (0, 3, 1, 2))
        x0 = x0 + x2
        if self.dropoutflag:
            x0 = self.dropout(x0)
        x0 = self.activation1(x0)



        x1 = torch.cat((mean_x0, x1), dim=1)
        x1 = torch.permute(x1, (0, 2, 3, 1))
        if downflag:
            x1 = self.fullyconnect2(x1)
        else:
            x1 = self.fullyconnect4(x1)
        x1 = torch.permute(x1, (0, 3, 1, 2))
        x1 = self.activation2(x1)

        return [x0, x1]


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, lambdaflag=False, n=9, norm_flag=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, lambdaflag=lambdaflag, dropoutflag=False, n=n, norm_flag=norm_flag)

    def forward(self, x):
        return self.conv([self.maxpool(x[0]), x[1]])


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, bilinear=True, lambdaflag=False, dropoutflag=False, n=18):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=kernel_size, lambdaflag=lambdaflag, dropoutflag=False, n=n)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, lambdaflag=lambdaflag, dropoutflag=False,)

    def forward(self, x1, x2):
        x1_glob = x1[1]
        x1 = x1[0]
        x2 = x2[0]

        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x_glob = x1_glob
        return self.conv([x, x_glob], downflag=False)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,  padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.conv_out(x)

class OutConv_glob(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(OutConv_glob, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=math.floor(kernel_size/2)),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=math.floor(kernel_size/2)))
        self.activation = nn.Tanh()
        self.fullyconnect = nn.Linear(in_channels, out_channels)


    def forward(self, x):
        x0 = x[0]
        x1 = x[1]
        x0 = self.conv(x0)

        x1 = torch.permute(x1, (0, 2, 3, 1))
        x1 = self.fullyconnect(x1)
        x1 = torch.permute(x1, (0, 3, 1, 2))

        return self.activation(x0+x1)
