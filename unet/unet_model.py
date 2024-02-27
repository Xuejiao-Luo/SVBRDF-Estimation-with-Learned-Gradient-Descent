from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.init_conv = InitConv(n_channels, 64, kernel_size=3)
        self.down1 = Down(64, 128, kernel_size=3)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512,lambdaflag=False, n=19)
        factor = 1
        self.down4 = Down(512, 512 // factor,lambdaflag=False, n=9)
        self.down5 = Down(512, 512 // factor,lambdaflag=False, n=9)
        self.down6 = Down(512, 512 // factor,lambdaflag=False, n=9)
        self.down7 = Down(512, 512 // factor,lambdaflag=False, n=9, norm_flag=False)

        self.up1 = Up(1024, 512 // factor, bilinear=bilinear,lambdaflag=False, n=9)
        self.up2 = Up(1024, 512 // factor, bilinear=bilinear,lambdaflag=False, n=9)
        self.up3 = Up(1024, 512 // factor, bilinear=bilinear,lambdaflag=False, n=9)
        self.up4 = Up(1024, 512 // 2, bilinear=bilinear,lambdaflag=False, n=9)
        self.up5 = Up(512, 256 // 2, bilinear=bilinear,lambdaflag=False, dropoutflag=True, n=37)
        self.up6 = Up(256, 128 // 2, bilinear=bilinear, dropoutflag=True)
        self.up7 = Up(128, 64, bilinear=bilinear, kernel_size=5, dropoutflag=True)
        self.outc = OutConv_glob(64, n_classes)

    def forward(self, x):
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        logits = self.outc(x)
        return logits
