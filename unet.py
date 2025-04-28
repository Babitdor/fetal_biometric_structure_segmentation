import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.Down_Conv_1 = DownSample(in_channels, 64)
        self.Down_Conv_2 = DownSample(64, 128)
        self.Down_Conv_3 = DownSample(128, 256)
        self.Down_Conv_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.Up_1 = UpSample(1024, 512)
        self.Up_2 = UpSample(512, 256)
        self.Up_3 = UpSample(256, 128)
        self.Up_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.Down_Conv_1(x)
        down_2, p2 = self.Down_Conv_2(p1)
        down_3, p3 = self.Down_Conv_3(p2)
        down_4, p4 = self.Down_Conv_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.Up_1(b, down_4)
        up_2 = self.Up_2(up_1, down_3)
        up_3 = self.Up_3(up_2, down_2)
        up_4 = self.Up_4(up_3, down_1)

        out = self.out(up_4)
        return out


# Debugging

# if __name__ == "__main__":
#     dou_conv = DoubleConv(256, 256)
#     print(dou_conv)

#     input_img = torch.rand((1, 3, 512, 512))
#     model = UNet(3, 10)
#     output = model(input_img)
#     print(output.size())
