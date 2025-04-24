import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, type='down'):
        super(DeepwiseConv, self).__init__()

        if type == 'down':
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                          padding=(dilation * (kernel_size - 1)) // 2,
                          dilation=dilation,
                          groups=in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=(dilation * (kernel_size - 1)) // 2,
                          dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                          padding=(dilation * (kernel_size - 1)) // 2,
                          dilation=dilation,
                          groups=in_channels // out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=(dilation * (kernel_size - 1)) // 2,
                          dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, down_size, kernel_size, dilation):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool1d(kernel_size=down_size, stride=down_size),
            DeepwiseConv(in_channels, out_channels, kernel_size, dilation)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, up_size, kernel_size, dilation):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=up_size, mode='linear', align_corners=True)
        self.conv = DeepwiseConv(in_channels * 2, out_channels, kernel_size, dilation, type='up')

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = DeepwiseConv(32, 64, kernel_size=7, dilation=2)
        self.down1 = Down(64, 128, down_size=5, kernel_size=5, dilation=2)
        self.down2 = Down(128, 256, down_size=2, kernel_size=3, dilation=2)

        self.down3 = Down(256, 256, down_size=2, kernel_size=3, dilation=1)

        self.up1 = Up(256, 128, up_size=2, kernel_size=3, dilation=2)
        self.up2 = Up(128, 64, up_size=2, kernel_size=5, dilation=2)
        self.up3 = Up(64, 32, up_size=5, kernel_size=7, dilation=2)

        self.outc = OutConv(32, 32)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)

        return logits


if __name__ == '__main__':
    input = torch.randn(size=(1, 12, 1000))
    model = UNet()

    print(model(input).shape)
