import torch
import torch.nn as nn
import torch.nn.functional as F

from pvt import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bn=True,
                 relu=True):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TransLayer, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 1),
            BasicConv2d(
                out_channel, out_channel, 3, padding=3, dilation=3
            )
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 1),
            BasicConv2d(
                out_channel, out_channel, 3, padding=5, dilation=5
            )
        )
        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class FPN(nn.Module):
    def __init__(self, channel):
        super(FPN, self).__init__()
        self.conv2 = BasicConv2d(channel, channel, 3, 1, 1)
        self.conv3 = BasicConv2d(channel, channel, 3, 1, 1)
        self.conv4 = BasicConv2d(channel, channel, 3, 1, 1)
        self.upsample = nn.Upsample(
            scale_factor=2.0, mode='bilinear', align_corners=False
        )

    def forward(self, x1, x2, x3, x4):
        x3 = self.upsample(self.conv4(x4)) + x3
        x2 = self.upsample(self.conv3(x3)) + x2
        x1 = self.upsample(self.conv2(x2)) + x1

        return x1


class BUNet(nn.Module):
    def __init__(self, channel=32):
        super(BUNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        self.Translayer1 = TransLayer(64, channel)
        self.Translayer2 = TransLayer(128, channel)
        self.Translayer3 = TransLayer(320, channel)
        self.Translayer4 = TransLayer(512, channel)

        self.fpn = FPN(channel)

        self.conv_out = nn.Conv2d(channel, 1, 3, 1, 1)

    def load_backbone_weights(self, path):
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        HW = x.shape[-2:]
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]     # (b, 64, 88, 88)
        x2 = pvt[1]     # (b, 128, 44, 44)
        x3 = pvt[2]     # (b, 320, 22, 22)
        x4 = pvt[3]     # (b, 512, 11, 11)

        x1 = self.Translayer1(x1)     # (b, 32, 88, 88)
        x2 = self.Translayer2(x2)     # (b, 32, 44, 44)
        x3 = self.Translayer3(x3)     # (b, 32, 22, 22)
        x4 = self.Translayer4(x4)     # (b, 32, 11, 11)

        x = self.fpn(x1, x2, x3, x4)

        fg = self.conv_out(x)
        fg = F.interpolate(fg, size=HW, mode='bilinear', align_corners=False)

        return fg


if __name__ == '__main__':
    model = BUNet()
    input_tensor = torch.randn(1, 3, 352, 352)

    out = model(input_tensor)
    print(out.shape)
