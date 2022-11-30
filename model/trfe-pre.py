import torch
import torch.nn as nn
import torchvision.models as models


class DeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch,
                kernel_size=2,
                stride=2,
                bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.conv(inputs)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=3,
                padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.conv(inputs)


class ResUNet50(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool

        # get some layer from resnet to make skip connection
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4


        self.conv_5t = Conv(2048, 512)
        self.conv_4t = Conv(1536, 512)
        self.conv_3t = Conv(768, 256)
        self.conv_2t = Conv(384, 128)
        self.conv_1t = Conv(128, 64)
        self.deconv4t = DeConv(512, 512)
        self.deconv3t = DeConv(512, 256)
        self.deconv2t = DeConv(256, 128)
        self.deconv1t = DeConv(128, 64)
        self.deconv0t = DeConv(64, 32)

        self.outt = nn.Conv2d(32, n_classes, 1)

        self.conv_5 = Conv(2048, 512)
        self.conv_4 = Conv(1536, 512)
        self.conv_3 = Conv(768, 256)
        self.conv_2 = Conv(384, 128)
        self.conv_1 = Conv(128, 64)

        self.deconv4 = DeConv(512, 512)
        self.deconv3 = DeConv(512, 256)
        self.deconv2 = DeConv(256, 128)
        self.deconv1 = DeConv(128, 64)
        self.deconv0 = DeConv(64, 32)

        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        down_1 = self.relu(x)
        x = self.maxpool(down_1)
        down_2 = self.layer1(x)
        down_3 = self.layer2(down_2)
        down_4 = self.layer3(down_3)
        x5 = self.layer4(down_4)
        x5 = self.conv_5(x5)

        x4_t = self.deconv4t(x5)
        x4_t = torch.cat([x4_t, down_4], dim=1)
        x4_t = self.conv_4t(x4_t)

        x3_t = self.deconv3t(x4_t)
        x3_t = torch.cat([x3_t, down_3], dim=1)
        x3_t = self.conv_3t(x3_t)

        x2_t = self.deconv2t(x3_t)
        x2_t = torch.cat([x2_t, down_2], dim=1)
        x2_t = self.conv_2t(x2_t)

        x1_t = self.deconv1t(x2_t)
        x1_t = torch.cat([x1_t, down_1], dim=1)
        x1_t = self.conv_1t(x1_t)

        x0_t = self.deconv0t(x1_t)
        gland = self.outt(x0_t)

        x4 = self.deconv4(x5)
        x4 = torch.cat([x4, down_4], dim=1)
        x4 = self.conv_4(x4)

        x3 = self.deconv3(x4)
        x3 = torch.cat([x3, down_3], dim=1)
        x3 = self.conv_3(x3)

        x2 = self.deconv2(x3)
        x2 = torch.cat([x2, down_2], dim=1)
        x2 = self.conv_2(x2)

        x1 = self.deconv1(x2)
        x1 = torch.cat([x1, down_1], dim=1)
        x1 = self.conv_1(x1)

        x0 = self.deconv0(x1)
        nodule = self.out(x0)

        return nodule, gland 


if __name__ == "__main__":
    net = ResUNet()
    input = torch.ones((8, 3, 224, 224))
    outputs = net(input)
    print(outputs.shape)
