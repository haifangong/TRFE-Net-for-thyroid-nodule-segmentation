import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class TRFENet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TRFENet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

        self.up6_align = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up7_align = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up8_align = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up9_align = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up6t = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6t = DoubleConv(1024, 512)
        self.up7t = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7t = DoubleConv(512, 256)
        self.up8t = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8t = DoubleConv(256, 128)
        self.up9t = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9t = DoubleConv(128, 64)
        self.conv10t = nn.Conv2d(64, out_ch, 1)

        self.reduction6 = nn.Conv2d(1024, 512, 1, padding=1)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6t = self.up6t(c5)
        merge6t = torch.cat([up_6t, c4], dim=1)
        c6t = self.conv6t(merge6t)

        up_7t = self.up7t(c6t)
        merge7t = torch.cat([up_7t, c3], dim=1)
        c7t = self.conv7t(merge7t)

        up_8t = self.up8t(c7t)
        merge8t = torch.cat([up_8t, c2], dim=1)
        c8t = self.conv8t(merge8t)

        up_9t = self.up9t(c8t)
        merge9t = torch.cat([up_9t, c1], dim=1)
        c9t = self.conv9t(merge9t)

        thyroid = self.conv10t(c9t)

        thyroid_norm = nn.Sigmoid()(thyroid)

        c8_mask = F.interpolate(thyroid_norm, scale_factor=0.5, mode='nearest')
        c7_mask = F.interpolate(c8_mask, scale_factor=0.5, mode='nearest')
        c6_mask = F.interpolate(c7_mask, scale_factor=0.5, mode='nearest')  

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        c6f = c6.mul(c6_mask)
        c6f = c6+c6f

        up_7 = self.up7(c6f)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        c7f = c7.mul(c7_mask)
        c7f = c7+c7f

        up_8 = self.up8(c7f)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        c8f = c8.mul(c8_mask)
        c8f = c8+c8f

        up_9 = self.up9(c8f)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        nodule = self.conv10(c9)

        # out = nn.Sigmoid()(c10)
        return nodule, thyroid

# import torch
# import torch.nn as nn
# import torchvision.models as models


# class DeConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DeConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_ch, out_ch,
#                 kernel_size=2,
#                 stride=2,
#                 bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.LeakyReLU(inplace=True)
#         )

#     def forward(self, inputs):
#         return self.conv(inputs)


# class Conv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_ch, out_ch,
#                 kernel_size=3,
#                 padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(
#                 out_ch, out_ch,
#                 kernel_size=3,
#                 padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.LeakyReLU(inplace=True)
#         )

#     def forward(self, inputs):
#         return self.conv(inputs)


# class TRFENet(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()

#         self.resnet = models.resnet50(pretrained=True)
#         self.conv1 = self.resnet.conv1
#         self.bn1 = self.resnet.bn1
#         self.relu = self.resnet.relu
#         self.maxpool = self.resnet.maxpool

#         # get some layer from resnet to make skip connection
#         self.layer1 = self.resnet.layer1
#         self.layer2 = self.resnet.layer2
#         self.layer3 = self.resnet.layer3
#         self.layer4 = self.resnet.layer4

#         self.conv_5t = Conv(2048, 512)
#         self.conv_4t = Conv(1536, 512)
#         self.conv_3t = Conv(768, 256)
#         self.conv_2t = Conv(384, 128)
#         self.conv_1t = Conv(128, 64)
#         self.deconv4t = DeConv(512, 512)
#         self.deconv3t = DeConv(512, 256)
#         self.deconv2t = DeConv(256, 128)
#         self.deconv1t = DeConv(128, 64)
#         self.deconv0t = DeConv(64, 32)

#         self.outt = nn.Conv2d(32, out_ch, 1)

#         self.conv_5 = Conv(2048, 512)
#         self.conv_4 = Conv(1536, 512)
#         self.conv_3 = Conv(768, 256)
#         self.conv_2 = Conv(384, 128)
#         self.conv_1 = Conv(128, 64)

#         self.deconv4 = DeConv(512, 512)
#         self.deconv3 = DeConv(512, 256)
#         self.deconv2 = DeConv(256, 128)
#         self.deconv1 = DeConv(128, 64)
#         self.deconv0 = DeConv(64, 32)

#         self.out = nn.Conv2d(32, out_ch, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         down_1 = self.relu(x)
#         x = self.maxpool(down_1)
#         down_2 = self.layer1(x)
#         down_3 = self.layer2(down_2)
#         down_4 = self.layer3(down_3)
#         x5 = self.layer4(down_4)
#         x5 = self.conv_5(x5)

#         x4_t = self.deconv4t(x5)
#         x4_t = torch.cat([x4_t, down_4], dim=1)
#         x4_t = self.conv_4t(x4_t)

#         x3_t = self.deconv3t(x4_t)
#         x3_t = torch.cat([x3_t, down_3], dim=1)
#         x3_t = self.conv_3t(x3_t)

#         x2_t = self.deconv2t(x3_t)
#         x2_t = torch.cat([x2_t, down_2], dim=1)
#         x2_t = self.conv_2t(x2_t)

#         x1_t = self.deconv1t(x2_t)
#         x1_t = torch.cat([x1_t, down_1], dim=1)
#         x1_t = self.conv_1t(x1_t)

#         x0_t = self.deconv0t(x1_t)
#         gland = self.outt(x0_t)

#         gland_norm = torch.sigmoid(gland)

#         c8_mask = F.interpolate(gland_norm, scale_factor=0.5, mode='nearest')
#         c7_mask = F.interpolate(c8_mask, scale_factor=0.5, mode='nearest')
#         c6_mask = F.interpolate(c7_mask, scale_factor=0.5, mode='nearest')  

#         x4 = self.deconv4(x5)
#         x4 = torch.cat([x4, down_4], dim=1)
#         x4 = self.conv_4(x4)

#         x3 = self.deconv3(x4)
#         x3 = torch.cat([x3, down_3], dim=1)
#         x3 = self.conv_3(x3)

#         x3f = x3.mul(c6_mask)
#         x3 = x3+x3f

#         x2 = self.deconv2(x3)
#         x2 = torch.cat([x2, down_2], dim=1)
#         x2 = self.conv_2(x2)

#         x2f = x2.mul(c7_mask)
#         x2 = x2+x2f

#         x1 = self.deconv1(x2)
#         x1 = torch.cat([x1, down_1], dim=1)
#         x1 = self.conv_1(x1)

#         x1f = x1.mul(c8_mask)
#         x1 = x1+x1f

#         x0 = self.deconv0(x1)
#         nodule = self.out(x0)

#         return nodule, gland 


# if __name__ == "__main__":
#     net = ResUNet()
#     input = torch.ones((8, 3, 224, 224))
#     outputs = net(input)
#     print(outputs.shape)
