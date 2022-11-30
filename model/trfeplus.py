import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ResNet101 import ResNet50


class Classifier(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Classifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


class ARPG(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(ARPG, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(8, mip)
        self.act = nn.LeakyReLU(0.2)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.gn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = a_w * a_h
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class TRFEPLUS(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TRFEPLUS, self).__init__()

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
        self.conv10 = nn.Conv2d(64, 32, 1)

        self.scale_pred = Classifier(1024, 1)

        self.up6t = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6t = DoubleConv(1024, 512)
        self.up7t = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7t = DoubleConv(512, 256)
        self.up8t = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8t = DoubleConv(256, 128)
        self.up9t = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9t = DoubleConv(128, 64)
        self.conv10t = nn.Conv2d(64, out_ch, 1)

        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv1 = nn.Conv2d(32, 32, 3, padding=1)

        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, out_ch, 3, padding=1)

        self.conv6f = ARPG(1024, 512)
        self.conv7f = ARPG(512, 256)
        self.conv8f = ARPG(256, 128)

        self.reduction6 = nn.Conv2d(1024, 512, 1, padding=1)

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

        scale = self.scale_pred(c5)
        scale = torch.sigmoid(scale)

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

        # thyroid_norm = nn.Sigmoid()(thyroid)
        # c6t = torch.sigmoid(c6t)
        # c7t = torch.sigmoid(c7t)
        # c8t = torch.sigmoid(c8t)
        # c9t = torch.sigmoid(c9t)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        c6f = torch.cat([c6, c6t], dim=1)
        c6f = self.conv6f(c6f)
        c6f = c6 + c6f

        up_7 = self.up7(c6f)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        c7f = torch.cat([c7, c7t], dim=1)
        c7f = self.conv7f(c7f)
        c7f = c7 + c7f

        up_8 = self.up8(c7f)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        c8f = torch.cat([c8, c8t], dim=1)
        c8f = self.conv8f(c8f)
        c8f = c8 + c8f

        up_9 = self.up9(c8f)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        nodule = self.conv10(c9)
        nodule = self.finalrelu1(nodule)
        nodule = self.finalconv1(nodule)
        nodule = self.finalrelu2(nodule)
        nodule = self.finalconv2(nodule)

        # out = nn.Sigmoid()(c10)
        return nodule, thyroid, scale
