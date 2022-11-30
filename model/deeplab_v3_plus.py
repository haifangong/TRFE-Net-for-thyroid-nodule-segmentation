'''
    deeplab_v3+ :

        "Encoder-Decoder with Atrous Separable Convolution for Semantic Image
        Segmentation"
        Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
        (https://arxiv.org/abs/1802.02611)

    according to [mobilenetv2_coco_voc_trainaug / mobilenetv2_coco_voc_trainval]
    https://github.com/lizhengwei1992/models/tree/master/research/deeplab
    we use MobileNet_v2 as feature exstractor

    These codes are motified frome https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/networks/deeplab_xception.py

Author: Zhengwei Li
Data: July 1 2018
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ResNet101 import ResNet101, ResNet50
from model.ResNet34 import ResNet34
from model.SPP import ASPP_simple, ASPP
from model.utils import load_pretrain_model


# -------------------------------------------------------------------------------------------------
# Deeplabv3plus
#
# feature exstractor : MobileNet_v2, Xception, VggNet, ResNet
# -------------------------------------------------------------------------------------------------

class Deeplabv3plus(nn.Module):
    def __init__(self, nInputChannels, n_classes, os, backbone_type):

        super(Deeplabv3plus, self).__init__()

        # mobilenetv2 feature
        self.os = os
        self.backbone_type = backbone_type

        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if backbone_type == 'resnet101':
            self.backbone_features = ResNet101(nInputChannels, os, pretrained=True)
            asppInputChannels = 2048
            asppOutputChannels = 256
            lowInputChannels = 256
            lowOutputChannels = 48

            self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)
            self.last_conv = nn.Sequential(
                nn.Conv2d(asppOutputChannels + lowOutputChannels,
                          256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
            )
        elif backbone_type == 'resnet50':
            self.backbone_features = ResNet50(nInputChannels, os, pretrained=True)
            asppInputChannels = 2048
            asppOutputChannels = 256
            lowInputChannels = 256
            lowOutputChannels = 48

            self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)
            self.last_conv = nn.Sequential(
                nn.Conv2d(asppOutputChannels + lowOutputChannels,
                          256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
            )
        elif backbone_type == 'resnet34':
            self.backbone_features = ResNet34()
            asppInputChannels = 512
            asppOutputChannels = 256
            lowInputChannels = 64
            lowOutputChannels = 48

            self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)
            self.last_conv = nn.Sequential(
                nn.Conv2d(asppOutputChannels + lowOutputChannels,
                          256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
            )

        else:
            raise NotImplementedError

        # low_level_features to 48 channels
        self.conv2 = nn.Conv2d(lowInputChannels, lowOutputChannels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(lowOutputChannels)

        # init weights
        if backbone_type == 'mobilenetv2':
            self._init_weight()
        ## You CANNOT use this to init xception.

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # x : 1/1 512 x 512

        x, low_level_features = self.backbone_features(input)
        # x : 1/os 512/os x 512/os

        if self.os == 32:
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        x = self.aspp(x)
        # 1/4 128 x 128
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        x = F.interpolate(x, low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def load_backbone(self, model_path):
        if self.backbone_type == 'mobilenetv2':
            self.backbone_features = load_pretrain_model(self.backbone_features, torch.load(model_path))
        elif self.backbone_type == 'xception':
            self.backbone_features.load_xception_pretrained(model_path)
        elif self.backbone_type == 'resnet101':
            self.backbone_features = load_pretrain_model(self.backbone_features, torch.load('/media/SecondDisk/chenguanqi/thyroid_seg/pre_train/resnet101-5d3b4d8f.pth'))
            print('Already load the backbone of resnet101')
        elif self.backbone_type == 'resnet50':
            self.backbone_features = load_pretrain_model(self.backbone_features, torch.load('/media/SecondDisk/chenguanqi/thyroid_seg/pre_train/resnet50-19c8e357.pth'))
            print('Already load the backbone of resnet50')
        elif self.backbone_type == 'resnet34':
            self.backbone_features = load_pretrain_model(self.backbone_features, torch.load(model_path))
            print('Already load the backbone of resnet34')
        else:
            raise NotImplementedError


if __name__ == "__main__":
    model = Deeplabv3plus(3, 1, 32, 'resnet101')
    print("wnet have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1e6))

    # indata = torch.rand(4, 3, 224, 224)
    # _ = wnet(indata)
