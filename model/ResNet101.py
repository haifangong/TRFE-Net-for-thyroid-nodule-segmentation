import torch
from torch import nn


class Bottleneck(nn.Module):
    """
    通过 _make_layer 来构造Bottleneck
    具体通道变化：
    inplanes -> planes -> expansion * planes 直连 out1
    inplanes -> expansion * planes 残差项 res
    由于多层bottleneck级连 所以inplanes = expansion * planes 
    总体结构 expansion * planes -> planes -> expansion * planes 
    注意：
    1.输出 ReLu(out1 + res)
    2.与普通bottleneck不同点在于 其中的stride是可以设置的
    3.input output shape是否相同取决于stride   
      out:[x+2rate-3]/stride + 1 
      res:[x-1]/stride + 1
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None, gn=False):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if gn:
            self.bn1 = nn.GroupNorm(32, planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        if gn:
            self.bn2 = nn.GroupNorm(32, planes)
        else:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if gn:
            self.bn3 = nn.GroupNorm(32, planes* 4)
        else:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=32, pretrained=True, GN=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        elif os == 32:
            strides = [1, 2, 2, 2]
            rates = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if GN:
            self.bn1 = nn.GroupNorm(32, 64)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0], gn=GN)#64， 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1], gn=GN)#128 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2], gn=GN)#256 23
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], rate=rates[3], gn=GN)

        self._load_pretrained_model(GN)


    def _make_layer(self, block, planes, blocks, stride=1, rate=1, gn=False):
        """
        block class: 未初始化的bottleneck class
        planes:输出层数
        blocks:block个数
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if gn:
                downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),
            )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample, gn=gn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def gn_init(m, zero_init=False):
        assert isinstance(m, GroupNorm)
        m.weight.data.fill_(0. if zero_init else 1.)
        m.bias.data.zero_()

    def _load_pretrained_model(self, gn):
        if gn:
            pretrain_dict = torch.load('model/pretrain/ImageNet-ResNet50-GN.pth')
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrain_dict.items():            
                if k in state_dict:
                    model_dict[k] = v

            state_dict.update(model_dict)
            self.load_state_dict(state_dict)
            print("successfully load the gn weights")
        else:
            pretrain_dict = torch.load('/home/liguanbin/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrain_dict.items():            
                if k in state_dict:
                    # print(k)
                    # if 'bn' in k:
                    #     print(model_dict[k])
                    #     model_dict[k] = 0
                        # continue
                    
                    model_dict[k] = v
                    # if 'bn' in k:
                    #     model_dict[k] = torch.zeros((state_dict[k].shape))
            state_dict.update(model_dict)
            self.load_state_dict(state_dict)
            print("successfully load the weights")

def ResNet101(nInputChannels=3, os=32, pretrained=True):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model

def ResNet50(nInputChannels=3, os=32, pretrained=True, GN=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained, GN=GN)
    return model
