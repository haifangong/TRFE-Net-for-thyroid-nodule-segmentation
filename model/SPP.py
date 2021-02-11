import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP_simple(nn.Module):
    """docstring for ASPP_simple, simple means no ReLU """

    def __init__(self, inplanes, planes, rates=[1, 6, 12, 18]):
        super(ASPP_simple, self).__init__()

        self.aspp0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                             stride=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=rates[1], dilation=rates[1], bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=rates[2], dilation=rates[2], bias=False),
                                   nn.BatchNorm2d(planes))
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                             stride=1, padding=rates[3], dilation=rates[3], bias=False),
                                   nn.BatchNorm2d(planes))

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))

        self.reduce = nn.Sequential(
            nn.Conv2d(planes * 5, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x = self.reduce(x)
        return x


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rates):
        super(ASPP, self).__init__()

        self.aspp1 = ASPP_module(inplanes, planes, rate=rates[0])
        self.aspp2 = ASPP_module(inplanes, planes, rate=rates[1])
        self.aspp3 = ASPP_module(inplanes, planes, rate=rates[2])
        self.aspp4 = ASPP_module(inplanes, planes, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU()
                                             )

        self.conv1 = nn.Conv2d(planes * 5, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
