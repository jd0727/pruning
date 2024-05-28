import torch.nn as nn
import math
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, shortcut=None,drop_rate=0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = shortcut
        self.stride = stride

        self.drop1 = nn.Dropout2d(drop_rate)
        self.drop2 = nn.Dropout2d(drop_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = out+residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, shortcut=None,drop_rate=0.2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        self.stride = stride
        #
        self.drop1 = nn.Dropout2d(drop_rate)
        self.drop2 = nn.Dropout2d(drop_rate)
        self.drop3 = nn.Dropout2d(drop_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.drop3(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = out+residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_cls=1000, act='relu',drop_rate=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_layer(block, 64, num_blocks[0],drop_rate=drop_rate)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2,drop_rate=drop_rate)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2,drop_rate=drop_rate)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], stride=2,drop_rate=drop_rate)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.linear = nn.Linear(512 * block.expansion, num_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,drop_rate=0):
        shortcut = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, shortcut,drop_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def resnetI(num_layer=18,num_cls=1000,act='relu',drop_rate=0):
    para_dict={
        18: {'block': BasicBlock, 'num_blocks':[2, 2, 2, 2]},
        34: {'block': BasicBlock, 'num_blocks': [3, 4, 6, 3]},
        50: {'block': Bottleneck, 'num_blocks': [3, 4, 6, 3]},
        101: {'block': Bottleneck, 'num_blocks': [3, 4, 23, 3]},
        152: {'block': Bottleneck, 'num_blocks': [3, 8, 36, 3]},
               }
    para=para_dict[num_layer]
    return ResNet(**para, num_cls=num_cls,act=act,drop_rate=drop_rate)


if __name__ == '__main__':
    test_x=torch.ones(size=(1,3,64,64))
    model=resnetI(num_layer=50,num_cls=1000,act='relu')

    y=model(test_x)
# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 models.
#
#     Args:
#         pretrained (bool): If True, returns a models pre-trained on ImageNet
#     """
#     models = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         models.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return models
#
#
# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 models.
#
#     Args:
#         pretrained (bool): If True, returns a models pre-trained on ImageNet
#     """
#     models = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         models.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return models
#
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 models.
#
#     Args:
#         pretrained (bool): If True, returns a models pre-trained on ImageNet
#     """
#     models = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         models.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return models
#
#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 models.
#
#     Args:
#         pretrained (bool): If True, returns a models pre-trained on ImageNet
#     """
#     models = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         models.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return models
#
#
# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 models.
#
#     Args:
#         pretrained (bool): If True, returns a models pre-trained on ImageNet
#     """
#     models = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         models.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return models



