import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop1=nn.Dropout(drop_rate)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes)
            )
            # nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut=None

        self.droprate = drop_rate
        self.equalInOut = (in_planes == planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        res = self.shortcut(x) if not self.shortcut is None else x
        x = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x)))
        out=self.drop1(out)
        out = self.conv2(out)
        out=out+res
        return out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [64, 64*widen_factor, 128*widen_factor, 256*widen_factor, 512*widen_factor]
        assert((depth - 2) % 8 == 0)
        n = (depth - 2) // 8
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.pool1=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.stage2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.stage3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.stage4 = NetworkBlock(n, nChannels[3], nChannels[4], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[4])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d([1, 1])
        self.linear = nn.Linear(nChannels[4], num_classes)
        self.nChannels = nChannels[4]

    def _make_layer(self, block, planes, num_blocks, stride,act='relu'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,act=act))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.relu(self.bn1(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out=self.linear(out)
        return out

if __name__ == '__main__':
    model=WideResNet(34,10)