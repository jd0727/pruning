'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    #DEFAULT OPTION B
    def __init__(self, in_planes, planes, stride=1,act='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #激活函数
        if act=='relu':
            self.act= F.relu
        elif act=='lk_relu':
            self.act =partial(F.leaky_relu,negative_slope=0.1)
        else:
            print('err type act')
            return
        #
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.shortcut=None

    def forward(self, x):
        #res link
        res = self.shortcut(x) if not self.shortcut is None else x
        #
        out = self.bn1(self.conv1(x))
        out = self.act(out)
        out = self.bn2(self.conv2(out))
        #shortcut path
        out = out + res
        #
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_cls=10,act='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_layer(block, 16, num_blocks[0], stride=1,act=act)
        self.stage2 = self._make_layer(block, 32, num_blocks[1], stride=2,act=act)
        self.stage3 = self._make_layer(block, 64, num_blocks[2], stride=2,act=act)
        self.pool = nn.AdaptiveAvgPool2d([1, 1])
        self.linear = nn.Linear(64, num_cls)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride,act='relu'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,act=act))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnetC(num_layer=20,num_cls=10,act='relu'):
    para_dict={
        20: {'block': BasicBlock, 'num_blocks': [3, 3, 3]},
        32: {'block': BasicBlock, 'num_blocks': [5, 5, 5]},
        44: {'block': BasicBlock, 'num_blocks': [7, 7, 7]},
        56: {'block': BasicBlock, 'num_blocks': [9, 9, 9]},
        110: {'block': BasicBlock, 'num_blocks': [18, 18, 18]},
               }
    assert num_layer in para_dict.keys(),'need in keys'
    para=para_dict[num_layer]
    return ResNet(**para, num_cls=num_cls,act=act)



if __name__ == "__main__":
    pass