import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes)
            )
            # nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut=None


    def forward(self, x):
        res = self.shortcut(x) if not self.shortcut is None else x
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out+res
        return out

class WideResNet(nn.Module):
    def __init__(self,block, num_blocks, drop_rate=0.5, num_cls=10, widen=1):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.conv1 =nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.stage1 = self._wide_layer(block, 16*widen, num_blocks[0], drop_rate, stride=1)
        self.stage2 = self._wide_layer(block, 32*widen, num_blocks[1], drop_rate, stride=2)
        self.stage3 = self._wide_layer(block, 64*widen, num_blocks[2], drop_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(64*widen, momentum=0.9)
        self.pool = nn.AdaptiveAvgPool2d([1, 1])
        self.linear = nn.Linear(64*widen, num_cls)
        #初始化
        self.apply(_weights_init)

    def _wide_layer(self, block, planes, num_block, drop_rate, stride):
        strides = [stride] + [1]*(int(num_block) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, drop_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn1(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def wrnC(num_layer=22,num_cls=10,widen=1,drop_rate=0):
    para_dict={
        22: {'block': BasicBlock, 'num_blocks': [3, 3, 3]},
        34: {'block': BasicBlock, 'num_blocks': [5, 5, 5]},
        46: {'block': BasicBlock, 'num_blocks': [7, 7, 7]},
        58: {'block': BasicBlock, 'num_blocks': [9, 9, 9]},
        70: {'block': BasicBlock, 'num_blocks': [11, 11, 11]},
               }
    assert num_layer in para_dict.keys(),'need in keys'
    para=para_dict[num_layer]
    return WideResNet(**para, num_cls=num_cls,widen=widen,drop_rate=drop_rate)


if __name__ == '__main__':
    model=wrnC(22, 10, 1, 0)
    x=torch.randn(7,3,32,32)
    y = model(x)

    print(y.size())