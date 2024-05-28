import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
import torch
import math

# VGG类，输入特征层和类别个数，得到类别向量

class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=True, num_cls=1000,input_size=32,drop_rate=0.4):
        super(VGG, self).__init__()
        self.features =  self.make_layers(cfg, batch_norm=batch_norm,drop_rate=drop_rate)  # 从参数输入特征提取流程 也即一堆CNN等
        pool_size=int(math.ceil(input_size/32))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        if num_cls>=200:
            flnumber = 2048
        else:
            flnumber = 512

        self.classifier = nn.Sequential(  # 定义分类器
            nn.Linear(512 * pool_size * pool_size, flnumber),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(flnumber, flnumber),
            nn.ReLU(True),
            nn.Dropout(p=0.0),
            nn.Linear(flnumber, num_cls),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # 把输入图像通过feature计算得到特征层
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x,start_dim= 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # isinstance 检查m是哪一个类型
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self,cfg,batch_norm=True,drop_rate=0.4):
        layers = []
        in_channels = 3
        for i,v in enumerate(cfg):
            if v == 'M':  # 最大池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v=='D':
                layers += [nn.Dropout(drop_rate)]
            else:
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1,bias=False),
                               nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True),
                               # nn.Dropout(0.4)
                               ]
                    if i+1<len(cfg) and cfg[i+1]!='M':
                        layers.append(nn.Dropout(drop_rate))
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1,bias=True),
                               nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)



def vggX(arch='D', num_cls=10, batch_norm=True, drop_rate=0.4):
    # cfg 用字母作为字典索引，方便获得对应的列表  vgg16对应D 详见论文
    cfgs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    cfg=cfgs[arch]
    model = VGG(cfg, batch_norm=batch_norm, num_cls=num_cls,drop_rate=drop_rate)
    return model


if __name__ == '__main__':
    model=vggX(num_cls=10)
    x=torch.rand(1,3,32,32)
    y=model(x)