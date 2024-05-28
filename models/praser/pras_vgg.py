import functools
import torch
from models.vgg import vggX
import torch.nn as nn

def ext_vgg(model):
    binding_dicts=[]
    last_conv_ind=-1
    for i,m in enumerate(model.features):
        if not isinstance(m,nn.Conv2d):
            continue
        if last_conv_ind ==-1:
            last_conv_ind = 0
            continue
        last_m=model.features[last_conv_ind]
        size=last_m.weight.data.size()[0]
        #
        binding_dict={
            'size':size,
            'in':[m],
            'out':[last_m],
            'type':'inner'
        }
        bn=model.features[last_conv_ind+1]
        if isinstance(bn,nn.BatchNorm2d):
            binding_dict['pth']=[bn]
        binding_dicts.append(binding_dict)
        last_conv_ind=i
    #记录bn
    last_m = model.features[last_conv_ind]
    last_bn = model.features[last_conv_ind+1]
    #线性
    for i, m in enumerate(model.classifier):
        if not isinstance(m,nn.Linear):
            continue
        size = last_m.weight.data.size()[0]
        binding_dict={
            'size':size,
            'in':[m],
            'out':[last_m],
            'type': 'inner'
        }
        if i==0 and isinstance(last_bn, nn.BatchNorm2d):
            binding_dict['pth']=[last_bn]
        binding_dicts.append(binding_dict)
        last_m=m
    return binding_dicts

if __name__ == '__main__':
    model = vggX(num_cls=10)
    bd=ext_vgg(model)