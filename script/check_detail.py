from data import *
from tools import *
import copy
from models import *


# num_cls=100
# test_loader= cifar_loader(set='test',num_cls=num_cls, batch_size=256)
# train_loader= cifar_loader(set='train',num_cls=num_cls, batch_size=256)
# input_size = (32, 32)

# test_loader= svhn_loader(set='test',batch_size=128)
# train_loader= svhn_loader(set='train',batch_size=128)

# test_loader= img_loader(set='test', batch_size=64)
# train_loader= img_loader(set='train', batch_size=64)

num_cls=200
test_loader= timg_loader(set='test', batch_size=128)
train_loader= timg_loader(set='train', batch_size=128)
input_size = (64, 64)

# dict_name='../chk/c10_res56_pxx'
# dict_name='../chk/sv_res56'
# dict_name='../chk/sv_vgg16'
# dict_name='../chk/c10_res110'
# dict_name='../chk/img_vgg16'
# dict_name='../chk/c100_vgg16_t50_px'
# dict_name='../chk/c10_vgg16'
dict_name='../chk/img_res50_p/img_res50_p35'

# model=CLSframe(backbone='vgg',num_cls=num_cls,drop_rate=0.1)
# model=CLSframe(backbone='resC',num_cls=num_cls,num_layer=56)
model=CLSframe(backbone='resI',num_cls=num_cls,drop_rate=0.1,num_layer=50)
model.load_wei(dict_name)


# model2onnx(model,file_name='../chk/img_res50.onnx',input_size=64)
# #
acc=test_model(model,test_loader,nums=[1,5])
print('Acc',acc)
# flop_cut, param_cut = calc_flop_para(model, input_size=input_size, ignore_zero=True)

# check_grad(model,train_loader)
#
for i in range(2):
    msg = train_SGD_StepLR(model, train_loader, test_loader, total_epoch=20, lr=0.0001, momentum=0.9, weight_decay=5e-4,
                            milestones=[10], gamma=0.1, file_name=dict_name, reg_loss=None, save_process=False)

# model.save_wei(dict_name)

# check_grad(model,train_loader)

# name='backbone.features.0.weight'
# names=str.split(name,'.')
# tar=model
# for n in names:
#     tar=getattr(tar,n)

