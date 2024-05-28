from models import *
import torch.nn as nn
from models.utils import *
from models.praser import *
from models import vgg
from models import resnet_imagenet
from models import resnet_cifar
# 分类框架
class CLSframe(nn.Module):
    def __init__(self ,backbone ,num_cls=10 ,device=None ,**kws):
        super(CLSframe, self).__init__()
        if not isinstance(backbone ,str):
            self.backbone =backbone
        elif backbone=='resC':
            self.backbone = resnet_cifar.resnetC(num_cls=num_cls ,**kws)
        elif backbone=='resI':
            self.backbone = resnet_imagenet.resnetI(num_cls=num_cls ,**kws)
        elif backbone=='vgg':
            self.backbone =vgg.vggX(num_cls=num_cls, **kws)
        else:
            self.backbone =None
        # device
        if device is None:
            if torch.cuda.is_available():
                inds =ditri_gpu(min_thres=0.1,one_thres=0.6)
                if inds is None:
                    raise Exception('No GPU')
                print('CLSframe: Put models on cuda ' +str(inds[0]))
                self.device = torch.device("cuda:" +str(inds[0]))
                print('CLSframe: Put DataParallel on cuda ' +str(inds))
                # self.backbone = nn.DataParallel(self.backbone, device_ids=inds)
            else:
                self.device = torch.device("cpu")
        else:
            self.device =torch.device(device)
        self.backbone.to(self.device)
        # cert
        self.cert= nn.CrossEntropyLoss(reduction='mean')

    def get_loss(self, imgs, lbs):
        imgs = imgs.to(self.device)
        lbs = lbs.to(self.device)
        pred = self.backbone(imgs)
        loss = self.cert(pred, lbs)
        return loss

    def imgs2lb(self ,imgs):
        imgs = imgs.to(self.device)
        pred = self.backbone(imgs)
        return pred

    def forward(self ,imgs):
        return self.imgs2lb(imgs)

    #得到通道
    def get_dicts(self,meth='all'):
        backbone=self._get_backbone()
        if isinstance(backbone,resnet_imagenet.ResNet):
            dicts=ext_resnet(backbone, meth=meth, input_size=(64, 64))
        elif isinstance(backbone,resnet_cifar.ResNet):
            dicts=ext_resnet(backbone, meth=meth, input_size=(32, 32))
        elif isinstance(backbone, vgg.VGG):
            dicts=ext_vgg(backbone)
        else:
            print('Err type')
            return None
        return dicts
    #############################################################################权重存取
    def _get_backbone(self):
        if 'cuda' in self.device.type and isinstance(self.backbone ,nn.DataParallel):
            backbone = self.backbone.module
        else:
            backbone = self.backbone
        return backbone

    def save_wei(self, filename):
        backbone = self._get_backbone()
        torch.save(backbone.state_dict(), filename)

    def load_wei(self, filename):
        backbone = self._get_backbone()
        state_dict = torch.load(filename, map_location=self.device)
        result =load_fmt(backbone, sd_ori=state_dict, character='fname', only_fullmatch=True)
        if result:
            refine_chans(backbone)
            return None
        print('Struct changed, Try to match')
        result = load_fmt(backbone, sd_ori=state_dict, character='size', only_fullmatch=True)
        if result:
            refine_chans(backbone)
            return None
        result = load_fmt(backbone, sd_ori=state_dict, character='lname', only_fullmatch=True)
        if result:
            refine_chans(backbone)
            return None
        print('Tolerates imperfect matches')
        load_fmt(backbone, sd_ori=state_dict, character='size fname', only_fullmatch=False)
        refine_chans(backbone)
        # state_dict = torch.load(filename, map_location=self.device)
        # backbone.load_state_dict(state_dict, strict=False)
        return None


if __name__ == '__main__':
    model =CLSframe(backbone='vgg')
    model.load_wei('../chk/c10_vgg16')