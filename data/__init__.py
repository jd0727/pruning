from .cifar import get_loader as cifar_loader
from .svhn import get_loader as svhn_loader
from .imagenet import get_loader as img_loader
from .tiny_imagenet import get_loader as timg_loader
import functools

def get_loader(name='c10',set='train', batch_size=128):
    loader_funcs={
        'c10':functools.partial(cifar_loader,num_cls=10),
        'c100':functools.partial(cifar_loader,num_cls=100),
        'sv':svhn_loader,
        'img':timg_loader
    }
    loader_func=loader_funcs[name]
    loader=loader_func(set=set,batch_size=batch_size)
    return loader