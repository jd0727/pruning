import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
import torch.utils.data as data


# svhn_pth='/home/user/JD/torchvision_dataset_download'
# svhn_pth='/home/user/JD/dataset'
svhn_pth='/home/exspace/dataset/SVHN'


def get_loader(set='train', batch_size=128,num_workers=0):
    if set== 'train':
        trans=transforms.Compose((
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
        ))
    elif set=='test':
        trans=transforms.Compose((
            transforms.ToTensor(),
            ))
    else:
        raise Exception('set err')
    dataset = SVHN(
        split=set,
        root=svhn_pth,
        download=False,
        transform=trans
    )
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory = True
    )
    return loader

if __name__ == '__main__':
    train_loader=get_loader(set='train',batch_size=128)
    test_loader = get_loader(set='test', batch_size=5)
    x,y=next(iter(train_loader))
    v=torch.mean(x)
    t=torch.max(y)