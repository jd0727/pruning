import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data



# timg_pth='/home/data-storage/Tiny-ImageNet/'
timg_pth='/home/exspace/dataset/Tiny-ImageNet/tiny-imagenet-200/'

image_train_pth = os.path.join(timg_pth,'train')
image_test_pth = os.path.join(timg_pth,'val')


def get_loader(set='train', batch_size=128,num_workers=0):
    data_pth=''
    if set== 'train':
        trans=transforms.Compose((
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, 8, padding_mode='constant'),
            transforms.ToTensor(),
        ))
        data_pth=image_train_pth
    elif set=='test':
        trans=transforms.Compose((
            transforms.ToTensor(),
            ))
        data_pth = image_test_pth
    else:
        raise Exception('set err')

    dataset = datasets.ImageFolder(
        root=data_pth,
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
    test_loader=get_loader(set='test',batch_size=5)
    train_loader = get_loader(set='train', batch_size=5)
    x,y=next(iter(test_loader))
    x, y = next(iter(train_loader))