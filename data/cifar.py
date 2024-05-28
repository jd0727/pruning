import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

cifar_pth='/home/exspace/dataset/CIFAR'
# cifar_pth='/home/user/JD/torchvision_dataset_download'
# cifar_pth='/home/user/JD/dataset'


def get_loader(set='train', num_cls=10, batch_size=128,num_workers=0):
    if set== 'train':
        istrain=True
        trans=transforms.Compose((
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            normalize,
        ))
    elif set=='test':
        istrain = False
        trans=transforms.Compose((
            transforms.ToTensor(),
            normalize
            ))
    else:
        raise Exception('set err')
    if num_cls==10:
        Dataset= torchvision.datasets.CIFAR10
    elif num_cls==100:
        Dataset = torchvision.datasets.CIFAR100
    else:
        raise Exception('cls err')

    dataset = Dataset(
        root=cifar_pth,
        train=istrain,
        transform=trans,
        download=False
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
    train_loader=get_loader(set='train',num_cls=10,batch_size=5)
    test_loader = get_loader(set='test', num_cls=10, batch_size=5)
    x,y=next(iter(train_loader))