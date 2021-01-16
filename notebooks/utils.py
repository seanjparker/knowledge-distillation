import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader


def weight_reset(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        model.reset_parameters()


def calc_accuracy(test_dataset, model, device):
    correct = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_dataset):
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds = model(imgs)
            pred_class = torch.argmax(preds, dim=1)
            correct += torch.eq(pred_class, labels).sum()

    print(f'test accuracy: {(correct * 100) / len(test_dataset.dataset):.3f}%')


def load_cifar():
    train_transforms = [
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]

    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]

    cifar_train = datasets.CIFAR10('./datasets', train=True, download=True,
                                   transform=transforms.Compose(train_transforms))
    cifar_test = datasets.CIFAR10('./datasets', train=False, download=True,
                                  transform=transforms.Compose(test_transforms))

    batch_size = 256
    train_dataset = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = DataLoader(cifar_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataset, test_dataset


def load_mnist():
    all_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    mnist_train = datasets.MNIST('./datasets', train=True, download=True,
                                 transform=transforms.Compose(all_transforms))
    mnist_test = datasets.MNIST('./datasets', train=False, download=True,
                                transform=transforms.Compose(all_transforms))

    batch_size = 256
    train_dataset = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataset, test_dataset
