from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import (Dataset,
                               DataLoader,
                               TensorDataset)

import torch
from torch import nn, optim
import tqdm
from torchvision.datasets import FashionMNIST
from torchvision import transforms, models
import torchvision

from my_resnet import ResNet, PlainNet
from train_resnet import train_net
from matplotlib import pyplot as plt


## data loader
transform = transforms.Compose([
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) # 이렇게 전처리하겠다고 지정

# trainset, testset을 전처리하면서 가져온다.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# 전처리한 이미지셋을 batch단위로 묶어준다.
# 논문에서는 cifar10 : mini batch = 128
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2) # batch단위로 묶어
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=True, num_workers=2)

# 클래스를 지정한다.
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# net 가져오는법 : ResNet
res = ResNet()
res.to("cuda:0")
# train_net(res,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_acc=res.train_acc, val_acc=res.test_acc)
# train_net(res,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_acc=res.train_acc, val_acc=res.test_acc)
# train_net(res,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_acc=res.train_acc, val_acc=res.test_acc)

train_net(res,trainloader, testloader, n_iter=8 ,device="cuda:0", lr=0.1, train_err=res.train_err, val_err=res.test_err)
train_net(res,trainloader, testloader, n_iter=4 ,device="cuda:0", lr=0.01, train_err=res.train_err, val_err=res.test_err)
train_net(res,trainloader, testloader, n_iter=4 ,device="cuda:0", lr=0.001, train_err=res.train_err, val_err=res.test_err)

# plain = PlainNet()
# plain.to("cuda:0")
# train_net(plain,trainloader, testloader, n_iter=30 ,device="cuda:0")

# plot띄우는 부분
# 여기다가 matplotlib 쓰면 된다.
plt.plot(res.train_err)
plt.plot(res.test_err)
plt.ylabel('accuracy')
plt.ylim((0,1))
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.title('resnet-20')
plt.show()


