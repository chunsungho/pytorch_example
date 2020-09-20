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

# from my_resnet import ResNet
from resnet32_44_56_110 import ResNet
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
res20 = ResNet(20)
res32 = ResNet(32)
res44 = ResNet(44)
res56 = ResNet(56)

res20.to("cuda:0")
res32.to("cuda:0")
res44.to("cuda:0")
res56.to("cuda:0")

train_net(res20,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=res20.train_err, val_err=res20.test_err)
train_net(res20,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=res20.train_err, val_err=res20.test_err)
test_err_res20 = train_net(res20,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res20.train_err, val_err=res20.test_err)

train_net(res32,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=res32.train_err, val_err=res32.test_err)
train_net(res32,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=res32.train_err, val_err=res32.test_err)
test_err_res32 = train_net(res32,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res32.train_err, val_err=res32.test_err)

train_net(res44,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=res44.train_err, val_err=res44.test_err)
train_net(res44,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=res44.train_err, val_err=res44.test_err)
test_err_res44 =train_net(res44,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res44.train_err, val_err=res44.test_err)

train_net(res56,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=res56.train_err, val_err=res56.test_err)
train_net(res56,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=res56.train_err, val_err=res56.test_err)
test_err_res56 = train_net(res56,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res56.train_err, val_err=res56.test_err)


# writedata.py
f = open("./log", 'w')
data = str(test_err_res20) + '\n' + \
       str(test_err_res32) + '\n' + \
       str(test_err_res44) + '\n' + \
       str(test_err_res56)
f.write(data)
f.close()

# plain = PlainNet()
# plain.to("cuda:0")
# train_net(plain,trainloader, testloader, n_iter=30 ,device="cuda:0")

# plot띄우는 부분
# 여기다가 matplotlib 쓰면 된다.
plt.plot(res20.test_err)
plt.plot(res32.test_err)
plt.plot(res44.test_err)
plt.plot(res56.test_err)
plt.ylabel('Error')
plt.ylim((0,1))
plt.xlabel('epoch')
plt.legend(['resnet20','resnet32','resnet44','resnet56'])
plt.title('resnet test data error')
plt.show()
