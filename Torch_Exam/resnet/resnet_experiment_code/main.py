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
from my_resnet import PlainNet
import pickle


## data loader
transform_randomcrop = transforms.Compose([
    transforms.RandomCrop(size=32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) # 이렇게 전처리하겠다고 지정

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) # 이렇게 전처리하겠다고 지정

# trainset, testset을 전처리하면서 가져온다.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_randomcrop)
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

is_plain = True
is_resNet = False
draw_plot = True

if is_resNet:
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
    train_net(res20,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res20.train_err, val_err=res20.test_err)

    # train, test error log 저장
    with open('./resnet_data/resNet/list_res20_train(weight init+padding)', 'wb') as f:
        pickle.dump(res20.train_err, f)
    with open('./resnet_data/resNet/list_res20_test(weight init+padding)', 'wb') as f:
        pickle.dump(res20.test_err, f)

    train_net(res32,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=res32.train_err, val_err=res32.test_err)
    train_net(res32,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=res32.train_err, val_err=res32.test_err)
    train_net(res32,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res32.train_err, val_err=res32.test_err)

    with open('./resnet_data/resNet/list_res32_train(weight init+padding)', 'wb') as f:
        pickle.dump(res32.train_err, f)
    with open('./resnet_data/resNet/list_res32_test(weight init+padding)', 'wb') as f:
        pickle.dump(res32.test_err, f)

    print(min(res32.train_err))
    print(min(res32.test_err))

    train_net(res44,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=res44.train_err, val_err=res44.test_err)
    train_net(res44,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=res44.train_err, val_err=res44.test_err)
    train_net(res44,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res44.train_err, val_err=res44.test_err)

    with open('./resnet_data/resNet/list_res44_train(weight init+padding)', 'wb') as f:
        pickle.dump(res44.train_err, f)
    with open('./resnet_data/resNet/list_res44_test(weight init+padding)', 'wb') as f:
        pickle.dump(res44.test_err, f)

    train_net(res56,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=res56.train_err, val_err=res56.test_err)
    train_net(res56,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=res56.train_err, val_err=res56.test_err)
    train_net(res56,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=res56.train_err, val_err=res56.test_err)

    with open('./resnet_data/resNet/list_res56_train(weight init+padding)', 'wb') as f:
        pickle.dump(res56.train_err, f)
    with open('./resnet_data/resNet/list_res56_test(weight init+padding)', 'wb') as f:
        pickle.dump(res56.test_err, f)

    # training중 최상의 test 결과를 log로 기록
    f = open("./resnet_data/resNet/log_res20,32,44,56(weight_init+padding)", 'w')
    data = str(min(res20.test_err)) + '\n' + \
           str(min(res32.test_err)) + '\n' + \
           str(min(res44.test_err)) + '\n' + \
           str(min(res56.test_err))
    f.write(data)
    f.close()

    if draw_plot:
        # plot띄우는 부분
        # 여기다가 matplotlib 쓰면 된다.
        plt.plot(res20.test_err)
        plt.plot(res32.test_err)
        plt.plot(res44.test_err)
        plt.plot(res56.test_err)
        plt.ylabel('Error')
        plt.ylim((0,0.5))
        plt.xlabel('epoch')
        # plt.legend(['resnet20','resnet32'])
        plt.legend(['resnet20','resnet32','resnet44','resnet56'])
        plt.title('resnet test data error(weight init+padding)')
        plt.savefig('/home/jsh/PycharmProjects/Torch_Exam/resnet_data/resNet/resnet test data error(weight init+padding).png')
        plt.show()

if is_plain:
    plain20 = PlainNet(20)
    plain32 = PlainNet(32)
    plain44 = PlainNet(44)
    plain56 = PlainNet(56)

    plain20.to("cuda:0")
    plain32.to("cuda:0")
    plain44.to("cuda:0")
    plain56.to("cuda:0")

    # train plainNet
    train_net(plain20,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=plain20.train_err, val_err=plain20.test_err)
    train_net(plain20,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=plain20.train_err, val_err=plain20.test_err)
    train_net(plain20,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=plain20.train_err, val_err=plain20.test_err)

    # save plain net error list
    path = "/home/jsh/PycharmProjects/Torch_Exam/resnet_data/plainNet/WI+Pad"
    with open(path+'/list_plain20_testErr', 'wb') as f:
        pickle.dump(plain20.test_err, f)
    with open(path+'/list_plain20_trainErr', 'wb') as f:
        pickle.dump(plain20.train_err, f)

    train_net(plain32,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=plain32.train_err, val_err=plain32.test_err)
    train_net(plain32,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=plain32.train_err, val_err=plain32.test_err)
    train_net(plain32,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=plain32.train_err, val_err=plain32.test_err)

    with open(path+'/list_plain32_testErr', 'wb') as f:
        pickle.dump(plain32.test_err, f)
    with open(path+'/list_plain32_trainErr', 'wb') as f:
        pickle.dump(plain32.train_err, f)

    train_net(plain44,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=plain44.train_err, val_err=plain44.test_err)
    train_net(plain44,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=plain44.train_err, val_err=plain44.test_err)
    train_net(plain44,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=plain44.train_err, val_err=plain44.test_err)

    with open(path+'/list_plain44_testErr', 'wb') as f:
        pickle.dump(plain44.test_err, f)
    with open(path+'/list_plain44_trainErr', 'wb') as f:
        pickle.dump(plain44.train_err, f)

    train_net(plain56,trainloader, testloader, n_iter=80 ,device="cuda:0", lr=0.1, train_err=plain56.train_err, val_err=plain56.test_err)
    train_net(plain56,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.01, train_err=plain56.train_err, val_err=plain56.test_err)
    train_net(plain56,trainloader, testloader, n_iter=40 ,device="cuda:0", lr=0.001, train_err=plain56.train_err, val_err=plain56.test_err)

    with open(path+'/list_plain56_testErr', 'wb') as f:
        pickle.dump(plain56.test_err, f)
    with open(path+'/list_plain56_trainErr', 'wb') as f:
        pickle.dump(plain56.train_err, f)


    # training중 최상의 test 결과를 log로 기록
    f = open(path + "/log_plainnet", 'w')
    data = "plain20 : " + str(min(plain20.test_err)) + '\n' \
           + "plain32 : " + str(min(plain32.test_err)) + '\n' \
           + "plain44 : " + str(min(plain44.test_err)) + '\n' \
           + "plain56 : " + str(min(plain56.test_err)) + '\n'

    f.write(data)
    f.close()

    if draw_plot :
        # draw plot
        plt.plot(plain20.test_err)
        plt.plot(plain32.test_err)
        plt.plot(plain44.test_err)
        plt.plot(plain56.test_err)
        plt.ylabel('Error')
        plt.ylim((0,0.5))
        plt.xlabel('epoch')
        plt.legend(['plain20','plain32','plain44','plain56'])
        plt.title('plain net test data error')
        #plt.savefig('plain net test data error.png')
        plt.show()

