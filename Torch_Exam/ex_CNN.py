import torch
from torch import nn, optim
from torch.utils.data import (Dataset,
                               DataLoader,
                               TensorDataset)
import tqdm

from torchvision.datasets import FashionMNIST
from torchvision import transforms

# class 정의
# fc가 아니라 fc들어가기 위한 밑작업 layer.
# 근데 내용이 이해가 안됨.
class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1)

def eval_net(net,data_loader,device="cpu"):
    net.eval()
    ys=[] # for accuracy
    ypreds=[]   # for accuracy
    for x,y in data_loader:
        x = x.to(device) # x,y 를 cpu로 처리
        y = y.to(device) # x,y 를 cpu로 처리

        with torch.no_grad(): # with 는 뭐지?
            _, y_pred = net(x).max(1)   # 여러 클래스중에서 가장 확률이 높은 녀석을 y_pred로 채택한다.
        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys) # 여기서 cat을 한 것은 list 자료형을 torch.Tensor로 변환하기 위함으로 보인다. 아래에서 tensor자료형이 아니면 acc 계산이 불가하다.
    ypreds = torch.cat(ypreds)

    acc = (ys==ypreds).float().sum() / len(ys)  # 정확도를 계산하는 법
    return acc.item()

def train_net(net, train_loader, test_loader,
              optimizer_cls=optim.Adam,
              loss_fn=nn.CrossEntropyLoss(),
              n_iter=10, device="cpu"):
    train_losses=[]
    train_acc=[]
    val_acc = []
    optimizer = optimizer_cls(net.parameters())

    for epoch in range(n_iter):
        running_loss = 0.0

        net.train()
        n = 0
        n_acc = 0

        for i, (xx,yy) in tqdm.tqdm(enumerate(train_loader),
            total=len(train_loader)):
            xx=xx.to(device)
            yy=yy.to(device)
            h=net(xx)
            loss=loss_fn(h,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            n+=len(xx) # 이렇게 하는 이유는 batch가 딱 맞아떨어지지 않기 때문이다.
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item() # 이게 맞는건가?

        train_losses.append(running_loss/i) # 1 epoch loss

        train_acc.append(n_acc/n) # 모든 batch에서 계산한 정확도를 1 epoch때 나누어서 계산

        val_acc.append(eval_net(net, test_loader,device))

        print(epoch, train_losses[-1], train_acc[-1],
              val_acc[-1], flush=True)

        # for epoch 끝


if __name__ == '__main__':
    ## 학습용 데이터 받기
    myPath = "/media/jsh/CA02176502175633/Users/Desktop/Documents/data"
    fashion_mnist_train = FashionMNIST(myPath+"/FashionMNIST",
                                      train=True,download=True,
                                      transform=transforms.ToTensor())

    # 테스트용 데이터 받기
    # Tensor형태로 받는다. ==> dataset로 만들어 버리기 위함.
    fashion_mnist_test = FashionMNIST(myPath+"/FashionMNIST",
                                      train=False,download=True,
                                      transform=transforms.ToTensor())



    batch_size=128 # batch size 설정

    ## 데이터를 batch 단위로 묶어놓는다.
    train_loader = DataLoader(fashion_mnist_train,
                              batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(fashion_mnist_test,
                              batch_size=batch_size,shuffle=False)

    # fc들어가기 전 이미지가 들어가게 될 cnn을 생성한다.
    conv_net = nn.Sequential(
        nn.Conv2d(1, 32, 5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.25),

        nn.Conv2d(32, 64, 5),  # 어떻게 이렇게 되는지 알아봐야한다.
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.25),
        FlattenLayer()
    )
    # CNN으로 부터 나오는 bottleneck feature에 맞추기 위해 유동적으로 output size를 아래와 같이 맞춘다.
    test_input = torch.ones(1, 1, 28, 28)  # 28x28이미지, 1 channel 짜리가 1개 있다.
    conv_output_size = conv_net(test_input).size()[-1]

    # FC layer
    # output class : 100개
    mlp = nn.Sequential(
        nn.Linear(conv_output_size, 200),
        nn.ReLU(),
        nn.BatchNorm1d(200),
        nn.Dropout(0, 25),
        nn.Linear(200, 100)
    )

    # total CNN
    net = nn.Sequential(
        conv_net,
        mlp
    )

    net.to("cuda:0")
    # 모델을 학습한다.
    train_net(net,train_loader,test_loader,n_iter=20,
              device="cuda:0")

