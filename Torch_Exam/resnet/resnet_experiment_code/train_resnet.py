import torch
from torch import nn, optim
from torch.utils.data import (Dataset,
                               DataLoader,
                               TensorDataset)
import tqdm

def eval_net(net,test_loader, device="cpu"):
    net.eval()
    ys = []  # for accuracy
    ypreds = []  # for accuracy
    for x, y in test_loader:
        x = x.to(device)  # x,y 를 cpu로 처리
        y = y.to(device)  # x,y 를 cpu로 처리

        with torch.no_grad():  # with 는 뭐지?
            _, y_pred = net(x).max(1)  # 여러 클래스중에서 가장 확률이 높은 녀석을 y_pred로 채택한다.
        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)  # 여기서 cat을 한 것은 list 자료형을 torch.Tensor로 변환하기 위함으로 보인다. 아래에서 tensor자료형이 아니면 acc 계산이 불가하다.
    ypreds = torch.cat(ypreds)

    acc = (ys == ypreds).float().sum() / len(ys)  # 정확도를 계산하는 법
    return acc.item()


def train_net(net, train_loader, test_loader, _optimizer=optim.SGD,
          loss_fn=nn.CrossEntropyLoss(), n_iter=10, device="cpu",
              lr=0.01, momentum=0.9, train_err=[], val_err=[]):
    train_losses = []
    # train_acc = []
    # val_acc = []
    optimizer = _optimizer(net.parameters(), lr=lr, momentum=momentum,weight_decay=0.0001)

    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        n_acc = 0

        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)  # 이렇게 하는 이유는 batch가 딱 맞아떨어지지 않기 때문이다.

            _, y_pred = h.max(1) # max(1)은 1차원축 방향으로의 원소중에서 가장 큰 원소를 고르는 명령어.
                                # flatten 을 (batch size,-1) 로 했기 때문에 max(1)이 맞다.

            n_acc += (yy == y_pred).float().sum().item()  # 이게 맞는건가?

        train_losses.append(running_loss / i)  # 1 epoch loss

        train_error = 1 - (n_acc/n)
        train_err.append(train_error)  # 모든 batch에서 계산한 정확도를 1 epoch때 나누어서 계산
        test_error = 1 - eval_net(net, test_loader, device)
        val_err.append(test_error)

        print(lr, epoch, train_losses[-1], train_err[-1],
              val_err[-1], flush=True)

    return val_err[-1]
