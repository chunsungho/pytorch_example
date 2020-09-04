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

# image를 일부는 32, 일부는 128사이즈로 나누는 클래스
class DownSizedPairImageFolder(ImageFolder):
    def __init__(self,root, transform=None,
                 large_size=128, small_size=32,**kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize(large_size) # 128개는 large 할당
        self.small_resizer = transforms.Resize(small_size)

    # 인덱스에 접근할 때 마다 호출되는 메서드
    def __getitem__(self,index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        # 읽은 이미지를 128x128픽셀과 32x32 픽셀로 리사이즈
        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)

        # transform하라고 명령이 들어오면 transform한다.
        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)

        # 32픽셀의 이미지와 128픽셀의 이미지 반환
        return small_img, large_img


myPath = "/home/jsh/PycharmProjects/Torch_Exam/data"

# data 가져와서 전처리 하는 작업
train_data = DownSizedPairImageFolder(
    myPath+"/lfw-deepfunneled/train",
    transform=transforms.ToTensor()
)

test_data = DownSizedPairImageFolder(
    myPath +"/lfw-deepfunneled/test",
    transform=transforms.ToTensor()
)


# 전처리된 데이터를 data loader로 batch화 시킨
batch_size=32
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size,
                         shuffle=False, num_workers=4)

# 이미지를 확대하는 net이다.
# 근데 왜 conv으로 축소시키고서 확대시키는거지?
net = nn.Sequential(
    nn.Conv2d(3,256,4,stride=2, padding=1),  # 이미지를 축소시키는 부분
    nn.ReLU(),
    nn.BatchNorm2d(256),

    nn.Conv2d(256,512,4,stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(512),

    nn.ConvTranspose2d(512,256,4,stride=2,padding=1), # deconvolution 부분이라고 보면 된다.
    nn.ReLU(),
    nn.BatchNorm2d(256),

    nn.ConvTranspose2d(256,128,4,stride=2,padding=1), # deconvolution 부분이라고 보면 된다.
    nn.ReLU(),
    nn.BatchNorm2d(128),

    nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),

    nn.ConvTranspose2d(64,3,4,stride=2,padding=1)
)

import math
def psnr(mse, max_v=1.0):
    return 10 * math.log10(max_v**2 / mse)

# 평가 헬퍼 함수
def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys = [] # 비어있는 tensor를 만들기 위한
    ypreds = []

    for x,y in data_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = net(x)
        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds) # 여기서 문제가 생긴다.

    # nn.MSELoss()가 아닌 다른 function적용.
    # PSNR 이란?
    score = nn.functional.mse_loss(ypreds,ys).item() # acc가 아닌 loss를 비교한다.
    return score

# 매개인자 : 전이학습 유무, loss, optim, net, dataLoader, device, epoch
def train_net(net,train_loader,test_loader, optimizer_cls = optim.Adam, loss_fn = nn.MSELoss(), n_iter=10, device = "cpu"):
    train_losses = [] # loss 수집해서 그래프확인용
    train_acc = [] # 출력용
    val_acc = []
    optimizer =optimizer_cls(net.parameters())

    for epoch in range(n_iter):
        running_loss = 0.0
        net.train() # dropout, bn사용 설정, 매 epoch마다 사용설정을 선언해준다.
        n = 0
        score = 0

        for i, (xx,yy) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
            '''
            batch 단위대로 데이터 뽑아서 학습
            forward, backward시행
            loss계산, grad계산, loss 수집, grad 업데이트
            '''
            xx = xx.to(device)
            yy = yy.to(device)
            y_pred = net(xx)
            loss = loss_fn(y_pred,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
        train_losses.append(running_loss/n)
        # val_acc.append(eval_net(net,train_loader,device))

        # print(epoch, train_losses[-1], psnr(train_losses[-1]), psnr(val_acc[-1]), flush=True) # 원래 이 코드로 검증해야하는데 GPU의 용량때문에 이 부분 스킵한다.
        print(epoch, train_losses[-1], psnr(train_losses[-1]), flush=True) # test 데이터로 검증하는 코드를 뺀 부분

net.to("cuda:0")
train_net(net, train_loader, test_loader, device="cuda:0")

from torchvision.utils import save_image
random_test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

it = iter(random_test_loader)
x, y = next(it)

bl_recon = torch.nn.functional.upsample(x, 128, mode="bilinear", align_corners=True)
# CNN으로 확대
yp = net(x.to("cuda:0")).to("cpu") # x는 cuda, net은 cpu여도 되는구나

# torch.cat로 원본, Bilinear, CNN 이미지를 결합하고
# save_image로 결합한 이미지를 출력
save_image(torch.cat([y, bl_recon, yp], 0), "cnn_upsale.jpg", nrow=4)
