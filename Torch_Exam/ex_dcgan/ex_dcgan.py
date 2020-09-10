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

# 데이터 전처리
from torchvision.utils import save_image

myPath = "/home/jsh/PycharmProjects/Torch_Exam/"
img_data = ImageFolder(myPath+"oxford-102/",
                       transform=transforms.Compose([
                           transforms.Resize(80),
                           transforms.CenterCrop(64),
                           transforms.ToTensor()
                       ]))

# data loader --> batch단위로 데이터를 묶어준다.
batch_size = 64
img_loader = DataLoader(img_data, batch_size = batch_size,
                        shuffle=True)

nz = 100
ngf = 32

# z vector를 가지고 이미지데이터로 만들것이므로 이미지가 확대되는 과정인 decoding과정이 된다.
class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias=False),

            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2,1,bias=False),

            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),

            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, 4,2,1,bias=False),

            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf,3,4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self,x):
        out = self.main(x)
        return out

# out_size = (in_size-1)*stride-2*padding \
#     +kernel_size+output_padding

in_size = 1
stride = 1
padding = 0
kernel_size = 4
output_padding = 0

ndf = 32

# GNet을 통해서 latent vector -> image data가 만들어져서 DNet으로 들어온다.
# DNet에서는 이 이미지를 분석해서 class로 판별할 것이므로 encoding과정(축소과정)이 된다.
class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4,2,1,bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1,4,1,0,bias=False)
        )

    def forward(self,x):
        out = self.main(x)
        return out.squeeze()

d = DNet().to("cuda:0")
g = GNet().to("cuda:0")

opt_d = optim.Adam(d.parameters(),
                   lr=0.0002, betas=(0.5,0.999))
opt_g = optim.Adam(g.parameters(),
                   lr=0.0002, betas=(0.5,0.999))

# 크로스 엔트로피를 계산하기 위한 보조 변수 등
ones = torch.ones(batch_size).to("cuda:0") # 데이터 결과랑 이거랑 차이 비교해서 loss비교할려고 만든 임시데이터
zeros = torch.zeros(batch_size).to("cuda:0")
loss_f = nn.BCEWithLogitsLoss()

# 모니터링용 z
fixed_z = torch.randn(batch_size,nz,1,1).to("cuda:0") # 모니터링용이 뭐지? 이게 나중에 어떻게 쓰이는거지?

from statistics import mean
def train_dcgan(g,d,opt_g,opt_d, loader):
    log_loss_g = []
    log_loss_d = []
    for real_img, _ in tqdm.tqdm(loader):
        batch_len=len(real_img)

        real_img = real_img.to("cuda:0")

        # latent z 생성
        z = torch.randn(batch_len, nz, 1, 1).to("cuda:0")
        # generator에서 생성
        fake_img = g(z)

        # detach()가 뭐하는거지?
        fake_img_tensor = fake_img.detach()

        # discriminator에서 fake이미지를 판별 ==> 추후에 이거를 loss에 넣어서 학습할거야
        out = d(fake_img)

        loss_g = loss_f(out, ones[:batch_len]) # generator는 discriminator를 속여서 real이라고 판단하게끔 하는게 목적이니까 out과 1을 비교한다.
        log_loss_g.append(loss_g.item())

        g.zero_grad()
        loss_g.backward() # 일단 지금까지 d(g(z)) 만으로 generator학습이 가능하다.
        opt_g.step()

        real_out = d(real_img)
        loss_d_real = loss_f(real_out, ones[:batch_len])

        fake_img = fake_img_tensor

        fake_out = d(fake_img_tensor)
        loss_d_fake = loss_f(fake_out, zeros[:batch_len])

        loss_d = loss_d_real + loss_d_fake
        log_loss_d.append(loss_d.item())

        # 식별 모델의 미분 계산과 파라미터 갱신
        d.zero_grad()
        loss_d.backward()
        opt_d.step()

    return mean(log_loss_g), mean(log_loss_d)

output_path = myPath+"gan_out/"
# learning
for epoch in range(300):
    train_dcgan(g,d,opt_g, opt_d, img_loader)

    if epoch % 10 == 0:
        torch.save(
            g.state_dict(),
            output_path+"g_{:03d}.prm".format(epoch),
            pickle_protocol=4
        )
        torch.save(
            d.state_dict(),
            output_path+"d_{:03d}.prm".format(epoch),
            pickle_protocol=4
        )

        generated_img=g(fixed_z)
        save_image(generated_img,
                   output_path+"{:03d}.jpg".format(epoch))