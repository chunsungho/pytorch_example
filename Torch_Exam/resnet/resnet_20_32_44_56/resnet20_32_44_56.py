import torch
from torch import nn, optim


class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1) # size[0] : mini-batch size

class ResidualBlock(nn.Module):
    def __init__(self,inChannel, outChannel, stride = 1): # in, out, stride, 같은거 쓴다.
        super().__init__()
        self.stride = stride
        self.out_channel = outChannel
        self.conv1 = nn.Conv2d(inChannel,outChannel,3,padding=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.relu1 = nn.ReLU() # 인자를 넣어야 하나

        self.conv2 = nn.Conv2d(outChannel, outChannel, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(outChannel)

        # shortcuts의 dimension을 맞춰주기 위한 layer
        self.conv1x1 = nn.Conv2d(inChannel, outChannel, 1, stride=stride)
        self.bn_sh = nn.BatchNorm2d(outChannel)
        self.relu_sh = nn.ReLU()

    def forward(self,x):
        # 하나의 블럭 안에서 어떻게 실행되는지 선언한다.
        shortcuts = x
        out = self.conv1(x) # 1 블럭 ####################
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out) # 2 블럭
        out = self.bn2(out)

        # 여기서 stride 판단해서 shortcuts도 downsampling할지 말지 결정하기
        if self.stride == 2:
            # 1x1 conv 이용한다
            shortcuts = self.conv1x1(x)
            shortcuts = self.bn_sh(shortcuts)

        out += shortcuts # H(x) = F(x) + x
        out = self.relu_sh(out) # 마지막은 shortcuts과 합쳐준 이후에 relu를 거치고 끝낸다.
        return out

# CIFAR-10
# version : 20, 32, 44, 56 이 있다.
class ResNet(nn.Module):
    def __init__(self, version):
        super().__init__()
        # conv, bn, relu, layer, flatten, fc layer 선언
        channels = [16,32,64]
        num_classes = 10
        self.train_err = []
        self.test_err = []
        self.version = version

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.ReLU()

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # CIFAR10으로는 하지말자. 원래 이미지 사이즈가 너무 작으니까
        self.layer1a = ResidualBlock(channels[0], channels[0], 1)
        self.layer1b = ResidualBlock(channels[0], channels[0], 1) # 32x32
        self.layer1c = ResidualBlock(channels[0], channels[0], 1)
        self.layer1d = ResidualBlock(channels[0], channels[0], 1)
        self.layer1e = ResidualBlock(channels[0], channels[0], 1)
        self.layer1f = ResidualBlock(channels[0], channels[0], 1)
        self.layer1g = ResidualBlock(channels[0], channels[0], 1)
        self.layer1h = ResidualBlock(channels[0], channels[0], 1)
        self.layer1i = ResidualBlock(channels[0], channels[0], 1)

        self.layer2a = ResidualBlock(channels[0], channels[1], 2) # 16x16
        self.layer2b = ResidualBlock(channels[1], channels[1], 1)
        self.layer2c = ResidualBlock(channels[1], channels[1], 1)
        self.layer2d = ResidualBlock(channels[1], channels[1], 1)
        self.layer2e = ResidualBlock(channels[1], channels[1], 1)
        self.layer2f = ResidualBlock(channels[1], channels[1], 1)
        self.layer2g = ResidualBlock(channels[1], channels[1], 1)
        self.layer2h = ResidualBlock(channels[1], channels[1], 1)
        self.layer2i = ResidualBlock(channels[1], channels[1], 1)

        self.layer3a = ResidualBlock(channels[1], channels[2], 2) # 8x8
        self.layer3b = ResidualBlock(channels[2], channels[2], 1)
        self.layer3c = ResidualBlock(channels[2], channels[2], 1)
        self.layer3d = ResidualBlock(channels[2], channels[2], 1)
        self.layer3e = ResidualBlock(channels[2], channels[2], 1)
        self.layer3f = ResidualBlock(channels[2], channels[2], 1)
        self.layer3g = ResidualBlock(channels[2], channels[2], 1)
        self.layer3h = ResidualBlock(channels[2], channels[2], 1)
        self.layer3i = ResidualBlock(channels[2], channels[2], 1)

        self.avgPool = nn.AvgPool2d(8,stride=1) # 8x8 -> 1x1 축소
        self.flatten = FlattenLayer()
        self.fc = nn.Linear(in_features=channels[2] * 1 * 1, out_features=num_classes) # avgpool로 인해 img size = 1x1.
        self.softmax = nn.Softmax(dim=1) # 아마 1차원 방향의 원소 합을 1이 되게끔 softmax하는것 일듯

    def forward(self,input):

        if self.version >= 20:
            out = self.conv1(input)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.layer1a(out)
            out = self.layer1b(out)
            out = self.layer1c(out)

            if self.version >= 32:
                out = self.layer1d(out)
                out = self.layer1e(out)

                if self.version >= 44:
                    out = self.layer1f(out)
                    out = self.layer1g(out)

                    if self.version >= 56:
                        out = self.layer1h(out)
                        out = self.layer1i(out)

            out = self.layer2a(out)
            out = self.layer2b(out)
            out = self.layer2c(out)

            if self.version >= 32:
                out = self.layer2d(out)
                out = self.layer2e(out)

                if self.version >= 44:
                    out = self.layer1f(out)
                    out = self.layer1g(out)

                    if self.version >= 56:
                        out = self.layer1h(out)
                        out = self.layer1i(out)

            out = self.layer3a(out)
            out = self.layer3b(out)
            out = self.layer3c(out)

            if self.version >= 32:
                out = self.layer3d(out)
                out = self.layer3e(out)

                if self.version >= 44:
                    out = self.layer1f(out)
                    out = self.layer1g(out)

                    if self.version >= 56:
                        out = self.layer1h(out)
                        out = self.layer1i(out)

            out = self.avgPool(out)
            out = self.flatten(out)
            out = self.fc(out)
            out = self.softmax(out)

        return out


