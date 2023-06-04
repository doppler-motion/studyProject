import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride=1, downsampling=None):
        super(ResidualBlock, self).__init__()
        self.downsampling = downsampling

        self.net = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannels)
        )

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsampling is None else self.downsampling(x)

        out += residual
        return F.relu(out)


class Bottleneck(nn.Module):
    def __init__(self, inplaces, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.downsampling = downsampling
        self.expansion = expansion

        self.net = nn.Sequential(
            nn.Conv2d(inplaces, places, stride=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places * self.expansion, stride=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(places * self.expansion)
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplaces, places * self.expansion, stride=stride, kernel_size=1, bias=1),
                nn.BatchNorm2d(places * self.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if not self.downsampling else self.downsample(x)
        out = self.net(x)

        out += residual
        return self.relu(out)


class ResNetBase1(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(ResNetBase1, self).__init__()

        # 前面的几层，转换图像
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 重复的layer，resnet34每layer分别有3， 4， 6， 3个residual block
        self.layer1 = self._make_layer(64, 64, block[0])
        self.layer2 = self._make_layer(64, 128, block[1], stride=2)
        self.layer3 = self._make_layer(128, 256, block[2], stride=2)
        self.layer4 = self._make_layer(256, 512, block[3], stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inplaces, places, block_num, stride=1):
        layers = []

        shortcut = nn.Sequential(
            nn.Conv2d(inplaces, places, 1, stride, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU()
        )

        layers.append(ResidualBlock(inplaces, places, stride, shortcut))

        for _ in range(1, block_num):
            layers.append(ResidualBlock(places, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class ResNetBase2(nn.Module):
    def __init__(self, block, num_classes=1000, expansion=4):
        super(ResNetBase2, self).__init__()
        self.expansion = expansion

        # 前面几层，转换图像
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 重复的layer：分别有 3， 4， 6， 3个bottleneck
        self.layer1 = self._make_layer(64, 64, 1, block[0])
        self.layer2 = self._make_layer(256, 128, 2, block[1])
        self.layer3 = self._make_layer(512, 256, 2, block[2])
        self.layer4 = self._make_layer(1024, 512, 2, block[3])

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        # 分类用的全连接层
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 1)
                nn.init.uniform_(m.bias, 0)

    def _make_layer(self, inplaces, places, stride, block_num):
        layers = []
        layers.append(Bottleneck(inplaces, places, stride, downsampling=True))

        for _ in range(1, block_num):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18():
    return ResNetBase1([2, 2, 2, 2])


def resnet34():
    return ResNetBase1([3, 4, 6, 3])


def resnet50():
    return ResNetBase2([3, 4, 6, 3])


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model34 = resnet34()
    model34.to(device)
    summary(model34, (3, 224, 224))

    model50 = resnet50()
    data = torch.randn(3, 3, 224, 224)
    out = model50(data)
    print(out.shape)
