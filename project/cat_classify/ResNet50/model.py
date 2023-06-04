import torch.nn as nn
import torchvision


class myRes(nn.Module):
    def __init__(self):
        super(myRes, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=False)

    def forward(self, x):
        x = self.resnet(x)
        return x
