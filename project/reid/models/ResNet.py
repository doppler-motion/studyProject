from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from IPython import embed


# __all__ = ['ResNet50', 'ResNet101', 'ResNet50M']

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(weights=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048  # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)

        # 做归一化
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)

        if not self.training:
            return f

        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


if __name__ == "__main__":
    resnet50 = ResNet50(152)
    img = torch.Tensor(32, 3, 256, 128)
    f = resnet50(img)
    # print(resnet50)
