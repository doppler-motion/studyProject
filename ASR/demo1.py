import torch
import torchaudio
from torch.nn import functional as F


def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


tf = torchaudio.transforms.MFCC(sample_rate=8000)


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, (1, 3), (1, 2), (0, 1)),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, 3, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, (8, 1)),
        )

    def forward(self, x):
        h = self.seq(x)
        return h.reshape(-1, 8)


if __name__ == '__main__':

    data_loader = torch.utils.data.DataLoader(torchaudio.datasets.YESNO('.', download=True), batch_size=1, shuffle=True)

    net = Net()
    opt = torch.optim.Adam(net.parameters())

    loss_fn = torch.nn.MSELoss()

    for epoch in range(100000):
        datas = []
        tags = []
        for data, _, tag in data_loader:
            tag = torch.stack(tag, dim=1).float()
            specgram = normalize(tf(data))
            datas.append(F.adaptive_avg_pool2d(specgram, (32, 256)))
            tags.append(tag)



        specgrams = torch.cat(datas, dim=0)
        tags = torch.cat(tags, dim=0)
        y = net(specgrams)
        loss = loss_fn(y, tags)

        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
