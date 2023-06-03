# https://zhuanlan.zhihu.com/p/482229114

import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from utils.model import M5
from utils.if_cuda_available import device
from utils.dataloader import test_loader, train_loader
from utils.data_process import transformed, transform, labels
from utils.train import train
from utils.test import test

model = M5(n_input=transformed.shape[0], n_output=len(labels))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# reduce the learning after 20 epochs by a factor of 10
log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

# Let's plot the training loss versus the number of iteration.
plt.plot(losses)
plt.title("training loss")
plt.show()
