import os.path
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.log_config import logger


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout=1):
        dout *= self.out(1 - self.out)
        return None



if __name__ == "__main__":
    x = 11
    y = sigmoid(x)
    print(y)
