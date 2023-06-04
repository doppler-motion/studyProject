import sys, os
import numpy as np

project_path = os.path.dirname(__file__)[
               :os.path.dirname(__file__).index("scripts")] + "/scripts"  # scripts 是我project的名字，根据project要改名字
sys.path.append(project_path)
from common_utils.common_log import logger

from common.layers import MatMul

if __name__ == "__main__":
    c = np.array([1, 0, 0, 0, 0, 0, 0])
    W = np.random.randn(7, 3)
    layer = MatMul(W)
    h = layer.forward(c)
    logger.info(h)
