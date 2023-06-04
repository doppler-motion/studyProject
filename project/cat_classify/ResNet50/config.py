import os
import logging

# 设置log全局变量
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 定义日志的级别。Info显示当程序运行时期望的一些信息

# 生成log文件夹
log_path = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)

# 生成log文件
log_name = os.path.join(log_path, 'train_log')  # log文件名称定义为 train_log
fh = logging.FileHandler(log_name, mode='w')  # file handler 以 写 模式将文件分配至正确的目的地
fh.setLevel(logging.DEBUG)

# 声明日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")  # levelname指前面添加的'INFO'，message指下方添加的train_core1
fh.setFormatter(formatter)
logger.addHandler(fh)
