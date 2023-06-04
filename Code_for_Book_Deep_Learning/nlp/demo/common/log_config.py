import logging
import logging.handlers
import os
import time


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    white = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(name)s:%(filename)s:%(lineno)d : %(levelname)s --> %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Log(object):
    def __init__(self, logger=None, log_cate="daily"):
        """指定保存日志的路径，日志级别，以及调用文件，将日志存入到指定的文件中"""

        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        self.log_time = "common"  # time.strftime("%Y_%m_%d")

        # file_dir = os.getcwd() + "/../log"
        # if not os.path.exists(file_dir):
        #     os.mkdir(file_dir)

        self.log_path = os.path.dirname(__file__)[:os.path.dirname(__file__).index("scripts")] + "scripts/log"  # 文件路径
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        self.log_name = log_cate + "_" + self.log_time + ".log"  # 文件名字

        self.log_file_name = os.path.join(self.log_path, self.log_name).replace("\\", "/")  # 可直接修改此处的path

        # fh = logging.FileHandler(self.log_file_name, "a", encoding="utf-8")  # 文件输出
        fh = logging.handlers.RotatingFileHandler(self.log_file_name,
                                                  mode="a",
                                                  encoding="utf-8",
                                                  maxBytes=10 * 1024 * 1024,
                                                  backupCount=5)  # 按照大小自动切割文件
        fh.setLevel(logging.INFO)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()  # 控制台输出
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s %(name)s:%(filename)s:%(lineno)d : %(levelname)s --> %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(CustomFormatter())

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 添加下面一句，添加日志后移除句柄
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)
        # 关闭打开的文件
        fh.close()
        ch.close()

    def get_logger(self):
        return self.logger


logger = Log(__name__).get_logger()

# print(os.path.dirname(__file__)[:os.path.dirname(__file__).index("scripts")] + "scripts/log")