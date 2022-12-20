"""
日志类
搬迁自原来的main_v4.py
copied by: 陈希峻 2022/12/20
"""

import logging
import os
import sys


class LOGGER(object):
    """
    logger 类
    """

    def __init__(self, stdout):
        # 创建一个logger
        import time
        logger_name = time.strftime('%Y-%m-%d %H-%M-%S')
        self.terminal = sys.stdout
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        # 创建一个handler，用于写入日志文件
        log_path = os.path.abspath(os.getcwd()) + "/logs/"  # 指定文件输出路径
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        LogName = log_path + logger_name + '.log'  # 指定输出的日志文件名
        fh = logging.FileHandler(LogName, encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
        fh.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        fh.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.stdout = stdout

    def write(self, message):
        # self.terminal.write(message)
        self.stdout.write(message)
        if message == '\n':
            return
        if "[ERROR]" in message:
            self.logger.error(message)
        elif "[WARNING]" in message:
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def flush(self):
        self.stdout.flush()
        pass

    def __del__(self):
        pass
