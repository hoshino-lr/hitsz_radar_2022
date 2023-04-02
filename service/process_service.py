"""
处理线程
在这里将处理来自相机的图像，结合雷达等信息提供侦测预警
created by 陈希峻 2022/12/22
"""

import time
import numpy as np
from threading import Thread
from typing import Callable
from net.network_pro import Predictor
from abstraction.provider import GenericProvider
from utils.fps_counter import FpsCounter
from loguru import logger
from service.abstract_service import StartStoppableTrait


class ProcessService(StartStoppableTrait):
    def __init__(self, img_size: tuple[int, int], frame_lambda: Callable[[], np.ndarray], name: str = "构建时未填写"):
        self._frame_lambda = frame_lambda
        self._name = name
        self._img_size = img_size
        self._predictor = None
        self._is_terminated = True
        self._thread = None
        self._fps_counter = FpsCounter()

        self._self_increment_identifier = 0
        # TODO: 为什么是 tuple[np.ndarray, np.ndarray]
        self._provider: GenericProvider[np.ndarray] = GenericProvider()

    def get_net_data_provider(self, timeout=1000):
        identifier = self._self_increment_identifier
        self._self_increment_identifier += 1
        return lambda: self._provider.latest(timeout, identifier)

    def start(self):
        self._predictor = Predictor(self._name, self._img_size)
        self._is_terminated = False
        self._fps_time = time.time()
        self._thread = Thread(target=self._spin, name=self._name)
        self._thread.start()
        logger.info(f"为 {self._name} 运行神经网络的线程已启动")
        pass

    def stop(self):
        self._is_terminated = True
        self._thread.join()
        self._predictor.stop()
        logger.info(f"为 {self._name} 运行神经网络的线程已停止")
        pass

    def __del__(self):
        self.stop()
        self._thread.join()
        pass

    # @property
    # def is_terminate(self):
    #     return self._is_terminated

    # @property
    # def fps(self):
    #     return self._fps

    # def _fps_update(self):
    #     """
    #     更新帧率
    #     """
    #     if self._fps_count >= 10:
    #         self._fps = self._fps_count / (time.time() - self._fps_time)
    #         self._fps_count = 0
    #         self._fps_time = time.time()
    #     else:
    #         self._fps_count += 1

    def get_fps_getter(self):
        return lambda: self._fps_counter.fps

    def _spin(self):
        # logger.info(f"子线程开始: {self._name}")
        while not self._is_terminated:
            frame = self._frame_lambda()
            if frame is not None:
                res = self._predictor.detect_cars(frame)
                self._provider.push(res)
                self._fps_counter.update()
                # self._fps_update()
