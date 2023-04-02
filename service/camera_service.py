import numpy as np
from functools import partial
from loguru import logger
from threading import Thread
from typing import Optional

from abstraction.provider import GenericProvider
from config_type import HikCameraConfig
from service.abstract_service import StartStoppableTrait
from utils.fps_counter import FpsCounter

ERR_TOLERANCE = 10


class HikCameraService(StartStoppableTrait):
    """
    海康机器人相机线程封装
    """

    def __init__(self, name: str, config: HikCameraConfig):
        logger.info(f"正在初始化相机 {name}")
        self._err_cnt = 0
        from camera.cam_hk_v3 import Camera_HK
        self._name = name
        self._camera: Optional[Camera_HK] = None
        self._spinner: Optional[Thread] = None
        self._config = config
        self._frame_provider: GenericProvider[np.ndarray] = GenericProvider()
        self._getter_increment = 0
        self._is_terminated = False

        self._fps_counter = FpsCounter()

    def get_latest_frame_getter(self, timeout: int | None = 1000):
        """
        获取一个获取最新画面的函数
        :return: 一个函数，调用该函数会返回最新画面，假如未更新会阻塞到有新画面
        """
        identifier = self._getter_increment
        self._getter_increment += 1
        # return lambda: self._frame_provider.latest(timeout, identifier)
        return partial(self._frame_provider.latest, timeout=timeout, identifier=identifier)
    
    def get_fps_getter(self):
        """
        获取一个获取帧率的函数
        :return: 一个函数，调用该函数会返回帧率，保证不会阻塞
        """
        return lambda: self._fps_counter.fps  # 写法是安全的吗？仅随手在 REPL 里试验过

    def start(self):
        logger.info(f"正在启动相机 {self._name} 的线程")
        from camera.cam_hk_v3 import Camera_HK
        self._camera = Camera_HK(self._config)
        self._is_terminated = False
        self._spinner = Thread(target=self._spin, name=f"CameraThread-{self._name}")
        self._spinner.start()

    def stop(self):
        """
        停止相机线程，会阻塞直到线程退出
        :return: 啥也不返回
        """
        # self._frame_provider.end()
        logger.info(f"正在停止相机 {self._name} 的线程")
        self._is_terminated = True
        if self._spinner is not None:
            self._spinner.join()
            self._spinner = None
        if self._camera is not None:
            self._camera.destroy()
            self._camera = None

    def __del__(self):
        if not self._is_terminated:
            logger.warning(f"相机 {self._name} 非正常退出")
            self.stop()

    def _spin(self):
        while not self._is_terminated:
            result, frame = self._camera.get_img()
            if result:
                self._fps_counter.update()
                self._frame_provider.push(frame)
            else:
                self._mark_error()

    def _mark_error(self):
        """
        错误计数
        """
        self._err_cnt += 1
        if self._err_cnt >= ERR_TOLERANCE:
            self._err_cnt = 0
            self._on_error()

    def _on_error(self):
        """
        错误重启
        """
        logger.warning(f"相机 {self._name} 错误达到 {ERR_TOLERANCE}，重启")
        self.stop()
        self.start()

