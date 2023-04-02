"""
Pyrdr 接收线程
接受 C++ 部分的处理结果
created by 陈希峻 2022/12/22
"""

from functools import partial
from threading import Lock, Thread
from queue import Queue
from typing import Optional

import numpy as np

from pyrdr.client import ImageClient, ImageAndArmorClient
from config_type import RdrConfig, RdrReceive
from utils.fps_counter import FpsCounter
from abstraction.provider import GenericProvider
from loguru import logger
from service.abstract_service import StartStoppableTrait


class RdrThread(StartStoppableTrait):
    """pyrdr 接收线程"""

    def __init__(self, config: RdrConfig):
        logger.info(f"正在初始化 RdrThread")
        self._recv_armor: bool = config.net_process is RdrReceive
        self._client: Optional[ImageClient | ImageAndArmorClient] = None
        self._spinner: Optional[Thread] = None
        self._config = config
        self._frame_provider: GenericProvider[np.ndarray] = GenericProvider()
        # TODO: 类型不对应要改
        self._armor_provider: GenericProvider[tuple[np.ndarray, np.ndarray]] = GenericProvider()
        self._getter_increment = 0
        self._is_terminated = False
        self._fps_counter = FpsCounter()

    def get_fps_getter(self):
        """
        获取一个获取帧率的函数
        :return: 一个函数，调用该函数会返回帧率，保证不会阻塞
        """
        return lambda: self._fps_counter.fps

    def get_latest_frame_getter(self):
        """
        获取一个获取最新画面的函数
        :return: 一个函数，调用该函数会返回最新画面，假如未更新会阻塞到有新画面
        """
        identifier = self._getter_increment
        self._getter_increment += 1
        return partial(self._frame_provider.latest, identifier=identifier)

    def get_latest_armor_getter(self):
        """
        获取一个获取最新网络识别装甲的函数
        :return: 一个函数，调用该函数会返回最新画面，假如未更新会阻塞到有新装甲数据
        """
        identifier = self._getter_increment
        self._getter_increment += 1
        return partial(self._armor_provider.latest, identifier=identifier)

    def start(self):
        logger.info(f"正在启动连接到 {self._config.endpoint} 的线程")
        if self._recv_armor:
            self._client = ImageAndArmorClient(self._config.endpoint)
        else:
            self._client = ImageClient(self._config.endpoint)
        self._is_terminated = False
        self._spinner = Thread(target=self._spin, name=f"RdrThread-{self._config.endpoint}")
        self._spinner.start()

    def stop(self):
        """
        停止相机线程，会阻塞直到线程退出
        :return: 啥也不返回
        """
        # self._frame_provider.end()
        logger.info(f"正在停止连接到 {self._config.endpoint} 的线程")
        self._is_terminated = True
        if self._spinner is not None:
            self._spinner.join()
            self._spinner = None
            logger.info(f"停止了连接到 {self._config.endpoint} 的线程")
        else:
            logger.warning(f"连接到 {self._config.endpoint} 的线程不存在，是否已经停止了？")

    def __del__(self):
        if not self._is_terminated:
            logger.warning(f"连接到 {self._config.endpoint} 的线程未被正常停止！")
            self.stop()

    def _spin(self):
        while not self._is_terminated:
            self._fps_counter.update()
            # logger.info(f"正在等待 {self._config.endpoint} 的消息")
            msg = self._client.recv()
            if msg is None:
                logger.warning(f"等待 {self._config.endpoint} 的消息超时")
            else:
                if self._recv_armor:
                    # img, armor = self._client.recv()
                    img, armor = msg
                    self._frame_provider.push(img)
                    self._armor_provider.push(armor)
                else:
                    # img = self._client.recv()
                    img = msg
                    self._frame_provider.push(img)
                # logger.info(f"收到了 {self._config.endpoint} 的消息并推送了出去")

