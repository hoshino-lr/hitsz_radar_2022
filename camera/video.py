"""
视频类
用于利用已录制视频进行调试
created by 陈希峻 2022/11/27
"""

import cv2
import numpy as np
import time
import config


class VideoCap:
    """
    视频类

    实现了固定帧率的视频读取
    """
    def __init__(self, video_path, size, roi, event_list):
        self.__size = size
        self.__roi = roi
        self.__img = np.ndarray((self.__size[1], self.__size[0], 3), dtype="uint8")
        self.__cap = cv2.VideoCapture(video_path)
        self.__ori_spf = 1.0 / self.__cap.get(cv2.CAP_PROP_FPS)     # Seconds Per Frame
        self.__spf = self.__ori_spf
        self.__is_paused = False
        self.__event_list = event_list
        self.__last_time = time.time()                              # 时间戳 浮点又不是不能用

    def get_frame(self) -> [bool, np.ndarray]:
        """
        获取一帧图像
        """
        while config.global_pause:
            time.sleep(0.1)
        result, frame = self.__cap.read()
        if result:
            self.__img = cv2.copyMakeBorder(
                frame[self.__roi[1]:self.__roi[3] + self.__roi[1], :, :], 0,
                self.__roi[1],
                0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if self.__last_time + self.__spf > time.time():
            time.sleep(self.__last_time + self.__spf - time.time())
        self.__last_time = time.time()
        return bool(result), self.__img

    def update(self):
        self._check_and_set('speed', lambda: self.set_speed(config.global_speed))

    def _check_and_set(self, name, func):
        """
        检查事件并执行
        :param name: 事件名
        :param func: 待执行函数
        :return:
        """
        if self.__event_list is not None and name in self.__event_list and self.__event_list[name].is_set():
            func()
            self.__event_list[name].clear()

    def set_speed(self, speed: float) -> None:
        """
        设置播放速度
        """
        self.__spf = self.__ori_spf / speed

    def __del__(self):
        if isinstance(self.__cap, cv2.VideoCapture):
            self.__cap.release()
