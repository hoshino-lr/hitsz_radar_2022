"""
视频类
用于利用已录制视频进行调试
created by 陈希峻 2022/11/27

将视频流从 Camer_HK 中分离出来, 便于无驱动使用
edited by 陈希峻 2022/12/7
"""

import cv2
import numpy as np
import time
import config
from config import cam_config
from camera.cam import Camera


class Camera_Video(Camera):
    """
    视频模拟相机类
    """

    def __init__(self, type_, event_list=None):
        """
        @param type_: 相机左右类型
        @param event_list: 事件列表
        """
        self.__type = type_
        self.__camera_config = cam_config[self.__type]
        self.__size = self.__camera_config['size']
        self.__roi = self.__camera_config['roi']
        self.__img = np.ndarray((self.__size[1], self.__size[0], 3), dtype="uint8")
        self.cap = VideoCap(self.__camera_config["video_path"], self.__size, event_list)
        self.init_ok = True

    def get_img(self) -> (bool, np.ndarray):
        if self.init_ok:
            self.cap.update()
            result, self.__img = self.cap.get_frame()
            return result, self.__img
        else:
            # print("init is failed dangerous!!!")
            return False, self.__img

    def destroy(self) -> None:
        del self.cap
        self.init_ok = False


class VideoCap:
    """
    视频类

    实现了固定帧率的视频读取
    """
    def __init__(self, video_path, size, event_list):
        self.__size = size
        self.__img = np.ndarray((self.__size[1], self.__size[0], 3), dtype="uint8")
        print(f"打开视频: {video_path}")
        self.__cap = cv2.VideoCapture(video_path)
        if not self.__cap.isOpened():
            print("视频打开失败")
            return None
        self.__ori_spf = 1.0 / self.__cap.get(cv2.CAP_PROP_FPS)     # Seconds Per Frame
        self.__spf = self.__ori_spf
        self.__is_paused = False
        self.__event_list = event_list
        self.__last_time = time.time()                              # 时间戳 浮点又不是不能用

    def get_frame(self) -> (bool, np.ndarray):
        """
        获取一帧图像
        """
        while config.global_pause:
            time.sleep(0.1)
        result, frame = self.__cap.read()
        if result:
            self.__img = frame
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
