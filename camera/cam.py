# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : cam.py
# Time       ：2022/4/12 下午4:36
# Author     ：李龙
# version    ：python 3.8
# Description：
"""
from abc import ABC, abstractmethod
import numpy as np

class Camera(ABC):
    """
    相机类
    """
    camera_ids = None
    init_ok = True
    __gain = 15
    __exposure = 5000.0

    @abstractmethod
    def __init__(self, type_, event_list=None):
        pass

    @abstractmethod
    def get_img(self):
        return NotImplemented

    @abstractmethod
    def destroy(self):
        return NotImplemented


def create_camera(_type, using_video:bool, event_list=None) -> Camera:
    '''
    创建相机
    :param _type:
    :param using_video: 是否使用视频
    :param event_list: 事件列表
    '''
    if using_video:
        from camera.cam_video import Camera_Video
        return Camera_Video(_type, event_list)
    else:
        from camera.cam_hk_v3 import Camera_HK
        return Camera_HK(_type, event_list)
