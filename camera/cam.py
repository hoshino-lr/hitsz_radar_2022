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
