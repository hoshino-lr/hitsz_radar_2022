# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : eco_forecast.py
# Time       ：2022/7/1 下午10:44
# Author     ：author name
# version    ：python 3.8
# Description：
"""

# 经济预测类
import numpy as np
from mapping.drawing import draw_message


class eco_forecast(object):
    def __init__(self, text_api):
        self.ori_pic = np.array([])
        self.text_api = text_api
        self.init_ok = False

    def guess(self) -> int:
        guess_num = -1
        if not self.init_ok:
            pass
        else:
            pass
        return guess_num

    def eco_detect(self, pic) -> bool:
        # 用帧差法
        result = False
        if self.init_ok and isinstance(pic, np.ndarray):
            pass
        return False

    def update_ori(self, pic):
        self.ori_pic = pic
