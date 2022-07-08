# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : decision.py
# Time       ：2022/7/1 下午10:28
# Author     ：author name
# version    ：python 3.8
# Description：
"""
import numpy as np
from mapping.drawing import draw_message


class decision_tree(object):

    def __init__(self, text_api):
        self._enemy_position = np.array([])  # 敌方位置
        self._our_position = np.array([])  # 我方位置
        self._enemy_blood = np.array([])  # 敌方血量
        self._out_blood = np.array([])  # 我方血量
        self._state = False  # 增益
        self.text_api = text_api

    def decision_alarm(self):
        # abandon
        self.clear_state()
        self._engineer_alarm()
        self._target_attack()
        self._run_alarm()
        self._fly_alarm()

    def _engineer_alarm(self):
        pass

    def _run_alarm(self):
        pass

    def _target_attack(self):
        pass

    def _fly_alarm(self):
        pass

    def clear_state(self):
        pass

    def update(self, last_points, now_points, state: int):
        pass
