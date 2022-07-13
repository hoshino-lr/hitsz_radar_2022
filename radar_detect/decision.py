# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : decision.py
# Time       ：2022/7/1 下午10:28
# Author     ：author name
# version    ：python 3.8
# Description：
"""
import time

import numpy as np
from mapping.drawing import draw_message
from config import enemy_color


class decision_tree(object):

    def __init__(self, text_api):
        self.init_flag = False

        self._enemy_position = np.zeros((5, 2))  # 敌方位置
        self._our_position = np.zeros((5, 2))  # 我方位置
        self._enemy_blood = np.zeros((8, 1))  # 敌方血量
        self._our_blood = np.zeros((8, 1))  # 我方血量
        self._last_blood = np.zeros((8, 1))  # 我方血量
        self._our_blood_max = np.ones((8, 1))  # 我方血量上限
        self._rp_alarming = {}
        self._state = 0  # 增益
        self.text_api = text_api
        self._car_decision = np.zeros((5, 2))
        self._last_decision = np.zeros((5, 2))
        if enemy_color:
            self.start_x = 28.
        else:
            self.start_x = 0
        self._start_time = 0.
        self._engineer_flag = False
        self._fly_flag = False
        self._tou_flag = False
        self._tou_qsz = False

    def decision_alarm(self):
        # abandon
        if self.init_flag:
            self._engineer_alarm()
            self._target_attack()
            self._run_alarm()
            self._tou_alarm()

        else:
            self.clear_state()

    def _engineer_alarm(self):
        flag = False
        eng_pos = self._enemy_position[1]
        if eng_pos.all():
            x_abs = abs(eng_pos[0] - self.start_x)
            if x_abs < 10:
                flag = True
        for r in self._rp_alarming.keys():
            # 格式解析
            _, _, _, location, _ = r.split('_')
            if location in ["我方公路区", "哨兵前盲道"]:
                if 2 in self._rp_alarming[r]:
                    flag = True
        self._engineer_flag = flag

    def _run_alarm(self):
        if self._state == 2:
            pass
        elif self._state == 4:
            decision = self._our_blood / self._our_blood_max <= 0.4
            for i in range(5):
                if decision[i][0]:
                    self._car_decision[i][0] = 2
        else:
            decision = self._our_blood / self._our_blood_max <= 0.25
            for i in range(5):
                if decision[i][0]:
                    self._car_decision[i][0] = 2

    def _tou_alarm(self):
        self.tou_flag = False
        car_flag = False
        for i in range(5):
            if self._our_position[i][0] != 0 and i != 1:
                distance = abs(self._enemy_position[i][0] - self.start_x)
                if distance <= 8:
                    car_flag = True
                    break
        if not car_flag:
            for i in range(5):
                if self._enemy_position[i][0] != 0 and i != 1:
                    distance = abs(self._enemy_position[i][0] - self.start_x)
                    if distance <= 8:
                        self.tou_flag = True
                        break

    def _target_attack(self):
        choose = np.bitwise_and(self._enemy_blood <= 100,
                                np.bitwise_and(self._enemy_position != 0,
                                               abs(self._enemy_position - self.start_x) <= 14)[:5, :])
        if choose.any():
            best_choose = 0
            for i in range(5):
                if choose[i][0] and self._enemy_blood[i] < self._enemy_blood[best_choose] and i != 1:
                    best_choose = i
            for i in self._car_decision:
                i[0] = best_choose + 1

    def _blood_alarm(self):
        blood_minus = self._last_blood - self._our_blood
        if blood_minus[5] < 0:
            self._tou_flag = True
        if -100 > blood_minus[6] > -600:
            self._tou_qsz = True
        if 0 > blood_minus[7] > -1000:
            self._tou_flag = True

    def _fly_alarm(self):
        if self._fly_flag:
            for i in range(self._our_blood.shape[0]):
                self._car_decision[i][0] = 3

    def clear_state(self):
        self._enemy_position = np.zeros((5, 2))  # 敌方位置
        self._our_position = np.zeros((5, 2))  # 我方位置
        self._enemy_blood = np.zeros((8, 1))  # 敌方血量
        self._our_blood = np.zeros((8, 1))  # 我方血量
        self._last_blood = np.zeros((8, 1))  # 我方血量
        self._our_blood_max = np.ones((8, 1))  # 我方血量上限
        self._rp_alarming = {}
        self._state = 0  # 增益
        self._car_decision = np.zeros((5, 2))
        self._last_decision = np.zeros((5, 2))
        self._engineer_flag = False
        self._fly_flag = False

    def update_start_time(self, t_: float):
        self._start_time = t_

    def update_position(self, our_position: np.ndarray, enemy_position: np.ndarray):
        self._our_position = our_position
        self._enemy_position = enemy_position

    def update_blood(self, our_blood: np.ndarray, enemy_blood: np.ndarray):
        self._our_blood = our_blood
        self._enemy_blood = enemy_blood

    def update_state(self, state: int, fly_flag: bool):
        self._state = state
        self._fly_flag = fly_flag
