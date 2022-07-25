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
        self._position_2d = None
        self._rp_alarming = {}
        self._state = 0  # 增益
        self.text_api = text_api
        self._car_decision = np.zeros((6, 4)).astype(np.uint8)
        self._last_decision = np.zeros((6, 4)).astype(np.uint8)
        if enemy_color:
            self.start_x = 0
        else:
            self.start_x = 28.
        self._start_time = time.time()
        self._energy_time = time.time()
        self._guard_time = time.time()
        self._fly_numbers = np.array([])
        self._engineer_flag = False
        self._fly_flag = False
        self._tou_flag = False
        self._tou_qsz = False
        self._remain_time = 0

    def decision_alarm(self):
        # abandon
        if self.init_flag:
            self._engineer_alarm()
            self._target_attack()
            self._run_alarm()
            self._tou_alarm()
        else:
            self.clear_state()
        self.generate_information()

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
            if location in ["3号高地下我方盲道及公路区", "前哨站我方盲道"]:
                if 2 in self._rp_alarming[r]:
                    flag = True
        self._engineer_flag = flag

    def _run_alarm(self):
        if self._state == 2:
            pass
        elif self._state == 4:
            decision = (self._our_blood / self._our_blood_max <= 0.4).reshape(-1)[:6]
            self._car_decision[decision, 0] = 2
        else:
            decision = (self._our_blood / self._our_blood_max <= 0.25).reshape(-1)[:6]
            self._car_decision[decision, 0] = 2

    def _tou_alarm(self):
        self._tou_flag = False
        car_flag = False
        # for i in range(5):
        #     if self._our_position[i][0] != 0:
        #         distance = abs(self._enemy_position[i][0] - self.start_x)
        #         if distance <= 8:
        #             car_flag = True
        #             break
        if not car_flag:
            result = np.bitwise_and(self._enemy_position[:, 0] != 0,
                                    abs(self._enemy_position[:, 0] - self.start_x) <= 10)
            result[1] = False
            if result.any():
                self._tou_flag = True

    def _target_attack(self):
        choose = np.bitwise_and(self._enemy_blood[:5, 0] <= 100,
                                np.bitwise_and(self._enemy_position[:, 0] != 0,
                                               abs(self._enemy_position - self.start_x)[:, 0] <= 14))
        choose[1] = False
        if choose.any():
            valid_c = np.argwhere(choose == True).reshape(-1).tolist()
            best_c = np.argmin(self._enemy_blood[valid_c, 0])
            best_c = valid_c[best_c]
            self._car_decision[:, 1] = best_c + 1

    def _guard_decision(self):
        self._car_decision[5][2:] = [0, 0]
        for r in self._rp_alarming.keys():
            # 格式解析
            _, _, _, location, _ = r.split('_')
            if location in ["我方3号高地", "3号高地下我方盲道及公路区", "前哨站我方盲道"]:
                if self._rp_alarming[r].shape[0] != 0:
                    if location == "我方3号高地":
                        self._car_decision[5][2:] = [1, 1]
                    elif location == "前哨站我方盲道":
                        self._car_decision[5][2:] = [2, 1]
                    else:
                        self._car_decision[5][2:] = [2, 2]

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
            self._car_decision[:, 0] = 3

    def _show_energy(self):
        self.text_api(draw_message("remain_time", 1,
                                   ("{0}:{1}".format(self._remain_time // 60, self._remain_time % 60), (1400, 120)),
                                   "critical"))
        remain_time = int(time.time() - self._energy_time)
        if self._state == 0:
            return
        elif self._state == 2:
            text = "BIG"
            remain_time = 45 - remain_time
            level = "critical"
        elif self._state == 1:
            text = "SMALL"
            remain_time = 45 - remain_time
            level = "critical"
        elif self._state == 4:
            text = "BIG"
            remain_time = 45 - remain_time
            level = "critical"
        elif self._state == 3:
            text = "SMALL"
            remain_time = 45 - remain_time
            level = "critical"
        else:
            text = "UNABLE"
            remain_time = 30 - remain_time
            level = "info"
            self.text_api(draw_message("energy_time", 1,
                                       ("", (1600, 120)),
                                       "critical"))
        self.text_api(draw_message("energy_time", 1,
                                   ("{0}  {1}".format(text, remain_time), (1600, 120)),
                                   level))

    def clear_state(self):
        self._enemy_position = np.zeros((5, 2))  # 敌方位置
        self._our_position = np.zeros((5, 2))  # 我方位置
        self._enemy_blood = np.zeros((8, 1))  # 敌方血量
        self._our_blood = np.zeros((8, 1))  # 我方血量
        self._last_blood = np.zeros((8, 1))  # 我方血量
        self._our_blood_max = np.ones((8, 1))  # 我方血量上限
        self._position_2d = None
        self._rp_alarming = {}
        self._state = 0  # 增益
        self._car_decision = np.zeros((6, 4)).astype(np.uint8)
        self._last_decision = np.zeros((6, 4)).astype(np.uint8)
        self._engineer_flag = False
        self._fly_flag = False
        self._fly_numbers = np.array([])

    def update_serial(self, our_position: np.ndarray, our_blood: np.ndarray,
                      enemy_blood: np.ndarray, state: list, remain_time: float, high_light: bool):

        self._our_position = our_position
        self._our_blood = our_blood.reshape((8, 1))
        self._enemy_blood = enemy_blood.reshape((8, 1))
        self._stage, self._state, self._energy_time = state
        self._remain_time = remain_time
        self._high_light = high_light

    def generate_information(self):
        self._show_energy()
        if self._fly_flag:
            self.text_api(draw_message("fly", 0, f"FLY_{self._fly_numbers}", "critical"))
            if isinstance(self._position_2d, np.ndarray):
                count = self._position_2d[:, 11]
                arg = np.argwhere(count == self._fly_numbers)[0][0]
                self.text_api(
                    draw_message("fly_alarm", 2, self._position_2d[arg][0:4].astype(int).tolist(), "critical"))
        if self._engineer_flag:
            self.text_api(draw_message("engineer", 0, "Engineer", "critical"))
        if self._tou_qsz:
            self.text_api(draw_message("tou_qsz", 0, "HERO QSZ!!!", "warning"))
        if self._tou_flag:
            self.text_api(draw_message("tou", 0, "DEFEND!!!", "critical"))
        self._guard_decision()
        if isinstance(self._position_2d, np.ndarray):
            if (self._remain_time < 600 or self._high_light) and self._position_2d.size > 0:
                arg = np.argwhere(self._position_2d[:, 11] == 1).reshape(-1, 1)
                if arg.size > 0:
                    self.text_api(
                        draw_message("hero", 2, self._position_2d[arg[0][0]][0:4].astype(int).tolist(), "critical"))
                # else:
                #     if self._high_light:
                #         self.text_api(draw_message(f"{i[11]}", 2, i[0:4].astype(int).tolist(), "warning"))

    def update_information(self, enemy_position: np.ndarray, fly_flag: bool, fly_numbers: np.ndarray, hero_r3: bool,
                           position_2d):
        self._enemy_position = enemy_position
        self._hero_r3 = hero_r3
        self._fly_flag = fly_flag
        self._fly_numbers = fly_numbers
        self._position_2d = position_2d
        self.init_flag = True

    def get_decision(self) -> np.ndarray:
        return self._car_decision
