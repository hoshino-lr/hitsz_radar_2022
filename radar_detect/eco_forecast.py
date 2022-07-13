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
import time
from config import enemy_color
import numpy as np
from mapping.drawing import draw_message
import cv2 as cv


class eco_forecast(object):
    def __init__(self, text_api):
        self.ori_pic = np.array([])
        self.cut = self.cut
        self.text_api = text_api
        self._init_flag = False
        self._intensity_bound = 50
        self._detect_threshold = 0.3
        self._dis_weight = 0.6
        self._time_weight = 0.4
        self._fore_threshold = 5
        if enemy_color:
            self._start_dis = 0
        else:
            self._start_dis = 28.
        self._last_Dtime = time.time()
        self._detect_frequency = 5

    def _guess(self, detect_message: np.ndarray) -> None:
        """

        Args:
            detect_message: [car_num[last_x, last_y, time]]

        Returns:

        """
        if time.time() - self._last_Dtime > self._detect_frequency:
            self._last_Dtime = time.time()
        else:
            return
        guess_num = -1
        score = np.zeros(3)
        now_time = time.time()
        if not self._init_flag:
            for i in range(score.size):
                score[i] = self._get_score(detect_message[i], now_time)
            if score[np.argmax(score)] > self._fore_threshold:
                guess_num = np.argmax(score) + 3
            else:
                guess_num = 0
        else:
            pass
        if guess_num != -1 and guess_num != 0:
            message = draw_message("eco_forecast", 0, f"检测到敌方加弹，推测为{guess_num}号", "critical")
            self.text_api(message)
        elif guess_num == 0:
            message = draw_message("eco_forecast", 0, f"检测到敌方加弹", "critical")
            self.text_api(message)

    def _get_score(self, sub_message: np.ndarray, now_time) -> float:
        # 距离判断，仅根据x
        distance = abs(self._start_dis - sub_message[0])
        if distance > 14:
            d_score = 0
        elif distance > 8:
            d_score = 5
        else:
            d_score = 10

        # 时间判断
        period = now_time - sub_message[2]
        if period > 15:
            t_score = 10
        elif period > 10:
            t_score = 5
        elif period > 5:
            t_score = 2.5
        else:
            t_score = 0
        score = t_score * self._time_weight + d_score * self._dis_weight
        return score

    def eco_detect(self, pic, detect_message: np.ndarray) -> None:
        # 用帧差法
        detect_flag = False
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        if self._init_flag and isinstance(pic, np.ndarray):
            cut_pic = pic[self.cut[0]:(self.cut[0] + self.cut[2]), self.cut[1]:(self.cut[1] + self.cut[3])].copy()
            gray_pic = cv.cvtColor(cut_pic, cv.COLOR_BGR2GRAY)
            current_frame_gray = cv.GaussianBlur(gray_pic, (7, 7), 0)
            frame_diff = cv.absdiff(current_frame_gray, self.ori_pic)  # 进行帧差
            _, frame_diff = cv.threshold(frame_diff, self._intensity_bound, 255, cv.THRESH_BINARY)
            frame_diff = cv.erode(frame_diff, kernel)
            frame_diff = cv.dilate(frame_diff, kernel)
            white_pixels = float(np.sum(frame_diff[frame_diff == 255])) / frame_diff.size
            if white_pixels > self._detect_threshold:
                detect_flag = True
        if detect_flag:
            self._guess(detect_message)

    def update_ori(self, pic: np.ndarray, cut: list):
        """
        Args:
            pic:
            cut: xywh
        """
        cut_pic = pic[cut[0]:(cut[0] + cut[2]), cut[1]:(cut[1] + cut[3])].copy()
        gray_pic = cv.cvtColor(cut_pic, cv.COLOR_BGR2GRAY)
        self.ori_pic = cv.GaussianBlur(gray_pic, (7, 7), 0)
        self.cut = cut
        self._init_flag = True
