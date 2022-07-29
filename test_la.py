# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : test.py
# Time       ：2022/7/27 下午7:41
# Author     ：author name
# version    ：python 3.8
# Description：
"""
import pickle
import pickle as pkl
import time
from radar_detect.solve_pnp import SolvePnp
from radar_detect.location_alarm import Alarm
from config import INIT_FRAME_PATH, \
    MAP_PATH, map_size, USEABLE, \
    enemy_color, cam_config, using_video, enemy2color
import cv2 as cv

if __name__ == "__main__":
    touch_api = lambda x: cv.imshow("map", x)
    loc_alarm = Alarm(enemy=enemy_color, api=touch_api, touch_api=None,
                      state_=USEABLE['locate_state'], _save_data=False, debug=False)  # 绘图及信息管理类
    try:
        sp = SolvePnp(None)  # pnp解算
        sp.read(f'cam_left_{enemy2color[enemy_color]}')
        loc_alarm.push_T(sp.rvec, sp.tvec, 0)
    except Exception as e:
        print(f"[ERROR] {e}")
        loc_alarm.push_T(cam_config["cam_left"]["rvec"], cam_config["cam_left"]["tvec"], 0)
    try:
        with open("resources/location_data.dat", "rb") as f:
            while True:
                location_left, location_right, rp_alarming = pickle.load(f)
                loc_alarm.two_camera_merge_update(location_left, location_right, rp_alarming)
                loc_alarm.check()
                loc_alarm.show()
                cv.waitKey(30)
    except Exception as e:
        print(e)
