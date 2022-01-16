"""
雷达主程序
"""

from camera.cam import Camera
from net.network_pro import Predictor
from resources.config import DEBUG, config_init, logger, USEABLE
from mapping.mainEntry import Mywindow
from radar_detect.reproject import Reproject

from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets
import logging
import cv2 as cv
import numpy as np
import sys


def spin_once():
    pic_left = None
    pic_right = None
    res_left, res_right = None,None
    if USEABLE['cam_left']:
        frame = cam.get_img()
        if frame is not None:
            im1 = frame.copy()
            res_left,frame = Predictor1.detect_cars(frame)
    if USEABLE['cam_right']:
        frame = cam.get_img()
        if frame is not None:
            im1 = frame.copy()
            res_right,frame = Predictor1.detect_cars(frame)
    result = [None, None]
    if res_left is None:
        pass
    else:
        armors = res_left[:, [11, 13, 6, 7, 8, 9]]
        cars = res_left[:,[11, 0, 1, 2, 3]]
        result = repo_left.check(armors, cars)
        print(result)
    if not myshow.view_change:
        pic_info = frame
    else:
        pic_info = frame
    if pic_info is not None:
        repo_left.update(result[0],pic_info)
        myshow.set_image(pic_info, "main_demo")
    if myshow.close:
        if USEABLE['cam_left']:
            Predictor1.stop()
        if USEABLE['cam_right']:
            Predictor1.stop()
        sys.exit(app.exec_())


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # ui
    app = QtWidgets.QApplication(sys.argv)
    myshow = Mywindow()
    repo_left = Reproject(DEBUG,"cam_left")
    timer_main = QTimer()
    timer_serial = QTimer()
    myshow.show()
    logger.info("初始化程序开始运行")
    config_init()
    if USEABLE['cam_left']:
        Predictor1 = Predictor('cam_left')
        cam = Camera('cam_left', True)
    if USEABLE['cam_right']:
        Predictor1 = Predictor('cam_right')
        cam = Camera('cam_right', True)
    if USEABLE['Lidar']:
        pass
    if USEABLE['serial']:
        pass
    timer_main.timeout.connect(spin_once)
    timer_main.start(0)
    sys.exit(app.exec_())
