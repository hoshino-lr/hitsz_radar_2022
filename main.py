"""
雷达主程序
"""
import datetime
import multiprocessing
import os
import signal
from multiprocessing import Process
import numpy as np
import sys
from camera.cam import Camera
from net.network_pro import Predictor
import time
from PyQt5 import QtWidgets
import cv2 as cv
from camera.cam import Camera
from resources.config import DEBUG, config_init, logger, USEABLE
from mul_manager.pro_manager import sub, pub
from mapping.mainEntry import Mywindow
from PyQt5.QtCore import QTimer

app = QtWidgets.QApplication(sys.argv)
myshow = Mywindow()

PRE1 = None
PRE2 = None


def process_detect(event, que, event_Close, name):
    # 多线程接收写法
    logger.info(f"子线程开始:{name}")
    print(f"子线程开始:{name}")
    predictor1 = Predictor(name)
    cam = Camera(name, DEBUG)
    count = 0
    t1 = time.time()

    while not event_Close.is_set():
        frame = cam.get_img()
        if frame is not None:
            im1 = frame.copy()
            res = predictor1.detect_cars(frame)
            pub(event, que, res)
            count += 1
            t2 = time.time()
            if t2 - t1 > 1:
                fps = float(count) / (t2 - t1)
                print(f'fps: {fps}')
                t1 = time.time()
                count = 0
    print("子进程准备退出")
    predictor1.stop()
    print("子进程退出完成")


def main_init():
    global PRE1
    global PRE2
    logger.info("初始化程序开始运行")
    config_init()
    logger.info("初始化成功")
    logger.info("开始主线程")
    if USEABLE['Lidar']:
        pass
    if USEABLE['serial']:
        pass


def spin_once():
    global PRE1
    global PRE2
    pic_left = None
    pic_right = None
    if USEABLE['cam_left']:
        res_left = sub(event_left, que_left)
    if USEABLE['cam_right']:
        res_right = sub(event_right, que_right)
    if not myshow.view_change:
        pic_info = res_left[1]
    else:
        pic_info = res_right[1]
    if pic_info is not None:
        myshow.set_image(pic_info, "main_demo")
    if myshow.close:
        event_close.set()
        if USEABLE['cam_left']:
            PRE1.join(timeout=5)
            os.kill(PRE1.pid, signal.SIGINT)
        if USEABLE['cam_right']:
            PRE2.join(timeout=5)
            os.kill(PRE2.pid, signal.SIGINT)
        sys.exit(app.exec_())


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    date = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # ui
    que_left = multiprocessing.get_context('spawn').Queue()
    event_right = multiprocessing.get_context('spawn').Event()
    que_right = multiprocessing.get_context('spawn').Queue()
    event_left = multiprocessing.get_context('spawn').Event()
    event_close = multiprocessing.get_context('spawn').Event()
    if USEABLE['cam_left']:
        PRE1 = Process(target=process_detect, args=(event_left, que_left, event_close, 'cam_left',))
        PRE1.daemon = True
        PRE1.start()
    if USEABLE['cam_right']:
        PRE1 = Process(target=process_detect, args=(event_right, que_right, event_close, 'cam_right',))
        PRE1.daemon = True
        PRE1.start()
    timer_main = QTimer()
    timer_serial = QTimer()
    myshow.show()
    main_init()
    timer_main.timeout.connect(spin_once)
    id = timer_main.start(0)
    sys.exit(app.exec_())
