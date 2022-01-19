"""
雷达主程序 use cv waitKey
created by 李龙 2022/1/15
"""
import multiprocessing
import os
import signal

import numpy as np
import sys
from net.network_pro import Predictor
from radar_detect.Linar import Radar
from radar_detect.reproject import Reproject
from radar_detect.location_alarm import Alarm
import time
from PyQt5 import QtWidgets
import cv2 as cv
from unuse.cam import Camera
from resources.config import DEBUG, LOGGER, USEABLE, enemy_color, test_region, cam_config
from mul_manager.pro_manager import sub, pub
from mapping.mainEntry import Mywindow
from PyQt5.QtCore import QTimer


class Radar_main(object):
    """
    雷达主进程
    """
    que_left = multiprocessing.get_context('spawn').Queue()
    event_right = multiprocessing.get_context('spawn').Event()
    que_right = multiprocessing.get_context('spawn').Queue()
    event_left = multiprocessing.get_context('spawn').Event()
    event_close = multiprocessing.get_context('spawn').Event()

    def __init__(self):
        self.logger = LOGGER(lambda x, t: myshow.set_text(x, t))
        self.text_api = lambda x, y, z: self.logger.add_text(x, y, z)
        self.show_api = lambda x: myshow.set_image(x, "main_demo")
        self.show_map = lambda x: myshow.set_image(x, "map")
        if USEABLE['cam_left']:
            self.PRE_left = multiprocessing.Process(target=process_detect, args=(
                self.event_left, self.que_left, self.event_close,
                'cam_left',))
            self.PRE_left.daemon = True
            self.repo_left = Reproject('cam_left')
        if USEABLE['cam_right']:
            self.PRE_right = multiprocessing.Process(target=process_detect,
                                                     args=(self.event_right, self.que_right, self.event_close,
                                                           'cam_right',))
            self.PRE_right.daemon = True
            self.repo_right = Reproject('cam_right')
        self.lidar = Radar('cam_right', text_api=self.text_api)
        if USEABLE['Lidar']:
            self.lidar.start()
        else:
            # 读取预定的点云文件
            self.lidar.preload()
        if USEABLE['serial']:
            pass
        self.loc_alarm = Alarm(enemy=enemy_color, api=self.show_map, touch_api=self.text_api,
                               region=test_region, debug=False)
        T = np.eye(4)
        rvec = cam_config['cam_left']['rvec']
        tvec = cam_config['cam_left']['tvec']
        T[:3, :3] = cv.Rodrigues(rvec)[0]  # 旋转向量转化为旋转矩阵
        T[:3, 3] = tvec.reshape(-1)  # 加上平移向量
        T = np.linalg.inv(T)  # 矩阵求逆
        self.loc_alarm.push_T(T, (T @ (np.array([0, 0, 0, 1])))[:3], 0)

    def start(self):
        self.PRE_left.start()

    def spin_once(self):
        while not myshow.close:
            pic_left = None
            pic_right = None
            if USEABLE['cam_left']:
                res_left = sub(self.event_left, self.que_left)
            if USEABLE['cam_right']:
                res_right = sub(self.event_right, self.que_right)
            result = [None, None]
            if res_left[0] is None:
                pass
            else:
                armors = res_left[0][:, [11, 13, 6, 7, 8, 9]]
                cars = res_left[0][:, [11, 0, 1, 2, 3]]
                result = self.repo_left.check(armors, cars)
                if result[0] is not None:
                    self.text_api.api_info(str(result[0].keys()))
            if not myshow.view_change:
                pic_info = res_left[1]
            else:
                pic_info = res_right[1]
            if pic_info is not None:
                self.repo_left.update(result[0], pic_info)
                myshow.set_image(pic_info, "main_demo")
            if res_left[0] is None:
                pass
            else:
                armors = res_left[0][:, [11, 6, 7, 8, 9]]
                armors[:, 3] = armors[:, 3] - armors[:, 1]
                armors[:, 4] = armors[:, 4] - armors[:, 2]
                armors = armors[np.logical_not(np.isnan(armors[:, 0]))]
                if armors.shape[0] != 0:
                    dph = self.lidar.detect_depth(rects=armors[:, 1:].tolist()).reshape(-1, 1)
                    x0 = (armors[:, 1] + armors[:, 3] / 2).reshape(-1, 1)
                    y0 = (armors[:, 2] + armors[:, 4] / 2).reshape(-1, 1)
                    xyz = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, dph], axis=1)
                    location = np.zeros((10, 4)) * np.nan
                    for i in xyz:
                        location[int(i[0]), :] = i
                    points = self.loc_alarm.update(location)
                    self.loc_alarm.show()
            self.text_api.

    def close(self):
        self.event_close.set()
        if USEABLE['cam_left']:
            self.PRE_left.join(timeout=5)
            os.kill(self.PRE_left.pid, signal.SIGINT)
        if USEABLE['cam_right']:
            self.PRE_left.join(timeout=5)
            os.kill(self.PRE_right.pid, signal.SIGINT)

def process_detect(event, que, event_Close, name):
    # 多线程接收写法
    print("子线程开始:{name}")
    predictor = Predictor(name)
    cam = Camera(name, DEBUG)
    count = 0
    t1 = time.time()
    while not event_Close.is_set():
        frame = cam.get_img()
        if frame is not None:
            im1 = frame.copy()
            res = predictor.detect_cars(frame)
            pub(event, que, res)
            count += 1
            t2 = time.time()
            if t2 - t1 > 1:
                fps = float(count) / (t2 - t1)
                print(f'fps: {fps}')
                t1 = time.time()
                count = 0

    print(f"相机网络子进程\t {name} 准备退出")
    predictor.stop()
    print(f"相机网络子进程\t {name} 退出完成")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # ui
    app = QtWidgets.QApplication(sys.argv)
    myshow = Mywindow()
    radar_main = Radar_main()
    radar_main.start()
    timer_serial = QTimer()  # 串口使用的线程
    myshow.show()
    radar_main.spin_once()
    radar_main.close()
    sys.exit(app.exec_())
