"""
雷达主程序
created by 李龙 2022/1/15
"""
import sys
import time
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
import cv2 as cv
import numpy as np
import multiprocessing
import threading
import serial

from camera.cam_new import Camera
from net.network_pro import Predictor
from radar_detect.Linar import Radar
from radar_detect.reproject import Reproject
from radar_detect.location_alarm import Alarm
from resources.config import DEBUG, LOGGER, USEABLE, enemy_color, test_region, cam_config
from mul_manager.pro_manager import sub, pub
from mapping.mainEntry import Mywindow
from Serial.UART import read, write
from Serial.HP_show import HP_scene


class Radar_main(object):
    """
    雷达主进程
    """
    _que_left = multiprocessing.get_context('spawn').Queue()
    _event_right = multiprocessing.get_context('spawn').Event()
    _que_right = multiprocessing.get_context('spawn').Queue()
    _event_left = multiprocessing.get_context('spawn').Event()
    _event_close = multiprocessing.get_context('spawn').Event()

    __res_left = [None, None]
    __res_right = [None, None]

    __cam_left = False
    __cam_right = False
    __Lidar = False
    __serial = False
    __record_state = False
    __record_object = []  # 视频录制对象列表，先用None填充
    # 录制保存位置
    __save_title = ''  # 当场次录制文件夹名

    def __init__(self):
        self.logger = LOGGER(lambda tp, pos, mes: myshow.set_text(tp, pos, mes))
        self.text_api = lambda x, y, z: self.logger.add_text(x, y, z)
        self.show_api = lambda x: myshow.set_image(x, "main_demo")
        self.show_map = lambda x: myshow.set_image(x, "map")
        self.__cam_left = USEABLE['cam_left']
        self.__cam_right = USEABLE['cam_right']
        self.__Lidar = USEABLE['Lidar']
        self.__serial = USEABLE['serial']
        if self.__cam_left:
            self.PRE_left = multiprocessing.Process(target=process_detect, args=(
                self._event_left, self._que_left, self._event_close,
                'cam_left',))
            self.PRE_left.daemon = True
            self.repo_left = Reproject('cam_left')
        if self.__cam_right:
            self.PRE_right = multiprocessing.Process(target=process_detect,
                                                     args=(self._event_right, self._que_right, self._event_close,
                                                           'cam_right',))
            self.PRE_right.daemon = True
            self.repo_right = Reproject('cam_right')
        self.lidar = Radar('cam_right', text_api=self.text_api)
        # self.__cam_far = Camera("cam_far", DEBUG)
        if self.__Lidar:
            self.lidar.start()
        else:
            # 读取预定的点云文件
            self.lidar.preload()
        if self.__serial:
            self.hp_sence = HP_scene(enemy_color, self.show_api)
            ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
            self.read_thr = threading.Thread(target=read, args=(ser,))
            self.write_thr = threading.Thread(target=write, args=(ser,))
            self.read_thr.setDaemon(True)
            self.write_thr.setDaemon(True)
            self.write_thr.start()
            self.read_thr.start()
        self.loc_alarm = Alarm(enemy=enemy_color, api=self.show_map, touch_api=self.text_api,
                               region=test_region, debug=False)
        T = np.eye(4)
        rvec = cam_config['cam_left']['rvec']
        tvec = cam_config['cam_left']['tvec']
        T[:3, :3] = cv.Rodrigues(rvec)[0]  # 旋转向量转化为旋转矩阵
        T[:3, 3] = tvec.reshape(-1)  # 加上平移向量
        T = np.linalg.inv(T)  # 矩阵求逆
        self.loc_alarm.push_T(T, (T @ (np.array([0, 0, 0, 1])))[:3], 0)

        myshow.Eapi = getattr(self, "update_epnp")
        myshow.Rapi = getattr(self, "record")

    def record(self):
        if myshow.record_state:
            time_ = time.localtime(time.time())
            self.__save_title = f"resources/records/{time_.tm_mday}_{time_.tm_hour}_" \
                                f"{time_.tm_min}"
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            self.__record_object.append(cv.VideoWriter(self.__save_title + "_left", fourcc, 25, ()))
            self.__record_object.append(cv.VideoWriter(self.__save_title + "_right", fourcc, 25, ()))
            self.__record_object.append(cv.VideoWriter(self.__save_title + "_far", fourcc, 25, ()))
            self.__record_state = True
        else:
            self.__record_state = False
            self.__record_object.clear()

    def start(self):
        if self.__cam_left:
            self.PRE_left.start()
        if self.__cam_right:
            self.PRE_right.start()

    def update_image(self):
        if not myshow.view_change:
            myshow.set_image(self.__res_left[1], "main_demo")
            if self.__res_right[1] is not None:
                myshow.set_image(self.__res_right[1][:, 2000:3000].copy(), "side_demo")
        else:
            myshow.set_image(self.__res_right[1], "main_demo")
            if self.__res_left[1] is not None:
                myshow.set_image(self.__res_left[1][:, 500:1700].copy(), "side_demo")

    def update_epnp(self, tvec, rvec, side):
        if side:
            T, T_ = self.repo_left.push_T(rvec, tvec)
            self.loc_alarm.push_T(T, T_, 0)
        else:
            self.repo_right.push_T(rvec, tvec)

    def update_reproject(self):
        res_temp = self.__res_left[0]
        if res_temp is None:
            pass
        else:
            armors = res_temp[:, [11, 13, 6, 7, 8, 9]]
            cars = res_temp[:, [11, 0, 1, 2, 3]]
            # result = self.repo_left.check(armors, cars)
            # if result[0] is not None:
            #     self.text_api("INFO", "Repro left", str(result[0].keys()))
            self.repo_left.update(None, self.__res_left[1])
        res_temp = self.__res_right[0]
        if res_temp is None:
            pass
        else:
            armors = res_temp[:, [11, 13, 6, 7, 8, 9]]
            cars = res_temp[:, [11, 0, 1, 2, 3]]
            # result = self.repo_left.check(armors, cars)
            # if result[0] is not None:
            #     self.text_api("INFO", "Repro right", str(result[0].keys()))
            self.repo_right.update(None, self.__res_right[1])

    def update_location_alarm(self):
        if self.__res_left[0] is None:
            pass
        else:
            armors = self.__res_left[0][:, [11, 6, 7, 8, 9]]
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
                self.loc_alarm.update(location)
                self.loc_alarm.show()

    def spin_once(self):
        if self.__cam_left:
            self.__res_left = sub(self._event_left, self._que_left)
        if self.__cam_right:
            self.__res_right = sub(self._event_right, self._que_right)
        if self.__record_state:
            if self.__res_left[1] is not None:
                self.__record_object[0].write(self.__res_left[1])
            if self.__res_right[1] is not None:
                self.__record_object[1].write(self.__res_right[1])
        if myshow.epnp_mode:
            self.update_image()
        else:
            self.update_reproject()
            self.update_image()
            self.update_location_alarm()
        if myshow.terminate:
            self._event_close.set()
            if self.__cam_left:
                self.PRE_left.join(timeout=3)
            if self.__cam_right:
                self.PRE_right.join(timeout=3)
            sys.exit()


def process_detect(event, que, Event_Close, name):
    # 多线程接收写法
    print(f"子线程开始:{name}")
    predictor = Predictor(name)
    cam = Camera(name, DEBUG)
    # count = 0
    # t1 = 0
    while not Event_Close.is_set():
        frame = cam.get_img()
        if frame is not None:
            #       t3 = time.time()
            res = predictor.detect_cars(frame)
            pub(event, que, res)
    #      t1 = t1 + time.time() - t3
    #     count += 1

    predictor.stop()
    # fps = float(count) / t1
    # print(f'{name} count:{count} fps: {int(fps)}')
    print(f"相机网络子进程:{name} 退出")
    sys.exit()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # ui
    app = QtWidgets.QApplication(sys.argv)
    myshow = Mywindow()
    radar_main = Radar_main()
    radar_main.start()
    timer_main = QTimer()  # 主循环使用的线程
    timer_serial = QTimer()  # 串口使用的线程
    myshow.show()
    timer_main.timeout.connect(radar_main.spin_once)
    timer_main.start(0)
    sys.exit(app.exec_())
