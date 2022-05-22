"""
UI类
created by 李龙 2022/4
"""
import sys
import time
import os
import cv2 as cv
import numpy as np
import multiprocessing
import threading
import serial
import logging
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QRect

from resources.config import INIT_FRAME_PATH, \
    MAP_PATH, map_size, DEBUG, USEABLE, \
    enemy_color, cam_config, num2cam, using_video, enemy2color

from mapping.ui_v2 import Ui_MainWindow
from camera.cam_hk import Camera_HK
from camera.cam_dh import Camera_DH
from net.network_pro import Predictor
from radar_detect.Linar import Radar
from radar_detect.reproject import Reproject
from radar_detect.location_alarm import Alarm
from mul_manager.pro_manager import sub, pub
from Serial.UART import read, write
from Serial.HP_show import HP_scene
from Serial.port_operation import Port_operate
from radar_detect.missile import Missile
from radar_detect.solve_pnp import SolvePnp
from mapping.pc_show import pc_show


def process_detect(event, que, Event_Close, name):
    # 多线程接收写法
    print(f"子线程开始: {name}")
    predictor = Predictor(name)
    cam = Camera_HK(name, using_video)
    count = 0
    count_error = 0
    t1 = 0
    try:
        while not Event_Close.is_set():
            result, frame = cam.get_img()
            if result and frame is not None:
                t3 = time.time()
                res = predictor.detect_cars(frame)
                pub(event, que, res)
                t1 = t1 + time.time() - t3
                count += 1
                if count == 100:
                    fps = float(count) / t1
                    print(f'{name} count:{count} fps: {int(fps)}')
                    count = 0
                    t1 = 0
            else:
                count_error += 1
                pub(event, que, [result, frame])
                if count_error == 100:
                    cam.destroy()
                    del cam
                    cam = Camera_HK(name, using_video)
                    count_error = 0
        predictor.stop()
        cam.destroy()
        print(f"相机网络子进程:{name} 退出")
    except Exception as e:
        print(e)
        print(f"相机网络子进程:{name} 寄了")
    sys.exit()


class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):  # 这个地方要注意Ui_MainWindow

    _que_left = multiprocessing.get_context('spawn').Queue()
    _event_right = multiprocessing.get_context('spawn').Event()
    _que_right = multiprocessing.get_context('spawn').Queue()
    _event_left = multiprocessing.get_context('spawn').Event()
    _event_close = multiprocessing.get_context('spawn').Event()

    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)
        self.hp_scene = HP_scene(enemy_color, lambda x: self.set_image(x, "blood"))
        self.text_api = lambda x, y, z: self.set_text(x, y, z)
        self.board_api = lambda x, y, z: self.set_board_text(x, y, z)
        self.pnp_api = lambda x, y, z: self.set_pnp_text(x, y, z)
        self.show_map = lambda x: self.set_image(x, "map")

        self.__ui_init()
        self.record_object = []

        self.epnp_image = QImage()
        self.item = None
        self.scene = QtWidgets.QGraphicsScene()  # 创建场景
        self.pnp_zoom = 1

        self.__res_right = False
        self.__res_left = False
        self.__res_far = False
        self.__pic_right = None
        self.__pic_left = None
        self.__pic_far = None

        self.__cam_left = USEABLE['cam_left']
        self.__cam_right = USEABLE['cam_right']
        self.__cam_far = USEABLE['cam_far']
        self.__Lidar = USEABLE['Lidar']
        self.__serial = USEABLE['serial']
        self.__using_d = USEABLE['using_d']
        if self.__cam_left:
            self.PRE_left = multiprocessing.Process(target=process_detect, args=(
                self._event_left, self._que_left, self._event_close,
                'cam_left',))
            self.PRE_left.daemon = True

        if self.__cam_right:
            self.PRE_right = multiprocessing.Process(target=process_detect,
                                                     args=(self._event_right, self._que_right, self._event_close,
                                                           'cam_right',))
            self.PRE_right.daemon = True

        if self.__cam_far:
            self.__CamFar = Camera_DH("cam_far", using_video)

        self.missile = Missile(enemy_color, self.text_api, self.board_api)
        self.lidar = Radar('cam_left', text_api=self.text_api, imgsz=cam_config['cam_left']["size"], queue_size=100)
        self.e_location = np.array([])

        if self.__Lidar:
            self.lidar.start()
        else:
            # 读取预定的点云文件
            self.lidar.preload()

        if self.__serial:
            ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
            self.read_thr = threading.Thread(target=read, args=(ser,))
            self.write_thr = threading.Thread(target=write, args=(ser,))
            self.read_thr.setDaemon(True)
            self.write_thr.setDaemon(True)
            self.write_thr.start()
            self.read_thr.start()

        self.repo_left = Reproject('cam_left', self.text_api)
        self.repo_right = Reproject('cam_right', self.text_api)
        self.loc_alarm = Alarm(enemy=enemy_color, api=self.show_map, touch_api=self.text_api,
                               using_Delaunay=self.__using_d, debug=False)

        try:
            self.sp.read(f'cam_left_{enemy2color[enemy_color]}')
            T, T_ = self.repo_left.push_T(self.sp.rvec, self.sp.tvec)
            self.loc_alarm.push_RT(self.sp.rvec, self.sp.tvec, 0)
            self.sp.read(f'cam_right_{enemy2color[enemy_color]}')
            self.repo_right.push_T(self.sp.rvec, self.sp.tvec)
        except Exception as e:
            print("[ERROR] using default data")
            T, T_ = self.repo_left.push_T(cam_config["cam_left"]["rvec"], cam_config["cam_left"]["tvec"])

        self.loc_alarm.push_T(T, T_, 0)
        self.start()

    def __ui_init(self):
        self.view_change = 0  # 视角切换控制符
        frame = cv.imread(INIT_FRAME_PATH)
        frame_m = cv.imread(MAP_PATH)
        self.terminate = False
        # 小地图翻转
        if enemy_color:
            frame_m = cv.rotate(frame_m, cv.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame_m = cv.rotate(frame_m, cv.ROTATE_90_CLOCKWISE)
        frame_m = cv.resize(frame_m, map_size)
        self.set_image(frame, "side_demo")
        self.set_image(frame, "far_demo")
        self.set_image(frame, "main_demo")
        self.set_image(frame_m, "map")
        del frame, frame_m

        self.time_textBrowser = {}
        self.feedback_textBrowser = {}  # textBrowser信息列表
        self.board_textBrowser = {}
        self.pnp_textBrowser = {}

        self.sp = SolvePnp(self.pnp_api)
        # 反馈信息栏，显示初始化
        # 雷达和位姿估计状态反馈栏，初始化为全False
        self.record_state = False  # 0:停止 1:开始
        self.epnp_mode = False  # 0:停止 1:开始
        self.terminate = False
        self.show_pc_state = False
        self.record.setText("录制")
        self.ChangeView.setText("切换视角")
        self.ShutDown.setText("终止程序")

    def ChangeView_on_clicked(self) -> None:
        """
        切换视角
        """
        from resources.config import USEABLE
        if self.view_change:
            if USEABLE['cam_left']:
                self.view_change = 0
            else:
                return
        else:
            if USEABLE['cam_right']:
                self.view_change = 1
            else:
                return

    def record_on_clicked(self) -> None:
        if not self.record_state:
            self.record.setText("录制 正在录制")
            fourcc = cv.VideoWriter_fourcc(*'MP42')
            time_ = time.localtime(time.time())
            save_title = f"resources/records/{time_.tm_mday}_{time_.tm_hour}_" \
                         f"{time_.tm_min}"
            self.record_object.append(cv.VideoWriter(save_title + "_left.avi", fourcc, 12,
                                                     cam_config['cam_left']['size']))
            self.record_object.append(cv.VideoWriter(save_title + "_right.avi", fourcc, 12,
                                                     cam_config['cam_right']['size']))
            self._record_thr = threading.Thread(target=self.mul_record)
            self._record_thr.setDaemon(True)
            self.record_state = True
            self._record_thr.start()
        else:
            self.record.setText("录制")
            self.record_state = False
            self._record_thr.join()
            self.record_object.clear()

    def showpc_on_clicked(self) -> None:
        self.show_pc_state = not self.show_pc_state
        self.epnp_mode = False

    def CloseProgram_on_clicked(self) -> None:
        """
        关闭程序
        """
        self.terminate = True

    def epnp_on_clicked(self) -> None:
        if self.tabWidget.currentIndex() == 1:
            self.epnp_mode = True
            self.show_pc_state = False
            time.sleep(0.1)
            if not self.view_change:
                frame = self.__pic_left
            else:
                frame = self.__pic_right
            if frame.shape[2] == 3:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                return

            self.epnp_image = QImage(rgb, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
            width = self.pnp_demo.width()
            height = self.pnp_demo.height()
            pix = QPixmap.fromImage(self.epnp_image).scaled(width, height)
            self.item = QtWidgets.QGraphicsPixmapItem(pix)
            self.scene.clear()
            self.scene.addItem(self.item)
            self.pnp_demo.setScene(self.scene)

            self.sp.sel_cam(self.view_change)
        else:
            self.epnp_mode = False

        self.sp.clc()
        self.pnp_textBrowser.clear()

    def epnp_calculate(self) -> None:
        result = self.sp.locate_pick()
        if result:
            self.sp.save()
            self.sp.clc()
            self.pnp_textBrowser.clear()
            self.update_epnp(self.sp.translation, self.sp.rotation, self.view_change)
            self.pnp_zoom = 1

    def epnp_mouseEvent(self, event) -> None:
        if self.epnp_mode:
            if event.button() == Qt.LeftButton:
                x = event.x() / self.pnp_zoom
                y = event.y() / self.pnp_zoom
                pointF = self.pnp_demo.mapToScene(QRect(0, 0, self.pnp_demo.viewport().width(),
                                                        self.pnp_demo.viewport().height()))[0]
                x_offset = pointF.x() / self.pnp_zoom
                y_offset = pointF.y() / self.pnp_zoom
                x = int((x + x_offset) / self.pnp_demo.width() * self.epnp_image.width())
                y = int((y + y_offset) / self.pnp_demo.height() * self.epnp_image.height())
                self.sp.add_point(x, y)

    def epnp_wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()  # 竖直滚过的距离
        if angleY > 0 and self.pnp_zoom < 8:
            self.pnp_zoom = self.pnp_zoom * 2
        elif angleY < 0 and self.pnp_zoom > 1:  # 滚轮下滚
            self.pnp_zoom = self.pnp_zoom / 2
        self.item.setScale(self.pnp_zoom)

    def pc_mouseEvent(self, event) -> None:
        if self.show_pc_state:
            # TODO: 这里的3072是个常值，最好改了
            x = int(event.x() / self.main_demo.width() * 3072)
            y = int(event.y() / self.main_demo.height() * 2048)
            w = 10
            h = 5
            # 格式定义： [N, [bbox(xyxy), conf, cls, bbox(xyxy), conf, cls, col, N]
            self.repo_left.check(np.array([x, y, w, h, 1., 1, x, y, w, h, 1., 1., 1., 0]).reshape(1, -1))
            if not self.__using_d:
                dph = self.lidar.detect_depth(rects=[[x - 5, y - 2.5, w, h]]).reshape(-1, 1)
                if not np.any(np.isnan(dph)):
                    self.loc_alarm.pc_location(np.concatenate([np.array([[x, y]]), dph], axis=1))
            else:
                self.loc_alarm.pc_location(
                    np.concatenate([np.ones((1, 1)), np.array([[x, y]]), np.zeros((1, 1))], axis=1))

    # 暂时不支持 epnp_change_view
    def epnp_change_view(self) -> None:
        pass

    def epnp_next_on_clicked(self) -> None:
        self.sp.step(1)

    def epnp_back_on_clicked(self) -> None:
        self.sp.step(-1)

    def epnp_del(self) -> None:
        self.sp.del_point()

    def epnp_clear_on_clicked(self) -> None:
        self.sp.clc()

    def set_image(self, frame, position="") -> bool:
        """
        Image Show Function

        :param frame: the image to show
        :param position: where to show
        :return: a flag to indicate whether the showing process have succeeded or not
        """
        if frame is None:
            return False
        if position not in ["main_demo", "map", "far_demo", "side_demo", "blood"]:
            print("[ERROR] The position isn't a member of this UIwindow")
            return False
        if position == "main_demo":
            width = self.main_demo.width()
            height = self.main_demo.height()
        elif position == "far_demo":
            width = self.far_demo.width()
            height = self.far_demo.height()
        elif position == "side_demo":
            width = self.side_demo.width()
            height = self.side_demo.height()
        elif position == "map":
            width = self.map.width()
            height = self.map.height()
        else:
            width = self.blood.width()
            height = self.blood.height()

        if frame.shape[2] == 3:
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        elif frame.shape[2] == 2:
            rgb = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            return False

        # allocate the space of QPixmap
        temp_image = QImage(rgb, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)

        temp_pixmap = QPixmap(temp_image).scaled(width, height)

        # set the image to the QPixmap location to show the image on the UI
        if position == "main_demo":
            self.main_demo.setPixmap(temp_pixmap)
            self.main_demo.setScaledContents(True)
        elif position == "far_demo":
            self.far_demo.setPixmap(temp_pixmap)
            self.far_demo.setScaledContents(True)
        elif position == "side_demo":
            self.side_demo.setPixmap(temp_pixmap)
            self.side_demo.setScaledContents(True)
        elif position == "map":
            self.map.setPixmap(temp_pixmap)
            self.map.setScaledContents(True)
        elif position == "blood":
            self.blood.setPixmap(temp_pixmap)
            self.blood.setScaledContents(True)

        return True

    def set_text(self, _type: str, position: str, message: str) -> bool:
        """
        to set text in the QtLabel
        :param message: message that will be put on the screen;
        :param _type: message level [ERROR,INFO,WARNING];
        :param position: message that will put on position [LOCATION,ALARM_POSITION,TIMEING]
        :return:
        a flag to indicate whether the showing process have succeeded or not
        """
        # Using "<br \>" to combine the contents of the message list to a single string message
        if message == "":
            if position in self.feedback_textBrowser.keys():
                self.feedback_textBrowser.pop(position)
        else:
            if _type == "ERROR":
                msg = f"<font color='#FF0000'><b>{message}</b></font>"
            elif _type == "INFO":
                msg = f"<font color='#404040'><b>{message}</b></font>"
            else:
                msg = f"<font color='#FF8033'><b>{message}</b></font>"
            self.feedback_textBrowser[position] = msg.replace('\n', "<br />")
            self.feedback_textBrowser[position] = msg
            self.time_textBrowser[position] = time.time()
        return True

    def set_board_text(self, _type: str, position: str, message: str) -> bool:
        """
        to set text in the QtLabel
        :param message: message that will be put on the screen;
        :param _type: message level [ERROR,INFO,WARNING];
        :param position: message that will put on position [LOCATION,ALARM_POSITION,TIMEING]
        :return:
        a flag to indicate whether the showing process have succeeded or not
        """
        # Using "<br \>" to combine the contents of the message list to a single string message
        if message == "":
            if position in self.board_textBrowser.keys():
                self.board_textBrowser.pop(position)
        else:
            if _type == "ERROR":
                msg = f"<font color='#FF0000'><b>{message}</b></font>"
            elif _type == "INFO":
                msg = f"<font color='#404040'><b>{message}</b></font>"
            else:
                msg = f"<font color='#FF8033'><b>{message}</b></font>"
            self.board_textBrowser[position] = msg.replace('\n', "<br />")
            self.board_textBrowser[position] = msg
        return True

    def set_pnp_text(self, _type: str, position: str, message: str) -> bool:
        """
        to set text in the QtLabel
        :param message: message that will be put on the screen;
        :param _type: message level [ERROR,INFO,WARNING];
        :param position: message that will put on position [LOCATION,ALARM_POSITION,TIMEING]
        :return:
        a flag to indicate whether the showing process have succeeded or not
        """
        # Using "<br \>" to combine the contents of the message list to a single string message
        if message == "":
            if position in self.pnp_textBrowser.keys():
                self.pnp_textBrowser.pop(position)
        else:
            if _type == "ERROR":
                msg = f"<font color='#FF0000'><b>{message}</b></font>"
            elif _type == "INFO":
                msg = f"<font color='#404040'><b>{message}</b></font>"
            else:
                msg = f"<font color='#FF8033'><b>{message}</b></font>"
            self.pnp_textBrowser[position] = msg.replace('\n', "<br />")
        text = "<br \>".join(list(self.pnp_textBrowser.values()))  # css format to replace \n
        self.pnp_condition.setText(text)
        return True

    def start(self) -> None:
        if self.__cam_left:
            self.PRE_left.start()
        if self.__cam_right:
            self.PRE_right.start()

    def update_state(self) -> None:
        self.set_board_text("INFO", "当前相机", f"当前相机为:{num2cam[self.view_change]}")
        if isinstance(self.__res_left, bool):
            self.set_board_text("ERROR", "左相机", "左相机寄了")
        else:
            self.set_board_text("INFO", "左相机", "左相机正常工作")

        if isinstance(self.__res_right, bool):
            self.set_board_text("ERROR", "右相机", "右相机寄了")
        else:
            self.set_board_text("INFO", "右相机", "右相机正常工作")

        if not self.__res_far:
            self.set_board_text("ERROR", "第三相机", "远相机寄了")
        else:
            self.set_board_text("INFO", "第三相机", "远相机正常工作")
        temp_t = time.time()
        for pos in list(self.feedback_textBrowser.keys()):
            if temp_t - self.time_textBrowser[pos] > 2:
                self.feedback_textBrowser.pop(pos)
        text = "<br \>".join(list(self.feedback_textBrowser.values()))  # css format to replace \n
        self.textBrowser.setText(text)
        text = "<br \>".join(list(self.board_textBrowser.values()))  # css format to replace \n
        self.condition.setText(text)

    def update_image(self) -> None:
        if not self.view_change:
            self.set_image(self.__pic_left, "main_demo")
            if self.__pic_right is not None:
                self.set_image(self.__pic_right[:, 2000:3000].copy(), "side_demo")
        else:
            self.set_image(self.__pic_right, "main_demo")
            if self.__pic_left is not None:
                self.set_image(self.__pic_left[:, 500:1700].copy(), "side_demo")
        if self.__pic_far is not None:
            self.set_image(self.__pic_far, "far_demo")

    def update_epnp(self, tvec: np.ndarray, rvec: np.ndarray, side: int) -> None:
        if not side:
            T, T_ = self.repo_left.push_T(rvec, tvec)
            self.loc_alarm.push_T(T, T_, 0)
            self.loc_alarm.push_RT(rvec, tvec, 0)
        else:
            self.repo_right.push_T(rvec, tvec)
            self.loc_alarm.push_RT(rvec, tvec, 1)

    def update_reproject(self) -> None:
        if self.__cam_left:
            self.repo_left.check(self.__res_left)
            if not self.record_state:
                self.repo_left.update(self.__pic_left)
            self.e_location = self.repo_left.get_pred_box()
        if self.__cam_right:
            self.repo_right.check(self.__res_right)
            if not self.record_state:
                self.repo_right.update(self.__pic_right)

    def update_missile(self) -> None:
        if isinstance(self.__res_left, bool) and not self.__res_left:
            self.missile.detect_api(self.__pic_left, None)

    def update_location_alarm(self) -> None:
        t_loc = None
        e_loc = None
        if isinstance(self.__res_left, np.ndarray):
            if self.__res_left.shape[0] != 0:
                if not self.__using_d:
                    armors = self.__res_left[:, [11, 6, 7, 8, 9]]
                    armors[:, 3] = armors[:, 3] - armors[:, 1]
                    armors[:, 4] = armors[:, 4] - armors[:, 2]
                    armors = armors[np.logical_not(np.isnan(armors[:, 0]))]
                    if armors.shape[0] != 0:
                        dph = self.lidar.detect_depth(rects=armors[:, 1:].tolist()).reshape(-1, 1)
                        x0 = (armors[:, 1] + armors[:, 3] / 2).reshape(-1, 1)
                        y0 = (armors[:, 2] + armors[:, 4] / 2).reshape(-1, 1)
                        t_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, dph], axis=1)
                    if self.e_location.shape[0] != 0:
                        armors = self.e_location[np.logical_not(np.isnan(self.e_location[:, 0]))]
                        dph = self.lidar.detect_depth(rects=armors[:, 1:].tolist()).reshape(-1, 1)
                        x0 = (armors[:, 1] + armors[:, 3] / 2).reshape(-1, 1)
                        y0 = (armors[:, 2] + armors[:, 4] / 2).reshape(-1, 1)
                        e_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, dph], axis=1)
                else:
                    armors = self.__res_left[:, [11, 6, 7, 8, 9]]
                    armors = armors[np.logical_not(np.isnan(armors[:, 0]))]
                    x0 = (armors[:, 1] + (armors[:, 3] - armors[:, 1]) / 2).reshape(-1, 1)
                    y0 = (armors[:, 2] + (armors[:, 4] - armors[:, 2]) / 2).reshape(-1, 1)
                    t_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, np.zeros(x0.shape)], axis=1)
                    if self.e_location.shape[0] != 0:
                        armors = self.e_location[np.logical_not(np.isnan(self.e_location[:, 0]))]
                        x0 = (armors[:, 1] + armors[:, 3] / 2).reshape(-1, 1)
                        y0 = (armors[:, 2] + armors[:, 4] / 2).reshape(-1, 1)
                        e_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, np.zeros(x0.shape)], axis=1)
        self.loc_alarm.update(t_loc, e_loc)
        self.loc_alarm.check()
        self.loc_alarm.show()

    def spin(self) -> None:
        # get images
        if self.__cam_left:
            self.__res_left, self.__pic_left = sub(self._event_left, self._que_left)
        if self.__cam_right:
            self.__res_right, self.__pic_right = sub(self._event_right, self._que_right)
        if self.__cam_far:
            self.__res_far, self.__pic_far = self.__CamFar.get_img()

        # if in epnp_mode , just show the images
        if not self.epnp_mode:
            if self.show_pc_state and isinstance(self.__pic_left, np.ndarray):
                if self.__using_d:
                    self.loc_alarm.pc_draw(self.__pic_left)
                else:
                    depth = self.lidar.read()
                    pc_show(self.__pic_left, depth)
                self.repo_left.update(self.__pic_left)
                self.repo_left.push_text()
                self.update_image()
                self.update_state()
            else:
                self.update_state()
                self.update_reproject()
                self.update_location_alarm()
                # self.update_missile()
                self.update_image()
        # update serial
        if self.__serial:
            Port_operate.get_message(self.hp_scene)
            self.hp_scene.show()
            location = self.loc_alarm.get_location()
            Port_operate.gain_positions(location)

            if Port_operate.change_view:
                Port_operate.change_view = not Port_operate.change_view
                self.view_change = int(not self.view_change)

        # if close the program
        if self.terminate:
            if self.__cam_far:
                self.__CamFar.destroy()
            self._event_close.set()
            if self.__cam_left:
                self.PRE_left.join(timeout=1)
            if self.__cam_right:
                self.PRE_right.join(timeout=1)
            sys.exit()

    def mul_record(self):
        while True:
            if self.record_state:
                if not isinstance(self.__res_left, bool):
                    self.record_object[0].write(self.__pic_left)
                if not isinstance(self.__res_right, bool):
                    self.record_object[1].write(self.__pic_right)
            else:
                break


class LOGGER(object):
    """
    logger 类
    """

    def __init__(self):
        # 创建一个logger
        import time
        logger_name = time.strftime('%Y-%m-%d %H-%M-%S')
        self.terminal = sys.stdout
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        # 创建一个handler，用于写入日志文件
        log_path = os.path.abspath(os.getcwd()) + "/logs/"  # 指定文件输出路径
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logname = log_path + logger_name + '.log'  # 指定输出的日志文件名
        fh = logging.FileHandler(logname, encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
        fh.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        fh.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)

    def write(self, message):
        self.terminal.write(message)
        if message == '\n':
            return
        if "[ERROR]" in message:
            self.logger.error(message)
        elif "[WARNING]" in message:
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def flush(self):
        pass

    def __del__(self):
        pass


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # ui
    app = QtWidgets.QApplication(sys.argv)
    sys.stdout = LOGGER()
    myshow = Mywindow()
    myshow.show()

    timer_main = QTimer()  # 主循环使用的线程
    timer_main.timeout.connect(myshow.spin)
    timer_main.start(0)

    sys.exit(app.exec_())
