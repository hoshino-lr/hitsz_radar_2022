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
from PyQt5.QtCore import QTimer, Qt, QRect, QPoint

from resources.config import INIT_FRAME_PATH, \
    MAP_PATH, enemy_color, map_size, DEBUG, USEABLE, \
    enemy_color, test_region, cam_config

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
    if name == "cam_far":
        cam = Camera_DH(name, DEBUG)
    else:
        cam = Camera_HK(name, DEBUG)
    count = 0
    t1 = 0
    while not Event_Close.is_set():
        result, frame = cam.get_img()
        if result and frame is not None:
            t3 = time.time()
            res = predictor.detect_cars(frame)
            pub(event, que, res)
            t1 = t1 + time.time() - t3
            count += 1
        else:
            pub(event, que, [result, frame])
    cam.destroy()
    predictor.stop()
    fps = float(count) / t1
    print(f'{name} count:{count} fps: {int(fps)}')
    print(f"相机网络子进程:{name} 退出")
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
        self.__ui_init()

        self.hp_sence = HP_scene(enemy_color, lambda x: self.set_image(x, "blood"))
        self.text_api = lambda x, y, z: self.set_text(x, y, z)
        self.board_api = lambda x, y, z: self.set_board_text(x, y, z)
        self.show_map = lambda x: self.set_image(x, "map")

        self.record_object = []
        self.epnp_image = QImage()
        self.item = None
        self.scene = QtWidgets.QGraphicsScene()  # 创建场景
        self.__res_right = [False, None]
        self.__res_left = [False, None]
        self.__res_far = [False, None]

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
            self.__CamFar = Camera_DH("cam_far", DEBUG)

        self.pnp_zoom = 1

        self.missile = Missile(enemy_color, self.text_api, self.board_api)
        self.lidar = Radar('cam_left', text_api=self.text_api)
        self.e_location = {}

        if self.__Lidar:
            self.lidar.start()
        else:
            # 读取预定的点云文件
            self.lidar.preload()

        if self.__serial:
            ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
            self.read_thr = threading.Thread(target=read, args=(ser,))
            self.write_thr = threading.Thread(target=write, args=(ser,))
            self.read_thr.setDaemon(True)
            self.write_thr.setDaemon(True)
            self.write_thr.start()
            self.read_thr.start()

        self.repo_left = Reproject('cam_left')
        self.repo_right = Reproject('cam_right')
        self.loc_alarm = Alarm(enemy=enemy_color, api=self.show_map, touch_api=self.text_api,
                               using_Delaunay=self.__using_d, region=test_region, debug=False)

        T, T_ = self.repo_right.push_T(cam_config['cam_left']['rvec'], cam_config['cam_left']['tvec'])
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

        self.feedback_textBrowser = {}  # textBrowser信息列表
        self.board_textBrowser = {}
        self.pnp_textBrowser = {}

        self.sp = SolvePnp()
        # 反馈信息栏，显示初始化
        self.set_text("textBrowser", "TIMEING", "intializing...")
        # 雷达和位姿估计状态反馈栏，初始化为全False
        self.record_state = False  # 0:停止 1:开始
        self.epnp_mode = False  # 0:停止 1:开始
        self.terminate = False
        self.show_pc_state = False
        self.record.setText("开始录制")
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
                self.ChangeView.setText("切换视角")
            else:
                pass
        else:
            if USEABLE['cam_right']:
                self.view_change = 1
                self.ChangeView.setText("切换视角")
            else:
                pass

    def record_on_clicked(self) -> None:
        if not self.record_state:
            time_ = time.localtime(time.time())
            self.__save_title = f"resources/records/{time_.tm_mday}_{time_.tm_hour}_" \
                                f"{time_.tm_min}"
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            self.record_object.append(cv.VideoWriter(self.__save_title + "_left", fourcc, 25,
                                                     cam_config['cam_left']['size']))
            self.record_object.append(cv.VideoWriter(self.__save_title + "_right", fourcc, 25,
                                                     cam_config['cam_right']['size']))
            self.record_object.append(cv.VideoWriter(self.__save_title + "_far", fourcc, 25,
                                                     cam_config['cam_far']['size']))
            self.record_state = True
        else:
            self.record_state = False
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
                frame = self.__res_left[1]
            else:
                frame = self.__res_right[1]
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
            text = self.sp.sel_cam(self.view_change)
            self.set_pnp_text("INFO", f"输入", text)
        else:
            self.epnp_mode = False
        self.sp.clc()
        self.pnp_textBrowser.clear()

    def epnp_calculate(self) -> None:
        result = self.sp.locate_pick()
        if result:
            self.sp.clc()
            self.pnp_textBrowser.clear()
            self.update_epnp(self.sp.translation, self.sp.rotation, self.view_change)
            self.set_pnp_text("INFO", "EPNP", "EPNP success!!!")
        else:
            self.set_pnp_text("ERROR", "EPNP", "Invalid action!!!")

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
                text = self.sp.add_point(x, y)
                self.set_pnp_text("INFO", f"Point{self.sp.count}", f"x: {x} y: {y}")
                self.set_pnp_text("INFO", f"输入", text)

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
            pass

    # 暂时不支持 epnp_change_view
    def epnp_change_view(self) -> None:
        pass

    def epnp_next_on_clicked(self) -> None:
        text = self.sp.step(1)
        self.set_pnp_text("INFO", f"输入", text)

    def epnp_back_on_clicked(self) -> None:
        text = self.sp.step(-1)
        self.set_pnp_text("INFO", f"输入", text)

    def epnp_del(self) -> None:
        num = self.sp.count
        text = self.sp.del_point()
        self.set_pnp_text("INFO", f"Point{self.sp.count}", f"x:  y: ")
        self.set_pnp_text("INFO", f"输入", text)

    def epnp_clear_on_clicked(self) -> None:
        text = self.sp.clc()
        self.pnp_textBrowser.clear()
        self.set_pnp_text("INFO", f"输入", text)

    def set_image(self, frame, position="") -> bool:
        """
        Image Show Function

        :param frame: the image to show
        :param position: where to show
        :return: a flag to indicate whether the showing process have succeeded or not
        """
        if frame is None:
            return False
        if not position in ["main_demo", "map", "far_demo", "side_demo", "blood"]:
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
        elif position == "blood":
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
                msg = f"<font color='#FF0000'><b>[ERROR] {message}</b></font>"
            elif _type == "INFO":
                msg = f"<font color='#FF0000'><b>[INFO] {message}</b></font>"
            else:
                msg = f"<font color='#FF0000'><b>[WARNING] {message}</b></font>"
            self.feedback_textBrowser[position] = msg
        text = "<br \>".join(list(self.feedback_textBrowser.values()))  # css format to replace \n
        self.textBrowser.setText(text)
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
                msg = f"<font color='#FF0000'><b>[ERROR] {message}</b></font>"
            elif _type == "INFO":
                msg = f"<font color='#FF0000'><b>[INFO] {message}</b></font>"
            else:
                msg = f"<font color='#FF0000'><b>[WARNING] {message}</b></font>"
            self.board_textBrowser[position] = msg

        text = "<br \>".join(list(self.board_textBrowser.values()))  # css format to replace \n
        self.condition.setText(text)
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
                msg = f"<font color='#FF0000'><b>[ERROR] {message}</b></font>"
            elif _type == "INFO":
                msg = f"<font color='#FF0000'><b>[INFO] {message}</b></font>"
            else:
                msg = f"<font color='#FF0000'><b>[WARNING] {message}</b></font>"
            self.pnp_textBrowser[position] = msg
        text = "<br \>".join(list(self.pnp_textBrowser.values()))  # css format to replace \n
        self.pnp_condition.setText(text)
        return True

    def start(self) -> None:
        if self.__cam_left:
            self.PRE_left.start()
        if self.__cam_right:
            self.PRE_right.start()

    def update_image(self) -> None:
        if not self.view_change:
            self.set_image(self.__res_left[1], "main_demo")
            if self.__res_right[1] is not None:
                self.set_image(self.__res_right[1][:, 2000:3000].copy(), "side_demo")
        else:
            self.set_image(self.__res_right[1], "main_demo")
            if self.__res_left[1] is not None:
                self.set_image(self.__res_left[1][:, 500:1700].copy(), "side_demo")

    def update_epnp(self, tvec: np.ndarray, rvec: np.ndarray, side: int) -> None:
        if side:
            T, T_ = self.repo_left.push_T(rvec, tvec)
            self.loc_alarm.push_T(T, T_, 0)
            self.loc_alarm.push_RT(rvec, tvec, 0)
        else:
            self.repo_right.push_T(rvec, tvec)
            self.loc_alarm.push_RT(rvec, tvec, 1)

    def update_reproject(self) -> None:
        res_temp = self.__res_left[0]
        if isinstance(res_temp, np.ndarray):
            if res_temp.shape[0] == 0:
                self.text_api("INFO", "Repo left", "")
            else:
                armors = res_temp[:, [11, 13, 6, 7, 8, 9]]
                cars = res_temp[:, [11, 0, 1, 2, 3]]
                result = self.repo_left.check(armors, cars)
                self.repo_left.update(None, self.__res_left[1])
        res_temp = self.__res_right[0]
        if isinstance(res_temp, np.ndarray):
            if res_temp.shape[0] == 0:
                self.e_location.clear()
            else:
                armors = res_temp[:, [11, 13, 6, 7, 8, 9]]
                cars = res_temp[:, [11, 0, 1, 2, 3]]
                result = self.repo_left.check(armors, cars)
                self.e_location = result[1]
                self.repo_right.update(None, self.__res_right[1])

    def update_missile(self) -> None:
        if isinstance(self.__res_left[0], bool) and not self.__res_left[0]:
            self.missile.detect_api(self.__res_left[1], {})

    def update_location_alarm(self) -> None:
        if isinstance(self.__res_left[0], np.ndarray):
            t_loc = None
            e_loc = None
            if self.__res_left[0].shape[0] == 0:
                self.text_api("INFO", "location", "")
            else:
                if not self.using_d:
                    armors = self.__res_left[0][:, [11, 6, 7, 8, 9]]
                    armors[:, 3] = armors[:, 3] - armors[:, 1]
                    armors[:, 4] = armors[:, 4] - armors[:, 2]
                    armors = armors[np.logical_not(np.isnan(armors[:, 0]))]
                    if armors.shape[0] != 0:
                        dph = self.lidar.detect_depth(rects=armors[:, 1:].tolist()).reshape(-1, 1)
                        x0 = (armors[:, 1] + armors[:, 3] / 2).reshape(-1, 1)
                        y0 = (armors[:, 2] + armors[:, 4] / 2).reshape(-1, 1)
                        t_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, dph], axis=1)
                    armors = self.e_location[np.logical_not(np.isnan(self.e_location[:, 0]))]
                    if armors.shape[0] != 0:
                        dph = self.lidar.detect_depth(rects=armors[:, 1:].tolist()).reshape(-1, 1)
                        x0 = (armors[:, 1] + armors[:, 3] / 2).reshape(-1, 1)
                        y0 = (armors[:, 2] + armors[:, 4] / 2).reshape(-1, 1)
                        e_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, dph], axis=1)
                else:
                    armors = self.__res_left[0][:, [11, 6, 7, 8, 9]]
                    armors = armors[np.logical_not(np.isnan(armors[:, 0]))]
                    x0 = (armors[:, 1] + (armors[:, 3] - armors[:, 1]) / 2).reshape(-1, 1)
                    y0 = (armors[:, 2] + (armors[:, 4] - armors[:, 2]) / 2).reshape(-1, 1)
                    t_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, 0], axis=1)
                    armors = self.e_location[np.logical_not(np.isnan(self.e_location[:, 0]))]
                    if armors.shape[0] != 0:
                        x0 = (armors[:, 1] + armors[:, 3] / 2).reshape(-1, 1)
                        y0 = (armors[:, 2] + armors[:, 4] / 2).reshape(-1, 1)
                        e_loc = np.concatenate([armors[:, 0].reshape(-1, 1), x0, y0, 0], axis=1)
            self.loc_alarm.update(t_loc, e_loc)
            self.loc_alarm.show()

    def spin(self) -> None:

        # get images
        if self.__cam_left:
            self.__res_left = sub(self._event_left, self._que_left)
            if isinstance(self.__res_left[0], bool):
                if not self.__res_left[0]:
                    self.text_api("ERROR", "cam_left", "cam_left failed!")
                else:
                    self.text_api("INFO", "cam_left", "cam_right ok")
        if self.__cam_right:
            self.__res_right = sub(self._event_right, self._que_right)
            if isinstance(self.__res_right[0], bool):
                if not self.__res_right[0]:
                    self.text_api("ERROR", "cam_right", "cam_right failed!")
                else:
                    self.text_api("INFO", "cam_right", "cam_right ok")
        if self.__cam_far:
            self.__res_far = self.__CamFar.get_img()
            if not self.__res_far[0]:
                self.text_api("ERROR", "cam_far", "cam_far failed!")
            else:
                self.text_api("INFO", "cam_far", "cam_far ok")

        # record images
        if self.record_state:
            if isinstance(self.__res_left[1], np.ndarray):
                self.record_object[0].write(self.__res_left[1])
            if isinstance(self.__res_right[1], np.ndarray):
                self.record_object[1].write(self.__res_right[1])
            if not self.__res_far[0]:
                self.record_object[2].write(self.__res_far)

        # if in epnp_mode , just show the images
        if not self.epnp_mode:
            if self.show_pc_state and isinstance(self.__res_left[1], np.ndarray):
                depth = self.lidar.read()
                pc_show(self.__res_left[1], depth)
                self.update_image()
            else:
                self.update_image()
                self.update_reproject()
                self.update_location_alarm()
                # self.update_missile()

        # update serial
        if self.__serial:
            Port_operate.get_message(self.hp_sence)
            self.hp_sence.show()
            if Port_operate.change_view:
                Port_operate.change_view = not Port_operate.change_view
                self.view_change = int(not self.view_change)

        # if close the program
        if self.terminate:
            if self.__cam_far:
                self.__CamFar.destroy()
            self._event_close.set()
            if self.__cam_left:
                self.PRE_left.join(timeout=3)
            if self.__cam_right:
                self.PRE_right.join(timeout=3)
            del self.__res_left
            del self.__res_right
            del self.loc_alarm
            sys.exit()


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
