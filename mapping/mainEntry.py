'''
自定义UI类
使用Qt设计的自定义UI
对上海交通大学的代码进行了一些删改
created by 李龙 2021/1
'''
from re import T
import sys
import time
from datetime import datetime
import os
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from radar_detect.solve_pnp import SolvePnp
from resources.config import INIT_FRAME_PATH, \
    MAP_PATH, enemy_color, map_size

from mapping.ui import Ui_MainWindow


class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):  # 这个地方要注意Ui_MainWindow
    def __init__(self):
        super(Mywindow, self).__init__()
        self.Eapi = None
        self.Rapi = None
        self.setupUi(self)
        self.view_change = 0  # 视角切换控制符
        frame = cv2.imread(INIT_FRAME_PATH)
        frame_m = cv2.imread(MAP_PATH)
        self.terminate = False
        # 小地图翻转
        if enemy_color:
            frame_m = cv2.rotate(frame_m, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame_m = cv2.rotate(frame_m, cv2.ROTATE_90_CLOCKWISE)
        frame_m = cv2.resize(frame_m, map_size)
        self.set_image(frame, "side_demo")
        self.set_image(frame, "far_demo")
        self.set_image(frame, "main_demo")
        self.set_image(frame_m, "map")
        del frame, frame_m

        self.feedback_textBrowser = {}  # textBrowser信息列表

        self.sp = SolvePnp()
        # 反馈信息栏，显示初始化
        self.set_text("textBrowser", "TIMEING", "intializing...")
        # 雷达和位姿估计状态反馈栏，初始化为全False
        self.record_state = False  # 0:停止 1:开始
        self.epnp_mode = False  # 0:停止 1:开始
        self.record.setText("开始录制")
        self.ChangeView.setText("视角 左")
        self.epnp.setText("位姿估计")
        self.CloseProgram.setText("终止程序")

    def ChangeView_on_clicked(self):
        """
        切换视角
        """
        from resources.config import USEABLE
        if self.view_change:
            if USEABLE['cam_left']:
                self.view_change = 0
                self.ChangeView.setText("视角 左")
            else:
                pass
        else:
            if USEABLE['cam_right']:
                self.view_change = 1
                self.ChangeView.setText("视角 右")
            else:
                pass

    def record_on_clicked(self):
        self.record_state = not self.record_state

    def epnp_on_clicked(self):
        self.epnp_mode = not self.epnp_mode
        if self.epnp_mode:
            self.epnp.setText("jie shu")
        else:
            self.epnp.setText("位姿估计")
            self.sp.clc()

    def CloseProgram_on_clicked(self):
        """
        关闭程序
        """
        self.terminate = True

    def epnp_mouseEvent(self, event):
        if self.epnp_mode:
            if event.button() == Qt.LeftButton:
                pos_x, pos_y = self.sp.add_point(event.x(), event.y())
            if event.button() == Qt.RightButton:
                self.sp.del_point()

    def epnp_keyEvent(self, event):
        # 这里event.key（）显示的是按键的编码
        print("按下：" + str(event.key()))
        # 举例，这里Qt.Key_A注意虽然字母大写，但按键事件对大小写不敏感
        if event.key() == Qt.Key_Escape:
            self.sp.clc()
        if event.key() == Qt.Key_Enter:
            result = self.sp.locate_pick()
            if result:
                self.sp.clc()
                self.epnp.setText("位姿估计")
                self.Eapi(self.sp.tvec, self.sp.rvec, self.view_change)
                self.epnp_mode = not self.epnp_mode
            else:
                pass

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.shape[2] == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
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


if __name__ == "__main__":
    # demo of the window class
    from resources.config import config_init

    config_init()
    app = QtWidgets.QApplication(sys.argv)
    myshow = Mywindow()
    timer = QTimer()
    myshow.set_text("textBrowser", '<br \>'.join(['', "<font color='#FF0000'><b>base detect enermy</b></font>",
                                                  "<font color='#FF0000'><b>base detect enermy</b></font>",
                                                  f"哨兵:<font color='#FF0000'><b>{99:d}</b></font>"]))

    myshow.show()  # 显示
    MAP_PATH1 = os.path.join(absolute_path, MAP_PATH)
    frame_m = cv2.imread(MAP_PATH1)

    timer.timeout.connect(show_picture)
    timer.start(0)
    sys.exit(app.exec_())
