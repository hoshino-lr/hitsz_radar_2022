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
from PyQt5.QtCore import QTimer

from resources.config import INIT_FRAME_PATH,\
    MAP_PATH,enemy_color,map_size,absolute_path

from mapping.ui import Ui_MainWindow

class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):  # 这个地方要注意Ui_MainWindow
    def __init__(self):
        super(Mywindow, self).__init__()
        self.setupUi(self)
        self.view_change = 0  # 视角切换控制符
        INIT_FRAME_PATH1 = os.path.join(absolute_path, INIT_FRAME_PATH)
        MAP_PATH1 = os.path.join(absolute_path, MAP_PATH)
        frame = cv2.imread(INIT_FRAME_PATH1)
        frame_m = cv2.imread(MAP_PATH1)
        self.close = False
        # 小地图翻转

        if enemy_color:
            frame_m = cv2.rotate(frame_m, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame_m = cv2.rotate(frame_m, cv2.ROTATE_90_CLOCKWISE)
        frame_m = cv2.resize(frame_m, map_size)
        self.set_image(frame, "main_demo")
        self.set_image(frame_m, "map")
        del frame, frame_m

        self.feedback_textBrowser = []  # textBrowser信息列表

        self.record_object = None  # 视频录制对象列表，先用None填充
        # 录制保存位置
        self.save_title = ''  # 当场次录制文件夹名

        # 反馈信息栏，显示初始化
        self.set_text("textBrowser", "intializing...")
        # 雷达和位姿估计状态反馈栏，初始化为全False
        self.record_state = False  # 0:开始 1:停止
        # self.btn1.setText("开始录制")
        self.ChangeView.setText("视角")
        # self.btn3.setText("位姿估计")
        self.btn4.setText("终止程序")


    def btn1_on_clicked(self):
        '''
        切换视角
        '''
        from resources.config import USEABLE
        if self.view_change:
            if USEABLE['cam_left'] == True:
                self.view_change = 0
                self.ChangeView.setText("视角 左")
            else:
                pass
        else:
            if USEABLE['cam_right'] == True:
                self.view_change = 1
                self.ChangeView.setText("视角 右")
            else:
                pass

    def btn4_on_clicked(self):
        '''
        关闭程序
        '''
        self.close = True

    def set_image(self, frame, position=""):
        """
        Image Show Function

        :param frame: the image to show
        :param position: where to show
        :return: a flag to indicate whether the showing process have succeeded or not
        """
        if not position in ["main_demo", "map", "textBrowser"]:
            print("[ERROR] The position isn't a member of this UIwindow")
            return False

        # get the size of the corresponding window
        if position == "main_demo":
            width = self.main_demo.width()
            height = self.main_demo.height()
        elif position == "map":
            width = self.map.width()
            height = self.map.height()
        elif position == "textBrowser":
            width = self.textBrowser.width()
            height = self.textBrowser.height()

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
        elif position == "map":
            self.map.setPixmap(temp_pixmap)
            self.map.setScaledContents(True)
        elif position == "textBrowser":
            self.textBrowser.setPixmap(temp_pixmap)
            self.textBrowser.setScaledContents(True)
        return True

    def set_text(self, position: str, message=""):
        """
        to set text in the QtLabel

        :param position: must be one of the followings: "feedback", "textBrowser", "state"
        :param message: For feedback, a string you want to show in the next line;
        For the others, a string to show on that position , which will replace the origin one.
        :return:
        a flag to indicate whether the showing process have succeeded or not
        """
        if position not in ["feedback", "textBrowser", "state"]:
            print("[ERROR] The position isn't a member of this UIwindow")
            return False
        if position == "feedback":
            if len(self.feedback_textBrowser) >= 12:  # the feedback could contain at most 12 messages lines.
                self.feedback_textBrowser.pop(0)
            self.feedback_textBrowser.append(message)
            # Using "<br \>" to combine the contents of the message list to a single string message
            message = "<br \>".join(self.feedback_textBrowser)  # css format to replace \n
            self.feedback.setText(message)
            return True
        if position == "state":
            self.state.setText(message)
            return True
        if position == "textBrowser":
            self.textBrowser.setText(message)
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


