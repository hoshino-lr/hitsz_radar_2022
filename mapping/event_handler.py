import queue
import sys
import time
import os
import cv2 as cv
import numpy as np
import threading
import logging
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QRect

import config
from config import USEABLE, enemy_color, cam_config, using_video, enemy2color
from mapping.ui_v5 import Ui_MainWindow
from radar_detect.Linar_rs import Radar
from radar_detect.reproject import Reproject
from radar_detect.location_alarm import Alarm
from mul_manager.pro_manager import sub, process_detect, process_detect_rs
from Serial.HP_show import HP_scene
from Serial.port_operation import Port_operate
from radar_detect.solve_pnp import SolvePnp
from radar_detect.eco_forecast import eco_forecast
from radar_detect.decision import decision_tree
from mapping.drawing import drawing, draw_message


class UI_MainWindows_EventHandler(Ui_MainWindow):
    def condition_key_on_clicked(self, event) -> None:
        if event.key() == Qt.Key_Q:
            self.loc_alarm.change_mode(self.view_change)
            self._use_lidar = not self.loc_alarm.state[0]
            self.set_board_text("INFO", "定位状态", self.loc_alarm.get_mode())
        elif event.key() == Qt.Key_T:  # 将此刻设为start_time
            Port_operate.start_time = time.time()
        elif event.key() == Qt.Key_3:  # xiao能量机关激活
            Port_operate.set_state(3, time.time())
        elif event.key() == Qt.Key_4:  # da能量机关激活
            Port_operate.set_state(4, time.time())
        elif event.key() == Qt.Key_E:  # 添加工程预警
            self.text_api(draw_message("engineer", 0, "Engineer", "critical"))
        elif event.key() == Qt.Key_F:  # 添加飞坡预警
            self.text_api(draw_message("fly", 0, "Fly", "critical"))
        elif event.key() == Qt.Key_C:  # 添加清零
            self.draw_module.clear_message()
        else:
            pass

    def ChangeView_on_clicked(self) -> None:
        """
        切换视角
        """
        pass
        # if self.view_change:
        #     if self.__cam_left:
        #         self.view_change = 0
        #     else:
        #         return
        # else:
        #     if self.__cam_right:
        #         self.view_change = 1
        #     else:
        #         return

    def record_on_clicked(self) -> None:
        if not self.record_state:
            self.record.setText("录制 正在录制")
            self.record_state = True
            self._left_record.set()
            self._right_record.set()
        else:
            self.record.setText("录制")
            self.record_state = False
            self._left_record.set()
            self._right_record.set()

    def showpc_on_clicked(self) -> None:
        self.show_pc_state = not self.show_pc_state
        self.epnp_mode = False

    def CloseProgram_on_clicked(self) -> None:
        """
        关闭程序
        """
        self.terminate = True

    def SpeedUp_on_clicked(self) -> None:
        config.global_speed += 0.25
        self._event_speed.set()
        self.CurrentSpeed.setText(f"x{config.global_speed}")

    def SlowDown_on_clicked(self) -> None:
        if(config.global_speed <= 0.25):
            return
        config.global_speed -= 0.25
        self._event_speed.set()
        self.CurrentSpeed.setText(f"x{config.global_speed}")

    def Pause_on_clicked(self) -> None:
        config.global_pause = not config.global_pause
        if config.global_pause:
            self.Pause.setText("继续")
        else:
            self.Pause.setText("暂停")

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
            self._update_epnp(self.sp.translation, self.sp.rotation, self.view_change)
            self.draw_module.info_update_dly(self.loc_alarm.get_draw(0))
            self.draw_module.info_update_reproject(self.repo_left.get_scene_region())

    def epnp_mouseEvent(self, event) -> None:
        if self.epnp_mode:
            if event.button() == Qt.LeftButton:
                x = event.x()
                y = event.y()
                pointF = self.pnp_demo.mapToScene(QRect(0, 0, self.pnp_demo.viewport().width(),
                                                        self.pnp_demo.viewport().height()))[0]
                x_offset = pointF.x()
                y_offset = pointF.y()
                x = int((x + x_offset) / self.pnp_demo.width() * self.epnp_image.width())
                y = int((y + y_offset) / self.pnp_demo.height() * self.epnp_image.height())
                self.sp.add_point(x, y)

    def pc_mouseEvent(self, event) -> None:
        if self.show_pc_state:
            # TODO: 这里的3072是个常值，最好改了
            x = int(event.x() / self.main_demo.width() * cam_config['cam_left']['size'][0])
            y = int(event.y() / self.main_demo.height() * cam_config['cam_left']['size'][1]) + cam_config['cam_left']['roi'][1]
            w = 10
            h = 5
            # 格式定义： [N, [bbox(xyxy), conf, cls, bbox(xyxy), conf, cls, col, N]
            self.repo_left.check(np.array([x, y, w, h, 1., 1, x, y, w, h, 1., 1., 1., 0]).reshape(1, -1))
            dph = self.lidar.detect_depth(
                rects=[[x - 5, y - 2.5, w, h]]).reshape(-1, 1)
            if not np.any(np.isnan(dph)):
                self.loc_alarm.pc_location(self.view_change, np.concatenate(
                    [np.array([[1., x, y]]), dph], axis=1))

    def eco_key_on_clicked(self, event) -> None:
        step = 20
        if event.key() == Qt.Key_P:  # 换框
            self.set_board_text("INFO", "框框状态", f"{self._num_kk + 1}框框")
            self._num_kk = abs(1 - self._num_kk)
        elif event.key() == Qt.Key_W and self._Eco_cut[2] > step:
            self._Eco_cut[2] -= step
            self._Eco_cut[3] -= step
        elif event.key() == Qt.Key_S and self._Eco_cut[3] < 2048 - step:
            self._Eco_cut[2] += step
            self._Eco_cut[3] += step
        elif event.key() == Qt.Key_A and self._Eco_cut[0] > step:
            self._Eco_cut[0] -= step
            self._Eco_cut[1] -= step
        elif event.key() == Qt.Key_D and self._Eco_cut[1] < 3072 - step:
            self._Eco_cut[0] += step
            self._Eco_cut[1] += step

    def eco_mouseEvent(self, event) -> None:
        if self.__pic_left is None:
            return
        x = event.x()
        y = event.y()
        x = int(x / self.far_demo.width() * (self._Eco_cut[1] - self._Eco_cut[0]) + self._Eco_cut[0])
        y = int(y / self.far_demo.height() * (self._Eco_cut[3] - self._Eco_cut[2]) + self._Eco_cut[2])
        if event.button() == Qt.LeftButton:
            self._Eco_point[0] = x
            self._Eco_point[1] = y
        if event.button() == Qt.RightButton:
            self._Eco_point[2] = x
            self._Eco_point[3] = y
        if self._Eco_point[2] > self._Eco_point[0] and self._Eco_point[3] > self._Eco_point[1]:
            self.supply_detector.update_ori(self.__pic_left, self._Eco_point, self._num_kk)
            self.text_api(
                draw_message("eco_board", 2,
                             (self._Eco_point[0], self._Eco_point[1],
                              self._Eco_point[2],
                              self._Eco_point[3]), "info"))

    def epnp_next_on_clicked(self) -> None:
        self.sp.step(1)

    def epnp_back_on_clicked(self) -> None:
        self.sp.step(-1)

    def epnp_del(self) -> None:
        self.sp.del_point()

    def epnp_clear_on_clicked(self) -> None:
        self.sp.clc()