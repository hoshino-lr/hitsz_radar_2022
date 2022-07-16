# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : drawing.py
# Time       ：2022/7/7 下午10:43
# Author     ：author name
# version    ：python 3.8
# Description：
"""
import numpy as np
import time
import queue
from typing import Dict
import cv2 as cv
import math


class draw_message(object):
    valid_time = 3

    def __init__(self, title_, type_, message_, level_):
        """

        Args:
            title_: 标题
            type_: 类型
            message_: 内容 【message】【bbox】 【message+(point)】
            level_: 等级
        """
        self.title = title_
        self.type = type_
        self.message = message_
        self.level = level_
        self.time = time.time()

    def check_valid(self) -> bool:
        if time.time() - self.time > self.valid_time:
            return False
        else:
            return True


class message_box(object):
    queue_size = 4

    def __init__(self):
        self.__message_base = {
            "info": {},
            "warning:": {},
            "critical": {}
        }
        self.__info_queue = queue.Queue(self.queue_size)
        self.__warning_queue = queue.Queue(self.queue_size)
        self.__critical_queue = queue.Queue(self.queue_size)

    def add_message(self, message: draw_message) -> None:
        if message.level in ["info", "warning", "critical"]:
            self.__message_base[message][message.title] = message
        if message.type == 0:
            if message.level == "info":
                if self.__info_queue.full():
                    self.__info_queue.get()
                    self.__info_queue.put(message)
            if message.level == "warning":
                if self.__warning_queue.full():
                    self.__warning_queue.get()
                    self.__warning_queue.put(message)
            if message.level == "critical":
                if self.__critical_queue.full():
                    self.__critical_queue.get()
                    self.__critical_queue.put(message)

    def refresh_message(self):
        for key1 in self.__message_base.keys():
            item = self.__message_base[key1]
            for key2 in item.keys():
                if not item[key2].check_valid():
                    item.pop(key2)

    def clear_message(self):
        for key in self.__message_base.keys():
            self.__message_base[key].clear()
        while not self.__critical_queue.empty():
            self.__critical_queue.get()
        while not self.__info_queue.empty():
            self.__info_queue.get()
        while not self.__warning_queue.empty():
            self.__warning_queue.get()

    def get_top_message(self) -> str:
        if not self.__critical_queue.empty():
            item = self.__critical_queue.queue[self.__critical_queue.qsize() - 1]
            if item.check_valid():
                return item
            else:
                return ""
        else:
            return ""

    @staticmethod
    def _get_prefix(l_: draw_message) -> str:
        period = time.time() - l_.time
        prefix = ""
        if period > 60:
            prefix = "[无效] "
        elif period > 30:
            prefix = "[半分钟前] "
        elif period > 10:
            prefix = "[10秒钟前] "
        elif period > 0:
            prefix = "[刚刚] "

        return prefix + l_.message

    def get_all_messages(self) -> list:
        item = []
        if not self.__critical_queue.empty():
            for i in self.__critical_queue.queue:
                item.append(drawing.set_text("ERROR", self._get_prefix(i)))

        if not self.__warning_queue.empty():
            for i in self.__warning_queue.queue:
                item.append(drawing.set_text("WARNING", self._get_prefix(i)))

        if not self.__info_queue.empty():
            for i in self.__info_queue.queue:
                item.append(drawing.set_text("INFO", self._get_prefix(i)))
        return item

    def get_valid_messages(self):
        res = []
        for item in self.__message_base.keys():
            res.append(list(self.__message_base[item].values()))
        return res


class drawing(object):
    """
    画图类，用来处理画图功能
    """

    def __init__(self):
        self._message_alarm = message_box()
        self._message_board = message_box()
        self._message_text = message_box()

    @staticmethod
    def _draw_text(pic: np.ndarray, input_text) -> None:
        """

        Args:
            pic:
            input_text:

        Returns:

        """
        tl = 5  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        text_size = cv.getTextSize(input_text, 0, tl, tf)
        middle_point = (int((pic.shape[1] - text_size[0][0]) / 2), pic.shape[0] - text_size[0][1])
        cv.putText(pic, input_text, middle_point, 0, tl, (20, 20, 255), thickness=tf,
                   lineType=cv.LINE_AA)

    @staticmethod
    def _draw_alarm(pic: np.ndarray, message: list) -> None:
        """

        Args:
            pic:
            message:

        Returns:

        """
        tl = 6  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness
        point = message[1]
        point[1] = point[1] - 50
        cv.putText(pic, message[0], point, 0, tl / 3, (20, 20, 255), thickness=tf,
                   lineType=cv.LINE_AA)

    @staticmethod
    def _draw_board(pic: np.ndarray, message: list) -> None:
        """

        Args:
            pic: frame
            message: bbox

        Returns:

        """
        tl = 3  # line/font thickness
        c1 = (message[0], message[1])
        c2 = (message[0] + message[2], message[1] + message[3])
        cv.rectangle(pic, c1, c2, (255, 0, 255), thickness=tl, lineType=cv.LINE_AA)

    def update(self, message: draw_message) -> None:
        """

        Args:
            message: list [title, message, type, level]
                     type: 0:text 1:alarm 2:board
                     level: "info" "warning" "critical"
                     board: message -> bbox
                     alarm: message -> [text, (point)]

        Returns:

        """
        if message.type == 0:
            self._message_text.add_message(message)
        if message.type == 1:
            self._message_alarm.add_message(message)
        if message.type == 2:
            self._message_board.add_message(message)

    def info_update_dly(self, cam_points):
        self.cam_points = cam_points

    def info_update_reproject(self, scene_region):
        self.scene_region = scene_region

    def _refresh(self):
        self._message_alarm.refresh_message()
        self._message_text.refresh_message()
        self._message_board.refresh_message()

    def clear_message(self):
        self._message_board.clear_message()
        self._message_text.clear_message()
        self._message_alarm.clear_message()

    def draw_message(self, pic) -> None:
        """

        Args:
            pic: np.array

        Returns:

        """
        if not isinstance(pic, np.ndarray):
            return
        else:
            self._refresh()
            text = self._message_text.get_top_message()
            self._draw_text(pic, text)

            res = self._message_alarm.get_valid_messages()
            for item in res:
                for it in item:
                    self._draw_alarm(pic, item)

            res = self._message_board.get_valid_messages()
            for item in res:
                for it in item:
                    self._draw_board(pic, item)

    def browser_message(self) -> list:
        return self._message_text.get_all_messages()

    def draw_alarm_area(self, frame, rp_alarming: dict) -> None:
        """
        画预警区域

        Args:

            frame:
            rp_alarming:

        Returns:

        """
        from config import enemy_color, color2enemy
        for i in self.scene_region.keys():
            recor = self.scene_region[i]
            _, shape_type, team, location, height_type = i.split('_')
            if color2enemy[team] != enemy_color:
                continue
            else:
                if i in rp_alarming.keys():
                    cv.polylines(frame, [recor], isClosed=1, color=(0, 255, 0), lineType=0)
                else:
                    cv.polylines(frame, [recor], isClosed=1, color=(0, 0, 255), lineType=0)

    @staticmethod
    def get_rgb(max_distance: float, min_distance: float, distance: float) -> tuple:
        scale = (max_distance - min_distance) / 10
        if distance < min_distance:
            r = 0
            g = 0
            b = 0xff
        elif distance < min_distance + scale:
            r = 0
            g = int((distance - min_distance) / scale * 255) & 0xff
            b = 0xff
        elif distance < min_distance + scale * 2:
            r = 0
            g = 0xff
            b = int((distance - min_distance - scale) / scale * 255) & 0xff
        elif distance < min_distance + scale * 4:
            r = int((distance - min_distance - scale * 2) / (scale * 2) * 255) & 0xff
            g = 0xff
            b = 0
        elif distance < min_distance + scale * 7:
            r = 0xff
            g = int((distance - min_distance - scale * 4) / (scale * 3) * 255) & 0xff
            b = 0
        elif distance < min_distance + scale * 10:
            r = 0xff
            g = 0
            b = int((distance - min_distance - scale * 7) / (scale * 3) * 255) & 0xff
        else:
            r = 0xff
            g = 0
            b = 0xff

        return b, g, r

    @staticmethod
    def draw_pc(src: np.ndarray, depth: np.ndarray) -> None:
        if not src.shape[0]:
            return
        else:
            for i in range(0, depth.shape[0] - 1, 15):
                for j in range(0, depth.shape[1] - 1, 15):
                    v = depth[i][j]
                    if math.isnan(depth[i][j]):
                        continue
                    rgb = drawing.get_rgb(20, 4, depth[i][j])
                    cv.circle(src, (j, i), radius=3, color=rgb, thickness=-1)

    def draw_CamPoints(self, pic):
        for i in self.cam_points:
            cv.circle(pic, tuple(i), 10, (0, 255, 0), -1)

    @staticmethod
    def set_text(_type: str, message: str) -> str:
        """
        to set text in the QtLabel
        :param message: message that will be put on the screen;
        :param _type: message level [ERROR,INFO,WARNING];
        """
        # Using "<br \>" to combine the contents of the message list to a single string message
        if _type == "ERROR":
            msg = f"<font color='#FF0000'><b>{message}</b></font>"
        elif _type == "INFO":
            msg = f"<font color='#404040'><b>{message}</b></font>"
        else:
            msg = f"<font color='#FF8033'><b>{message}</b></font>"
        return msg.replace('\n', "<br />")
