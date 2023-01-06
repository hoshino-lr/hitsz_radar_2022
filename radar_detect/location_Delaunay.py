# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : location_Delaunay.py
# Time       ：2022/4/15 下午9:54
# Author     ：李龙
# version    ：python 3.8
# Description：
"""
import cv2 as cv
import numpy as np
from config import Delaunary_points, cam_config
from mapping.drawing import draw_message


class location_Delaunay(object):
    """

    """
    init_ok = False # 已经初始化德劳内投影点
    w_points = None # 投影点
    cam_rect = ()   # 相机能拍摄到的矩形范围
    rect_points = None  # 矩形范围的顶点
    debug_mode = False  # 调试模式

    def __init__(self, cam_side: str, debug: bool = False, choose=list(range(0, 80, 1)), rvec=None, tvec=None):
        self.debug_mode = debug
        if cam_side in ["cam_left", "cam_right"]:
            t = Delaunary_points[int(debug)][cam_side]
            self.cam_rect = t[0]
            # 德劳内标记点
            self.c_points = np.array(t[1]).reshape((-1, 3)) * 1000
            self.c_points = self.c_points[list(np.array(choose) - 1)]
            # 相机内外参
            self._rvec = cam_config[cam_side]["rvec"]
            self._tvec = cam_config[cam_side]["tvec"]
            self._K_O = cam_config[cam_side]['K_0']
            self._C_O = cam_config[cam_side]['C_0']
        else:
            return
        if rvec is not None:
            self._rvec = rvec
        if tvec is not None:
            self._tvec = tvec
        # 德劳内定位初始化操作
        self._get_region()
        self.Dly = cv.Subdiv2D(self.cam_rect)
        self.rect_points = np.array([
            [self.cam_rect[0], self.cam_rect[1]],
            [self.cam_rect[0] + self.cam_rect[2] - 1, self.cam_rect[1]],
            [self.cam_rect[0], self.cam_rect[1] + self.cam_rect[3] - 1],
            [self.cam_rect[0] + self.cam_rect[2] - 1, self.cam_rect[1] + self.cam_rect[3] - 1]
        ])
        # 往Subdiv2D实例中插入德劳内标记点的投影点，划分德劳内三角定位区域
        try:
            for i in self.cam_points:
                self.Dly.insert(tuple(i.tolist()))
            for i in self.rect_points:
                self.Dly.insert(tuple(i.tolist()))
        except Exception as e:
            print(f"[ERROR] {e}")
        self.init_ok = True

    def _get_region(self):
        """
        投影德劳内坐标点，获取图像上对应的点坐标
        """
        points = cv.projectPoints(self.c_points, self._rvec,
                                  self._tvec, self._K_O, self._C_O)[0].astype(int).reshape(-1, 2)  # 得到反投影坐标
        # 只选用在图像矩形中的点
        rows = (points[:, 0] >= self.cam_rect[0]) & (points[:, 0] <= self.cam_rect[0] + self.cam_rect[2]) \
               & (points[:, 1] >= self.cam_rect[1]) & (points[:, 1] <= self.cam_rect[1] + self.cam_rect[3])
        # 德劳内标记点的投影点
        self.cam_points = points[rows]
        self.w_points = self.c_points[rows] / 1000

    def push_T(self, rvec, tvec):
        """
        投影德劳内坐标点，并进行划分区域的初始化操作
        """
        self._rvec = rvec
        self._tvec = tvec
        self._get_region()
        del self.Dly
        self.Dly = cv.Subdiv2D(self.cam_rect)
        try:
            for i in self.cam_points:
                self.Dly.insert(tuple(i.tolist()))
            for i in self.rect_points:
                self.Dly.insert(tuple(i.tolist()))
        except Exception as e:
            print(f"[ERROR] {e}")
        # triangle = self.Dly.getTriangleList()
        self.init_ok = True

    def get_point_pos(self, l: np.ndarray, detect_type: int):
        """
        德劳内定位函数，
        """
        # 输入4维向量 输出3维向量（点坐标）
        w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
        if not self.init_ok or l.shape[0] != 4:
            pass
        else:
            # 最近点
            if detect_type == 2:
                try:
                    res = self.Dly.findNearest(tuple(l[1:3]))[1]
                    w_point = self._cal_pos_vertex(res)
                except Exception as e:
                    w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
                    print(f"[ERROR] {e}")
            elif detect_type == 1:
                try:
                    # 截去输入的第一维cls
                    # res[0]：类型 res[1]：边 res[2]：顶点
                    res = self.Dly.locate(tuple(l[1:3]))
                    value = res[0]
                    if value == cv.SUBDIV2D_PTLOC_ERROR:
                        return np.ndarray((3, 1), dtype=np.float64) * np.nan
                    if value == cv.SUBDIV2D_PTLOC_INSIDE:
                        # 在划分的德劳内三角形区域内
                        first_edge = res[1]
                        second_edge = self.Dly.getEdge(first_edge, cv.SUBDIV2D_NEXT_AROUND_LEFT)
                        third_edge = self.Dly.getEdge(second_edge, cv.SUBDIV2D_NEXT_AROUND_LEFT)
                        # 计算位置
                        w_point = self._cal_pos_triangle(
                            np.array([self.Dly.edgeDst(first_edge)[1],
                                      self.Dly.edgeDst(second_edge)[1],
                                      self.Dly.edgeDst(third_edge)[1],
                                      ]),
                            l[1:3]
                        )
                    if value == cv.SUBDIV2D_PTLOC_ON_EDGE:
                        # 在三角形边上
                        first_edge = res[1]
                        w_point = self._cal_pos_edge(
                            np.array([self.Dly.edgeOrg(first_edge)[1],
                                      self.Dly.edgeDst(first_edge)[1]
                                      ]),
                            l[1:3]
                        )
                    if value == cv.SUBDIV2D_PTLOC_VERTEX:
                        # 在三角形顶点
                        w_point = self._cal_pos_vertex(np.ndarray(self.cam_points[res[2]]))
                except Exception as e:
                    w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
                    print(f"[ERROR] {e}")
            else:
                w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
            if w_point.reshape(-1).shape[0] == 0:
                w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
        return w_point

    def _cal_pos_edge(self, pts: np.ndarray, pt) -> np.ndarray:
        """
        计算在边上的点的坐标
        :param pts: 边端点的投影坐标，(2,2) ndarray
        :param pt: 待定位点的图像坐标，长度为2的一维ndarray
        :return: 待定位点的定位结果，世界坐标，长度为3的一维ndarray
        """
        if not self._check(pts):
            return np.ndarray((3, 1), dtype=np.float64) * np.nan
        else:
            # 求分点比例
            magnitude1 = pts[1] - pts[0]
            magnitude3 = pt - pts[0]
            k_1 = np.dot(magnitude1, magnitude3) / self._mag_pow(magnitude1)
            # 德劳内投影点对应世界坐标点
            h_1 = self.w_points(np.where(self.cam_points == pts[0]))
            h_2 = self.w_points(np.where(self.cam_points == pts[1]))
            # 定比分点计算待定位点坐标
            return h_1 * (1 - k_1) + h_2 * k_1

    def _cal_pos_triangle(self, pts: np.ndarray, pt) -> np.ndarray:
        """
        计算三角形内点的坐标
        :param pts: 三角形顶点的投影坐标数组，(3,2) ndarray
        :param pt: 待定位点的图像坐标，长度为2的一维ndarray
        :return: 待定位点的定位结果，世界坐标，长度为3的一维ndarray
        """
        if not self._check(pts):
            return np.ndarray((3, 1), dtype=np.float64) * np.nan
        else:
            # 以pts0到pts1、pts2的向量为基底，使用向量运算计算pts0到pt的向量，从而计算待定位点pt的世界坐标
            magnitude1 = pts[1] - pts[0]
            magnitude2 = pts[2] - pts[0]
            magnitude3 = pt - pts[0]
            # 求向量模
            L1 = np.sqrt(magnitude1.dot(magnitude1))
            L2 = np.sqrt(magnitude2.dot(magnitude2))
            L3 = np.sqrt(magnitude3.dot(magnitude3))
            # 用基底表示pts0到pt的向量
            angle2 = np.arccos(magnitude3.dot(magnitude1) / (L3 * L1))
            angle1 = np.arccos(magnitude3.dot(magnitude2) / (L3 * L2))
            angle = np.arccos(magnitude2.dot(magnitude1) / (L2 * L1))
            k_1 = np.sin(angle1) / np.sin(angle) * L3 / L1
            k_2 = np.sin(angle2) / np.sin(angle) * L3 / L2
            # 寻找世界坐标点
            judge = (self.cam_points[:, 0] == pts[0][0]) & (self.cam_points[:, 1] == pts[0][1])
            num = np.where(judge == True)[0]
            h_1 = self.w_points[num]
            judge = (self.cam_points[:, 0] == pts[1][0]) & (self.cam_points[:, 1] == pts[1][1])
            num = np.where(judge == True)[0]
            h_2 = self.w_points[num]
            judge = (self.cam_points[:, 0] == pts[2][0]) & (self.cam_points[:, 1] == pts[2][1])
            num = np.where(judge == True)[0]
            h_3 = self.w_points[num]
            # 计算pt世界坐标
            return h_1 * (1 - k_1 - k_2) + h_2 * k_1 + h_3 * k_2

    def _cal_pos_vertex(self, pts) -> np.ndarray:
        """
        顶点处点坐标
        :param pts: 待定位坐标点，[x, y]
        :return: 世界坐标
        """
        if not self._check(pts):
            return np.ndarray((3, 1), dtype=np.float64) * np.nan
        else:
            # 找到图像坐标对应的世界坐标
            judge = (self.cam_points[:, 0] == pts[0]) & (self.cam_points[:, 1] == pts[1])
            num = np.where(judge == True)[0]
            return self.w_points[num]

    def _check(self, pts: np.ndarray) -> bool:
        """
        判断点是否未落在图像边缘
        :param pts: 待判断的点，[x, y]
        :return: 布尔值，False 代表落在边缘，True代表未落在边缘
        """
        for i in pts:
            res = i == self.rect_points
            rows = (res[:, 0] == True) & (res[:, 1] == True)
            if rows.any():
                return False
        return True

    @staticmethod
    def _mag_pow(magnitude):
        """
        计算二维向量的平方
        :param magnitude: 一个二维向量
        :return: magnitude的平方
        """
        return pow(magnitude[0], 2) + pow(magnitude[1], 2)

    def get_points(self):
        """
        获取德劳内标记点
        :return: (N,2) ndarray，表示N个德劳内标记点的图像坐标
        """
        return self.cam_points
