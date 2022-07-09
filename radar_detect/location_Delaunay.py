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
    init_ok = False
    w_points = None
    cam_rect = ()
    rect_points = None
    debug_mode = False

    def __init__(self, cam_side: str, debug: bool = False, choose=list(range(0, 80, 1)), rvec=None, tvec=None):
        self.debug_mode = debug
        if cam_side in ["cam_left", "cam_right"]:
            t = Delaunary_points[int(debug)][cam_side]
            self.cam_rect = t[0]
            self.c_points = np.array(t[1]).reshape((-1, 3)) * 1000
            self.c_points = self.c_points[list(np.array(choose)-1)]
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
        self._get_region()
        self.Dly = cv.Subdiv2D(self.cam_rect)
        self.rect_points = np.array([
            [self.cam_rect[0], self.cam_rect[1]],
            [self.cam_rect[0] + self.cam_rect[2] - 1, self.cam_rect[1]],
            [self.cam_rect[0], self.cam_rect[1] + self.cam_rect[3] - 1],
            [self.cam_rect[0] + self.cam_rect[2] - 1, self.cam_rect[1] + self.cam_rect[3] - 1]
        ])
        try:
            for i in self.cam_points:
                self.Dly.insert(tuple(i.tolist()))
            for i in self.rect_points:
                self.Dly.insert(tuple(i.tolist()))
        except Exception as e:
            print(f"[ERROR] {e}")
        self.init_ok = True

    def _get_region(self):
        points = cv.projectPoints(self.c_points, self._rvec,
                                  self._tvec, self._K_O, self._C_O)[0].astype(int).reshape(-1, 2)  # 得到反投影坐标
        rows = (points[:, 0] >= self.cam_rect[0]) & (points[:, 0] <= self.cam_rect[0] + self.cam_rect[2]) \
               & (points[:, 1] >= self.cam_rect[1]) & (points[:, 1] <= self.cam_rect[1] + self.cam_rect[3])
        self.cam_points = points[rows]
        self.w_points = self.c_points[rows] / 1000

    def push_T(self, rvec, tvec):
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
        w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
        if not self.init_ok or l.shape[0] != 4:
            pass
        else:
            if detect_type == 2:
                try:
                    res = self.Dly.findNearest(tuple(l[1:3]))[1]
                    w_point = self._cal_pos_vertex(res)
                except Exception as e:
                    w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
                    print(f"[ERROR] {e}")
            elif detect_type == 1:
                try:
                    res = self.Dly.locate(tuple(l[1:3]))
                    value = res[0]
                    if value == cv.SUBDIV2D_PTLOC_ERROR:
                        return np.ndarray((3, 1), dtype=np.float64) * np.nan
                    if value == cv.SUBDIV2D_PTLOC_INSIDE:
                        first_edge = res[1]
                        second_edge = self.Dly.getEdge(first_edge, cv.SUBDIV2D_NEXT_AROUND_LEFT)
                        third_edge = self.Dly.getEdge(second_edge, cv.SUBDIV2D_NEXT_AROUND_LEFT)
                        w_point = self._cal_pos_triangle(
                            np.array([self.Dly.edgeDst(first_edge)[1],
                                      self.Dly.edgeDst(second_edge)[1],
                                      self.Dly.edgeDst(third_edge)[1],
                                      ]),
                            l[1:3]
                        )
                    if value == cv.SUBDIV2D_PTLOC_ON_EDGE:
                        first_edge = res[1]
                        w_point = self._cal_pos_edge(
                            np.array([self.Dly.edgeOrg(first_edge)[1],
                                      self.Dly.edgeDst(first_edge)[1]
                                      ]),
                            l[1:3]
                        )
                    if value == cv.SUBDIV2D_PTLOC_VERTEX:
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
        if not self._check(pts):
            return np.ndarray((3, 1), dtype=np.float64) * np.nan
        else:
            magnitude1 = pts[1] - pts[0]
            magnitude3 = pt - pts[0]
            k_1 = np.dot(magnitude1, magnitude3) / self._mag_pow(magnitude1)
            h_1 = self.w_points(np.where(self.cam_points == pts[0]))
            h_2 = self.w_points(np.where(self.cam_points == pts[1]))
            return h_1 * (1 - k_1) + h_2 * k_1

    def _cal_pos_triangle(self, pts: np.ndarray, pt) -> np.ndarray:
        if not self._check(pts):
            return np.ndarray((3, 1), dtype=np.float64) * np.nan
        else:
            magnitude1 = pts[1] - pts[0]
            magnitude2 = pts[2] - pts[0]
            magnitude3 = pt - pts[0]
            L1 = np.sqrt(magnitude1.dot(magnitude1))
            L2 = np.sqrt(magnitude2.dot(magnitude2))
            L3 = np.sqrt(magnitude3.dot(magnitude3))
            angle2 = np.arccos(magnitude3.dot(magnitude1) / (L3 * L1))
            angle1 = np.arccos(magnitude3.dot(magnitude2) / (L3 * L2))
            angle = np.arccos(magnitude2.dot(magnitude1) / (L2 * L1))
            k_1 = np.sin(angle1) / np.sin(angle) * L3 / L1
            k_2 = np.sin(angle2) / np.sin(angle) * L3 / L2
            judge = (self.cam_points[:, 0] == pts[0][0]) & (self.cam_points[:, 1] == pts[0][1])
            num = np.where(judge == True)[0]
            h_1 = self.w_points[num]
            judge = (self.cam_points[:, 0] == pts[1][0]) & (self.cam_points[:, 1] == pts[1][1])
            num = np.where(judge == True)[0]
            h_2 = self.w_points[num]
            judge = (self.cam_points[:, 0] == pts[2][0]) & (self.cam_points[:, 1] == pts[2][1])
            num = np.where(judge == True)[0]
            h_3 = self.w_points[num]
            return h_1 * (1 - k_1 - k_2) + h_2 * k_1 + h_3 * k_2

    def _cal_pos_vertex(self, pts) -> np.ndarray:
        if not self._check(pts):
            return np.ndarray((3, 1), dtype=np.float64) * np.nan
        else:
            judge = (self.cam_points[:, 0] == pts[0]) & (self.cam_points[:, 1] == pts[1])
            num = np.where(judge == True)[0]
            return self.w_points[num]

    def _check(self, pts: np.ndarray) -> bool:
        for i in pts:
            res = i == self.rect_points
            rows = (res[:, 0] == True) & (res[:, 1] == True)
            if rows.any():
                return False
        return True

    @staticmethod
    def _mag_pow(magnitude):
        return pow(magnitude[0], 2) + pow(magnitude[1], 2)

    def get_points(self):
        return self.cam_points

