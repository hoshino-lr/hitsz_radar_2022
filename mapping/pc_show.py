# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : pc_show.py
# Time       ：2022/4/12 下午10:57
# Author     ：李龙
# version    ：python 3.8
# Description：
"""
import numpy as np
import cv2 as cv


def get_rgb(max_distance: float, min_distance: float, distance: float) -> tuple:
    scale = max_distance - min_distance
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


def pc_show(src: np.ndarray, depth: np.ndarray) -> None:
    if not src.shape[0]:
        return
    else:
        for i in range(0, depth.shape[0] - 1):
            for j in (0, depth.shape[1] - 1):
                if depth[i][j] is np.nan:
                    continue
                rgb = get_rgb(15, 2, depth[i][j])
                cv.circle(src, (i, j), radius=1, color=rgb, thickness=-1)
