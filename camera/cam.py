"""
相机类
__init.py__
用于打开相机并且管理相机进程
"""
import cv2 as cv
import cam_drive
import numpy as np


class camera_config(object):
    def __init__(self, fx, fy, cx, cy, k1, k2, p1, p2, k3):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.mtx = cv.Mat(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1
        self.dist = cv.Mat(1, 5) << k1, k2, p1, p2, k3
        # cv.initUndistortRectifyMap(self.mtx,self.dist,)
        self.FOCUS_PIXEL = (fx + fy) / 2


class camera(object):
    """
    相机类
    """

    def __init__(self, camera_ids, run_mode, debug=False):
        self.debug = debug
        self.camera_ids = camera_ids
        self.run_mode = run_mode


if __name__ == '__main__':
    cam = cam_drive.HKCamera("00F78889001")
    c = cam.init(0, 0, 1280, 1024, 5000.0, 15)
    b = cam.setParam(800, 800)
    a = cam.start()
    vis2 = np.ndarray((1024, 1280, 3), dtype="uint8")
    while 1:
        result = cam.read(vis2)
        if result:
            cv.imshow("asd", vis2)
            cv.waitKey(10)
        else:
            pass
