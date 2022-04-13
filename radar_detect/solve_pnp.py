"""
预警类
created by 黄继凡 2021/1
最新修改 by 李龙 2022/3/26
"""
import cv2
import numpy as np
from radar_detect.location import CameraLocation
from resources.config import objPoints, objNames, DEBUG, cam_config


class SolvePnp(CameraLocation):
    imgPoints = np.zeros((6, 2), dtype=np.float32)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    # 鼠标回调事件
    count = 0  # 计数，依次确定个点图像坐标

    def __init__(self):
        super(SolvePnp, self).__init__(self.rvec, self.tvec)
        self.debug = DEBUG
        self.sel_cam(0)

    def add_point(self, x: int, y: int) -> str:
        if self.count < self.count_max - 1:
            self.imgPoints[self.count, :] = [float(x), float(y)]
            self.count += 1
            text = f"现在需要输入第{self.count+1}个点：{self.names[self.count]}"
        elif self.count == self.count_max - 1:
            self.imgPoints[self.count, :] = [float(x), float(y)]
            self.count += 1
            text = f"please press enter!"
        else:
            self.imgPoints[self.count, :] = [float(x), float(y)]
            text = f"please press enter"
        return text

    def del_point(self) -> str:
        if self.count > 0:
            self.count -= 1
            text = f"now Point is {self.names[self.count]}"
        else:
            text = f"invalid !!! please press enter"
        return text

    def sel_cam(self, side) -> str:
        if side == 0:
            side_text = 'cam_left'
        else:
            side_text = 'cam_right'
        self.count_max = len(objNames[int(self.debug)][side_text])
        self.names = objNames[int(self.debug)][side_text]
        self.objPoints = objPoints[int(self.debug)][side_text]
        self.size = cam_config[side_text]['size']
        self.distCoeffs = cam_config[side_text]['C_0']
        self.cameraMatrix = cam_config[side_text]['K_0']
        self.count = 0
        text = f"现在需要输入第{self.count+1}个点： {self.names[self.count]}"
        return text

    def save(self) -> str:
        if self.size:
            self.save_to("left_cam")
            text = f"data save to left_cam"
        else:
            self.save_to("right_cam")
            text = f"data save to right_cam"
        return text

    def clc(self) -> str:
        self.imgPoints = np.zeros((6, 2))
        self.count = 0
        text = f"现在需要输入第{self.count+1}个点： {self.names[self.count]}"
        return text

    def step(self, num) -> str:
        if self.count + num >= self.count_max or self.count + num < 0:
            pass
        else:
            self.count = self.count + num
        text = f"现在需要输入第{self.count + 1}个点： {self.names[self.count]}"
        return text

    # 四点标定函数
    def locate_pick(self) -> bool:
        if self.count == self.count_max:
            _, rvec, tvec, _ = cv2.solvePnPRansac(objectPoints=self.objPoints,
                                                  distCoeffs=self.distCoeffs,
                                                  cameraMatrix=self.cameraMatrix,
                                                  imagePoints=self.imgPoints)
            self.rotation = rvec
            self.translation = tvec
            return True
        else:
            return False

