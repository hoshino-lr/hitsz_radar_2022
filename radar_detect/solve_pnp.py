"""
预警类
created by 黄继凡 2021/1
最新修改 by 李龙 2022/5/3
"""
import cv2
import numpy as np
from radar_detect.location import CameraLocation
from config import objPoints, objNames, DEBUG, cam_config, enemy2color, enemy_color


class SolvePnp(CameraLocation):
    imgPoints = np.zeros((6, 2), dtype=np.float32)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    # 鼠标回调事件
    count = 0  # 计数，依次确定个点图像坐标

    def __init__(self, text_api):
        super(SolvePnp, self).__init__(self.rvec, self.tvec)
        self.debug = DEBUG
        self._api = text_api
        self.sp_state = False
        self.side_text = ""

    def add_point(self, x: int, y: int) -> None:
        if self.count < self.count_max - 1:
            self.imgPoints[self.count, :] = np.array([float(x), float(y) + self.offset_y])
            self.count += 1
        elif self.count == self.count_max - 1:
            self.imgPoints[self.count, :] = np.array([float(x), float(y) + self.offset_y])
        self._update_info()

    def del_point(self) -> None:
        if self.count < self.count_max and self.count > 0:
            self.imgPoints[self.count] = np.array([0, 0])
        self._update_info()

    def sel_cam(self, side) -> None:
        if side == 0:
            side_text = f'cam_left_{enemy2color[enemy_color]}'
        else:
            side_text = f'cam_right_{enemy2color[enemy_color]}'
        self.side_text = side_text
        self.count_max = len(objNames[int(self.debug)][side_text])
        self.names = objNames[int(self.debug)][side_text]
        self.imgPoints = np.zeros((self.count_max, 2), dtype=np.float32)
        self.objPoints = objPoints[int(self.debug)][side_text] * 1000  # 米转换成毫米

        if side == 0:
            side_text = f'cam_left'
        else:
            side_text = f'cam_right'
        self.size = cam_config[side_text]['size']
        self.distCoeffs = cam_config[side_text]['C_0']
        self.cameraMatrix = cam_config[side_text]['K_0']
        self.offset_y = cam_config[side_text]['roi'][1]
        self.count = 0
        self._update_info()

    def save(self) -> None:
        if self.debug:
            text = "_debug"
        else:
            text = ""
        self.save_to(self.side_text + text)
        self._update_info()

    def read(self, name) -> None:
        if self.debug:
            text = "_debug"
        else:
            text = ""
        ca = self.from_checkpoint(f"{name}{text}")
        self.tvec = ca.translation
        self.rvec = ca.rotation

    def clc(self) -> None:
        self.imgPoints = np.zeros((self.count_max, 2), dtype=np.float32)
        self.count = 0
        self._update_info()

    def step(self, num) -> None:
        if self.count + num >= self.count_max or self.count + num < 0:
            pass
        else:
            self.count = self.count + num
        self._update_info()

    def _update_info(self):
        self._api("INFO", "side", f"当前相机位置：{self.side_text}")
        self._api("INFO", "sp+state", f"当前标注状态：{self.sp_state}")
        self._api("INFO", "count", f"当前点：{self.count + 1}")
        for i in range(1, self.count_max + 1):
            text = f"{self.names[i - 1]}\n" \
                   f"x : {self.imgPoints[i - 1][0]} y: {self.imgPoints[i - 1][1]}"
            if i - 1 != self.count:
                self._api("INFO", f"count{i}", text)
            else:
                self._api("ERROR", f"count{i}", text)  # 只是显示一种颜色

    # 四点标定函数
    def locate_pick(self) -> bool:
        if self.imgPoints.all():  # 粗暴的判断
            try:
                _, rvec, tvec, _ = cv2.solvePnPRansac(objectPoints=self.objPoints,
                                                      distCoeffs=self.distCoeffs,
                                                      cameraMatrix=self.cameraMatrix,
                                                      imagePoints=self.imgPoints,
                                                      iterationsCount=1000,
                                                      reprojectionError=3,
                                                      confidence=0.99,
                                                      flags=cv2.SOLVEPNP_EPNP)
            except Exception as e:
                print(f"[ERROR] {e}")
                self.sp_state = False
                self._update_info()
                return False
            self.rotation = rvec
            self.translation = tvec
            print(f"[INFO] rvec:{rvec}")
            print(f"[INFO] tvec:{tvec}")
            self.sp_state = True
            self._update_info()
            return True
        else:
            self.sp_state = False
            self._update_info()
            return False
