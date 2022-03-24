"""
预警类
created by 黄继凡 2021/1
最新修改 by  lilong 2022/3/20
"""
import cv2
import numpy as np
from .location import CameraLocation
from resources.config import test_objPoints, test_objNames, DEBUG, cam_config


class SolvePnp(CameraLocation):
    imgPoints = np.zeros((6, 2), dtype=np.float32)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    # 鼠标回调事件
    count = 0  # 计数，依次确定个点图像坐标

    def __init__(self):
        super(SolvePnp, self).__init__(self.rvec, self.tvec)
        self.debug = DEBUG
        if self.debug:
            self.count_max = len(test_objNames)
            self.names = test_objNames
            self.objPoints = test_objPoints
        else:
            self.count_max = len(test_objNames)
            self.names = test_objNames
            self.objPoints = test_objPoints
        self.size = cam_config['cam_left']['size']
        self.distCoeffs = cam_config['cam_left']['C_0']
        self.cameraMatrix = cam_config['cam_left']['K_0']

    def add_point(self, x, y) -> bool:
        self.imgPoints[self.count, :] = [float(x), float(y)]
        if self.count < self.count_max:
            self.count += 1
        else:
            return False
        print("the coordinate (x,y) is", x, y)

    def del_point(self) -> None:
        if self.count > 0:
            self.count -= 1

    def sel_cam(self, side) -> None:
        if side == 0:
            self.size = cam_config['cam_left']['size']
            self.distCoeffs = cam_config['cam_left']['C_0']
            self.cameraMatrix = cam_config['cam_left']['K_0']
        else:
            self.size = cam_config['cam_right']['size']
            self.distCoeffs = cam_config['cam_right']['C_0']
            self.cameraMatrix = cam_config['cam_right']['K_0']

    def save(self) -> None:
        pass

    def clc(self) -> None:
        self.imgPoints = np.zeros((6, 2))
        self.count = 0

    # 四点标定函数
    def locate_pick(self):
        _, rvec, tvec, _ = cv2.solvePnPRansac(objectPoints=self.objPoints,
                                              distCoeffs=self.distCoeffs,
                                              cameraMatrix=self.cameraMatrix,
                                              imagePoints=self.imgPoints)
        self.rvec = rvec
        self.tvec = tvec


if __name__ == "__main__":
    PIC = cv2.imread("/home/hoshino/CLionProjects/hitsz_radar/resources/beijing.png")
    cv2.namedWindow("PNP", cv2.WINDOW_NORMAL)
    cv2.imshow("PNP", PIC)
    rvec, tvec = locate_pick()
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]  # 旋转向量转化为旋转矩阵
    T[:3, 3] = tvec.reshape(-1)  # 加上平移向量
    T = np.linalg.inv(T)  # 矩阵求逆
    print(T, (T @ (np.array([0, 0, 0, 1])))[:3])
    cv2.destroyAllWindows()
