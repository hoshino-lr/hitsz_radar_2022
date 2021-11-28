"""
相机类
__init.py__
用于打开相机并且管理相机进程
"""
import cv2 as cv
import cam_drive
import numpy as np

Camera_Messages = {
    0: "Camera_OK",
    1: "unable to enum camera devices!",
    2: "no camera found!",
    3: "SN number not found!",
    4: "failed to create handle!",
    5: "failed to open device!",
    6: "failed to set some parameters!",
    7: "failed to start grabbing image!",
    8: "failed to stop grabbing image!",
    9: "failed to get img"
}


class CameraConfig(object):
    def __init__(self, fx, fy, cx, cy, k1, k2, p1, p2, k3):
        """
        @param debug:暂时没用
        @param run_mode:暂时没用
        @param camera_ids:
        """
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


class Camera(object):
    """
    相机类
    """
    img = np.ndarray((1024, 1024, 3), dtype="uint8")
    camera_ids = "00F78889001"
    __init_ok = False
    __set = False
    __ROI = [0, 0, 1024, 1024]
    __gain = 15
    __exprosure = 5000.0

    def __init__(self, camera_ids, run_mode, debug=False):
        """
        @param debug:暂时没用
        @param run_mode:暂时没用
        @param camera_ids:
        """
        self.debug = debug
        self.camera_ids = camera_ids
        self.run_mode = run_mode
        result = self.cam = cam_drive.HKCamera("00F78889001")
        if not result:
            self.error_handler(result)
        else:
            print(Camera_Messages[0])
            self.__init_ok = True

    def cam_init(self, ROI, exprosure, gain):
        """
        相机参数设置
        """
        if not self.__init_ok:
            print("init failed can't get img")
            return False
        if self.__set:
            print("the params have been set")
            return False
        self.__ROI = ROI
        self.__exprosure = exprosure
        self.__gain = gain
        result = cam.init(ROI[0], ROI[1], ROI[2], ROI[3], self.__exprosure, self.__gain)
        if not result:
            self.error_handler(result)
            return False
        self.__set = True

        def cam_set(self, exprosure, gain):
            self.__exprosure = exprosure
            self.__gain = gain
            result = cam.setParam(800, 800)
            if not result:
                self.error_handler(result)
                return False

    def error_handler(self, error_num):
        """
        错误信息输出
        """
        message = Camera_Messages[error_num]
        print(message)
        self.__init_ok = False

    def get_img(self):
        if not self.__init_ok:
            print("init failed can't get img")
            return None
        result = cam.read(self.img)
        if not result:
            print("failed to get img")
            return None
        return self.img

    def cam_start(self):
        if not self.__init_ok:
            print("init failed can't get img")
            return False
        result = self.cam.start()
        if not result:
            self.error_handler(result)
            return False
        return True


if __name__ == '__main__':
    cam = cam_drive.HKCamera("00F78889001")
    c = cam.init(0, 0, 1024, 1024, 5000.0, 15)
    b = cam.setParam(800, 800)
    a = cam.start()
    vis2 = np.ndarray((1024, 1024, 3), dtype="uint8")
    while 1:
        result = cam.read(vis2)
        if result:
            cv.imshow("asd", vis2)
            cv.waitKey(10)
        else:
            pass
