import time
import sys
import threading
import termios
from ctypes import *
import cv2 as cv
import numpy as np

sys.path.append("./MvImport")
from MvImport.MvCameraControl_class import *
from resources.config import cam_config


class Camera(object):
    """
    相机类
    """
    camera_ids = None
    __init_ok = True
    __gain = 15
    __exprosure = 5000.0

    def __init__(self, type_, debug=False):
        """
        @param debug:使用视频或者是相机
        @param type:相机左右类型
        """
        self.__debug = debug
        self.__type = type_
        self.__camera_config = cam_config[self.__type]
        self.__stDevInfo = self.__camera_config['id']
        self.__size = self.__camera_config['size']
        self.__img = np.ndarray((self.__size[0], self.__size[1], 3), dtype="uint8")
        self.__exprosure = self.__camera_config['exposure']
        self.__gain = self.__camera_config['gain']
        if not self.__debug:
            # ch:创建相机实例 | en:Creat Camera Object
            self.cam = MvCamera()

            # ch:选择设备并创建句柄 | en:Select device and create handle
            ret = self.cam.MV_CC_CreateHandle(self.__stDevInfo)
            if ret != 0:
                print("create handle fail! ret[0x%x]" % ret)
                self.__init_ok = False

            # ch:打开设备 | en:Open device
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print("open device fail! ret[0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print("set TriggerMode failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetEnumValue("ExposureMode", MV_EXPOSURE_AUTO_MODE_OFF)
            if ret != 0:
                print("set height failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
            if ret != 0:
                print("set GainAuto failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetBoolValue("BlackLevelEnable", False)
            if ret != 0:
                print("set BlackLevelEnable failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)
            if ret != 0:
                print("set BalanceWhiteAuto failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetEnumValue("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS)
            if ret != 0:
                print("set AcquisitionMode failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
            if ret != 0:
                print("set AcquisitionFrameRateEnable failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetIntValue("Height", self.__size(1))
            if ret != 0:
                print("set height failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetIntValue("Width", self.__size(0))
            if ret != 0:
                print("set width failed! ret [0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.__exprosure)
            if ret != 0:
                print("start grabbing fail! ret[0x%x]" % ret)
                self.__init_ok = False

            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.__exprosure)
            if ret != 0:
                print("start grabbing fail! ret[0x%x]" % ret)
                self.__init_ok = False

            # ch:获取数据包大小 | en:Get payload size
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                print("get payload size fail! ret[0x%x]" % ret)
                self.__init_ok = False
            self.__nPayloadSize = stParam.nCurValue
            # ch:开始取流 | en:Start grab image
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                print("start grabbing fail! ret[0x%x]" % ret)
                self.__init_ok = False
            self.__stDeviceList = MV_FRAME_OUT_INFO_EX()
            memset(byref(self.__stDeviceList), 0, sizeof(self.__stDeviceList))
            self.__data_buf = (c_ubyte * self.__nPayloadSize)()
        else:
            self.cap = cv.VideoCapture(self.__camera_config["video_path"])
            self.__init_ok = True

    def work_thread(self):
        ret = self.cam.MV_CC_GetOneFrameTimeout(self.__data_buf, self.__nPayloadSize, self.__stDeviceList, 1000)
        if ret == 0:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                self.__stDeviceList.nWidth, self.__stDeviceList.nHeight, self.__stDeviceList.nFrameNum))

            nRGBSize = self.__stDeviceList.nWidth * self.__stDeviceList.nHeight * 3
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.__stDeviceList.nWidth
            stConvertParam.nHeight = self.__stDeviceList.nHeight
            stConvertParam.pSrcData = self.__data_buf
            stConvertParam.nSrcDataLen = self.__stDeviceList.nFrameLen
            stConvertParam.enSrcPixelType = self.__stDeviceList.enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
            stConvertParam.nDstBufferSize = nRGBSize

            ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                return False
            else:
                self.__img = np.asarray(self.__data_buf)
                return True
        else:
            print("get one frame fail, ret[0x%x]" % ret)
            return False

    def get_img(self):
        if self.__init_ok:
            if not self.__debug:
                result = self.work_thread()
            else:
                result, frame = self.cap.read()
                return frame
            if not result:
                print("failed to get img")
        else:
            print("init is failed dangerous!!!")
            return self.__img

    def destroy(self):
        # ch:停止取流 | en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close deivce fail! ret[0x%x]" % ret)

        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
        self.__init_ok = False
