"""
相机类
用于打开相机并且输出图像
created by 李龙 2021/11
最终修改 by 李龙 2021/1/19
"""
import re
import time
import sys
import threading
import termios
from ctypes import *
import cv2 as cv
import numpy as np
from camera.MvImport.MvCameraControl_class import *
from resources.config import cam_config
from camera.cam import Camera


class Camera_HK(Camera):
    """
    相机类
    """

    def __init__(self, type_, debug=False):
        """
        @param debug:使用视频或者是相机
        @param type:相机左右类型
        """
        self.__debug = debug
        self.__type = type_
        self.__camera_config = cam_config[self.__type]
        self.__id = self.__camera_config['id']
        self.__size = self.__camera_config['size']
        self.__img = np.ndarray((self.__size[0], self.__size[1], 3), dtype="uint8")
        self.__exposure = self.__camera_config['exposure']
        self.__gain = self.__camera_config['gain']
        if not self.__debug:
            SDKVersion = MvCamera.MV_CC_GetSDKVersion()
            print("SDKVersion[0x%x]" % SDKVersion)

            deviceList = MV_CC_DEVICE_INFO_LIST()
            tlayerType = MV_USB_DEVICE

            # ch:枚举设备 | en:Enum device
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0:
                print("enum devices fail! ret[0x%x]" % ret)
                self.init_ok = False
                return

            if deviceList.nDeviceNum == 0:
                print("find no device!")
                self.init_ok = False

            Find = False

            for i in range(0, deviceList.nDeviceNum):
                mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                    strSerialNumber = ""
                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                        if per == 0:
                            break
                        strSerialNumber = strSerialNumber + chr(per)
                    if self.__id == strSerialNumber:
                        nConnectionNum = i
                        Find = True
                    print("user serial number: %s" % strSerialNumber)
            if Find:
                # ch:选择设备并创建句柄 | en:Select device and create handle
                self.__stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)],
                                           POINTER(MV_CC_DEVICE_INFO)).contents

                # ch:创建相机实例 | en:Creat Camera Object
                self.cam = MvCamera()

                ret = self.cam.MV_CC_CreateHandle(self.__stDeviceList)
                if ret != 0:
                    print("create handle fail! ret[0x%x]" % ret)
                    self.init_ok = False

                # ch:打开设备 | en:Open device
                ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                if ret != 0:
                    print("open device fail! ret[0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
                if ret != 0:
                    print("set TriggerMode failed! ret [0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetEnumValue("ExposureMode", MV_EXPOSURE_AUTO_MODE_OFF)
                if ret != 0:
                    print("set height failed! ret [0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
                if ret != 0:
                    print("set GainAuto failed! ret [0x%x]" % ret)
                    self.init_ok = False
                ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)
                if ret != 0:
                    print("set PixelFormat failed! ret [0x%x]" % ret)
                    self.init_ok = False
                ret = self.cam.MV_CC_SetBoolValue("BlackLevelEnable", False)
                if ret != 0:
                    print("set BlackLevelEnable failed! ret [0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)
                if ret != 0:
                    print("set BalanceWhiteAuto failed! ret [0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetEnumValue("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS)
                if ret != 0:
                    print("set AcquisitionMode failed! ret [0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
                if ret != 0:
                    print("set AcquisitionFrameRateEnable failed! ret [0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetIntValue("Height", int(self.__size[1]))
                if ret != 0:
                    print("set height failed! ret [0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetIntValue("Width", int(self.__size[0]))
                if ret != 0:
                    print("set width failed! ret [0x%x]" % ret)
                    self.init_ok = False
                ret = self.cam.MV_CC_SetIntValue("OffsetX", int(0))
                if ret != 0:
                    print("set width failed! ret [0x%x]" % ret)
                    self.init_ok = False
                ret = self.cam.MV_CC_SetIntValue("OffsetY", int(0))
                if ret != 0:
                    print("set OffsetY failed! ret [0x%x]" % ret)
                    self.init_ok = False
                ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.__exposure)
                if ret != 0:
                    print("start grabbing fail! ret[0x%x]" % ret)
                    self.init_ok = False

                ret = self.cam.MV_CC_SetFloatValue("Gain", float(self.__gain))
                if ret != 0:
                    print("start grabbing fail! ret[0x%x]" % ret)
                    self.init_ok = False

                # ch:获取数据包大小 | en:Get payload size
                stParam = MVCC_INTVALUE()
                memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
                ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
                if ret != 0:
                    print("get payload size fail! ret[0x%x]" % ret)
                    self.init_ok = False
                self.__nPayloadSize = stParam.nCurValue
                self.__data_buf = (c_ubyte * self.__nPayloadSize)()
                # ch:开始取流 | en:Start grab image
                ret = self.cam.MV_CC_StartGrabbing()
                if ret != 0:
                    print("start grabbing fail! ret[0x%x]" % ret)
                    self.init_ok = False
                self.__stDeviceList = MV_FRAME_OUT_INFO_EX()
                memset(byref(self.__stDeviceList), 0, sizeof(self.__stDeviceList))
            else:
                self.init_ok = False

        else:
            self.cap = cv.VideoCapture(self.__camera_config["video_path"])
            self.init_ok = True

    def work_thread(self) -> bool:
        ret = self.cam.MV_CC_GetOneFrameTimeout(self.__data_buf, self.__nPayloadSize, self.__stDeviceList, 100)
        if ret == 0:
            # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            #     self.__stDeviceList.nWidth, self.__stDeviceList.nHeight, self.__stDeviceList.nFrameNum))

            nRGBSize = self.__stDeviceList.nWidth * self.__stDeviceList.nHeight * 3
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.__stDeviceList.nWidth
            stConvertParam.nHeight = self.__stDeviceList.nHeight
            stConvertParam.pSrcData = self.__data_buf
            stConvertParam.nSrcDataLen = self.__stDeviceList.nFrameLen
            stConvertParam.enSrcPixelType = self.__stDeviceList.enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
            stConvertParam.nDstBufferSize = nRGBSize

            ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                return False
            else:
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                self.__img = np.asarray(img_buff)
                self.__img = self.__img.reshape((self.__size[1], self.__size[0], -1))
                return True
        else:
            print("get one frame fail, ret[0x%x]" % ret)
            return False

    def get_img(self) -> [bool, np.ndarray]:
        if self.init_ok:
            if not self.__debug:
                result = self.work_thread()
                return result, self.__img
            else:
                result, self.__img = self.cap.read()
                return bool(result), self.__img
        else:
            # print("init is failed dangerous!!!")
            return False, self.__img

    def destroy(self) -> None:
        if not self.__debug:
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
            self.init_ok = False
        else:
            self.cap.release()
            self.init_ok = False


if __name__ == "__main__":
    import time

    cv.namedWindow("test", cv.WINDOW_NORMAL)
    cam_test = Camera("cam_left")
    t1 = time.time()
    count = 0
    while True:
        if cam_test.init_ok:
            t2 = time.time()
            frame = cam_test.get_img()
            count += 1
            cv.imshow("test", frame)
            key = cv.waitKey(1)
            if t2 - t1 >= 5:
                fps = count / (t2 - t1)
                count = 0
                t1 = time.time()
                print(f"fps {fps}")
            if key == ord('q'):
                cam_test.destroy()
                break
        else:
            break

    cv.destroyAllWindows()
