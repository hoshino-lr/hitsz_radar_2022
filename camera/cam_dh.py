# !/usr/bin/env python3
# -*-coding:utf-8 -*-

"""
# File       : cam_dh.py
# Time       ：2022/4/12 下午4:36
# Author     ：李龙
# version    ：python 3.8
# Description：
"""
import gxipy as gx
from camera.cam import Camera
import numpy as np
from resources.config import cam_config
import cv2 as cv


class Camera_DH(Camera):
    """
    相机类
    """

    def __init__(self, type_, debug=False):
        self.__debug = debug
        self.__type = type_
        self.__camera_config = cam_config[self.__type]
        self.__id = self.__camera_config['id']
        self.__size = self.__camera_config['size']
        self.__img = np.ndarray((self.__size[0], self.__size[1], 3), dtype="uint8")
        self.__exposure = self.__camera_config['exposure']
        self.__gain = self.__camera_config['gain']
        if not debug:
            try:
                device_manager = gx.DeviceManager()
                dev_num, dev_info_list = device_manager.update_device_list()
                if dev_num == 0:
                    print("Number of enumerated devices is 0")
                    return

                self.cam = gx.DeviceManager.open_device_by_sn(sn=self.__id,
                                                              access_mode=gx.GxAccessMode.EXCLUSIVE)
                # exit when the camera is a mono camera
                if self.cam.PixelColorFilter.is_implemented() is False:
                    print("This sample does not support mono camera.")
                    self.cam.close_device()
                    return

                # set continuous acquisition
                self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

                # set exposure
                self.cam.ExposureTime.set(self.__exposure)

                # set gain
                self.cam.Gain.set(self.__gain)

                # get param of improving image quality
                if self.cam.GammaParam.is_readable():
                    gamma_value = self.cam.GammaParam.get()
                    gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
                else:
                    gamma_lut = None
                if self.cam.ContrastParam.is_readable():
                    contrast_value = self.cam.ContrastParam.get()
                    contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
                else:
                    contrast_lut = None
                if self.cam.ColorCorrectionParam.is_readable():
                    color_correction_param = self.cam.ColorCorrectionParam.get()
                else:
                    color_correction_param = 0

                # start data acquisition
                self.cam.stream_on()
                self.init_ok = True
            except Exception as exception:
                print("[ERROR] %s" % exception)
                self.init_ok = False

        else:
            self.cap = cv.VideoCapture(self.__camera_config["video_path"])
            self.init_ok = True

    def work_thread(self) -> bool:
        # print height, width, and frame ID of the acquisition image
        raw_image = self.cam.data_stream[0].get_image()

        # print("Frame ID: %d   Height: %d   Width: %d"
        #       % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))

        # get RGB image from raw image
        rgb_image = raw_image.convert("BGR")
        if rgb_image is None:
            print('[ERROR] Failed to convert RawImage to RGBImage')
            self.init_ok = False
            return False

        # create numpy array with data from rgb image
        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None:
            print('[ERROR] Failed to get numpy array from RGBImage')
            self.init_ok = False
            return False

        self.__img = numpy_image
        return True

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
        if not self.__debug and self.init_ok:
            # ch:停止取流 | en:Stop grab image
            # stop data acquisition
            self.cam.stream_off()

            # ch:关闭设备 | Close device
            self.cam.close_device()

            self.init_ok = False
        else:
            self.cap.release()
            self.init_ok = False
