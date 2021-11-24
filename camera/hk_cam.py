"""
hk_cam.py
相机类
海康相机驱动
"""

from ..resources import cam_drive

CameraMessages = {
    0: "Camera_OK",
    1: "unable to enum camera devices!",
    2: "no camera found!",
    3: "SN number not found!",
    4: "failed to create handle!",
    5: "failed to open device!",
    6: "failed to set some parameters!",
    7: "failed to start grabbing image!",
    8: "failed to stop grabbing image!",
}


