"""
配置文件的类型定义
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraConfig:
    """相机的通用设置"""
    enable: bool
    """是否启用"""
    net_process: bool | str
    """是否启用网络处理，如果是字符串则直接回放预先推理的模型输出"""
    k_0: np.matrix
    """内参"""
    c_0: np.matrix
    """畸变系数"""
    rotate_vec: np.matrix
    """旋转向量"""
    transform_vec: np.matrix
    """平移向量"""
    e_0: np.matrix
    """外参"""


@dataclass
class HikCameraDriverConfigExt:
    """海康相机驱动的设置，用作 :py:class:`CameraConfig` 的扩展"""
    roi: tuple[int, int, int, int]
    """ROI 设置 (x, y, w, h)"""
    camera_id: str
    """序列号"""
    exposure: int
    """曝光时间"""
    gain: int
    """增益"""


@dataclass
class HikCameraConfig(CameraConfig, HikCameraDriverConfigExt):
    """海康相机的设置"""
    pass


@dataclass
class VideoConfig(CameraConfig):
    """视频假相机的设置"""
    path: str
    """视频路径"""
