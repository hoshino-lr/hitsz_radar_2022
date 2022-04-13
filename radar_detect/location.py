"""
revc,tvec 储存读取类
created by 秦泽钊 2021/1
最新修改 by 李龙 2022/3/26
"""
from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

import numpy as np


class CameraLocation(object):
    """
    将原本的 rvec 与 tvec 整合为一个对象
    """
    checkpoint_dir = "../resources/cam_data/"

    @staticmethod
    def __last_checkpoint(cls) -> (int, str):
        """
        查找目录中最新的
        :return: id 和对应的文件名
        """
        pattern = re.compile(r"(d+)_.+\.[rt]vec")
        current_biggest_id = 0
        candidate = None
        for file_path in os.listdir(cls.checkpoint_dir):
            match = pattern.match(file_path)
            if match is not None:
                file_id = int(match.groups()[1])
                if file_id > current_biggest_id:
                    current_biggest_id = file_id
                    candidate = file_path
        if candidate is None:  # 无记录
            return -1, None
        else:
            return current_biggest_id, candidate.rsplit(".", 1)[0]

    def __init__(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        self.rotation: np.ndarray = rvec  # 旋转变量
        self.translation: np.ndarray = tvec  # 平移变量
        return

    @staticmethod
    def __get_file_path(file_name: str) -> (Path, Path):
        return (Path(CameraLocation.checkpoint_dir, file_name + ".rvec"),
                Path(CameraLocation.checkpoint_dir, file_name + ".tvec"))

    def save_to(self, file_name: str) -> None:
        """
        将两个变换向量保存到指定路径
        :param file_name: 指定路径（不带后缀名）
        """
        rotation_file_path, translation_file_path = self.__get_file_path(file_name)
        self.rotation.tofile(rotation_file_path)
        self.translation.tofile(translation_file_path)  # TODO: 是否要使用tofile来储存二进制？
        return

    def save_by_id(self) -> None:
        """
        将两个变换向量用自增序列号命名保存到默认路径
        """
        last_id, _ = self.__last_checkpoint(self)
        return self.save_to(os.path.join(self.checkpoint_dir,
                                         f"{last_id + 1}_{datetime.datetime.now()}"))

    @staticmethod
    def from_last_checkpoint(cls) -> CameraLocation:
        """
        从默认路径读取两个变换向量
        :return: 一个 PnpConverter 对象
        """
        _, path = cls.__last_checkpoint()
        if path is None:
            pass
        else:
            return cls.from_checkpoint(path)

    @staticmethod
    def from_checkpoint(file_name: str) -> CameraLocation:
        """
        从指定的路径读取两个变换向量
        :param file_name:str 文件名（不带后缀名）
        :return: 一个 PnpConverter 对象
        """
        # file_name = f"{datetime.datetime.now()}" if file_name is None else file_name
        rotation_file_path, translation_file_path = CameraLocation.__get_file_path(file_name)
        return CameraLocation(np.fromfile(rotation_file_path), np.fromfile(translation_file_path))


