import time
from functools import partial
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import av
import cv2
import numpy as np
from loguru import logger

from abstraction.provider import GenericProvider
from config_type import VideoConfig
from proto.record.record_pb2 import Record, NpArray
from utils.fps_counter import FpsCounter
from service.abstract_service import StartStoppableTrait

SLEEP_TIME = 0.1


class VideoReader(StartStoppableTrait):
    """
    录制读取管理器
    """

    def __init__(self, config: VideoConfig):
        self._config = config
        self._frame_provider: GenericProvider[np.ndarray] = GenericProvider()
        self._armor_provider: GenericProvider[np.ndarray] = GenericProvider()
        self._is_terminated = True
        self._thread: Optional[Thread] = None
        self._record_list: Optional[list[Record]] = None
        logger.info(f"正在打开 {Path(config.path).name}")
        self._video: av.container.InputContainer = av.container.open(config.path)
        video_streams: list[av.video.VideoStream] = self._video.streams.video
        if len(video_streams) == 0:
            raise ValueError(f"文件 {config.path} 中没有视频流")
        elif len(video_streams) > 1:
            logger.warning(f"{Path(config.path).name} 中含有 {len(video_streams)} 个视频流，选用第一个")
        self._video_stream: av.video.VideoStream = video_streams[0]
        self._resolution = (self._video_stream.codec_context.width, self._video_stream.codec_context.height)
        logger.info(f"视频流分辨率：{self._resolution}")
        # logger.info(f"视频流{dict(self._video_stream.codec_context)}")
        self._video_stream.thread_type = "AUTO"
        # self._cap: Optional[cv2.VideoCapture] = None
        self._ori_spf: float = 0.0
        self._total_frame: int = 0
        self._is_paused: bool = False
        self._speed: float = 1.0
        self._last_time: float = 0.0
        self._frame_pos = 0
        self._self_increment_identifier = 0
        self._fps_counter = FpsCounter()
        self._change_lock = Lock()

    def get_latest_frame_getter(self):
        identifier = self._self_increment_identifier
        self._self_increment_identifier += 1
        # return lambda: self._frame_provider.latest(timeout, identifier)
        return partial(self._frame_provider.latest, identifier=identifier)

    def get_latest_armor_getter(self):
        identifier = self._self_increment_identifier
        self._self_increment_identifier += 1
        # return lambda: self._frame_provider.latest(timeout, identifier)
        return partial(self._armor_provider.latest, identifier=identifier)

    def get_fps_getter(self):
        """
        获取一个获取帧率的函数
        :return: 一个函数，调用该函数会返回帧率，保证不会阻塞
        """
        return lambda: self._fps_counter.fps

    def _spin(self):
        for frame in self._video.decode(self._video_stream):
            if self._is_terminated:
                return
            while self._is_paused:
                time.sleep(SLEEP_TIME)
                continue
            if self._last_time + self._get_spf() > time.time():
                time.sleep(self._last_time + self._get_spf() - time.time())
            self._last_time = time.time()
            frame: av.video.VideoFrame
            if frame.pts is None:
                print(frame)
            ndarray = frame.to_ndarray(format="bgr24")
            ndarray = cv2.putText(ndarray, f"pts: {frame.pts}/{self._video_stream.frames}", (10, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            ndarray = cv2.putText(ndarray, f"estimate time: {float(frame.pts * self._video_stream.time_base)}",
                                  (10, 80),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            self._frame_provider.push(ndarray)
            self._fps_counter.update()
            # self._frame_pos += 1
            self._frame_pos = frame.pts
            if self._frame_pos >= self._total_frame:
                self.reset()
                return
        # while not self._is_terminated:
        #     for frame in self._video.decode(self._video_stream):
        #         self._frame_provider.push(frame.to_ndarray(format="bgr24"))
        #         self._frame_pos += 1
        #         if self._frame_pos >= self._total_frame:
        #             self.reset()
        #             return
        #     result, frame = self._read_video()
        #     if not result:
        #         self.reset()
        #         return
        #     self._frame_provider.push(frame)
        # TODO: Network replay
        # self._frame_provider.push(None)
        self._frame_provider.end()

    def start(self):
        self._is_terminated = False
        logger.info(f"开始读取 {Path(self._config.path).name}")
        # if self._config.net_process is str:
        #     record_file = open(self._config.net_process, 'rb')
        #     record_seq = Record.ParseFromString(record_file.read())
        #     record_file.close()
        #     self._record_list = list(record_seq)
        # self._cap = cv2.VideoCapture(self._config.path)
        self._ori_spf = 1 / self._video_stream.base_rate
        # self._ori_spf = 1.0 / self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frame = self._video_stream.frames
        # self._total_frame = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # self._frame_pos = self._video_stream.index
        # self._frame_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        self._is_paused = False
        self._speed = 1.0
        self._last_time = time.time()
        self._thread = Thread(target=self._spin, name="VideoReader")
        self._thread.start()

    def _get_spf(self):
        if self.speed > 0:
            return self._ori_spf / self.speed
        else:
            raise ValueError('speed must be positive')

    # def read_net(self) -> np.ndarray:
    #     """
    #     读取一帧网络数据
    #     :return: 网络数据
    #     """
    #     return self._record_list[self._frame_pos].net_data

    def stop(self):
        self._is_terminated = True
        self._thread.join()

    def __del__(self):
        if not self._is_terminated:
            logger.warning("ReadManager 未正常退出")
            self.stop()

    @property
    def resolution(self):
        return self._resolution

    @property
    def total_frame(self):
        return self._total_frame

    @property
    def frame_pos(self) -> int:
        return self._frame_pos

    @frame_pos.setter
    def frame_pos(self, frame):
        if frame < 0 or frame > self._total_frame:
            logger.warning(f"Frame out of range: {frame} not in [0, {self._total_frame}]")
            frame = 0 if frame < 0 else self._total_frame
            # raise ValueError('Frame out of range')
        self._frame_pos = frame
        self._video.seek(frame, stream=self._video_stream)
        # with self._change_lock:
        #     self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        #     self._frame_pos = frame

    @property
    def time_pos(self) -> float:
        return self.frame_pos / self._total_frame

    @time_pos.setter
    def time_pos(self, value: float):
        self.frame_pos = int(value * self._total_frame)

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, speed: float):
        if speed <= 0:
            self._speed = 0.001
        else:
            self._speed = speed

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @is_paused.setter
    def is_paused(self, is_paused: bool):
        self._is_paused = is_paused

    def reset(self):
        self._video.seek(0, stream=self._video_stream)
        # self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @property
    def total_time(self) -> float:
        return self._total_frame * self._ori_spf

    def __del__(self):
        self._video.close()
        # self._cap.release()

    @staticmethod
    def deserialize_ndarray(nparr: NpArray) -> np.ndarray:
        """
        反序列化
        """
        return np.frombuffer(nparr.data, dtype=np.dtype(nparr.dtype)).reshape(nparr.shape)


if __name__ == "__main__":
    rrm = VideoReader(VideoConfig(
        enable=True,
        net_process=True,
        k_0=np.mat([[2580.7380664637653, 0.0, 1535.9830165125002],
                    [0.0, 2582.8839945792183, 1008.784910706948],
                    [0.0, 0.0, 1.0]]),
        c_0=np.mat([[-0.0640364274094021], [0.04211319930460198], [0.0010490064499735965],
                    [-0.0003352752162304746], [0.27835581516135494]]),
        rotate_vec=np.mat([[1.69750257], [0.69091169], [-0.54474128]]),
        transform_vec=np.mat([[-11381.85466339], [-479.01247871], [9449.30328641]]),
        e_0=np.mat([
            [0.0185759, -0.999824, 0.00251985, -0.0904854],
            [0.0174645, -0.00219543, -0.999845, -0.132904],
            [0.999675, 0.018617, 0.0174206, -0.421934],
            [0, 0, 0, 1]
        ]),
        path="/home/shiroki/radar_data/16_4_33_left.avi"
    ))
    getter = rrm.get_latest_frame_getter()
    cv2.namedWindow("record_play")
    while True:
        if not rrm.is_paused:
            img = getter()
            if img is None:
                break
            cv2.imshow("record_play", img)
        key = cv2.pollKey()
        match key:
            case int(any_key) if any_key == ord('['):
                rrm.frame_pos -= 200
                logger.info(f"'[' is pressed, frame_pos changed to {rrm.frame_pos}")
            case int(any_key) if any_key == ord(']'):
                rrm.frame_pos += 200
                logger.info(f"']' is pressed, frame_pos changed to {rrm.frame_pos}")
            case int(any_key) if any_key == ord(' '):
                rrm.is_paused = not rrm.is_paused
                logger.info(f"paused is {rrm.is_paused}")
            case int(any_key) if any_key == ord(','):
                rrm.speed -= 0.25
                logger.info(f"',' is pressed, speed changed to {rrm.speed}")
            case int(any_key) if any_key == ord('.'):
                rrm.speed += 0.25
                logger.info(f"'.' is pressed, speed changed to {rrm.speed}")
            case int(any_key) if any_key > 0:
                logger.warning(f"Unknown key: {chr(any_key)}")
