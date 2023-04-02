import cv2
import numpy as np
import time
from threading import Lock, Thread
from typing import Optional

from abstraction.provider import GenericProvider
from config_type import VideoConfig
from proto.record.record_pb2 import Record, NpArray

SLEEP_TIME = 0.1


class RecordReadManager:
    """
    录制读取管理器
    """

    def __init__(self, config: VideoConfig):
        self._config = config
        self._frame_provider: GenericProvider[np.ndarray] = GenericProvider()
        self._net_provider: GenericProvider[np.ndarray] = GenericProvider()
        self._is_terminated = True
        self._thread: Optional[Thread] = None
        self._record_list: Optional[list[Record]] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._ori_spf: float = 0.0
        self._total_frame: int = 0
        self._is_paused: bool = False
        self._speed: float = 1.0
        self._last_time: float = 0.0
        self._frame_pos = 0
        self._self_increment_identifier = 0
        self._change_lock = Lock()
        self.start()

    def get_latest_frame_getter(self, timeout=1000):
        identifier = self._self_increment_identifier
        self._self_increment_identifier += 1
        return lambda: self._frame_provider.latest(timeout, identifier)

    def _spin(self):
        while not self._is_terminated:
            result, frame = self._read_video()
            if not result:
                self.reset()
                return
            self._frame_provider.push(frame)
        # TODO: Network replay

    def start(self):
        self._is_terminated = False
        print(f"开始读取：{self._config}")
        # if self._config.net_process is str:
        #     record_file = open(self._config.net_process, 'rb')
        #     record_seq = Record.ParseFromString(record_file.read())
        #     record_file.close()
        #     self._record_list = list(record_seq)
        self._cap = cv2.VideoCapture(self._config.path)
        self._ori_spf = 1.0 / self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frame = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._frame_pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
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

    def _read_video(self) -> (bool, np.ndarray):
        """
        读取一帧视频
        :return: (是否成功, 视频数据)
        """
        if not self._cap.isOpened():
            return False, None
        while self.is_paused:
            time.sleep(SLEEP_TIME)
        with self._change_lock:
            result, video = self._cap.read()
        if self._last_time + self._get_spf() > time.time():
            time.sleep(self._last_time + self._get_spf() - time.time())
        self._last_time = time.time()
        return result, video

    # def read_net(self) -> np.ndarray:
    #     """
    #     读取一帧网络数据
    #     :return: 网络数据
    #     """
    #     return self._record_list[self._frame_pos].net_data

    @property
    def total_frame(self):
        return self._total_frame

    @property
    def frame_pos(self) -> int:
        return self._frame_pos

    @frame_pos.setter
    def frame_pos(self, frame):
        if frame < 0 or frame > self._total_frame:
            raise ValueError('Frame out of range')
        with self._change_lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self._frame_pos = frame

    @property
    def time_pos(self) -> float:
        return self.frame_pos / self._total_frame

    @time_pos.setter
    def time_pos(self, time: float):
        self.frame_pos = int(time * self._total_frame)

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
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @property
    def total_time(self) -> float:
        return self._total_frame * self._ori_spf

    def __del__(self):
        self._cap.release()

    @staticmethod
    def deserialize_ndarray(nparr: NpArray) -> np.ndarray:
        """
        反序列化
        """
        return np.frombuffer(nparr.data, dtype=np.dtype(nparr.dtype)).reshape(nparr.shape)


if __name__ == "__main__":
    rrm = RecordReadManager(VideoConfig(
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
        cv2.imshow("record_play", getter())
        key = cv2.pollKey()
        match key:
            case 91:
                rrm.frame_pos -= 20
                print(f"pos is {rrm.frame_pos}")
            case 93:
                rrm.frame_pos += 20
                print(f"pos is {rrm.frame_pos}")
            case 32:
                rrm.is_paused = not rrm.is_paused
                print(f"paused is {rrm.is_paused}")
            case 44:
                rrm.speed -= 0.25
                print("sp",rrm.speed)
            case 46:
                rrm.speed += 0.25
                print("sp",rrm.speed)
            case int(any) if any > 0:
                print(any)
