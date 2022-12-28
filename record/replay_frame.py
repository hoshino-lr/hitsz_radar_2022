import numpy as np
import cv2
from config import cam_config
import time
from record.protobuf.record_pb2 import Record, NpArray

SLEEP_TIME = 0.1


class RecordReadManager:
    """
    录制读取管理器
    """

    def __init__(self, config):
        print(f"开始读取：{config}")
        if config['using_net_record']:
            record_file = open(config['pb_path'], 'rb')
            self._record_seq = Record.ParseFromString(record_file.read())
            record_file.close()
            self._record_frame = (record for record in self._record_seq)
        self._cap = cv2.VideoCapture(config['video_path'])
        self._ori_spf = 1.0 / self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frame = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._prop = {
            'is_paused': False,
            'is_reversed': False,
            'speed': 1.0,
            'index': 0,
        }
        self._last_time = time.time()

    def _get_spf(self):
        if self._prop['is_speed'] > 0:
            return self._ori_spf / self._prop['speed']
        else:
            raise ValueError('speed must be positive')

    def read_video(self) -> (bool, np.ndarray):
        """
        读取一帧视频
        :return: (是否成功, 视频数据)
        """
        if not self._cap.isOpened():
            return False, None
        if self._prop['index'] < 0 or self._prop['index'] >= self._total_frame:
            return False, None
        while self._prop['is_paused']:
            time.sleep(SLEEP_TIME)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._prop['index'])
        result, video = self._cap.read()
        if not self._prop['is_reversed']:
            self._prop['index'] += 1
        else:
            self._prop['index'] -= 1
        if self._last_time + self._get_spf() > time.time():
            time.sleep(self._last_time + self._get_spf() - time.time())
        return result, video

    def read_net(self) -> np.ndarray:
        """
        读取一帧网络数据
        :return: 网络数据
        """
        pass

    def set_prop(self, key, value):
        """
        设置属性
        :param key: 属性名
        :param value: 属性值
        """
        self._prop[key] = value

    def get_prop(self, key):
        """
        设置属性
        :param key: 属性名
        :return: 属性值
        """
        if key in self._prop:
            return self._prop[key]
        # 用于无害地返回只读属性
        elif hasattr(self, '_' + key):
            return getattr(self, '_' + key)

    def __del__(self):
        self._cap.release()

    @staticmethod
    def deserialize_ndarray(nparr: NpArray) -> np.ndarray:
        """
        反序列化
        """
        return np.frombuffer(nparr.data, dtype=np.dtype(nparr.dtype)).reshape(nparr.shape)
