import numpy as np
import cv2
from config import cam_config, global_replay_net, using_record
import time
from record.protobuf.record_pb2 import Record, NpArray


class RecordReadManager:
    """
    录制读取管理器
    """

    def __init__(self, cam_name):
        print("开始读取")
        if using_record:
            record_file = open(cam_config[cam_name]['pb_path'], 'rb')
            self._record_seq = Record.ParseFromString(record_file.read())
            self._record_file.close()
            self._record_gen = (record for record in self._record_seq)
        self._video_file = cv2.VideoCapture(cam_config[cam_name]['video_path'])
        self._frame_count = 0
        self._is_reversd = False
        self._speed = 1.0
        self._start_time = time.time()

    def read(self) -> (np.ndarray, np.ndarray):
        """
        读取一帧
        :return: (视频数据, 网络数据)
        """
        video = self._video_file.read()[1]
        # 采用带记录的回放方式
        if using_record:
            if self._record_gen is None:
                return None, None
            record_frame = self._record_seq[self._frame_count]
            self._frame_count += 1
            # 手动延迟
            if time.time() - self._start_time < record_frame.real_time / self._speed:
                time.sleep(record_frame.real_time / self._speed - (time.time() - self._start_time))
            net_data = RecordReadManager.deserialize_ndarray(record_frame.net_data)
            return video, net_data

        # 采用原始方式
        else:
            return video, None

    def read_next(self):
        """
        读取下一帧
        :return: (视频数据, 网络数据)
        """
        return self.read()

    @staticmethod
    def deserialize_ndarray(nparr: NpArray) -> np.ndarray:
        """
        反序列化
        """
        return np.frombuffer(nparr.data, dtype=np.dtype(nparr.dtype)).reshape(nparr.shape)
