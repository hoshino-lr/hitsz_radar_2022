"""
相机线程
在这里获取相机流并提供录制功能，可选视频替代
created by 陈希峻 2022/12/22
"""
import time
from abc import abstractmethod, ABC
from threading import Thread, Lock
from record.replay_frame import RecordReadManager

ERR_MAX = 10


class CameraThread(ABC):
    """
    相机线程
    """

    @abstractmethod
    def __init__(self, config):
        self._config = config
        self._name = config["name"]
        self._camera = None
        self._thread = None
        self._is_terminated = True
        self._lock = Lock()
        self._latest_frame = None
        self._fps = 0
        self._fps_time = 0
        self._fps_count = 0

        self.start()

    def _fps_update(self):
        """
        更新帧率
        """
        if self._fps_count >= 10:
            self._fps = self._fps_count / (time.time() - self._fps_time)
            self._fps_count = 0
            self._fps_time = time.time()
        else:
            self._fps_count += 1

    @abstractmethod
    def start(self):
        self._is_terminated = False
        self._fps_time = time.time()
        pass

    @abstractmethod
    def stop(self):
        self._is_terminated = True
        pass

    def __del__(self):
        self.stop()
        pass

    @property
    def name(self):
        return self._name

    @property
    def latest_frame(self):
        with self._lock:
            return self._latest_frame.copy()

    @property
    def config(self):
        return self._config

    @property
    def real_fps(self):
        return self._fps



class CameraThread_Real(CameraThread):
    """
    真实相机
    """

    def __init__(self, config):
        self._err_cnt = 0
        super().__init__(config)

    def start(self):
        from camera.cam_hk_v3 import Camera_HK
        super().start()
        self._err_cnt = 0
        self._camera = Camera_HK(self._name)
        self._thread = Thread(target=self._spin, name=self._name)
        self._thread.start()

    def stop(self):
        super().stop()
        self._thread.join()
        self._camera.destroy()

    def _spin(self):
        while not self._is_terminated:
            result, frame = self._camera.get_img()
            with self._lock:
                self._latest_frame = frame
            self._fps_update()

    def _mark_error(self):
        """
        错误计数
        """
        self._err_cnt += 1
        if self._err_cnt >= ERR_MAX:
            self._err_cnt = 0
            self._on_error()

    def _on_error(self):
        """
        错误重启
        """
        print(f"相机:{self._name} 错误重启")
        self.stop()
        self.start()


class CameraThread_Video(CameraThread):
    """
    视频相机
    """

    def __init__(self, config, record: RecordReadManager):
        self._record = record
        super().__init__(config)

    def start(self):
        super().start()
        self._thread = Thread(target=self._spin, name=self._name)
        self._thread.start()

    def stop(self):
        super().stop()
        self._thread.join()

    def _spin(self):
        while not self._is_terminated:
            result, frame = self._record.read_video()
            if not result:
                self._record.reset()
                continue
            with self._lock:
                self._latest_frame = frame
            self._fps_update()

    def set_prop(self, key, value):
        """
        修改录制读取属性
        :param key: 属性名
        :param value: 属性值
        """
        self._record.set_prop(key, value)

    def get_prop(self, key):
        """
        获取录制读取属性
        :param key: 属性名
        :return: 属性值
        """
        return self._record.get_prop(key)
