"""
处理线程
在这里将处理来自相机的图像，结合雷达等信息提供侦测预警
created by 陈希峻 2022/12/22
"""

import time
from threading import Event, Thread
from queue import Queue

import cv2

from threads.camera_thread import CameraThread
from net.network_pro import Predictor
from record.record_frame import RecordWriteManager


class ProcessThread():
    def __init__(self, camera: CameraThread):
        self._camera = camera
        self._roi = camera.config['roi']
        self._name = camera.name
        self._predictor = None
        self._is_terminated = True
        self._event = Event()
        self._queue = Queue()
        self._thread = None
        self._fps = 0
        self._fps_time = 0
        self._fps_count = 0

        self.start()

    def start(self):
        if not isinstance(self._camera, CameraThread):
            raise TypeError("处理线程相机对象错误")
        self._event.clear()
        self._queue.queue.clear()
        self._predictor = Predictor(self._camera.name)
        self._is_terminated = False
        self._fps_time = time.time()
        self._thread = Thread(target=self._spin, name=self._name)
        self._thread.start()
        print(f"子线程开始: {self._name}")
        pass

    def stop(self):
        self._predictor.stop()
        print(f"处理线程:{self._camera.name} 退出")
        pass

    def sub(self):
        has_data = self._event.wait(timeout=0.1)
        if has_data:
            try:
                data = self._queue.get_nowait()
            except:
                print("处理线程接受数据出错")
                data = False, None
            self._event.clear()
            return data
        else:
            return False, None

    def __del__(self):
        self.stop()
        self._thread.join()
        pass

    @property
    def is_terminate(self):
        return self._is_terminated

    @property
    def fps(self):
        return self._fps

    def _pub(self, data):
        if self._event.is_set():
            return
        if self._queue.full():
            try:
                self._queue.get()
            except:
                pass
        self._queue.put(data)
        self._event.set()

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

    def _spin(self):
        # count = 0
        # count_error = 0
        # t1 = 0
        print(f"子线程开始: {self._name}")
        while not self._is_terminated:
            # TODO: 录制功能升级
            frame = self._camera.latest_frame
            if frame is not None:
                img = cv2.copyMakeBorder(
                    frame[self._roi[1]:self._roi[3] + self._roi[1], :, :], 0,
                    self._roi[1],
                    0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                res = self._predictor.detect_cars(img)
                self._pub((res, img))
                self._fps_update()
