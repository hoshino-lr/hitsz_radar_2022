"""
处理线程
在这里将处理来自相机的图像，结合雷达等信息提供侦测预警
created by 陈希峻 2022/12/22
"""

import time
from threading import Event, Thread
from queue import Queue
from threads.camera_thread import CameraThread
from net.network_pro import Predictor
from record.record_frame import RecordWriteManager


class ProcessThread():
    def __init__(self, camera):
        self._camera = camera
        self._name = camera.name
        self._predictor = None
        self._is_terminated = True
        self._event = Event()
        self._queue = Queue()
        self._thread = None

    def start(self):
        if not isinstance(self._camera, CameraThread):
            raise TypeError("处理线程相机对象错误")
        self._event.clear()
        self._queue.queue.clear()
        self._predictor = Predictor(self._camera.name)
        self._is_terminated = False
        self._thread = Thread(target=self._spin, name=self._name)
        self._thread.start()
        print(f"子线程开始: {self._name}")
        pass

    def stop(self):
        self._predictor.stop()
        print(f"处理线程:{self._camera.name} 退出")
        pass

    def sub(self):
        self._event.wait()
        self._event.clear()
        return self._queue.get()

    def check(self):
        # TODO: 检查相机是否正常
        pass

    def __del__(self):
        self.stop()
        self._thread.join()
        pass

    @property
    def is_terminate(self):
        return self._is_terminated

    def _pub(self, data):
        self._queue.put(data)
        self._event.set()

    def _spin(self):
        # count = 0
        # count_error = 0
        # t1 = 0
        while not self._is_terminated:
            # TODO: 录制功能升级
            # if event_list['record'].is_set():
            #     # predictor.record_on_clicked()
            #     if not is_record:
            #         record = RecordWriteManager(name)
            #         is_record = True
            #     else:
            #         del record
            #         is_record = False
            #     event_list['record'].clear()

            result, frame = self._camera.get_img()
            if result and frame is not None:
                # t3 = time.time()
                res, frame = self._predictor.detect_cars(frame)
                # if is_record:
                #     record.write(video_data=frame, net_data=res)
                self._pub((res, frame))
                # TODO: 让UI干计数功能
                # time.sleep(0.04)
                # t1 = t1 + time.time() - t3
                # count += 1
                # if count == 100:
                #     fps = float(count) / t1
                #     print(f'{self._name} count:{count} fps: {int(fps)} 传输等待队列: {self._queue.qsize()}')
                #     count = 0
                #     t1 = 0
            # TODO: 让相机自己监控自己
            # else:
            #     # 错误计数
            #     count_error += 1
            #     if count_error == 10:
            #         # TODO: 改相机记得看看这
            #         self._camera.restart()
            #         # print(f"相机出错重启:{name}")
            #         # cam.destroy()
            #         # del cam
            #         # cam = create_camera(name, using_video, event_list)
            #         count_error = 0
