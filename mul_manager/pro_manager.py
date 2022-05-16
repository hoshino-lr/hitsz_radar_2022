"""
多线程管理类
mul_manager.py
用于多进程间的信息发布与接收
created by 李龙 in 2022/1
最终修改 by 李龙 in 2022/1/15
"""
from os import wait
from multiprocessing import Queue, Process, Event
import cv2 as cv
import time

def thread_detect(event, que, name):
    """
    debug 使用
    """
    from net.network_pro import Predictor
    import cv2
    import time
    # 多线程接收写法
    predictor1 = Predictor(name)
    cap = cv2.VideoCapture("/home/hoshino/CLionProjects/hitsz_radar/resources/two_cam/1.mp4")
    count = 0
    t1 = time.time()
    t3 = time.time()
    print("副线程开始")
    print(id(event))
    while True:
        if t1 - t3 > 20:
            break
        frame = sub(event, que)
        if frame is not None:
            im1 = frame.copy()
            predictor1.detect_cars(frame)
            count += 1
            t2 = time.time()
            if t2 - t1 > 1:
                fps = float(count) / (t2 - t1)
                print(f'fps: {fps}')
                t1 = time.time()
                count = 0
    predictor1.stop()


def pub(event, que, data):
    """
    pub 函数
    :param event 多进程事件
    :param que 传输队列
    :param data 待传输数据
    """
    if event.is_set():
        return
    if que.full():
        try:
            que.get()
        except:
            pass
    que.put(data)
    event.set()


def sub(event, que):
    """
    sub 函数
    :param event 多进程事件
    :param que 传输队列
    """
    _wait = event.wait(timeout=0.04)
    if _wait:
        try:
            re_data = que.get_nowait()
        except:
            print("[ERROR] sub process can't get data!")
            re_data = False, None
        event.clear()
        return re_data
    else:
        return False, None


