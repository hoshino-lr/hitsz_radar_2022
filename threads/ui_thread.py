"""
UI线程
在这里显示更新UI

QT限制 弃用
created by 陈希峻 2022/12/22
"""

from threading import Thread
from ui.mainwindow.mainwindow import MainWindow


class UiThread():
    def __init__(self, window: MainWindow):
        self._window = window
        self._is_terminated = True
        self._thread = None

    def start(self):
        self._is_terminated = False
        self._thread = Thread(target=self._spin, name="ui_thread")
        self._thread.start()
        print("UI线程开始")

    def stop(self):
        self._is_terminated = True
        print("UI线程退出")

    def __del__(self):
        self.stop()
        self._thread.join()
        pass

    def _spin(self):
        while not self._is_terminated:
            self._window.spin()

    @property
    def window(self):
        return self._window
