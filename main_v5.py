"""
UI类
created by 李龙 2022/8
last edited by 陈希峻 2022/11
"""

import sys
import config

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer

from logger.logger import LOGGER
from threads.ui_thread import UiThread
from ui.mainwindow.mainwindow import MainWindow

from record.replay_frame import RecordReadManager
from threads.camera_thread import CameraThread_Real, CameraThread_Video
from threads.process_thread import ProcessThread


class ThreadManager():
    def __init__(self, cam_configs):
        self._cam_init(cam_configs)
        self._ui_init()

    def _cam_init(self, cam_configs):
        """
        初始化相机
        :param cam_configs: 相机配置
        """
        self._cameras = {}
        for name, config in cam_configs.items():
            if not config['enable']:
                continue
            self._cameras[name] = {}
            if config['using_video']:
                self._cameras[name]['record'] = RecordReadManager(config)
                self._cameras[name]['camera'] = CameraThread_Video(config, self._cameras[name]['record'])
            else:
                self._cameras[name]['record'] = None
                self._cameras[name]['camera'] = CameraThread_Real(config)
            self._cameras[name]['process'] = ProcessThread(self._cameras[name]['camera'])

    def _ui_init(self):
        """
        初始化UI
        """
        self._window = MainWindow(self._cameras)
        self._window.show()
        self._timer_main = QTimer()
        self._timer_main.timeout.connect(self._window.spin)
        self._timer_main.start(0)
        # self._ui.start()


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    _stdout = sys.stdout
    sys.stdout = LOGGER(_stdout)

    thread_manager = ThreadManager(config.cam_config)

    # public.L_CAMERA_THREAD = CameraThread_Real("")
    # TODO: 改成响应式
    sys.exit(app.exec_())
