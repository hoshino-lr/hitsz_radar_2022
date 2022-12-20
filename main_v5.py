"""
UI类
created by 李龙 2022/8
last edited by 陈希峻 2022/11
"""

import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer

from logger.logger import LOGGER
from ui.mainwindow.mainwindow import Mywindow


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    # ui
    app = QtWidgets.QApplication(sys.argv)
    _stdout = sys.stdout
    sys.stdout = LOGGER(_stdout)
    MyShow = Mywindow()
    MyShow.show()

    # TODO: 改成响应式
    timer_main = QTimer()  # 主循环使用的线程
    timer_main.timeout.connect(MyShow.spin)
    timer_main.start(0)

    sys.exit(app.exec_())
