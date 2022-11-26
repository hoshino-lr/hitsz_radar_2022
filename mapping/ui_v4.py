# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Demo_v4yRcOuC.ui'
##
## Created by: Qt User Interface Compiler version 5.15.6
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *  # type: ignore
from PyQt5.QtGui import *  # type: ignore
from PyQt5.QtWidgets import *  # type: ignore


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1881, 1060)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 1869, 1048))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.tab.setMinimumSize(QSize(0, 0))
        self.widget = QWidget(self.tab)
        self.widget.setObjectName(u"widget")
        self.middle = QGridLayout(self.widget)
        self.middle.setSpacing(2)
        self.middle.setObjectName(u"middle")
        self.middle.setContentsMargins(2, 2, 2, 2)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.far_demo = QLabel(self.widget)
        self.far_demo.setObjectName(u"far_demo")
        self.far_demo.setMinimumSize(QSize(476, 270))
        self.far_demo.setFrameShape(QFrame.StyledPanel)
        self.far_demo.setAlignment(Qt.AlignCenter)
        self.far_demo.setMargin(5)
        self.far_demo.setOpenExternalLinks(False)

        self.horizontalLayout_2.addWidget(self.far_demo)

        self.blood = QLabel(self.widget)
        self.blood.setObjectName(u"blood")
        self.blood.setMinimumSize(QSize(320, 270))
        self.blood.setFrameShape(QFrame.StyledPanel)
        self.blood.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.blood)

        self.hero_demo = QLabel(self.widget)
        self.hero_demo.setObjectName(u"hero_demo")
        self.hero_demo.setMinimumSize(QSize(320, 270))
        self.hero_demo.setFrameShape(QFrame.StyledPanel)
        self.hero_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.hero_demo)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)

        self.middle.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.left_demo = QLabel(self.widget)
        self.left_demo.setObjectName(u"left_demo")
        self.left_demo.setMinimumSize(QSize(146, 693))
        self.left_demo.setFrameShape(QFrame.StyledPanel)
        self.left_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.left_demo)

        self.main_demo = QLabel(self.widget)
        self.main_demo.setObjectName(u"main_demo")
        self.main_demo.setMinimumSize(QSize(1170, 693))
        self.main_demo.setFrameShape(QFrame.StyledPanel)
        self.main_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.main_demo)

        self.right_demo = QLabel(self.widget)
        self.right_demo.setObjectName(u"right_demo")
        self.right_demo.setMinimumSize(QSize(146, 693))
        self.right_demo.setFrameShape(QFrame.StyledPanel)
        self.right_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.right_demo)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 8)
        self.horizontalLayout_4.setStretch(2, 1)

        self.middle.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)

        self.middle.setRowStretch(0, 7)
        self.widget1 = QWidget(self.tab)
        self.widget1.setObjectName(u"widget1")
        self.widget1.setGeometry(QRect(1480, 10, 357, 971))
        self.verticalLayout_2 = QVBoxLayout(self.widget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.map = QLabel(self.widget1)
        self.map.setObjectName(u"map")
        self.map.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.map.sizePolicy().hasHeightForWidth())
        self.map.setSizePolicy(sizePolicy)
        self.map.setMinimumSize(QSize(320, 718))
        self.map.setMaximumSize(QSize(320, 16777215))
        self.map.setFrameShape(QFrame.StyledPanel)
        self.map.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.map)

        self.condition = QLabel(self.widget1)
        self.condition.setObjectName(u"condition")
        self.condition.setMinimumSize(QSize(355, 73))
        self.condition.setFrameShape(QFrame.StyledPanel)
        self.condition.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.condition)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.Pause = QPushButton(self.widget1)
        self.Pause.setObjectName(u"Pause")

        self.gridLayout.addWidget(self.Pause, 1, 0, 2, 1)

        self.SpeedUp = QPushButton(self.widget1)
        self.SpeedUp.setObjectName(u"SpeedUp")

        self.gridLayout.addWidget(self.SpeedUp, 1, 1, 2, 1)

        self.SlowDown = QPushButton(self.widget1)
        self.SlowDown.setObjectName(u"SlowDown")

        self.gridLayout.addWidget(self.SlowDown, 1, 2, 2, 1)

        self.ShowLidar = QPushButton(self.widget1)
        self.ShowLidar.setObjectName(u"ShowLidar")
        self.ShowLidar.setMinimumSize(QSize(115, 30))

        self.gridLayout.addWidget(self.ShowLidar, 0, 2, 1, 2)

        self.ChangeView = QPushButton(self.widget1)
        self.ChangeView.setObjectName(u"ChangeView")
        self.ChangeView.setMinimumSize(QSize(115, 30))

        self.gridLayout.addWidget(self.ChangeView, 0, 0, 1, 2)

        self.record = QPushButton(self.widget1)
        self.record.setObjectName(u"record")
        self.record.setMinimumSize(QSize(115, 30))

        self.gridLayout.addWidget(self.record, 3, 0, 1, 2)

        self.ShutDown = QPushButton(self.widget1)
        self.ShutDown.setObjectName(u"ShutDown")
        self.ShutDown.setMinimumSize(QSize(115, 30))

        self.gridLayout.addWidget(self.ShutDown, 3, 2, 1, 2)

        self.CurrentSpeed = QLabel(self.widget1)
        self.CurrentSpeed.setObjectName(u"CurrentSpeed")

        self.gridLayout.addWidget(self.CurrentSpeed, 1, 3, 2, 1)


        self.verticalLayout_2.addLayout(self.gridLayout)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.horizontalLayoutWidget_2 = QWidget(self.tab_2)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(0, 10, 1841, 971))
        self.horizontalLayout_3 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.epnp_begin = QPushButton(self.horizontalLayoutWidget_2)
        self.epnp_begin.setObjectName(u"epnp_begin")
        self.epnp_begin.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.epnp_begin)

        self.epnp_clear = QPushButton(self.horizontalLayoutWidget_2)
        self.epnp_clear.setObjectName(u"epnp_clear")
        self.epnp_clear.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.epnp_clear)

        self.epnp_back = QPushButton(self.horizontalLayoutWidget_2)
        self.epnp_back.setObjectName(u"epnp_back")
        self.epnp_back.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.epnp_back)

        self.epnp_next = QPushButton(self.horizontalLayoutWidget_2)
        self.epnp_next.setObjectName(u"epnp_next")
        self.epnp_next.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.epnp_next)

        self.pnp_condition = QLabel(self.horizontalLayoutWidget_2)
        self.pnp_condition.setObjectName(u"pnp_condition")
        self.pnp_condition.setFrameShape(QFrame.StyledPanel)
        self.pnp_condition.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.verticalLayout.addWidget(self.pnp_condition)


        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.pnp_demo = QGraphicsView(self.horizontalLayoutWidget_2)
        self.pnp_demo.setObjectName(u"pnp_demo")

        self.horizontalLayout_3.addWidget(self.pnp_demo)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 8)
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.ChangeView.clicked.connect(MainWindow.ChangeView_on_clicked)
        self.epnp_begin.clicked.connect(MainWindow.epnp_calculate)
        self.epnp_back.clicked.connect(MainWindow.epnp_back_on_clicked)
        self.epnp_clear.clicked.connect(MainWindow.epnp_clear_on_clicked)
        self.epnp_next.clicked.connect(MainWindow.epnp_next_on_clicked)
        self.record.clicked.connect(MainWindow.record_on_clicked)
        self.ShowLidar.clicked.connect(MainWindow.showpc_on_clicked)
        self.ShutDown.clicked.connect(MainWindow.CloseProgram_on_clicked)
        self.pnp_demo.mousePressEvent = MainWindow.epnp_mouseEvent
        self.main_demo.mouseMoveEvent = MainWindow.pc_mouseEvent
        self.far_demo.mousePressEvent = MainWindow.eco_mouseEvent
        self.far_demo.keyPressEvent = MainWindow.eco_key_on_clicked
        self.condition.keyPressEvent = MainWindow.condition_key_on_clicked
        self.condition.setFocusPolicy(Qt.ClickFocus)
        self.far_demo.setFocusPolicy(Qt.ClickFocus)
        self.tabWidget.currentChanged.connect(MainWindow.epnp_on_clicked)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.far_demo.setText(QCoreApplication.translate("MainWindow", u"\u654c\u65b9\u57fa\u5730\u89c6\u89d2", None))
        self.blood.setText(QCoreApplication.translate("MainWindow", u"\u8840\u91cf\u663e\u793a\u754c\u9762", None))
        self.hero_demo.setText(QCoreApplication.translate("MainWindow", u"\u4e09\u500d\u955c\u663e\u793a\u754c\u9762", None))
        self.left_demo.setText(QCoreApplication.translate("MainWindow", u"left_demo", None))
        self.main_demo.setText(QCoreApplication.translate("MainWindow", u"main_demo", None))
        self.right_demo.setText(QCoreApplication.translate("MainWindow", u"right_demo", None))
        self.map.setText(QCoreApplication.translate("MainWindow", u"\u5c0f\u5730\u56fe", None))
        self.condition.setText(QCoreApplication.translate("MainWindow", u"\u96f7\u8fbe\u7ad9\u8fd0\u884c\u72b6\u6001", None))
        self.Pause.setText(QCoreApplication.translate("MainWindow", u"\u6682\u505c", None))
        self.SpeedUp.setText(QCoreApplication.translate("MainWindow", u"\u52a0\u901f", None))
        self.SlowDown.setText(QCoreApplication.translate("MainWindow", u"\u51cf\u901f", None))
        self.ShowLidar.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u6fc0\u5149\u96f7\u8fbe", None))
        self.ChangeView.setText(QCoreApplication.translate("MainWindow", u"\u6539\u53d8\u89c6\u89d2", None))
        self.record.setText(QCoreApplication.translate("MainWindow", u"\u5f55\u50cf", None))
        self.ShutDown.setText(QCoreApplication.translate("MainWindow", u"\u9000\u51fa", None))
        self.CurrentSpeed.setText(QCoreApplication.translate("MainWindow", u"x 1.0", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u4e3b\u9875\u9762", None))
        self.epnp_begin.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb", None))
        self.epnp_clear.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u7a7a", None))
        self.epnp_back.setText(QCoreApplication.translate("MainWindow", u"back", None))
        self.epnp_next.setText(QCoreApplication.translate("MainWindow", u"next", None))
        self.pnp_condition.setText(QCoreApplication.translate("MainWindow", u"pnp_condion", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"EPNP", None))
    # retranslateUi

