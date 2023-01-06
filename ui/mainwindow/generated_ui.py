# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'qt_genyWyFZN.ui'
##
## Created by: Qt User Interface Compiler version 5.15.7
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
        MainWindow.resize(1412, 840)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.tab.setMinimumSize(QSize(0, 0))
        self.gridLayout_2 = QGridLayout(self.tab)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setSizeConstraint(QLayout.SetMinimumSize)
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.far_demo = QLabel(self.tab)
        self.far_demo.setObjectName(u"far_demo")
        self.far_demo.setMinimumSize(QSize(476, 270))
        self.far_demo.setMaximumSize(QSize(16777215, 270))
        self.far_demo.setFrameShape(QFrame.StyledPanel)
        self.far_demo.setAlignment(Qt.AlignCenter)
        self.far_demo.setMargin(5)
        self.far_demo.setOpenExternalLinks(False)

        self.horizontalLayout_2.addWidget(self.far_demo)

        self.blood = QLabel(self.tab)
        self.blood.setObjectName(u"blood")
        self.blood.setMinimumSize(QSize(320, 270))
        self.blood.setMaximumSize(QSize(16777215, 270))
        self.blood.setFrameShape(QFrame.StyledPanel)
        self.blood.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.blood)

        self.hero_demo = QLabel(self.tab)
        self.hero_demo.setObjectName(u"hero_demo")
        self.hero_demo.setMinimumSize(QSize(320, 270))
        self.hero_demo.setMaximumSize(QSize(16777215, 270))
        self.hero_demo.setFrameShape(QFrame.StyledPanel)
        self.hero_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.hero_demo)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)

        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setSizeConstraint(QLayout.SetMinimumSize)
        self.left_demo = QLabel(self.tab)
        self.left_demo.setObjectName(u"left_demo")
        self.left_demo.setMinimumSize(QSize(140, 400))
        self.left_demo.setFrameShape(QFrame.StyledPanel)
        self.left_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.left_demo)

        self.main_demo = QLabel(self.tab)
        self.main_demo.setObjectName(u"main_demo")
        self.main_demo.setMinimumSize(QSize(800, 400))
        self.main_demo.setFrameShape(QFrame.StyledPanel)
        self.main_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.main_demo)

        self.right_demo = QLabel(self.tab)
        self.right_demo.setObjectName(u"right_demo")
        self.right_demo.setMinimumSize(QSize(140, 400))
        self.right_demo.setFrameShape(QFrame.StyledPanel)
        self.right_demo.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.right_demo)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 8)
        self.horizontalLayout_4.setStretch(2, 1)

        self.gridLayout_2.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)

        self.playControlGridLayout = QGridLayout()
        self.playControlGridLayout.setObjectName(u"playControlGridLayout")
        self.playControlGridLayout.setHorizontalSpacing(2)
        self.TogglePlay = QPushButton(self.tab)
        self.TogglePlay.setObjectName(u"TogglePlay")

        self.playControlGridLayout.addWidget(self.TogglePlay, 1, 0, 1, 1)

        self.SpeedSpinBox = QDoubleSpinBox(self.tab)
        self.SpeedSpinBox.setObjectName(u"SpeedSpinBox")
        self.SpeedSpinBox.setEnabled(False)
        self.SpeedSpinBox.setMinimum(0.010000000000000)
        self.SpeedSpinBox.setMaximum(20.000000000000000)
        self.SpeedSpinBox.setSingleStep(0.100000000000000)
        self.SpeedSpinBox.setValue(1.000000000000000)

        self.playControlGridLayout.addWidget(self.SpeedSpinBox, 1, 3, 1, 1)

        self.speedLabel = QLabel(self.tab)
        self.speedLabel.setObjectName(u"speedLabel")

        self.playControlGridLayout.addWidget(self.speedLabel, 1, 2, 1, 1)

        self.FpsStatus = QLabel(self.tab)
        self.FpsStatus.setObjectName(u"FpsStatus")

        self.playControlGridLayout.addWidget(self.FpsStatus, 1, 8, 1, 1)

        self.TimeSlider = QSlider(self.tab)
        self.TimeSlider.setObjectName(u"TimeSlider")
        self.TimeSlider.setMaximum(999)
        self.TimeSlider.setOrientation(Qt.Horizontal)

        self.playControlGridLayout.addWidget(self.TimeSlider, 0, 1, 1, 10)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.playControlGridLayout.addItem(self.horizontalSpacer, 1, 1, 1, 1)

        self.ResetSpeed = QPushButton(self.tab)
        self.ResetSpeed.setObjectName(u"ResetSpeed")

        self.playControlGridLayout.addWidget(self.ResetSpeed, 1, 6, 1, 1)

        self.PlayStatus = QLabel(self.tab)
        self.PlayStatus.setObjectName(u"PlayStatus")

        self.playControlGridLayout.addWidget(self.PlayStatus, 0, 0, 1, 1)

        self.CustomSpeed = QCheckBox(self.tab)
        self.CustomSpeed.setObjectName(u"CustomSpeed")

        self.playControlGridLayout.addWidget(self.CustomSpeed, 1, 7, 1, 1)

        self.SpeedSlider = QSlider(self.tab)
        self.SpeedSlider.setObjectName(u"SpeedSlider")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SpeedSlider.sizePolicy().hasHeightForWidth())
        self.SpeedSlider.setSizePolicy(sizePolicy)
        self.SpeedSlider.setMinimumSize(QSize(150, 0))
        self.SpeedSlider.setMaximum(6)
        self.SpeedSlider.setValue(3)
        self.SpeedSlider.setOrientation(Qt.Horizontal)
        self.SpeedSlider.setTickPosition(QSlider.TicksBelow)
        self.SpeedSlider.setTickInterval(1)

        self.playControlGridLayout.addWidget(self.SpeedSlider, 1, 4, 1, 1)


        self.gridLayout_2.addLayout(self.playControlGridLayout, 2, 0, 1, 1)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setSizeConstraint(QLayout.SetMaximumSize)
        self.map = QLabel(self.tab)
        self.map.setObjectName(u"map")
        self.map.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.map.sizePolicy().hasHeightForWidth())
        self.map.setSizePolicy(sizePolicy1)
        self.map.setMinimumSize(QSize(250, 500))
        self.map.setMaximumSize(QSize(320, 16777215))
        self.map.setFrameShape(QFrame.StyledPanel)
        self.map.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.map)

        self.condition = QLabel(self.tab)
        self.condition.setObjectName(u"condition")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.condition.sizePolicy().hasHeightForWidth())
        self.condition.setSizePolicy(sizePolicy2)
        self.condition.setMinimumSize(QSize(250, 73))
        self.condition.setFrameShape(QFrame.StyledPanel)
        self.condition.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.condition)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.Record = QPushButton(self.tab)
        self.Record.setObjectName(u"Record")

        self.gridLayout.addWidget(self.Record, 3, 0, 1, 1)

        self.ChangeView = QPushButton(self.tab)
        self.ChangeView.setObjectName(u"ChangeView")

        self.gridLayout.addWidget(self.ChangeView, 0, 0, 1, 1)

        self.ShutDown = QPushButton(self.tab)
        self.ShutDown.setObjectName(u"ShutDown")

        self.gridLayout.addWidget(self.ShutDown, 4, 0, 1, 2)

        self.DisplayLidar = QCheckBox(self.tab)
        self.DisplayLidar.setObjectName(u"DisplayLidar")

        self.gridLayout.addWidget(self.DisplayLidar, 0, 1, 1, 1)

        self.UseNet = QCheckBox(self.tab)
        self.UseNet.setObjectName(u"UseNet")
        self.UseNet.setChecked(False)

        self.gridLayout.addWidget(self.UseNet, 3, 1, 1, 1)


        self.verticalLayout_2.addLayout(self.gridLayout)


        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 1, 3, 1)

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

        self.horizontalLayout.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.ShutDown.clicked.connect(MainWindow.close)

        self.tabWidget.setCurrentIndex(0)


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
        self.TogglePlay.setText(QCoreApplication.translate("MainWindow", u"\u6682\u505c", None))
        self.speedLabel.setText(QCoreApplication.translate("MainWindow", u"\u901f\u5ea6\uff1ax", None))
        self.FpsStatus.setText(QCoreApplication.translate("MainWindow", u"\u7f51\u7edcFPS\uff1a0 \u76f8\u673aFPS\uff1a0", None))
        self.ResetSpeed.setText(QCoreApplication.translate("MainWindow", u"\u91cd\u7f6e1x", None))
        self.PlayStatus.setText(QCoreApplication.translate("MainWindow", u"00:00/00:00", None))
        self.CustomSpeed.setText(QCoreApplication.translate("MainWindow", u"\u81ea\u5b9a\u4e49\u901f\u5ea6", None))
        self.map.setText(QCoreApplication.translate("MainWindow", u"\u5c0f\u5730\u56fe", None))
        self.condition.setText(QCoreApplication.translate("MainWindow", u"\u96f7\u8fbe\u7ad9\u8fd0\u884c\u72b6\u6001", None))
        self.Record.setText(QCoreApplication.translate("MainWindow", u"\u5f55\u50cf", None))
        self.ChangeView.setText(QCoreApplication.translate("MainWindow", u"\u6539\u53d8\u89c6\u89d2", None))
        self.ShutDown.setText(QCoreApplication.translate("MainWindow", u"\u9000\u51fa", None))
        self.DisplayLidar.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u6fc0\u5149\u96f7\u8fbe", None))
        self.UseNet.setText(QCoreApplication.translate("MainWindow", u"\u4f7f\u7528\u5f55\u5236\u7f51\u7edc", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u4e3b\u9875\u9762", None))
        self.epnp_begin.setText(QCoreApplication.translate("MainWindow", u"\u5f00\u59cb", None))
        self.epnp_clear.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u7a7a", None))
        self.epnp_back.setText(QCoreApplication.translate("MainWindow", u"back", None))
        self.epnp_next.setText(QCoreApplication.translate("MainWindow", u"next", None))
        self.pnp_condition.setText(QCoreApplication.translate("MainWindow", u"pnp_condion", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"EPNP", None))
    # retranslateUi

