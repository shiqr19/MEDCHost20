# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\大二上\科协\新生赛\MEDC2020\2020新生赛上位机\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        mainwindow.setObjectName("mainwindow")
        mainwindow.resize(1563, 1215)
        self.centralwidget = QtWidgets.QWidget(mainwindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ball_val_L = QtWidgets.QSlider(self.centralwidget)
        self.ball_val_L.setGeometry(QtCore.QRect(100, 20, 240, 22))
        self.ball_val_L.setMaximum(255)
        self.ball_val_L.setProperty("value", 160)
        self.ball_val_L.setOrientation(QtCore.Qt.Horizontal)
        self.ball_val_L.setObjectName("ball_val_L")
        self.BVL = QtWidgets.QLabel(self.centralwidget)
        self.BVL.setGeometry(QtCore.QRect(20, 20, 90, 20))
        self.BVL.setObjectName("BVL")
        self.arena = QtWidgets.QLabel(self.centralwidget)
        self.arena.setGeometry(QtCore.QRect(640, 10, 720, 720))
        self.arena.setAlignment(QtCore.Qt.AlignCenter)
        self.arena.setObjectName("arena")
        self.ser_selection = QtWidgets.QComboBox(self.centralwidget)
        self.ser_selection.setGeometry(QtCore.QRect(200, 130, 120, 30))
        self.ser_selection.setObjectName("ser_selection")
        self.SERL = QtWidgets.QLabel(self.centralwidget)
        self.SERL.setGeometry(QtCore.QRect(20, 130, 108, 24))
        self.SERL.setObjectName("SERL")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 200, 108, 24))
        self.label.setObjectName("label")
        self.serial_open = QtWidgets.QPushButton(self.centralwidget)
        self.serial_open.setGeometry(QtCore.QRect(50, 920, 150, 46))
        self.serial_open.setObjectName("serial_open")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(371, 131, 120, 24))
        self.label_2.setObjectName("label_2")
        self.ser_status = QtWidgets.QLabel(self.centralwidget)
        self.ser_status.setGeometry(QtCore.QRect(503, 131, 161, 24))
        self.ser_status.setObjectName("ser_status")
        self.serial_close = QtWidgets.QPushButton(self.centralwidget)
        self.serial_close.setGeometry(QtCore.QRect(250, 920, 150, 46))
        self.serial_close.setObjectName("serial_close")
        self.cam_label = QtWidgets.QLabel(self.centralwidget)
        self.cam_label.setGeometry(QtCore.QRect(200, 200, 161, 24))
        self.cam_label.setObjectName("cam_label")
        self.cam_open = QtWidgets.QPushButton(self.centralwidget)
        self.cam_open.setGeometry(QtCore.QRect(250, 990, 150, 46))
        self.cam_open.setObjectName("cam_open")
        self.ser_update = QtWidgets.QPushButton(self.centralwidget)
        self.ser_update.setGeometry(QtCore.QRect(50, 990, 150, 46))
        self.ser_update.setObjectName("ser_update")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(400, 20, 108, 24))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(400, 80, 108, 24))
        self.label_4.setObjectName("label_4")
        self.X = QtWidgets.QLabel(self.centralwidget)
        self.X.setGeometry(QtCore.QRect(490, 20, 108, 24))
        self.X.setObjectName("X")
        self.Y = QtWidgets.QLabel(self.centralwidget)
        self.Y.setGeometry(QtCore.QRect(490, 80, 108, 24))
        self.Y.setObjectName("Y")
        self.SERL_2 = QtWidgets.QLabel(self.centralwidget)
        self.SERL_2.setGeometry(QtCore.QRect(20, 280, 108, 24))
        self.SERL_2.setObjectName("SERL_2")
        self.task_selection = QtWidgets.QComboBox(self.centralwidget)
        self.task_selection.setGeometry(QtCore.QRect(200, 280, 120, 30))
        self.task_selection.setObjectName("task_selection")
        self.task_selection.addItem("")
        self.task_selection.addItem("")
        self.task_selection.addItem("")
        self.task_start = QtWidgets.QPushButton(self.centralwidget)
        self.task_start.setGeometry(QtCore.QRect(50, 850, 150, 46))
        self.task_start.setObjectName("task_start")
        self.timing_start = QtWidgets.QPushButton(self.centralwidget)
        self.timing_start.setGeometry(QtCore.QRect(50, 780, 150, 46))
        self.timing_start.setObjectName("timing_start")
        self.task_sta = QtWidgets.QLabel(self.centralwidget)
        self.task_sta.setGeometry(QtCore.QRect(370, 280, 108, 24))
        self.task_sta.setObjectName("task_sta")
        self.SERL_3 = QtWidgets.QLabel(self.centralwidget)
        self.SERL_3.setGeometry(QtCore.QRect(20, 360, 141, 24))
        self.SERL_3.setObjectName("SERL_3")
        self.total_time = QtWidgets.QLabel(self.centralwidget)
        self.total_time.setGeometry(QtCore.QRect(240, 360, 141, 24))
        self.total_time.setObjectName("total_time")
        self.task_terminate = QtWidgets.QPushButton(self.centralwidget)
        self.task_terminate.setGeometry(QtCore.QRect(250, 850, 150, 46))
        self.task_terminate.setObjectName("task_terminate")
        self.result_text = QtWidgets.QTextEdit(self.centralwidget)
        self.result_text.setGeometry(QtCore.QRect(640, 790, 751, 301))
        self.result_text.setObjectName("result_text")
        self.SERL_4 = QtWidgets.QLabel(self.centralwidget)
        self.SERL_4.setGeometry(QtCore.QRect(20, 430, 151, 24))
        self.SERL_4.setObjectName("SERL_4")
        self.current_time_text = QtWidgets.QLabel(self.centralwidget)
        self.current_time_text.setGeometry(QtCore.QRect(240, 430, 141, 24))
        self.current_time_text.setObjectName("current_time_text")
        self.SERL_5 = QtWidgets.QLabel(self.centralwidget)
        self.SERL_5.setGeometry(QtCore.QRect(20, 500, 171, 24))
        self.SERL_5.setObjectName("SERL_5")
        self.current_point_text = QtWidgets.QLabel(self.centralwidget)
        self.current_point_text.setGeometry(QtCore.QRect(240, 500, 141, 24))
        self.current_point_text.setText("")
        self.current_point_text.setObjectName("current_point_text")
        self.SERL_6 = QtWidgets.QLabel(self.centralwidget)
        self.SERL_6.setGeometry(QtCore.QRect(20, 70, 108, 24))
        self.SERL_6.setObjectName("SERL_6")
        self.tag_id_text = QtWidgets.QLabel(self.centralwidget)
        self.tag_id_text.setGeometry(QtCore.QRect(200, 70, 141, 24))
        self.tag_id_text.setObjectName("tag_id_text")
        mainwindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainwindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1563, 30))
        self.menubar.setObjectName("menubar")
        mainwindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainwindow)
        self.statusbar.setObjectName("statusbar")
        mainwindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainwindow)
        QtCore.QMetaObject.connectSlotsByName(mainwindow)

    def retranslateUi(self, mainwindow):
        _translate = QtCore.QCoreApplication.translate
        mainwindow.setWindowTitle(_translate("mainwindow", "仪器设计大赛新生组"))
        self.BVL.setText(_translate("mainwindow", "Ball_val_L"))
        self.arena.setText(_translate("mainwindow", "arena"))
        self.SERL.setText(_translate("mainwindow", "Serial"))
        self.label.setText(_translate("mainwindow", "Camera"))
        self.serial_open.setText(_translate("mainwindow", "打开串口"))
        self.label_2.setText(_translate("mainwindow", "串口状态："))
        self.ser_status.setText(_translate("mainwindow", "关闭"))
        self.serial_close.setText(_translate("mainwindow", "关闭串口"))
        self.cam_label.setText(_translate("mainwindow", "0"))
        self.cam_open.setText(_translate("mainwindow", "打开相机"))
        self.ser_update.setText(_translate("mainwindow", "更新串口"))
        self.label_3.setText(_translate("mainwindow", "Xsend"))
        self.label_4.setText(_translate("mainwindow", "Ysend"))
        self.X.setText(_translate("mainwindow", "0"))
        self.Y.setText(_translate("mainwindow", "0"))
        self.SERL_2.setText(_translate("mainwindow", "Task"))
        self.task_selection.setItemText(0, _translate("mainwindow", "Task0"))
        self.task_selection.setItemText(1, _translate("mainwindow", "Task1"))
        self.task_selection.setItemText(2, _translate("mainwindow", "Task2"))
        self.task_start.setText(_translate("mainwindow", "选择任务"))
        self.timing_start.setText(_translate("mainwindow", "开始计时"))
        self.task_sta.setText(_translate("mainwindow", "无任务"))
        self.SERL_3.setText(_translate("mainwindow", "Total time:"))
        self.total_time.setText(_translate("mainwindow", "0"))
        self.task_terminate.setText(_translate("mainwindow", "终止任务"))
        self.SERL_4.setText(_translate("mainwindow", "current time:"))
        self.current_time_text.setText(_translate("mainwindow", "0"))
        self.SERL_5.setText(_translate("mainwindow", "current point:"))
        self.SERL_6.setText(_translate("mainwindow", "group"))
        self.tag_id_text.setText(_translate("mainwindow", "1"))

