# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\大二上\科协\新生赛\MEDCHost20\v1.0\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        mainwindow.setObjectName("mainwindow")
        mainwindow.resize(1563, 923)
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
        self.BX = QtWidgets.QLabel(self.centralwidget)
        self.BX.setGeometry(QtCore.QRect(20, 80, 72, 15))
        self.BX.setObjectName("BX")
        self.BY = QtWidgets.QLabel(self.centralwidget)
        self.BY.setGeometry(QtCore.QRect(140, 80, 72, 15))
        self.BY.setObjectName("BY")
        self.TT = QtWidgets.QLabel(self.centralwidget)
        self.TT.setGeometry(QtCore.QRect(270, 80, 72, 15))
        self.TT.setObjectName("TT")
        self.time = QtWidgets.QLabel(self.centralwidget)
        self.time.setGeometry(QtCore.QRect(330, 80, 72, 15))
        self.time.setObjectName("time")
        self.ballx = QtWidgets.QLabel(self.centralwidget)
        self.ballx.setGeometry(QtCore.QRect(100, 80, 72, 15))
        self.ballx.setObjectName("ballx")
        self.bally = QtWidgets.QLabel(self.centralwidget)
        self.bally.setGeometry(QtCore.QRect(210, 80, 72, 15))
        self.bally.setObjectName("bally")
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
        self.serial_open.setGeometry(QtCore.QRect(20, 280, 150, 46))
        self.serial_open.setObjectName("serial_open")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(371, 131, 120, 24))
        self.label_2.setObjectName("label_2")
        self.ser_status = QtWidgets.QLabel(self.centralwidget)
        self.ser_status.setGeometry(QtCore.QRect(503, 131, 161, 24))
        self.ser_status.setObjectName("ser_status")
        self.serial_close = QtWidgets.QPushButton(self.centralwidget)
        self.serial_close.setGeometry(QtCore.QRect(190, 280, 150, 46))
        self.serial_close.setObjectName("serial_close")
        self.cam_label = QtWidgets.QLabel(self.centralwidget)
        self.cam_label.setGeometry(QtCore.QRect(200, 200, 161, 24))
        self.cam_label.setObjectName("cam_label")
        self.cam_open = QtWidgets.QPushButton(self.centralwidget)
        self.cam_open.setGeometry(QtCore.QRect(190, 350, 150, 46))
        self.cam_open.setObjectName("cam_open")
        self.ser_update = QtWidgets.QPushButton(self.centralwidget)
        self.ser_update.setGeometry(QtCore.QRect(20, 350, 150, 46))
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
        self.BX_2 = QtWidgets.QLabel(self.centralwidget)
        self.BX_2.setGeometry(QtCore.QRect(390, 190, 72, 21))
        self.BX_2.setObjectName("BX_2")
        self.fps_text = QtWidgets.QLabel(self.centralwidget)
        self.fps_text.setGeometry(QtCore.QRect(500, 194, 72, 21))
        self.fps_text.setText("")
        self.fps_text.setObjectName("fps_text")
        mainwindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainwindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1563, 37))
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
        self.BX.setText(_translate("mainwindow", "BallX:"))
        self.BY.setText(_translate("mainwindow", "BallY:"))
        self.TT.setText(_translate("mainwindow", "Time:"))
        self.time.setText(_translate("mainwindow", "time"))
        self.ballx.setText(_translate("mainwindow", "X"))
        self.bally.setText(_translate("mainwindow", "Y"))
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
        self.BX_2.setText(_translate("mainwindow", "fps"))

