from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from Ui_mainwindow import *
import serial
import cv2
import serial.tools.list_ports
import sys
import numpy as np
import math
import time
import pupil_apriltags


ser_sta = 0
start_time = time.clock()
last_fps_time = time.clock()
def tran_pos(corners, pos, D):
    """corners is a matrix with shape of 4*2 clockwise left-up to left-down
    pos is the coordinate of ball (x,y)
    D is the edge length of the square"""
    Y = np.array([
        [0, 0, 1],
        [D, 0, 1],
        [D, D, 1],
        [0, D, 1]], dtype="float32")

    X = np.concatenate([corners, np.ones((4, 1))], axis=1)
    AT = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    pre = np.array([
        [pos[0], pos[1], 1]
    ])

    after = np.dot(pre, AT)[0]
    return [int(after[0]), int(after[1])]


# def find(lh, ls, lv, hh, hs, hv):
def find(gray,ball_val_L,test):
    X = 0
    Y = 0
    Xsend = -100
    Ysend = -100
    # lower = np.array([lh, ls, lv])
    # upper = np.array([hh, hs, hv])
    # mask = cv2.inRange(HSV, lower, upper)   # 二值化
    mask = cv2.inRange(gray, ball_val_L.value(), 255)  # 二值化
    cv2.imshow('mask', mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找图像轮廓
    tempdist = 60000
    for i in range(len(contours)):  # 遍历每个轮廓
        M = cv2.moments(contours[i])    # 求轮廓矩
        if M['m00'] > 100 and M['m00'] < 10000:    # 面积不为0
            cx = M['m10'] / M['m00']  # 求中心坐标
            cy = M['m01'] / M['m00']
            pos0 = [cx, cy]

            a = tran_pos(test, pos0, 400)
            b = tran_pos(test, contours[i][0][0], 400)

            if 0 < a[0] < 400 and 0 < a[1] < 400 and 0 < b[0] < 400 and 0 < b[1] < 400 and (a[0] - 200) ** 2 + (a[1] - 200) ** 2 < tempdist:
                center, radius = cv2.minEnclosingCircle(contours[i])    # 最小覆盖圆
                if M['m00'] > 0.4*radius**2:
                # 距离中心最近，且圆度满足要求
                # print(contours[i][0][0])
                    tempdist = (a[0]-200)**2+(a[1]-200)**2  # 距离最近
                    X = int(cx)
                    Y = int(cy)
                    Xsend = a[0]
                    Ysend = a[1]
    return X, Y, Xsend, Ysend

#找四个角点
def find_corner_apriltag(at_detector,gray,image):
    tags = at_detector.detect(gray)
    X = []
    Y = []
    for tag in tags:
        X.append(int(tag.center[0]))
        Y.append(int(tag.center[1]))
        cv2.circle(image, (int(tag.center[0]), int(tag.center[1])), 3, (0, 0, 255), 6)
    return np.array(X), np.array(Y), len(X)


def camDialog():
    #选择摄像头
    cam_int, okPressed = QInputDialog.getInt(None,"选择相机","请选择相机（0/1）:",min=0,max=1)
    if okPressed:
        return cam_int
    else:
        QMessageBox.critical(None, "Error", "选择默认相机0")
        return 0


class CamOpenThread(QThread):
    #update_data = pyqtSignal(int,int)

    def __init__(self,cam_open,cam,ballx,bally,arena,time,ser,ball_val_L,X,Y,fps_text,parent=None):
        super(CamOpenThread,self).__init__(parent)
        self.cam =cam
        self.ballx = ballx
        self.bally = bally
        self.arena = arena
        self.time = time
        self.ser = ser
        self.ball_val_L = ball_val_L
        self.cam_open = cam_open
        self.X = X
        self.Y = Y
        self.fps_text = fps_text


    def run(self):
        WIDTH = 1280
        HEIGHT = 720
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cam.set(cv2.CAP_PROP_FPS, 30)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.cam_open.setEnabled(False)

        at_detector = pupil_apriltags.Detector(families='tag25h9',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

        i=0
        while self.cam.isOpened():
            i += 1
            global last_fps_time
            if i%25 == 0:
                time_present = float(time.clock())- float(start_time)
                fps = 25/(time.clock()-last_fps_time)
                last_fps_time = time.clock()
                self.fps_text.setText(str(fps))

            _, image = self.cam.read()
            # print(np.shape(image))
            image = image[:, (WIDTH-HEIGHT)//2:(WIDTH+HEIGHT)//2]
            # HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # CornerX, CornerY, num = find_corner(40, 60, 80, 60, 255, 255)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            CornerX, CornerY, num = find_corner_apriltag(at_detector,gray,image)
            if num == 4:
                XY0 = CornerX + CornerY
                XY1 = CornerX - CornerY
                quad1X = CornerX[np.argmin(XY0)]
                quad2X = CornerX[np.argmax(XY1)]
                quad3X = CornerX[np.argmax(XY0)]
                quad4X = CornerX[np.argmin(XY1)]
                quad1Y = CornerY[np.argmin(XY0)]
                quad2Y = CornerY[np.argmax(XY1)]
                quad3Y = CornerY[np.argmax(XY0)]
                quad4Y = CornerY[np.argmin(XY1)]
                test = np.array([
                [quad1X, quad1Y],
                [quad2X, quad2Y],
                [quad3X, quad3Y],
                [quad4X, quad4Y]]
                )
                # 画出场地
                cv2.line(image, (quad1X, quad1Y), (quad2X, quad2Y), (255, 0, 255), 1)
                cv2.line(image, (quad2X, quad2Y), (quad3X, quad3Y), (255, 0, 255), 1)
                cv2.line(image, (quad3X, quad3Y), (quad4X, quad4Y), (255, 0, 255), 1)
                cv2.line(image, (quad4X, quad4Y), (quad1X, quad1Y), (255, 0, 255), 1)
                #CenterX, CenterY, Xsend, Ysend = find(150)
                CenterX, CenterY, Xsend, Ysend = find(gray,self.ball_val_L,test)  # 找球
                # Xsend是变换后，CenterX是变换前

                if CenterX != 0 and CenterY != 0:
                    cv2.circle(image, (CenterX, CenterY), 3, (0, 255, 0), 6)
                    self.ballx.setText(str(Xsend))
                    self.bally.setText(str(Ysend))

                Xsend = 400 - Xsend
                Ysend = 400 - Ysend
                # print(Xsend, Ysend)

                global ser_sta
                if ser_sta == 1:
                    s = bytes([250, (Xsend >> 7) + 1, (Xsend & 0x7f) + 1, (Ysend >> 7) + 1, (Ysend & 0x7f) + 1])
                    try:
                        self.ser.write(s)
                        self.X.setText(str(Xsend))
                        self.Y.setText(str(Ysend))
                    except:
                        self.X.setText('ser error!')
                        self.Y.setText('ser error!')


            self.time.setText(str(time.clock()))
            # 显示图像，其实有ui就不需要它了，但去掉的话ui也显示不出来，不知道为啥，只能这样最小化了
            cv2.imshow("image", image)
            cv2.moveWindow("image", 0, 0)
            cv2.resizeWindow("image", 1, 1)
            cv2.waitKey(1)
            # 在UI上画图
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)  # 这句话只能这么写，查了一个小时……
            self.arena.setPixmap(QPixmap.fromImage(img))

            if i == 20:
                start = time.clock()
            if i == 120:
                print(time.clock() - start)














#mainwindow
#寻找可选串口
def ser_read():
    port_list = list(serial.tools.list_ports.comports())
    port_list_str = []
    if len(port_list) <= 0:
        port_list_str = None
    else:
        port_list_0 = list(map(list,port_list))
        port_list_str = [port_list_0[i][0] for i in range(len(port_list_0))]
    return port_list_str 

class MainWindow(QMainWindow, Ui_mainwindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.port_list_str = ser_read()
        self.setupUi(self)
        if self.port_list_str != None:
            self.ser_selection.addItems(self.port_list_str)
        else:
            self.ser_selection.addItem("无串口")
        self.ser = serial.Serial()
        self.serial_open.clicked.connect(self.openSerial)
        self.serial_close.clicked.connect(self.closeSerial)
        self.cam_open.clicked.connect(self.openCamera)
        self.ser_update.clicked.connect(self.updateSerlist)
        self.cam_int = camDialog()
        self.camera = cv2.VideoCapture(self.cam_int)
        self.cam_label.setText(str(self.cam_int))

    def openSerial(self):        
        global ser_sta
        try:
            if ser_sta == 1:
                try:
                    self.ser.close()
                    ser_sta = 0
                except:
                    self.ser_status.setText("port close error!")
            self.ser.port=self.ser_selection.currentText()
            self.ser.baudrate = 115200
            self.ser.timeout = 5
            try:
                self.ser.open()
            except:
                self.ser_status.setText("port open error!")
            self.ser_status.setText("using "+self.ser.name)
            ser_sta = 1
        except :
            self.ser_status.setText("port error!")
        return

    def closeSerial(self):
        global ser_sta
        try:
            if ser_sta == 1:
                self.ser.close()
                ser_sta = 0
                self.ser_status.setText("关闭")
        except:
            self.ser_status.setText("port close error!")
        return

    def openCamera(self):
        self.cam_th = CamOpenThread(self.cam_open,self.camera,self.ballx,self.bally,self.arena,self.time,self.ser,self.ball_val_L,self.X,self.Y,self.fps_text)
        self.cam_th.start()

    def updateSerlist(self):
        port_list_str = ser_read()
        self.ser_selection.clear()
        if self.port_list_str != None:
            self.ser_selection.addItems(port_list_str)
        else:
            self.ser_selection.addItem("无串口")
        return