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
task_sta = 0
timing_sta = 0
start_time = time.clock()
last_fps_time = time.clock()
in_area = 10
around_area = 20

def tran_pos(corners,pos,D):
    """corners is a matrix with shape of 4*2 clockwise left-up to left-down
    pos is the coordinate of ball (x,y)
    D is the edge length of the square"""
    Y = np.array([
        [0, 0],
        [D, 0],
        [D, D],
        [0, D]], dtype="float32")
    X=corners.astype('float32')
    M=cv2.getPerspectiveTransform(X,Y)

    pre=np.array([
        [pos[0],pos[1],1]
    ])
    after=np.dot(M,pre.T)
    return [int(after[0]/after[2]),int(after[1]/after[2])]

def inv_tran_pos(corners,pos,D):
    """corners is a matrix with shape of 4*2 clockwise left-up to left-down
    pos is the coordinate of ball (x,y)
    D is the edge length of the square"""
    Y = np.array([
        [0, 0],
        [D, 0],
        [D, D],
        [0, D]], dtype="float32")
    X=corners.astype('float32')
    M=cv2.getPerspectiveTransform(Y,X)

    pre=np.array([
        [pos[0],pos[1],1]
    ])
    after=np.dot(M,pre.T)
    return [int(after[0]/after[2]),int(after[1]/after[2])]


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
    #cv2.imshow('mask', mask)
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
def find_corner_apriltag(at_detector,gray,image,tag_id,tag_id_text):
    tags = at_detector.detect(gray)
    X = []
    Y = []
    for tag in tags:
        if tag_id == -1 or tag.tag_id == tag_id:
            X.append(int(tag.center[0]))
            Y.append(int(tag.center[1]))
            cv2.circle(image, (int(tag.center[0]), int(tag.center[1])), 3, (0, 0, 255), 6)
            tag_id_text.setText(str(tags[0].tag_id))
    return np.array(X), np.array(Y), len(X)


def camDialog():
    #选择摄像头
    cam_int, okPressed = QInputDialog.getInt(None,"选择相机","请选择相机（0/1）:",min=0,max=1)
    if okPressed:
        return cam_int
    else:
        QMessageBox.critical(None, "Error", "选择默认相机0")
        return 0

#选择Tag编号
def get_tag_id():
    tag_id, okPressed = QInputDialog.getInt(None,"选择组号","请选择组号（0-23）,-1为识别所有tag:",min=-1,max=23)
    if okPressed:
        return tag_id
    else:
        QMessageBox.critical(None, "Error", "识别所有tag")
        return -1


#评分机制
def judge_in_area(x_send,y_send,x_target,y_target):
    distance = math.sqrt((x_send-x_target)**2 + (y_send - y_target)**2)
    if distance < in_area:
        return 1
    elif distance < around_area:
        return 2
    else:
        return 0
# ----- 判断当前点是否在目标点周围某个半径的圆内。要走到的目标点半径是 in_area，要避开的点半径是 around_area。 
#  如果在要走到的点半径内，就返回1； 如果在要避开的点的半径内，就返回一个>0的值；否则返回0


class CamOpenThread(QThread):
    updated = QtCore.pyqtSignal(str)

    def __init__(self,cam_open,cam,arena,ser,ball_val_L,X,Y,total_time,result_text,task_start,current_point_text,
                 current_time_text,task_sta,tag_id,tag_id_text,fps_text,ui_sta_text,parent=None):
        super(CamOpenThread,self).__init__(parent)
        self.cam =cam
        self.arena = arena
        self.ser = ser
        self.ball_val_L = ball_val_L
        self.cam_open = cam_open
        self.X = X
        self.Y = Y
        self.total_time = total_time
        self.result_text = result_text
        self.task_start = task_start
        self.current_point_text = current_point_text
        self.current_time_text = current_time_text
        self.task_sta = task_sta
        self.tag_id = tag_id
        self.tag_id_text = tag_id_text
        self.fps_text = fps_text
        self.ui_sta_text = ui_sta_text
        self.tasks = [[[200,200]],[[200,200]],[[100,300],[300,300],[300,100],[100,100]]]
        self.point_count = [0,1,4]
        self.total_task = 0
        self.current_point = 0
        self.in_area_time = time.clock()
        self.in_area_sta = 0
        self.stop_timing_sta = 0
        self.total_time_list = []
        # ----- 下面这行是后来加的，在目标点周围停留的时间 【！！！！！】
        self.hold_time = 3.
        

    def __in_departure_area(self, x, y):
        # departure_area_list = [None, None, (100, 100), (100, 300)]  # ----- 在这里改出发区域位置 【！！！！！】
        departure_area_list = [None, None, (100, 100), (100, 200)] 
        if task_sta == 1:
            return True
        elif task_sta == 2 or task_sta == 3:
            return judge_in_area(x, y, departure_area_list[task_sta][0], departure_area_list[task_sta][1]) == 1

    def __in_target_area(self, x, y):
        target_point = self.target_point_list[-1]
        return judge_in_area(x, y, target_point[0], target_point[1]) == 1

    def __in_wrong_target_area(self, x, y):
        __wrong_target_point_list = self.target_point_list[0:-1]
        for wrong_point in __wrong_target_point_list:
            if judge_in_area(x, y, wrong_point[0], wrong_point[1]) == 1:
                return wrong_point
        return False

    def __in_ban_area(self, x, y):
        for ban_point in self.ban_point_list:
            if judge_in_area(x, y, ban_point[0], ban_point[1]) > 0:
                return True
        return False
    # ----- 上面这几个函数是后加的


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
            #display total time
            global timing_sta,start_time,task_sta,last_fps_time
            if i%25 == 0:
                time_present = float(time.clock())- float(start_time)
                if timing_sta == 1:
                    self.total_time.setText('%d'%time_present)
                    # ----- timing_sta = 1 表示计时已经开始， = 其它数值，表示计时还未开始
                    # ----- 注意这里用 timing_sta 表达的“总时间”total_time，而不是单个目标点的时间 current_time
                fps = 25/(time.clock()-last_fps_time)
                last_fps_time = time.clock()
                self.fps_text.setText(str(fps))


            i += 1
            _, image = self.cam.read()
            # print(np.shape(image))
            image = image[:, (WIDTH-HEIGHT)//2:(WIDTH+HEIGHT)//2]
            # HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # CornerX, CornerY, num = find_corner(40, 60, 80, 60, 255, 255)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            CornerX, CornerY, num = find_corner_apriltag(at_detector,gray,image,self.tag_id,self.tag_id_text)
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

                Xsend = 400 - Xsend
                Ysend = 400 - Ysend
                # print(Xsend, Ysend)

                #判断
                # ----- 评分在这里！！！
                # ----- timing_sta 表示的是现在是否在计时（总时间 total time，
                # 而不是当前目标点时间 current time），
                # timing_sta = 0 表示未开始计时，且小球不在出发区域内，
                # timing_sta = 1 表示已开始计时， timing_sta = 2 表示小球在出发区域内等待开始计时
                # ----- start_time 记录的是当前任务开始计时的系统时间
                # ---------- 以下的代码改过
                # ---------- 所有用来改参数的地方都用 【！！！！！】 标注了
                if task_sta != 0:  # task_sta == 0 似乎应该什么都不做
                    if timing_sta == 0:
                        if task_sta == 1:
                            timing_sta = 1
                            self.updated.emit('Task 1 started ... ')
                            self.target_point_list = [(200, 200)]  # ----- task1的目标点从这里改【！！！！！】 （可以有多个，按照倒序依次写成元组）
                            __current_point = self.target_point_list[-1]
                            self.current_point_text.setText("(%d, %d)" % (__current_point[0], __current_point[1]))
                            self.in_target_area_sta = False
                            self.in_ban_area_sta = False
                            start_time = time.clock()
                        elif task_sta == 2 or task_sta == 3:
                            if self.__in_departure_area(Xsend, Ysend):
                                timing_sta = 2
                                self.updated.emit('Ready ... ')
                                if task_sta == 2:
                                    self.target_point_list = [(300, 300)]  # ----- task2的目标点从这里改【！！！！！】 （可以有多个，按照倒序依次写成元组）
                                    self.ban_point_list = [(100, 300), (200, 200), (300, 100)]  # ----- task2 禁区在这里改【！！！！！】
                                    # ----- 下面这行是task2改规则之后加的
                                    self.task2_into_forbidden_area = False
                                    # -----
                                elif task_sta == 3:
                                    # self.target_point_list = [(x, 100 + 0.02 * (x - 200) ** 2) for x in range(125, 301, 25)] 
                                    self.target_point_list = [(x, round(200 + 100 * math.sin(0.01 * math.pi * x))) for x in range(125, 301, 25)] 
                                    # ----- task3的目标点从这里改 【！！！！！】（按顺序依次写各个目标点，不用逆序，因为下面一行逆序过了）
                                    self.target_point_list.reverse()
                                __current_point = self.target_point_list[-1]
                                self.current_point_text.setText("(%d, %d)" % (__current_point[0], __current_point[1]))
                                self.in_target_area_sta = False
                                self.in_ban_area_sta = False
                    elif timing_sta == 2:
                        # 此时一定有 task_sta == 2 or task_sta == 3
                        if not self.__in_departure_area(Xsend, Ysend):
                            timing_sta = 1
                            self.updated.emit('Task %d started ... ' % (task_sta))
                            start_time = time.clock()
                    elif timing_sta == 1:
                        # ----- task1
                        if task_sta == 1:
                            if self.in_target_area_sta == False:
                                if self.__in_target_area(Xsend, Ysend) == True:
                                    self.in_target_area_sta = True
                                    self.in_target_area_start_time = time.clock()
                            else:  # self.in_target_area_sta == True
                                if self.__in_target_area(Xsend, Ysend) == False:
                                    self.current_time_text.setText('0')
                                    self.in_target_area_sta = False
                                else:  # self.__in_target_area(Xsend, Ysend) == True
                                    current_point_time = time.clock() - self.in_target_area_start_time
                                    if current_point_time < self.hold_time:  # 停留时间还不够
                                        self.current_time_text.setText("%f" % (current_point_time))
                                    else:  # current_point_time >= self.hold_time  停留时间已经足够
                                        self.current_time_text.setText("0")
                                        __this_point = self.target_point_list.pop()
                                        self.updated.emit('Arrived at point (%d, %d)! ' % (__this_point[0], __this_point[1]))
                                        self.in_target_area_sta = False
                                        if len(self.target_point_list) == 0:  # 所有点都走完了
                                            task_total_time = time.clock() - start_time
                                            self.updated.emit('Task 1 finished! Total time: %f' % (task_total_time))
                                            task_sta = 0
                                            timing_sta = 0
                                            self.current_point_text.setText(" ")
                                            # -----
                                            self.total_time.setText("0")
                                            self.task_start.setEnabled(True)
                                            self.task_sta.setText('0')
                                        else:
                                            __current_point = self.target_point_list[-1]
                                            self.current_point_text.setText("(%d, %d)" % (__current_point[0], __current_point[1]))
                        
                        elif task_sta == 2:
                            # ----- task2
                            if self.in_target_area_sta == False and self.in_ban_area_sta == False:
                                if self.__in_target_area(Xsend, Ysend) == True and self.__in_ban_area(Xsend, Ysend) == False:
                                    self.in_target_area_sta = True
                                    self.in_target_area_start_time = time.clock()
                                elif self.__in_target_area(Xsend, Ysend) == False and self.__in_ban_area(Xsend, Ysend) == True:
                                    self.updated.emit('! Get into forbidden area!!! current coordinate: (%d, %d)' % (Xsend, Ysend))
                                    self.in_ban_area_sta = True
                                    # ----- 下面这些行是 task2 改规则之后加入的
                                    if False == self.task2_into_forbidden_area:
                                        self.task2_into_forbidden_area = True
                                        start_time -= 10
                                        self.updated.emit('Penalize 10s !')
                                    # ----- 上面这些行是 task2 改规则之后加入的
                                    self.in_ban_area_start_time = time.clock()
                                # else:  # __in_target_area 和 __in_ban_area 都返回False
                                    # 什么都不做
                            elif self.in_target_area_sta == True and self.in_ban_area_sta == False:
                                if self.__in_target_area(Xsend, Ysend) == True and self.__in_ban_area(Xsend, Ysend) == False:
                                    current_point_time = time.clock() - self.in_target_area_start_time
                                    if current_point_time < self.hold_time:  # 停留时间还不够
                                        self.current_time_text.setText("%f" % (current_point_time))
                                    else:  # current_point_time >= self.hold_time  停留时间已经足够
                                        self.current_time_text.setText("0")
                                        __this_point = self.target_point_list.pop()
                                        self.updated.emit('Arrived at point (%d, %d)! ' % (__this_point[0], __this_point[1]))
                                        self.in_target_area_sta = False
                                        if len(self.target_point_list) == 0:  # 所有点都走完了
                                            task_total_time = time.clock() - start_time
                                            self.updated.emit('Task 2 finished! Total time: %f' % (task_total_time))
                                            task_sta = 0
                                            timing_sta = 0
                                            self.current_point_text.setText(" ")
                                            # -----
                                            self.total_time.setText("0")
                                            self.task_start.setEnabled(True)
                                            self.task_sta.setText('0')
                                        else:
                                            __current_point = self.target_point_list[-1]
                                            self.current_point_text.setText("(%d, %d)" % (__current_point[0], __current_point[1]))
                                elif self.__in_target_area(Xsend, Ysend) == False and self.__in_ban_area(Xsend, Ysend) == False:
                                    self.current_time_text.setText('0')
                                    self.in_target_area_sta = False
                                else:  # self.__in_target_area(Xsend, Ysend) == False and self.__in_ban_area(Xsend, Ysend) == True:
                                    self.updated.emit('! Get into forbidden area!!! current coordinate: (%d, %d)' % (Xsend, Ysend))
                                    self.in_ban_area_sta = True
                                    self.in_target_area_sta = False
                                    self.in_ban_area_start_time = time.clock()
                            else:  # self.in_target_area_sta == False and self.in_ban_area_sta == True:
                                if self.__in_ban_area(Xsend, Ysend) == False:
                                    ban_total_time = time.clock() - self.in_ban_area_start_time
                                    self.in_ban_area_sta = False
                                    self.updated.emit('already moved out of forbidden area! total time spent in forbidden area: %f' % ban_total_time)
                                    start_time -= 10 * ban_total_time  # ----- 【！！！！！】 在这里该罚时的比例
                                # else:  # self.__in_target_area(Xsend, Ysend) == False and self.__in_ban_area(Xsend, Ysend) == True:
                                    # 什么都不做
                
                        elif task_sta == 3:        
                            # ----- task3
                            # if self.__in_target_area(Xsend, Ysend) == False and self.__in_wrong_target_area(Xsend, Ysend) == False:
                                # 什么也不做
                            if self.__in_target_area(Xsend, Ysend) == True:
                                __arrived_point = self.target_point_list.pop()
                                self.updated.emit('Arrived at point: (%d, %d)' % (__arrived_point[0], __arrived_point[1]))
                                if len(self.target_point_list) == 0:  # 所有点都走完了
                                    task_total_time = time.clock() - start_time
                                    self.updated.emit('Task 3 finished! Total time: %f' % (task_total_time))
                                    task_sta = 0
                                    timing_sta = 0
                                    self.current_point_text.setText(" ")
                                    # -----
                                    self.total_time.setText("0")
                                    self.task_start.setEnabled(True)
                                    self.task_sta.setText('0')
                                else:
                                    __current_point = self.target_point_list[-1]
                                    self.current_point_text.setText("(%d, %d)" % (__current_point[0], __current_point[1]))
                            else:
                                __wrong_point = self.__in_wrong_target_area(Xsend, Ysend)
                                if __wrong_point != False:
                                    while self.target_point_list[-1] != __wrong_point:
                                        __missed_point = self.target_point_list.pop()
                                        self.updated.emit('Missed point (%d, %d) !!!   10s penalized !' % (__missed_point[0], __missed_point[1]))
                                        start_time -= 10  # ----- penalize 10s for missed point !  要改task3的惩罚时间到这里来改！
                                    # ----- 现在“栈顶”是这次走到的点，可以用与上面self.__in_target_area(Xsend, Ysend) == True时相同的方法处理
                                    __arrived_point = self.target_point_list.pop()
                                    self.updated.emit('Arrived at point: (%d, %d)' % (__arrived_point[0], __arrived_point[1]))
                                    if len(self.target_point_list) == 0:  # 所有点都走完了
                                        task_total_time = time.clock() - start_time
                                        self.updated.emit('Task 3 finished! Total time: %f' % (task_total_time))
                                        task_sta = 0
                                        timing_sta = 0
                                        self.current_point_text.setText(" ")
                                        # -----
                                        self.total_time.setText("0")
                                        self.task_start.setEnabled(True)
                                        self.task_sta.setText('0')
                                    else:
                                        __current_point = self.target_point_list[-1]
                                        self.current_point_text.setText("(%d, %d)" % (__current_point[0], __current_point[1]))
                                    
                                    
                """                    
                self.total_task = self.point_count[task_sta]
                if task_sta == 0:
                    self.current_point = 0
                    self.in_area_sta = 0
                    self.stop_timing_sta = 0
                    self.total_time_list = []
                if task_sta != 0:
                    if timing_sta == 0:
                        if(task_sta == 1):
                            timing_sta = 1
                            start_time = time.clock()
                            self.current_point_text.setText('(100,100)')
                        if(task_sta == 2):
                            self.current_point_text.setText('(100,300)')
                            if(judge_in_area(Xsend,Ysend,100,100) == 1):
                                timing_sta = 2
                                self.total_time.setText('Ready...')
                    if(timing_sta == 2):
                        if(judge_in_area(Xsend,Ysend,100,100) != 1):
                            timing_sta = 1
                            start_time = time.clock()
                    if timing_sta == 1:
                        distance = judge_in_area(Xsend,Ysend,self.tasks[task_sta][self.current_point][0],self.tasks[task_sta][self.current_point][1])
                        if self.current_point < self.total_task:
                            if self.in_area_sta == 0:
                                if distance == 1:
                                    self.in_area_time = time.clock()
                                    self.in_area_sta = 1
                                    self.stop_timing_sta = 0
                                    self.in_area_time = time.clock()
                            else:
                                if distance == 0:
                                    self.in_area_sta = 0
                                    self.total_time_list = []
                                    self.current_time_text.setText('0')
                                elif distance == 2:
                                    if self.stop_timing_sta == 0:
                                        self.stop_timing_sta = 1
                                        self.total_time_list.append(float(time.clock())-float(self.in_area_time))
                                elif distance == 1:
                                    if self.stop_timing_sta == 1:
                                        self.stop_timing_sta = 0
                                        self.in_area_time = time.clock()
                                    total = sum(self.total_time_list) + float(time.clock())-float(self.in_area_time)
                                    self.current_time_text.setText(str(total))
                                    #显示
                                    if total > 3:
                                        #显示单点完成
                                        self.updated.emit('Task %d Point %d Done!'%(task_sta,self.current_point+1))
                                        self.current_point += 1
                                        self.in_area_sta = 0
                                        self.total_time_list = []
                                        self.current_time_text.setText('0')
                                        #判断是否完成全部任务
                                        if self.current_point == self.total_task:
                                            self.updated.emit('Task %d Finished! Total time:%.6f\n\n'%(task_sta,float(time.clock())-float(start_time)))
                                            #终止任务
                                            task_sta = 0
                                            self.task_start.setEnabled(True)
                                            timing_sta = 0
                                            self.current_point_text.setText(' ')
                                            self.current_point = 0
                                            self.task_sta.setText('0')
                """
                                            
                # ----- 评分到这里结束了！



                    

                #向串口发数据
                global ser_sta
                if ser_sta == 1:
                    # ----- 没有太懂这里为啥又要set一遍current point的text，注释掉了。
                    #   并且target = self.tasks[task_sta][0] 在 task_sta == 3 时越界了
                    # if task_sta != 0:
                        # target = self.tasks[task_sta][0]
                        # self.current_point_text.setText('(%s,%s)'%(str(self.tasks[task_sta][self.current_point][0]),str(self.tasks[task_sta][self.current_point][1])))
                    s = bytes([250, (Xsend >> 7) + 1, (Xsend & 0x7f) + 1, (Ysend >> 7) + 1, (Ysend & 0x7f) + 1])
                            
                    try:
                        self.ser.write(s)
                        self.X.setText(str(Xsend))
                        self.Y.setText(str(Ysend))
                    except:
                        self.X.setText('ser error!')
                        self.Y.setText('ser error!')
                    

            # 显示图像，其实有ui就不需要它了，但去掉的话ui也显示不出来，不知道为啥，只能这样最小化了
            #cv2.imshow("image", image)
            #cv2.moveWindow("image", 0, 0)
            #cv2.resizeWindow("image", 1, 1)
            #cv2.waitKey(1)
            # 在UI上画图
            if self.ui_sta_text.currentText() == '实时' or (i % 10 == 0):
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
        self.tag_id = get_tag_id()
        self.camera = cv2.VideoCapture(self.cam_int)
        self.cam_label.setText(str(self.cam_int))
        self.tag_id_text.setText(str(self.tag_id))
        self.task_start.clicked.connect(self.taskStart)
        self.task_terminate.clicked.connect(self.taskTerminate)

    #打开串口
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

    #关闭串口
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

    #进入打开相机线程
    def openCamera(self):
        self.cam_th = CamOpenThread(self.cam_open,self.camera,self.arena,self.ser,self.ball_val_L,self.X,self.Y,self.total_time,self.result_text,self.task_start,self.current_point_text,self.current_time_text,self.task_sta,self.tag_id,self.tag_id_text,self.fps_text,self.ui_sta_text)
        self.cam_th.updated.connect(self.updateResult)
        self.cam_th.start()

    def updateResult(self,text):
        self.result_text.append(text)

    #更新串口
    def updateSerlist(self):
        port_list_str = ser_read()
        self.ser_selection.clear()
        if self.port_list_str != None:
            self.ser_selection.addItems(port_list_str)
        else:
            self.ser_selection.addItem("无串口")
        return

    #选择任务
    def taskStart(self):
        self.task_start.setEnabled(False)
        global task_sta
        task = self.task_selection.currentText()
        if task == 'Task0':
            task_sta = 0
        elif task == 'Task1':
            task_sta = 1
        elif task == 'Task2':
            task_sta = 2
            self.result_text.append('Starting point of task 2:(100, 100). Please move to this point.')  # ----- 在这里改起始点的提示信息【！！！！！】
        elif task == "Task3":
            task_sta = 3
            self.result_text.append('Starting point of task 3:(100, 200). Please move to this point.')  # ----- 在这里改起始点的提示信息【！！！！！】
        self.task_sta.setText(str(task_sta))
        # self.result_text.append('Task %d start!'%task_sta)  # ----- 这里注释掉了：把task_start放到上面了
        return
        # ----- task_sta 表示正在运行某个task（是“正在运行”而不是“在选择框里选择了”）
        # ----- 点击“选择任务”按钮之后，才会把 task_sta 变成相应的任务值。点击之前其保持为0


    #终止任务
    def taskTerminate(self):
        self.task_start.setEnabled(True)
        global timing_sta,task_sta
        timing_sta = 0
        task_sta = 0
        self.task_sta.setText('0')
        self.total_time.setText("0")  # ----- 加了一句，终止任务要把total time 清零！
        return
        # ----- 点击“终止任务”之后，会把 task_sta 和 timing_sta 都置零（保证了某个非0的task开始时，timing_sta == 0）
