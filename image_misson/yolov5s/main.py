import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import untitled
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import time
from gui import Ui_MainWindow
from Depth_Cam import RS_Cam
from yolov5_cam import YoLov5TRT 


class MainCode(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #self.PrepParameters()
        self.bt.clicked.connect(self.on_Camera)
        self.open_flag=False
        self.video_stream=cv2.VideoCapture('test.mp4')
        self.Cam = RS_Cam() 
        engine_path = 'build/yolov5s.engine'
        self.yolo = YoLov5TRT(engine_path)
 #标签显示类别 result working button 显示的位置 改变result的逻辑

    def showtext(self,text):
        self.text.append(text)
    
    def on_Camera(self):
        self.label_state.setText("working")
        self.open_flag = True
        if self.rb1.isChecked()==True:#用于确定是第几轮
            self.round = 1
            self.judge = 0
        if self.rb2.isChecked()==True:
            self.round = 2
            self.judge = 1
        if self.rb3.isChecked()==True:
            self.round = 3
            self.judge = 0
        if self.rb4.isChecked()==True:
            self.round = 4
            self.judge = 1

    def PrepParameters(self):
        self.dictionary = {}    #用于统计预测结果出现次数，过滤误识别
        self.round = 1          #默认第一轮
        self.statistics = 0
        self.open_flag = False
        self.label_2.setText('空闲')
        self.sleep = 5          #休眠时间，等待摄像头稳定

    def paintEvent(self, a0: QtGui.QPaintEvent):
        if self.open_flag:
            frame = self.Cam()
            frame, _  = self.yolo.infer(frame)
            frame=cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            self.Qframe=QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[1]*3,QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))
            self.showtext("result")
            self.update()

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())
