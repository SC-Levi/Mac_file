import qimage2ndarray
from gui import Ui_MainWindow
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from Depth_Cam import RS_Cam
from yolov5_cam import YoLov5TRT
import time
class Show(Ui_MainWindow):
    def __del__(self):
        try:
            self.pipeline.release()
        except:
            return
    def __init__(self,img):
        self.img = img
        super().__init__()
        self.setupUi(self)
        self.CallBackFunctions()
        EP = 'build/yolov5s.engine'
        #self.yolo = YoLov5TRT(EP)

        # self.Timer.timeout.connect(self.detection)
        # self.Timer = QTimer()
        # self.Timer.timeout.connect(self.detection)


    def detection(self): 
        #sleep(1)
        #for i in range(30):
        #img, _ = self.yolo.infer(self.img)
        video = cv2.VideoCapture("CER.mp4")
        while(1):
            time.sleep(0.5)
            ret, img = video.read()
            #img, _ = self.yolo.infer(self.img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #ret, video = cv2.resize(self.video,(640,480),interpolation = cv2.INTER_AREA)
            frame = img
            self.Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format_RGB888) 
            self.update()
            #qimg = qimage2ndarray.array2qimage(img)
            
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))
            self.label.show()

    def CallBackFunctions(self):
        self.bt.clicked.connect(self.detection)









if __name__ == '__main__':
    Cam = RS_Cam()
    #while(1):
    raw_img = Cam(0)
    #print(type(raw_img))
        #cv2.imshow("a",raw_img)
       # key = cv2.waitKey(1)
    app = QApplication(sys.argv)
    W = Show(raw_img)
    W.show()
    sys.exit(app.exec())
