import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import MainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import time
import pyrealsense2 as rs
import numpy as np
class MainCode(QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        MainWindow.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.on_video)
        self.open_flag=False

        self.painter = QPainter(self)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def showtext(self,text):
        self.logs.append(text)
    
    def on_video(self):
        self.label_state.setText("working")
        if self.open_flag:
            self.pushButton.setText('open')
        else:
            self.pushButton.setText('close')
        self.open_flag = bool(1-self.open_flag)

    def paintEvent(self, a0: QtGui.QPaintEvent):
        if self.open_flag:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Stack both images horizontally
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images= np.hstack((color_image, depth_colormap))

            frame=cv2.cvtColor(images,cv2.COLOR_BGR2RGB)
            self.Qframe=QImage(frame.data,frame.shape[1],frame.shape[0],frame.shape[1]*3,QImage.Format_RGB888)
            self.img_label.setPixmap(QPixmap.fromImage(self.Qframe))
            self.showtext("result")
            self.update()

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())
