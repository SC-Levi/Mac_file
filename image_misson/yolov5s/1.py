#coding=utf-8

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import cv2
import time
class Main_Window(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 主窗口显示：400 200是窗口左上角在显示器上的位置，1000,550是长和宽
        self.setGeometry(400,200,1000,550)
        self.setWindowTitle('2333')
        # 日志显示
        self.logs = QTextBrowser(self)
        self.logs.setGeometry(QtCore.QRect(700, 20, 200, 400))
        # 图像显示
        self.img_label = QLabel(self)
        self.img_label.setGeometry(QtCore.QRect(10,10, 640, 480))
        self.img_label.setObjectName("img_label")
        self.img_label.setText("                                  图像显示区域")
        # 开始按钮
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(570, 500, 90, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("开始")
        self.pushButton.clicked.connect(self.network_process)

        self.img = 0
        self.text = ""
        # self.show()

    def get_image(self,image):
        self.img = image
    
    def showtext(self,text):
        self.logs.append(text)

    def network_process(self):
        # self.showtext("START")
        print("network processing")
        '''

        神经网络前传推理代码放这儿

        '''
        text = "Goal_ID=people;Num=5"
        self.showtext(text)
        self.showimg()

    def showimg(self):
        height, width, bytesPerComponent = self.img.shape
        bytesPerLine = 3 * width
        cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
        QImg = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.img_label.setPixmap(pixmap)
        self.img_label.setCursor(Qt.CrossCursor)
        self.update()
        self.showtext("END")

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    W = Main_Window()
    img = cv2.imread('dog.jpg')
    W.get_image(img)
    W.show()
    sys.exit(app.exec_())    
    