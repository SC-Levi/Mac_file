import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)



    def PrepParameters(self):
        self.dictionary = {}        #统计 过滤误识别
        self.statistics = 0         #
        self.round = 1              #从第一轮开始
        self.open_flag = False
        self.label_2.setText('空闲')
        self.sleep = 5

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(911, 604)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        self.logs = QTextBrowser(self.centralwidget)
        self.logs.setGeometry(QtCore.QRect(700, 20, 300, 400))

        # QtCore.QRect创建一个矩形，左上角坐标20,20，长和宽为101和31
        self.label_state = QtWidgets.QLabel(self.centralwidget)
        self.label_state.setGeometry(QtCore.QRect(720, 480, 101, 31))
        self.label_state.setObjectName("label")
        self.label_state.setFont(QFont('Microsoft YaHei',20))

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.label.setObjectName("label")

        self.bt = QtWidgets.QPushButton(self.centralwidget)
        self.bt.setGeometry(QtCore.QRect(170, 490, 181, 51))
        self.bt.setObjectName("bt")
        
        self.text = QtWidgets.QTextBrowser(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(650, 10, 256, 461))
        self.text.setObjectName("text")
        self.rb1 = QtWidgets.QRadioButton(self.centralwidget)
        self.rb1.setGeometry(QtCore.QRect(410, 510, 115, 19))
        self.rb1.setObjectName("rb1")
        self.rb2 = QtWidgets.QRadioButton(self.centralwidget)
        self.rb2.setGeometry(QtCore.QRect(530, 510, 115, 19))
        self.rb2.setObjectName("rb2")
        self.rb3 = QtWidgets.QRadioButton(self.centralwidget)
        self.rb3.setGeometry(QtCore.QRect(650, 510, 115, 19))
        self.rb3.setObjectName("rb3")
        self.rb4 = QtWidgets.QRadioButton(self.centralwidget)
        self.rb4.setGeometry(QtCore.QRect(770, 510, 115, 19))
        self.rb4.setObjectName("rb4")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 500, 91, 41))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 911, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:48pt; vertical-align:sub;\">图片显示区域</span></p></body></html>"))
        self.bt.setText(_translate("MainWindow", "开始检测"))
        self.rb1.setText(_translate("MainWindow", "第一轮"))
        self.rb2.setText(_translate("MainWindow", "第二轮"))
        self.rb3.setText(_translate("MainWindow", "第三轮"))
        self.rb4.setText(_translate("MainWindow", "第四轮"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">空闲</span></p></body></html>"))
if __name__ == '__main__':
	app = QApplication(sys.argv)
	W = Ui_MainWindow()
	img = cv2.imread("dog.jpg")
	W.get_image(img)
	W.show()
	sys.exit(app.exec())











