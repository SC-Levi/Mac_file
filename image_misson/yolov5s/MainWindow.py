# -*- coding: utf-8 -*-
 
# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
# QtCore模块涵盖了包的核心的非GUI功能，此模块被用于处理程序中涉及到的 time、文件、目录、数据类型、文本流、链接、mime、线程或进程等对象。
# QtGui模块涵盖多种基本图形功能的类; 包括但不限于：窗口集、事件处理、2D图形、基本的图像和界面 和字体文本。
# QtWidgets模块包含了一整套UI元素组件，用于建立符合系统风格的classic界面，非常方便，可以在安装时选择是否使用此功能。
 
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # 主窗口 大小设置为1280*720
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080,640)

        #QWidget（小部件）是pyqt5所有用户界面对象的基类。他为QWidget提供默认构造函数。默认构造函数没有父类。
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # Qlabel创建一个静态文本
        # setGeometry() 方法用于自行设置PyQt5窗口的几何形状。
        # QtCore.QRect创建一个矩形，左上角坐标20,20，长和宽为101和31
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 151, 31))
        self.label.setObjectName("label")
        self.label.setFont(QFont('Microsoft YaHei',15))
        # QtCore.QRect创建一个矩形，左上角坐标20,20，长和宽为101和31
        self.label_state = QtWidgets.QLabel(self.centralwidget)
        self.label_state.setGeometry(QtCore.QRect(720, 450, 101, 31))
        self.label_state.setObjectName("label")
        self.label_state.setFont(QFont('Microsoft YaHei',20))
        # QtWidgets.QPushButton创建一个按钮
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(720, 500, 93, 48))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setFont(QFont('Microsoft YaHei',20))
        # 输出日志
        self.logs = QTextBrowser(self.centralwidget)
        self.logs.setGeometry(QtCore.QRect(700, 20, 300, 400))
        # 基层部件，20,50是起始位置，后续封装都是相对这个坐标而言的
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 50, 640, 480))
        self.widget.setObjectName("widget")
        # Qlabel创建一个静态文本，绝对坐标其实是20,50
        self.img_label = QtWidgets.QLabel(self.widget)
        self.img_label.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.img_label.setObjectName("img_label")
        self.img_label.setFont(QFont('Microsoft YaHei',20))
        MainWindow.setCentralWidget(self.centralwidget)
        # QtWidgets.QMenuBar创建一个菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 713, 26))
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
        self.label.setText(_translate("MainWindow", "videostream"))
        self.label_state.setText(_translate("MainWindow", "free"))
        self.pushButton.setText(_translate("MainWindow", "open"))
        self.img_label.setText(_translate("MainWindow", "                                  视频播放区域"))