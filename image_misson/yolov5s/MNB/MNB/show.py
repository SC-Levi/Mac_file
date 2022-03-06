from gui import Ui_MainWindow
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
class Show(Ui_MainWindow):
	"""docstring for Show"""
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.Timer = QTimer()
		self.Timer.timeout.connect(self.detection)
		# self.Timer.timeout.connect(self.detection)


	def detection():

	 	









if __name__ == '__main__':
	app = QApplication(sys.argv)
	W = Show()
	img = cv2.imread("dog.jpg")
	W.get_image(img)
	W.show()
	sys.exit(app.exec())
