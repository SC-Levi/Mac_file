import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog

import time
from untitled import Ui_MainWindow

class Det_Gui(QMainWindow, Ui_MainWindow):
	"""docstring for ClassName"""
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.PrepParameters() #参数初始化
		self.Timer = QTimer()
		self.Timer.timeout.connect(self.detection)

	def initUI(self, MainWindow):
		# 设置主窗口： 显示位置 大小长宽高
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(911, 604)
		self.setGeometry(500, 200, 1000, 600)
		self.setWindowTitle("Window")
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		# 设置日志
		self.logs = QTextBrowser(self)
		self.logs.setGeometry(QtCore.QRect(700, 30, 200, 400))
		# 显示图像
		self.img_label = QLabel(self)
		self.img_label.setGeometry(QtCore.QRect(10, 10, 640,480))
		self.img_label.setText("					图像显示区域")
		# 设置标签
		self.label = QtWidgets.QLabel(self.centralwidget)
		self.label.setGeometry(QtCore.QRect(0, 0, 640, 480))
		self.label.setObjectName("label")
		self.label_2 = QtWidgets.QLabel(self.centralwidget)
		self.label_2.setGeometry(QtCore.QRect(30, 500, 91, 41))
		self.label_2.setObjectName("label_2")
		# 设置轮数分类标签
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
		# 设置按钮
		self.pushButton = QtWidgets.QPushButton(self)
		self.pushButton.setGeometry(QtCore.QRect(570, 500, 90, 70))
		self.pushButton.setText("开始")
		self.pushButton.clicked.connect(self.network_process)


		self.img = 0
		self.text = ""



	def PrepParameters(self):
		self.dictionary = {}		#统计 过滤误识别
		self.statistics = 0 		#
		self.round = 1 				#从第一轮开始
		self.open_flag = False
		self.label_2.setText('空闲')
		self.sleep = 5

	def detection(self):
		if self.open_flag == True:
			self.label_2.setText('识别中')
			frames = self.pipeline.wait_for_frame()
			color_frame = frame.get_color_frame()
			color_image = np.asanyaarray(color_frame.get_data())
			if self.statistic == 30:
				self.grayA = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
			if (self.statistic-30)%15==0 and self.statistic>59:
				self.garyB = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
				(score, diff) = compare_ssim(self.grayA, grayB, full = True)
				print(score)
				# 设置阈值
				if score>0.89:
					if self.round==2 or self.round==4:
						if self.judge == 1:
							self.savetxt()
							self.text.append("OK")
							self.label_2.setText("待转动")
							QApplication.processEvents()
							time.sleep(self.sleep)
						if self.judge == 2:
							self.SaveTxt()
							self.text.append('OK')
							self.label_2.setText('待转动')
							QApplication.processEvents()
							time.sleep(self.sleep)
						if self.judge ==3:
							self.SaveTxt()
							self.label_2.setText('结束')
							self.Timer.stop()
						self.judge += 1
						 #time.sleep(5)
					else:
						self.SaveTxt()
						self.text.append('OK')
						self.label_2.setText('结束')
						self.judge == 0
						self.Timer.stop()
			#读取检测代码




			#检测结束 保存结果 结束
			if self.round ==1 or self.round == 3:
				if time.process_time()-self.timelb >= 8:
					self.SaveTxt()
					self.label_2.setText('结束')
					self.Timer.stop()
			if self.round ==2 or self.round == 4:
				if time.process_time()-self.timelb >= 23.5-2*self.sleep and self.judge == 3:
					self.SaveTxt()
					self.label_2.setText('结束')
					self.Timer.stop()
				elif time.process_time()-self.timelb >= self.sleep+4 and self.judge == 2:
					self.SaveTxt()
					self.text.append('OK')
					self.label_2.setText('待转动')
					QApplication.processEvents()
					self.judge += 1
				elif time.process_time()-self.timelb >= 2 and self.judge == 1:
					self.SaveTxt()
					self.text.append('OK')
					self.label_2.setText('待转动')
					QApplication.processEvents()
					self.judge += 1





		
	def get_image(self, image):
		self.img = image

	def SaveTxt(self):
		self.text.clear()
		finallist = []
		modify = []
		for key, value in self.dictionary.items():
			if float(value)/float(self.statistics) > 0.4:
				finallist.append(key)
				modify.append([key[8:13], int(key[-1])])
		for i in range(len(modify)-1):
			for j in range(i+1, len(modify)):
				if modify[i][0] == modify[j][0]:
					if (modify[i][1]) > modify[j][1] and modify[i][1]<=3:
						finallist[j] = 0
					else:
						finallist[i] = 0


		if self.judge == 3 or self.judge ==2:
			with open("DUT-Det" + str(self.round) + ".txt", "r") as f:
				values = []
				ids = []
				lines = []
				for line in f.readlines():
					if len(line) > 15:
						lines.append(line[8:13])
						values.append(int(line[18:19]))
		with open("DUT-Det"+str(self.round)+'.txt','w+') as f:
			f.write('START\r\n')
			if self.judge == 3 or self.judge == 2:
				for id in finallist:
					if id != 0:
						for i in range(len(ids)):
							if id[8:13] == ids[i]:
								va = int(id[-1]) +values[i]
								lines[i] = 0
								nid = 'Goal_ID=%s;Num=%g' % (ids[i],va)
								f.write('%s\n'%(nid))
								self.text.append(str(nid))
								break
						else:
							f.write('%s\n' % (id))
							self.text.append(str(id))

				for line in lines:
					if line != 0 and len(line)>15:
						f.write('%s' % (line))
						self.text.append(str(line[:19]))
			else:
				for id in finallist:
					if id != 0:
						f.write('%s\n' % (id))
						self.text.append(str(id))
			if self.judge == 0 or self.judge == 3:
				f.write('END')
			self.statistics = 0
			self.dictionary = {}




	def showimg(self):
		height, width, bytesPerCompoment = self.img.shape
		bytesPerCompoment = 3 * width
		cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
		Qimg = QImage(self.img.data, width, height, bytesPerCompoment,QImage.Format_RGB888)
		pixmap = QPixmap.fromImage(QImg)
		pixmap = QPixmap.setPixmap(pixmap)
		self.img_label.setPixmap(pixmap)
		self.img_labelsetCursor(Qt.CrossCursor)
		self.update()
		self.showtext("END")


if __name__ == "__main__":
	app = QApplication(sys.argv)
	W = Det_Gui()
	img = cv2.imread()
	W.get_image(img)
	W.show()
	sys.exit(app.exec())





















