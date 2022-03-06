import cv2
import numpy as np
print('import success')

"""
img = cv2.imread('Resources/1.jpg')
kernel = np.ones((5,5),np.uint8)  # 核

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),0) #模糊
imgCanny = cv2.Canny(img,150,200)  # 边缘
imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)  #膨胀
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)

cv2.imshow('Gray Image',imgGray)
cv2.imshow('Blur Image',imgBlur)
cv2.imshow('Canny Image',imgCanny)
cv2.imshow('Dia Image',imgDialation)
cv2.imshow('Rro Image',imgEroded)
cv2.waitKey(0)
"""

kernel = np.ones((5,5),np.uint8)  # 核
#cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
#cap.set(10,100)
cap = cv2.VideoCapture('Resources/能量机关第一视角视频1.avi')

while True:
    succsee , img = cap.read()
    imgCanny = cv2.Canny(img, 150, 200)  # 边缘
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
    cv2.imshow('Video',imgDialation)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

