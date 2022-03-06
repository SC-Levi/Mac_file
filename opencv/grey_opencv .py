#opencv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_path = '/Users/levi/opencv/test.jpg'
img1 = cv.imread('img_path')
if img1 is None:
    print("img is none")
img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img3 = cv.resize(img2, (600, 600))
img0 = cv.imgread(img3, 0)
ret, th1 = cv.threshold(img0, 100, 255, cv.THRESH_BINARY)
fig = plt.figure(figsize = (50, 50))
thresh = cv.threshold(img1, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU) # 寻找二值图像的轮廓contours, 
hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
print(len(contours))
plt.imshow(th1, cmap('gray'))