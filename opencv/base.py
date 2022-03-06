import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
imagerandem = np.random.randint(0, 256, size=[256, 256, 3], dtype = np.uint8)


image = cv.imread('3.jpeg')
cv.namedWindow('image')
cv.imshow('image',image)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite('base.jpeg',image)

b, g, r = cv.split(image)
cv.imshow('b', b)
cv.waitKey()
cv.destroyAllWindows()
cv.imshow('g', g)
cv.waitKey()
cv.destroyAllWindows()
cv.imshow('r', r)
cv.waitKey()
cv.destroyAllWindows()
# merge()
image_bgr = cv.merge([b, g, r])
cv.imshow('image_bgr',image_bgr)
cv.waitKey()
cv.destroyAllWindows()
#image.shape image.size image.dtype

# RGB色彩空间 
#rgb bgr 
imgRGB = cv. cvtColor(image,  cv.COLOR_BGR2RGB)
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(imgRGB,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
imgRGB =  cv.polylines(imgRGB, [pts], True, (0, 255, 255))
cv.imshow('image_bgr', imgRGB)
cv.waitKey()
cv.destroyAllWindows()

# GRAY色彩空间
imgG = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(imgG, [pts], True, (0, 255, 255))
cv.imshow('gray', imgG)
cv.waitKey()
cv.destroyAllWindows()

# YCRCb 色彩空间
imgY = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
cv.imshow('imgY', imgY)
cv.waitKey()
cv.destroyAllWindows()

# HSV 色彩空间

imgH = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow('imgH', imgH)
cv.waitKey()
cv.destroyAllWindows()

# 仿射变换 平移 旋转 缩放
# 平移
h, w = image.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, -100]])
imgM = cv.warpAffine(image, M, (w ,h))
cv.imshow('Move', imgM)
cv.waitKey()
cv.destroyAllWindows()
# 缩放旋转
M2 = np.float32([[0.5, 0, 200], [0, 0.5, 200]])
imgM2 = cv.warpAffine(image, M2, (w, h))
cv.imshow('small', imgM2)
cv.waitKey()
cv.destroyAllWindows()
# 旋转
M3 =cv.getRotationMatrix2D((w/2, h/2), 180, 1) # 以图像的中心旋转 180度 缩放为1
imgMR = cv.warpAffine(image, M3, (w, h))
cv.imshow('Rotation', imgMR)
cv.waitKey()
cv.destroyAllWindows()


# 重映射 将一个图像的像素点放置在另一个图像的指定位置 叫做重映射 
# 映射函数： 查找新图像在原始图像内的位置。 cv2.remap()
# dst = cv2.remap(src, map1, map2, interpolation[, borderMode[,borderValue]])
# dst : target picture. interpolation : insert way. borderMode map1, map2 0
rand = np.random.randint(0, 256, size = [6, 6], dtype = np.uint8)
w, h = image.shape[:2]
x = np.zeros((w, h), np.float32)
y = np.zeros((w, h), np.float32)
for i in range(w):
	for j in range(h):
		x.itemset((i, j), j)
		y.itemset((i, j), i)
rst = cv.remap(image, x, y, cv.INTER_LINEAR) # 数组的复制  复制成功
cv.imshow('rst', rst)
cv.destroyAllWindows()

# 绕x轴旋转
'''
# 图像形态处理
# 腐蚀运算
k = np.ones((3, 3), np.uint8)
img_ed = cv.erode(image, k, iterations = 3)
cv.imshow('erode',img_ed)
cv.waitKey()
cv.destroyAllWindows()
# 膨胀
img1 = cv.dilate(image, k, iterations = 1)
img2 = cv.dilate(image, k, iterations = 2)
img3 = cv.
'''











