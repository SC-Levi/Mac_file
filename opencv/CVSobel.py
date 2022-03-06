'''
图像梯度及边缘检测 
Sobel算子 运算图像灰度函数的梯度近似值 消除噪声
平面卷积 获取横向和纵向的灰度差分值 得到单个方向的偏导数
dst = cv2.Sobel(src, ddepth, dx, dy[, ksize[, scale[, delta[, borderType]]]])
dst 目标函数图像
ddepth 输出图像的深度
ksize Sobel核 的大小
scale 缩放因子
borderType 边界样式
'''


import cv2 as cv
image = cv.imread("test.jpg")
# 设置参数dx=1，dy=0， 得到水平方向边缘信息
Sobelx = cv.Sobel(image, cv.CV_64F, 1, 0)
# 计算结果取绝对值
Sobelx = cv.convertScaleAbs(Sobelx)
# 设置参数dx=0，dy=1， 得到垂直方向边缘信息
Sobely = cv.Sobel(image, cv.CV_64F, 0, 1)
# 计算结果取绝对值
Sobely = cv.convertScaleAbs(Sobely)
# 设置参数dx=1， dy=1， 得到水平和垂直方向的边缘信息
Sobelxy = cv.Sobel(image, cv.CV_64F, 1, 1)
# 取绝对值
Sobelxy =cv.convertScaleAbs(Sobelxy)
# 加权函数 水平和垂直方向加权计算
Sobel_W = cv.addWeighted(Sobelx, 0.5, Sobely, 0.5, 0)
# 显示图像
cv.imshow('img', image)
# cv.imshow('imgx', Sobelx)
# cv.imshow('imgy', Sobely)
# cv.imshow('imgxy', Sobelxy)
cv.imshow('imgWeight', Sobel_W)
cv.imwrite('Sobel_W.jpeg', Sobel_W)
cv.waitKey()
cv.destroyAllWindows()





















