'''
Scharr算子 相同的计算速度 更高的计算精度
dst = cv2.Scharr (src, ddpeth, dx, dy[, scale[, delta[, borderType]]])
'''
import cv2 as cv
image = cv.imread('5.jpeg')
# 获取水平方向的边缘信息
scharrx = cv.Scharr(image, cv.CV_64F, 1, 0)
# 取绝对值
scharrx = cv.convertScaleAbs(scharrx)
# 获取垂直方向的边缘信息
scharry = cv.Scharr(image, cv.CV_64F, 0, 1)
# 取绝对值
scharry = cv.convertScaleAbs(scharry)
# 设置ksize为-1， 计算水平和垂直方向的边缘信息
scharr_Sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, -1)
scharr_Sobel_x = cv.convertScaleAbs(scharr_Sobel_x)
scharr_Sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, -1)
scharr_Sobel_y = cv.convertScaleAbs(scharr_Sobel_y)
# 加权
scharry_W = cv.addWeighted(scharr_Sobel_x, 0.5,scharr_Sobel_y, 0.5, 0)
# 显示图像
cv.imshow('img', image)
cv.imshow('imgWeight', scharry_W)
cv.imwrite('scharry_W.jpeg', scharry_W)
cv.waitKey()
cv.destroyAllWindows()





