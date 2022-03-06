import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('Resources/9.BMP')

plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()

# cv2.namedWindow('findCorners', 0)
# cv2.resizeWindow('findCorners', 1000, 460)
# img = cv2.imread('Resources/ImgPnp.jpg')
# cv2.imshow('findCorners',img)
# objectPoints特征点世界坐标
# 以特征点所在平面为世界坐标XY平面，并在该平面中确定世界坐标的原点，以我设计的二维码为例，
# 我设定二维码的中心为世界坐标的原点，并按照顺时针方向逐个输入四个角点的世界坐标。
objPoints = np.array([[-207.5, -182.5, 20],[207.5, -182.5, 20],[207.5, 182.5, 0],[-207.5, 182.5, 0]])
# imagePoints特征点在摄像头下的像素点坐标
# 在这儿将获得四个特征点对应2D的像素点坐标，而这个过程你可以人为的从图像中逐个点获得
imgPoints = np.array([[272.453,248.798],[327.341,247.492],[327.341,285.391],[273.76,289.311]])
cameraMatrix = np.array([[1.03110741e+03,0.00000000e+00,3.09627059e+02],
                         [0.00000000e+00,1.02988140e+03,2.42642582e+02],
                         [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
distCoeffs = None
# distCoeffs = np.array([-0.1408541617822344, 0.8248872589005009, 9.542278053040669e-05, 0.002492233684456267, -3.277725618255168])
# 1、R的第i行 表示摄像机坐标系中的第i个坐标轴方向的单位向量在世界坐标系里的坐标
# 2、R的第i列 表示世界坐标系中的第i个坐标轴方向的单位向量在摄像机坐标系里的坐标
# 3、t 表示世界坐标系的原点在摄像机坐标系的坐标
# 4、-R的转置 * t 表示摄像机坐标系的原点在世界坐标系的坐标
success, rotation_vector, translation_vector = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
R = cv2.Rodrigues(rotation_vector)[0]  # 由于solvePnP返回的是旋转向量，故用罗德里格斯变换变成旋转矩阵
t = translation_vector
position = -np.matrix(R).T * np.matrix(t) # 解算位置 ：-R的转置 * t 表示摄像机坐标系的原点在世界坐标系的坐标
print(position)
print(translation_vector)