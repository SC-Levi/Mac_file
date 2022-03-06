from numpy import *
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# 一切的前提是利用pnp反解出的摄像头的位置是无限接近（0，0，z）
# 一切的前提是利用pnp反解出的摄像头的位置是无限接近（0，0，z）
# 一切的前提是利用pnp反解出的摄像头的位置是无限接近（0，0，z）
"""
img = Image.open('Resources/ImgPnp.jpg')

plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()
"""

distance = 7639 # 机关与摄像头的距离
energy_center = (0,0) # 能量机关对于图片上的圆心
Ini_v = 18   # 初速度18m/s
g = 9.8
pre_time = 0.4 # 预测子弹飞行时间 经验所得
energy_r = 765 # 能量机关的半径
energy_pal = pi/3 # 能量机关的角速度为60°每秒

starting_point = array([-389.7314,-1454.8757,-7639.7267])
target_point = array([765.0,0.0,0.0])
# objectPoints特征点世界坐标
# 以特征点所在平面为世界坐标XY平面，并在该平面中确定世界坐标的原点，以我设计的二维码为例，
# 我设定二维码的中心为世界坐标的原点，并按照顺时针方向逐个输入四个角点的世界坐标。
obj_Point = np.array([[-115.0, -64.0, 0],[115.0, -70.0, 0],[115.0, 64.0, 0],[-115.0, 64.0, 0]])
# imagePoints特征点在摄像头下的像素点坐标
# 在这儿将获得四个特征点对应2D的像素点坐标，而这个过程你可以人为的从图像中逐个点获得
img_Point = np.array([[711.87,487.565],[244.12,492.941],[246.81,1022.51],[714.55,1027.89]])
cameraMatrix = np.array([[1.03110741e+03,0.00000000e+00,3.09627059e+02],
                         [0.00000000e+00,1.02988140e+03,2.22642582e+02],
                         [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
distCoeffs = np.array([-0.1408541617822344, 0.8248872589005009, 9.542278053040669e-05, 0.002492233684456267, -3.277725618255168])

img_Center = array([(img_Point[0][0]+img_Point[1][0])/2,(img_Point[0][1]+img_Point[3][1])/2]) # 对于图片的中心位置

def Pnp_solve(obj_Point,img_Point,cameraMatrix,distCoeffs):
    success, rotation_vector, translation_vector = cv2.solvePnP(obj_Point,img_Point,cameraMatrix,distCoeffs)
    R = cv2.Rodrigues(rotation_vector)[0]  # 由于solvePnP返回的是旋转向量，故用罗德里格斯变换变成旋转矩阵
    t = translation_vector
    # 1、R的第i行 表示摄像机坐标系中的第i个坐标轴方向的单位向量在世界坐标系里的坐标
    # 2、R的第i列 表示世界坐标系中的第i个坐标轴方向的单位向量在摄像机坐标系里的坐标
    # 3、t 表示世界坐标系的原点在摄像机坐标系的坐标
    # 4、-R的转置 * t 表示摄像机坐标系的原点在世界坐标系的坐标
    position = -np.matrix(R).T * np.matrix(t) # 解算位置 ：-R的转置 * t 表示摄像机坐标系的原点在世界坐标系的坐标
    return squeeze(array(position).T)  # 得到摄像机坐标系的原点在世界坐标系的坐标


Img_Point = array([277.0,800.0])

# 击打位置调整, 目标在图片坐标系下的三维坐标Img_Point：（x，y）
def Target_PreSolver(img_Center,Img_Point):
    act_r = linalg.norm(img_Center - Img_Point)
    scale = act_r/energy_r # 比例尺
    _ = Img_Point-img_Center
    if _[0]==0 or _[1]==0:
        _[0] += 0.001
        _[1] += 0.001
    elif _[0]>0 and _[1]>0: # 找象限
        rad = 0 + tanh(_[1]/_[0]) + energy_pal*pre_time
    elif _[0]<0 and _[1]>0:
        rad = pi/2 + tanh(-_[0]/_[1]) + energy_pal*pre_time
    elif _[0]<0 and _[1]<0:
        rad = pi + tanh(_[1]/_[0]) + energy_pal*pre_time
    elif _[0]>0 and _[1]<0:
        rad = 3*pi/2 + tanh(_[0]/-_[1]) + energy_pal*pre_time

    if rad <= pi/2:  # 预测落点
        Target_Point = img_Center + (act_r*cos(rad),act_r*sin(rad))
    elif rad <= pi:
        Target_Point = img_Center + (-act_r * sin(rad-pi/2), act_r * cos(rad-pi/2))
    elif rad <= 3*pi/2 :
        Target_Point = img_Center + (-act_r * cos(rad - pi), -act_r * sin(rad - pi))
    else:
        Target_Point = img_Center + (act_r * sin(rad - 3*pi/2), -act_r * cos(rad - 3*pi/2))

    return (Target_Point-img_Center)/scale # 推断世界坐标



# 落点位置解算，子弹初速度v，能量机关中心位置en_center(标定时把中心位置作为世界坐标的(0,0,0)点),
# 相机在世界坐标系下的位置坐标starting_point,目标在世界坐标系下的三维坐标target_point
def PointoffallSolver(v,starting_point,target_point):
    diffence = (starting_point-target_point)/1000 # 坐标差值
    x = diffence[0]
    d = sqrt(diffence[0]**2+diffence[2]**2)  # 重新计算距离
    y = diffence[1] # 垂直距离
    alpha = tanh((v**2-sqrt(v**4-g*(2*(v**2)*y+g*(d**2))))/(g*d)) # 弧度制的仰角
    t = d/(v*cos(alpha)) # 所需时间
    beta = cosh(x/(v*t*cos(alpha)))
    alpha_ang = alpha/pi*180
    beta_ang =  beta/pi*180
    return alpha_ang, beta_ang,t # 输出所需要的绝对角度


starting_point = Pnp_solve(obj_Point,img_Point,cameraMatrix,distCoeffs) # 得到摄像机坐标系的原点在世界坐标系的坐标
print(starting_point)
print(PointoffallSolver(Ini_v,starting_point,target_point))
print(Target_PreSolver(img_Center,Img_Point))