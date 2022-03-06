# -*- coding: utf-8 -*-
# @Time    :2021/7/17 21:16
# @Author  :LSY Dreampoet
# @SoftWare:PyCharm

import cv2
import numpy as np

if __name__ == '__main__':

    # 找棋盘格角点
    # 阈值
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 30, 0.001)

    #棋盘格模板规格
    w = 8
    h = 6
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = [] # 在世界坐标系中的三维点
    imgpoints = [] # 在图像平面的二维点

    #检测是否卡死
    I = 1
    J = 0
    rate = 0
    cap = cv2.VideoCapture(0)
    while True:
        flat,img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray",gray)
        key = cv2.waitKey(1)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
        # 如果找到足够点对，将其存储起来
        print("The ret is :",ret,"I=",I)
        if ret == True:
            print("咦，它中了！",J)
            J += 1
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w,h), corners, ret)
            cv2.imshow('findCorners',img)
        if key == ord('q'):
            break
        I += 1
        rate = J/I
        print("准确率是：",rate)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print (("ret:"),ret)
    print (("mtx:\n"),mtx)        # 内参数矩阵
    print (("dist:\n"),dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print (("rvecs:\n"),rvecs)    # 旋转向量  # 外参数
    print (("tvecs:\n"),tvecs)    # 平移向量  # 外参数
    # 去畸变
    img2 = cv2.imread('./Calibration_Success/0_calibration.jpg')
    h,w = img2.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h)) # 自由比例参数
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    # 根据前面ROI区域裁剪图片
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('./Calibration_Success/C_C.jpg',dst) #选择自己的图片
    cv2.imshow("C_C",dst)
    cv2.waitKey(0)


