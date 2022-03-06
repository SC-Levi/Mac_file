import time
import _thread
from scipy.spatial.distance import pdist, squareform
import serial # 串口
from numpy import *
import cv2
import numpy as np

flag = False # 是不是打动的物体
centerpoint = []
MIN = []
robot_yaw_angle = 0.0 #机器人云台水平角度值
ser = serial.Serial("COM7",115200,timeout=0.5)
robot_pitch_angle = 0.0 #
robot_shoot_speed = 0.0 #
distance = 7639 # 机关与摄像头的距离
energy_center = (0,0) # 能量机关对于图片上的圆心
Ini_v = 26  # 初速度18m/s
g = 9.8
pre_time = 0.4 # 预测子弹飞行时间 经验所得
energy_r = 765 # 能量机关的半径
energy_pal = pi/3 # 能量机关的角速度为60°每秒
detect_state = 0
starting_point = array([350,618.-82060051,8000.87440246])
img_Center = array([291.987,163.275]) # 对于图片的中心位置
ball_xy = (0,0)

# 击打位置调整, 目标在图片坐标系下的三维坐标Img_Point：（x，y）
# flag: 是否进行超前预测
def Target_PreSolver(img_Center,Img_Point,flag):
    act_r = linalg.norm(img_Center - Img_Point) # 得到对于图像上的半径
    scale = act_r/energy_r # 比例尺
    if flag == True:
        vector = Img_Point - img_Center
        angle_rot = energy_pal * pre_time
        rotation_matix = array([[cos(angle_rot), -sin(angle_rot)], [sin(angle_rot), cos(angle_rot)]])
        Target_Point = img_Center + dot(vector, rotation_matix)
    else:
        Target_Point = Img_Point
        """
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
        """

    x, y = (Target_Point-img_Center)/scale
    Target_Point = (x,y,0)
    return Target_Point # 推断世界坐标



# 落点位置解算，子弹初速度v，能量机关中心位置en_center(标定时把中心位置作为世界坐标的(0,y,z)点),
# 相机在世界坐标系下的位置坐标starting_point,目标在世界坐标系下的三维坐标target_point
def PointoffallSolver(v,starting_point,target_point):
    diffence = (target_point-starting_point)/1000 # 坐标差值
    x = diffence[0]
    d = sqrt(diffence[0]**2+diffence[2]**2)  # 重新计算距离
    y = -diffence[1] # 垂直距离
    print('y:',y)
    print('x:',x)
    # if y < 0.6 :
    #     y -= 0.3*y
    alpha = arctan((v**2-sqrt(v**4-g*(2*(v**2)*y+g*(d**2))))/(g*d)) # 弧度制的仰角
    t = d/(v*cos(alpha)) # 所需时间
    beta = arctan(x/d)
    alpha_ang = alpha/pi*180
    beta_ang =  beta/pi*180
    return alpha_ang, beta_ang,t # 输出所需要的绝对角度

def empty():
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
            hor[x] = np.hstack(imgArray[x])
        for x in range(0, rows):
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def get_data() :
    while True :
        data_count = ser.inWaiting()

        if data_count != 0 :
            if data_count == 7 :
                recv = ser.read(7)
                print(recv)
                tmp_yaw = int.from_bytes(recv[1 :3], byteorder='big', signed=True)
                tmp_pitch = int.from_bytes(recv[3 :5], byteorder='big', signed=True)
                tmp_shootspeed = int.from_bytes(recv[5 :6], byteorder='big', signed=False)

                robot_yaw_angle = tmp_yaw / 100.0
                robot_pitch_angle = tmp_pitch / 100.0
                robot_shoot_speed = tmp_shootspeed / 10.0

                print("yaw-", robot_yaw_angle, " pitch-", robot_pitch_angle, " shoot-", robot_shoot_speed)

            else :
                ser.reset_input_buffer()

        time.sleep(0.1)

# 发送数据
def Send_data(detect_state,yaw_angle,pitch_angle,shoot_control):
  # detect_state 有效状态
  # yaw_angle 左右偏向角
  # pitch_angle 俯仰角
  # shoot_control = 1  # 1：不发射  2：连续发射 3：单发

  buf=b'\xAA' + detect_state.to_bytes(length=1,byteorder='big',signed=False) + int(yaw_angle*100).to_bytes(length=2,byteorder='big',signed=True) + int(pitch_angle*100).to_bytes(length=2,byteorder='big',signed=True) + shoot_control.to_bytes(length=1,byteorder='big',signed=False)
  checksum = 0x00
  for i in range(1,7):
      checksum += buf[i]
  checksum &= 0xFF  # 最后一位的校验位

  buf += checksum.to_bytes(length=1,byteorder='big',signed=False)
  ser.write(buf)

def getContours(img):
    global detect_state
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # 保存了所有轮廓上的所有点，图像表现跟轮廓一致
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 算出面积
        if area>200: # 把一些噪点过滤掉
            cv2.drawContours(imgContour, cnt, -1, (255, 255, 0), 3)
            peri = cv2.arcLength(cnt,True)  # 轮廓线长度
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) # 找出轮廓的多边形拟合曲线
            x , y , w , h = cv2.boundingRect(approx)  # 当得到对象轮廓后，可用boundingRect()得到包覆此轮廓的最小正矩形
            rect = cv2.minAreaRect(cnt)  # 获取最小外接圆的半径
            width,height = cv2.boxPoints(rect)[1]
            centerpoint.append((x + (w // 2), y + (h // 2))) # 把各个轮廓的中心点记录


    if len(centerpoint)==0:
        target_Point = False
        detect_state = 0
    elif len(centerpoint)==1:
        objectType = 'Target'
        cv2.circle(imgContour, (int(centerpoint[0][0]),int(centerpoint[0][1])), 30, (0, 255, 0), 2)
        cv2.putText(imgContour, objectType, (int(centerpoint[0][0]),int(centerpoint[0][1])), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (255, 0, 0), 2)
        target_Point = centerpoint[0]
        detect_state = 1
    else:
        dis_point = squareform(pdist(centerpoint)) # 获得各个点与点之间的距离https://www.zhihu.com/question/291006944?sort=created
        for i in range(len(centerpoint)):
            _list = dis_point[:,i].tolist()
            _list.remove(0)
            _min = min(_list) # 找出距离最小的值
            MIN.append(_min)

        _max = max(MIN) # 找出最小距离里最大的那个
        target = MIN.index(_max)  # 它的下标

        objectType = 'Target'
        cv2.circle(imgContour, (int(centerpoint[target][0]),int(centerpoint[target][1])), 30, (0, 255, 0), 2)
        cv2.putText(imgContour, objectType, (int(centerpoint[target][0]),int(centerpoint[target][1])), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (255, 0, 0), 2)
        target_Point = centerpoint[target]
        detect_state = 1

    centerpoint.clear()
    MIN.clear()
    return target_Point

#img = cv2.imread(path)
# cap = cv2.VideoCapture('Resources/大风车_覆盖白光(高清).avi')
cap = cv2.VideoCapture(1)
#cap.set(3,640)
#cap.set(4,480)
#cap.set(10,100)

# 橙色0 179 33 255 75 255
cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars',640,480)
cv2.createTrackbar('Can_min','TrackBars',0,200,empty)  # can的阈值，找轮廓的
cv2.createTrackbar('Can_max','TrackBars',200,200,empty)
cv2.createTrackbar('Color_min','TrackBars',0,179,empty) # 色调
cv2.createTrackbar('Color_max','TrackBars',179,179,empty)
cv2.createTrackbar('Sat_min','TrackBars',50,255,empty) # 饱和度
cv2.createTrackbar('Sat_max','TrackBars',255,255,empty)
cv2.createTrackbar('Val_min','TrackBars',0,255,empty)  # 亮度
cv2.createTrackbar('Val_max','TrackBars',255,255,empty)

_thread.start_new_thread(get_data,()) # 开启接收线程，执行get_data方法

while True:
    succsee , imgOrigin = cap.read()
    B, G, R = cv2.split(imgOrigin)
    imgHSV = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2HSV)  # HSV通道
    # 图像灰度梯度 高于maxVal被认为是真正的边界，低于minVal的舍弃。两者之间的值要判断是否与真正的边界相连，相连就保留，不相连舍弃
    canny_min_threshold = cv2.getTrackbarPos('Can_min', 'TrackBars')
    canny_max_threshold = cv2.getTrackbarPos('Can_max', 'TrackBars')
    h_min = cv2.getTrackbarPos('Color_min', 'TrackBars')
    h_max = cv2.getTrackbarPos('Color_max', 'TrackBars')
    s_min = cv2.getTrackbarPos('Sat_min', 'TrackBars')
    s_max = cv2.getTrackbarPos('Sat_max', 'TrackBars')
    v_min = cv2.getTrackbarPos('Val_min', 'TrackBars')
    v_max = cv2.getTrackbarPos('Val_max', 'TrackBars')
    lower = np.array([h_min, s_min, v_min])  # 就是
    upper = np.array([h_max, s_max, v_max])  # 这三步
    mask = cv2.inRange(imgHSV, lower, upper)  # 得到特定的颜色通道
    img = cv2.bitwise_and(imgOrigin, imgOrigin, mask=mask)  # 还原到原图
    B, G, R = cv2.split(img)
    imgContour = img.copy()  # 画框结果图
    # 图片预处理结束
    color_channel = cv2.subtract(R, B) # 红蓝通道相减  对于红色机关
    _, imgnew = cv2.threshold(color_channel, 40, 255, cv2.THRESH_BINARY)  # 图片二值化处理

    ####
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imgnew = cv2.morphologyEx(imgnew, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算1
    # imgnew = cv2.morphologyEx(imgnew, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算1
    #### 至此获取单通道二值化图像 imgnew 结束

    energy_img = cv2.dilate(imgnew, kernel,1)  # 膨胀图像
    h, w = energy_img.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列
    cv2.floodFill(energy_img,mask,(0, 0), (255))
    energy_img = cv2.erode(energy_img, kernel,1)  # 腐蚀图像
    energy_img = cv2.bitwise_not(energy_img)
    # energy_img = cv2.dilate(energy_img, kernel,iterations=1)
    #### 至此获取能量机关二值图像结束energy_img

    imgBlur = cv2.GaussianBlur(energy_img,(7,7),1)  # 高斯滤波
    imgCanny = cv2.Canny(imgBlur,canny_min_threshold,canny_max_threshold) # 得到利用Canny得到的有轮廓的图像
    # # # # # # # #
    # # # # # # # #
    target_Point = getContours(imgCanny)   # 画框画在imgContour上 并且得到需要击打的图像坐标
    Target_Point = Target_PreSolver(img_Center, target_Point,flag) # 击打坐标预测并反馈到世界坐标上
    alpha_ang, beta_ang,_ = PointoffallSolver(Ini_v, starting_point, Target_Point) # 落点位置解算,输出所需要的绝对角度
    # # # # # # # #
    # # # # # # # #
    # alpha_ang = round((alpha_ang + robot_pitch_angle),2)
    # beta_ang = round((beta_ang + robot_pitch_angle),2)
    print('角度是：',alpha_ang,beta_ang,_)
    # print('Startpoint,Targetpoint',starting_point,Target_Point)
    Send_data(detect_state,beta_ang,alpha_ang,0) # 发送数据 连续发射
    time.sleep(0.5)
    Send_data(detect_state, 0, 0, 1)  # 发送数据 连续发射
    time.sleep(0.5)
    Send_data(detect_state, -beta_ang,-alpha_ang,0)  # 发送数据回到原来位置 不发射
    time.sleep(1)  # 打一会

    imgBlank = np.zeros_like(img)  # 空白图像
    imgStack = stackImages(0.3,([imgOrigin,imgContour,imgnew],
                               [imgCanny,energy_img,imgBlank]))

    cv2.imshow("Lookimg",imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break













