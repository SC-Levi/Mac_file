import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import serial as ser # 串口


def  fill_like_flood(image):
    h , w , c= image.shape
    mask = np.zeros((h+2,w+2),dtype=np.uint8)
    cv2.floodFill(image,mask,(10,10),(0,0,0),(50,50,50),(220,220,220),cv.FLOODFILL_FIXED_RANGE)
    return image

objPoints = np.array([[-115.0, -64.0, 0],[115.0, -70.0, 0],[115.0, 64.0, 0],[-115.0, 64.0, 0]]) # 装甲板的实际世界坐标
imgPoints = np.array([])
cameraMatrix = np.array([[1.03110741e+03,0.00000000e+00,3.09627059e+02],
                         [0.00000000e+00,1.02988140e+03,2.22642582e+02],
                         [0.00000000e+00,0.00000000e+00,1.00000000e+00]])
distCoeffs = np.array([-0.1408541617822344, 0.8248872589005009, 9.542278053040669e-05, 0.002492233684456267, -3.277725618255168])

centerpoint = []
MIN = []

# def calibration_pos():  # 位置标定



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
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
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


def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # 保存了所有轮廓上的所有点，图像表现跟轮廓一致
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 算出面积
        if area>200: # 把一些噪点过滤掉
            cv2.drawContours(imgContour, cnt, -1, (255, 255, 0), 3)
            peri = cv2.arcLength(cnt,True)  # 轮廓线长度
            approx = cv2.approxPolyDP(cnt,0.02*peri,True) # 找出轮廓的多边形拟合曲线
            # objCor = len(approx)
            x , y , w , h = cv2.boundingRect(approx)  # 当得到对象轮廓后，可用boundingRect()得到包覆此轮廓的最小正矩形
            # rect = cv2.minAreaRect(cnt)  # 获取最小外接圆的半径
            # width,height = cv2.boxPoints(rect)[1]
            centerpoint.append((x + (w // 2), y + (h // 2))) # 把各个轮廓的中心点记录
            lw_rate = w/h # 长宽比
            area_rate = area/(w*h)
            print(area_rate)
                #print('area_rate 1=',area_rate)
                #print('area_rate =',area_rate)
                #print('lw_rate =', lw_rate)
                #if objCor == 3: objectType = 'Tri'
                #elif objCor == 4 :
                #    aspRatio = w/float(h)
                #    if aspRatio > 0.95 and aspRatio <1.05:
                #        objectType = 'Square'  # 是不是正方形
                #    else:
                #        objectType = 'Rectangle'
            """
            if area<2000:
                objectType = 'Target'
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 0), 2)

            elif area >=3000:
                objectType = 'Already Target'
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 0), 2)   
            """
    print(centerpoint)

    if len(centerpoint)==0:
        pass
    elif len(centerpoint)==1:
        objectType = 'Target'
        cv2.circle(imgContour, (int(centerpoint[0][0]),int(centerpoint[0][1])), 30, (0, 255, 0), 2)
        cv2.putText(imgContour, objectType, (int(centerpoint[0][0]),int(centerpoint[0][1])), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (255, 0, 0), 2)
    else:
        dis_point = squareform(pdist(centerpoint)) # 获得各个点与点之间的距离https://www.zhihu.com/question/291006944?sort=created
        for i in range(len(centerpoint)):
            _list = dis_point[:,i].tolist()
            _list.remove(0)
            _min = min(_list) # 找出距离最小的值
            MIN.append(_min)

        _max = max(MIN) # 找出最小距离里最大的那个
        target = MIN.index(_max)  # 它的下标
        # print('target',target)

        objectType = 'Target'
        cv2.circle(imgContour, (int(centerpoint[target][0]),int(centerpoint[target][1])), 30, (0, 255, 0), 2)
        cv2.putText(imgContour, objectType, (int(centerpoint[target][0]),int(centerpoint[target][1])), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (255, 0, 0), 2)

    centerpoint.clear()
    MIN.clear()

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
cv2.createTrackbar('Sat_min','TrackBars',150,255,empty) # 饱和度
cv2.createTrackbar('Sat_max','TrackBars',255,255,empty)
cv2.createTrackbar('Val_min','TrackBars',100,255,empty)  # 亮度
cv2.createTrackbar('Val_max','TrackBars',255,255,empty)


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
    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])  # 就是
    upper = np.array([h_max, s_max, v_max])  # 这三步
    mask = cv2.inRange(imgHSV, lower, upper)  # 得到特定的颜色通道
    img = cv2.bitwise_and(imgOrigin, imgOrigin, mask=mask)  # 还原到原图
    B, G, R = cv2.split(img)
    imgContour = img.copy()  # 画框结果图
    # 图片预处理结束

    color_channel = cv2.subtract(R, B) # 红蓝通道相减  对于红色机关
    # color_channel = cv2.subtract(B, G)  # 蓝红通道相减 对于蓝色机关
    _, imgnew = cv2.threshold(color_channel, 40, 255, cv2.THRESH_BINARY)  # 图片二值化处理
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    # _, img_2 = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY) # 图片二值化处理
    # imgnew = cv2.bitwise_and(img_1, img_2)  # 两种处理方式相与，消除颜色光晕

    ####
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imgnew = cv2.morphologyEx(imgnew, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭运算1
    # imgnew = cv2.morphologyEx(imgnew, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算1
    #### 至此获取单通道二值化图像 imgnew 结束

    energy_img = cv2.dilate(imgnew, kernel,1)  # 膨胀图像
    h, w = energy_img.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)  # mask必须行和列都加2，且必须为uint8单通道阵列
    cv2.floodFill(energy_img,mask,(0, 0), (255))
    energy_img = cv2.erode(energy_img, kernel,2)  # 腐蚀图像
    energy_img = cv2.bitwise_not(energy_img)
    # energy_img = cv2.dilate(energy_img, kernel,iterations=1)
    #### 至此获取能量机关二值图像结束energy_img

    imgBlur = cv2.GaussianBlur(energy_img,(5,5),1)  # 高斯滤波
    imgCanny = cv2.Canny(imgBlur,canny_min_threshold,canny_max_threshold)
    getContours(imgCanny)   # 画框画在imgContour上
    imgBlank = np.zeros_like(img)  # 空白图像

    imgStack = stackImages(0.3,([imgOrigin,imgContour,imgnew],
                               [imgCanny,energy_img,imgBlank]))

    cv2.imshow("Lookimg",imgStack)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break