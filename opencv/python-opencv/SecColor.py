import cv2
import numpy as np

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

def empty():
    pass

img = cv2.imread('Resources/1.jpg')
#cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
#cap.set(10,100)


# 橙色0 179 33 255 75 255
cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars',640,480)
cv2.createTrackbar('HueMin','TrackBars',0,179,empty)
cv2.createTrackbar('HueMax','TrackBars',179,179,empty)
cv2.createTrackbar('SatMin','TrackBars',0,255,empty)
cv2.createTrackbar('SatMax','TrackBars',255,255,empty)
cv2.createTrackbar('ValMin','TrackBars',0,255,empty)
cv2.createTrackbar('ValMax','TrackBars',255,255,empty)





while True:
    # succsee , img = cap.read()
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)  #  灰度
    h_min = cv2.getTrackbarPos('HueMin','TrackBars')
    h_max = cv2.getTrackbarPos('HueMax', 'TrackBars')
    s_min = cv2.getTrackbarPos('SatMin', 'TrackBars')
    s_max = cv2.getTrackbarPos('SatMax', 'TrackBars')
    v_min = cv2.getTrackbarPos('ValMin', 'TrackBars')
    v_max = cv2.getTrackbarPos('ValMax', 'TrackBars')
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])  # 就是
    upper = np.array([h_max,s_max,v_max])  # 这三步
    mask = cv2.inRange(imgHSV,lower,upper) # 得到特定的颜色通道
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    cv2.imshow('imgStack', imgStack)
    cv2.waitKey(1)
    # cv2.imshow('mask', mask)
    # cv2.imshow('imgHSV', imgHSV)
    # cv2.imshow('img', img)
    # cv2.imshow('Video',imgResult)
    #if cv2.waitKey(1) & 0xFF ==ord('q'):
    #    break