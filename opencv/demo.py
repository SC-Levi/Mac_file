import numpy as np
import cv2 as cv

# cap = cv.VideoCapture(-1)
# cap = cv.VideoCapture(1)
cap = cv.VideoCapture(0)  # 传入一个设备索引,代表不同的摄像头,或者传入一个视频文件名
# cap = cv.VideoCapture('testVideo.mp4')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print('宽度和高度分别是:', width, height)  # 宽度和高度分别是: 640.0 480.0

ret = cap.set(cv.CAP_PROP_FRAME_WIDTH,320)  # 设定宽度
ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT,240)  # 设定高度

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()  # 返回True或者False,如果读取正确是True,可以用False来判断是否到达视频的末尾 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    print(type(frame))
    print(np.shape(frame))
    cv.imshow('LinMaZi-frame-LinZuQuan', frame)  # 设定显示窗口标题文字
    if cv.waitKey(1) == ord('q'):  # 判断用户是否输入q键
        break

# When everything done, release the capture
cap.release()  # 释放资源
cv.destroyAllWindows()

