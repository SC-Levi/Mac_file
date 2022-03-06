import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
imagerandem = np.random.randint(0, 256, size=[256, 256, 3], dtype = np.uint8)


image = cv.imread('3.jpeg')
imgRGB = cv. cvtColor(image,  cv.COLOR_BGR2RGB)
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
font = cv.FONT_HERSHEY_SIMPLEX
imgRGB = cv.putText(imgRGB,'OpenCV',(50,1000), font, 20,(0,255,255),9,cv.LINE_AA)
imgRGB =  cv.polylines(imgRGB, [pts], True, (0, 255, 255))
cv.imshow('image_bgr', imgRGB)
cv.waitKey()
cv.destroyAllWindows()