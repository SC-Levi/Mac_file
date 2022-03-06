import cv2
import numpy
cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    cv2.waitKey(2)
