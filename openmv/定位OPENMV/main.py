# Single Color RGB565 Blob Tracking Example
#
# This example shows off single color RGB565 tracking using the OpenMV Cam.

import sensor, image, time, math,struct
from pyb import UART
from pyb import LED
threshold_index = 0 # 0 for red, 1 for green, 2 for blue

thresholds = [(1, 21, -121, 127, -6, 127)] # 这里可以进行多颜色识别，目前只是用了识别黑色，黑色阈值是数组的第1个元素，即thresholds[0]
LED(1).on()
LED(2).on()
LED(3).on()

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time = 200)
sensor.set_auto_gain(True) # must be turned off for color tracking
sensor.set_auto_whitebal(True) # must be turned off for color tracking
clock = time.clock()

uart =UART(3, 115200)  #串口3，波特率115200
uart.init(115200, bits=8, parity=None, stop=1)  #8位数据位，无校验位，1位停止位

def send_data_packet(x, y,l,s):
    temp = struct.pack("<bbffff",                #格式为俩个字符俩个整型
                   0xAA,                       #帧头1
                   0xAE,                       #帧头2
                   float(x),
                   float(y),
                   float(l),
                   float(s))
    uart.write(temp)

K=5000
while(True):
    clock.tick()
    img = sensor.snapshot()
    b = img.find_blobs([thresholds[0]], pixels_threshold=1200, area_threshold=1200, merge=True)
    r = img.find_rects(threshold = 10000)
    c = img.find_circles(threshold = 3500, x_margin = 10, y_margin = 10, r_margin = 10,r_min = 2, r_max = 100, r_step = 2)
    if b:

        if c:
            for blob in b:
                img.draw_rectangle(blob.rect())
                img.draw_cross(blob.cx(),blob.cy())
                Lm = (blob[2] +blob[3])/2
                length = K/Lm
                send_data_packet(blob[5], blob[6],length,2.0)
                print(blob[5], blob[6],length,2.0)
        else:
            for blob in b:
                img.draw_rectangle(blob.rect())
                img.draw_cross(blob.cx(),blob.cy())
                Lm = (blob[2] +blob[3])/2
                length = K/Lm
                send_data_packet(blob[5], blob[6],length,1.0)
                print(blob[5], blob[6],length,1.0)
    else:
        send_data_packet(0.0, 0.0,0.0,0.0)

        print(0.0, 0.0,0.0,0.0)



