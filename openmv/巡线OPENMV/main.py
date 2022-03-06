THRESHOLD = (27, 88, 24, 121, -100, 79) # Grayscale threshold for dark things...
import sensor, image, time
from pyb import LED
import struct
from pyb import UART


LED(1).on()
LED(2).on()
LED(3).on()

sensor.reset()
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQQVGA) # 80x60 (4,800 pixels) - O(N^2) max = 2,3040,000.
#sensor.set_windowing([0,20,80,40])
sensor.skip_frames(time = 2000)     # WARNING: If you use QQVGA it may take seconds
clock = time.clock()                # to process a frame sometimes.


uart =UART(3, 115200)  #串口3，波特率115200
uart.init(115200, bits=8, parity=None, stop=1)  #8位数据位，无校验位，1位停止

def send_data_packet(x, y):
    temp = struct.pack("<bbff",                #格式为俩个字符俩个整型
                   0xAA,                       #帧头1
                   0xAE,                       #帧头2
                   float(x), # up sample by 4    #数据1
                   float(y)) # up sample by 4    #数据2
    uart.write(temp)                           #串口发送

while(True):
    clock.tick()
    img = sensor.snapshot().binary([THRESHOLD])
    line = img.get_regression([(100,100,0,0,0,0)], robust = True)

    if (line):
        rho_err = abs(line.rho())-img.width()/2
        if line.theta()>90:
            theta_err = line.theta()-180
        else:
            theta_err = line.theta()
        img.draw_line(line.line(), color = 127)
        print(rho_err,theta_err)
        send_data_packet(rho_err, theta_err)

    else:

        pass

