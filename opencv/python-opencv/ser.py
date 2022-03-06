import serial
import time
import _thread

ser = serial.Serial("COM7",115200,timeout=0.5)

robot_yaw_angle = 0.0 #机器人云台水平角度值
robot_pitch_angle = 0.0 #
robot_shoot_speed = 0.0 #

def get_data():
  while True:
    data_count = ser.inWaiting()
    print(data_count)
    if data_count != 0 :
      if data_count == 7 :
        recv = ser.read(7)
        # print(recv)
        tmp_yaw = int.from_bytes(recv[1:3], byteorder='big', signed=True)
        tmp_pitch = int.from_bytes(recv[3:5], byteorder='big', signed=True)
        tmp_shootspeed = int.from_bytes(recv[5:6], byteorder='big', signed=False)
        
        robot_yaw_angle = tmp_yaw/100.0
        robot_pitch_angle = tmp_pitch/100.0
        robot_shoot_speed = tmp_shootspeed/10.0
        
        print("yaw-",robot_yaw_angle," pitch-",robot_pitch_angle," shoot-",robot_shoot_speed)

      else:
        ser.reset_input_buffer()  # 齐输入缓冲器，放弃其所有内容。
    
    time.sleep(0.1)


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

if __name__ == '__main__':
  
  _thread.start_new_thread(get_data,()) # 开启接收线程，执行get_data方法

  robot_yaw_angle = 0.0  # 机器人云台水平角度值
  robot_pitch_angle = 0.0  #
  robot_shoot_speed = 0.0

  #Send_data(1,10,10,0)

  while True :

      pass
    # alpha_ang = round((1 + robot_pitch_angle), 2)
    # beta_ang = round((1 + robot_pitch_angle), 2)
    # print('角度是：', alpha_ang, beta_ang)
    # print('Startpoint,Targetpoint',starting_point,Target_Point)
    # Send_data(1, beta_ang, alpha_ang, 0)  # 发送数据 连续发射







