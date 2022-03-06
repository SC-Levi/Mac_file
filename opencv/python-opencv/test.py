detect_state = 1  # 0- 1-
yaw_angle = 0.0
pitch_angle = 0.0
shoot_control = 1  # # 1：不发射  2：连续发射 3：单发

# print(int(yaw_angle*100).to_bytes(length=2, byteorder='big', signed=False))
# print("{:04X}".format(int(yaw_angle*100)))

buf = b'\xAA' + detect_state.to_bytes(length=1, byteorder='big', signed=False) + int(yaw_angle * 100).to_bytes(length=2,
                                                                                                               byteorder='big',
                                                                                                               signed=True) + int(
    pitch_angle * 100).to_bytes(length=2, byteorder='big', signed=True) + shoot_control.to_bytes(length=1,
                                                                                                 byteorder='big',
                                                                                                 signed=False)

# buf = 'aa'+"{:04X}".format(int(yaw_angle*100))+"{:04X}".format(int(pitch_angle*100))+"{:02X}".format(int(shoot_control))
checksum = 0x00
for i in range(1, 7) :
    checksum += buf[i]
checksum &= 0xFF  # 最后一位的校验位

buf += checksum.to_bytes(length=1, byteorder='big', signed=False)
buf = int.from_bytes(buf, byteorder = 'big')
buf = hex(buf)
print(buf)