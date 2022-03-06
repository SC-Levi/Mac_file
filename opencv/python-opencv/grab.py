import cv2 as cv
import numpy as np
import serial
import time

ser = serial.Serial("COM5",115200,timeout=0.5)

robot_yaw_angle = 0.0 #机器人云台水平角度值
robot_pitch_angle = 0.0 #
robot_shoot_speed = 0.0 #

def get_data():
  while True:
    data_count = ser.inWaiting()

    if data_count != 0 :
      if data_count == 7 :
        recv = ser.read(7)
        print(recv)
        tmp_yaw = int.from_bytes(recv[1:3], byteorder='big', signed=True)
        tmp_pitch = int.from_bytes(recv[3:5], byteorder='big', signed=True)
        tmp_shootspeed = int.from_bytes(recv[5:6], byteorder='big', signed=False)
        
        robot_yaw_angle = tmp_yaw/100.0
        robot_pitch_angle = tmp_pitch/100.0
        robot_shoot_speed = tmp_shootspeed/10.0
        
        print("yaw-",robot_yaw_angle," pitch-",robot_pitch_angle," shoot-",robot_shoot_speed)  

      else:
        ser.reset_input_buffer()
    
    time.sleep(0.1)

def  fill_like_flood(image):
    h , w , c= image.shape
    mask = np.zeros((h+2,w+2),dtype=np.uint8)
    cv.floodFill(image,mask,(10,10),(0,0,0),(50,50,50),(220,220,220),cv.FLOODFILL_FIXED_RANGE)
    return image

def bag_of_image(image):
    _size = image.size
    image = np.ascontiguousarray(image,dtype=np.int32)
    black_num = np.sum(image == 0)
    white_num = _size - black_num
    return [black_num,white_num]

def point_predict(detected_point):
    center_x = 472
    center_y = 197
    center_point = [center_x,center_y]
    center_point = np.array(center_point)
    detected_point = np.array(detected_point)
    t = 10 # ms
    speed = 60 # °/s
    angle = speed * t / 1000
    rotation_matrix = np.array([[np.cos(angle),-1 * np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    prediction_point = center_point + np.dot(rotation_matrix,(detected_point - center_point))
    scale = 70 / 92 # cm/pixel
    prediction_point_3d = (prediction_point - center_point) * scale
    prediction_point_3d = np.array([prediction_point_3d[0],prediction_point_3d[1],0])
    return prediction_point_3d

# pitch angle : up and down ; yaw angle : left and right
def angle_predict(prediction_point_3d): 
	v = 15 # m/s
	g = 9.8 # m/s^2
	camera_point_3d = np.array([-90,80,800])
	print(camera_point_3d.shape)
	xz_camera = np.array([camera_point_3d[0],camera_point_3d[2]])
	xz_prediction = np.array([prediction_point_3d[0],prediction_point_3d[2]])
	hypotenuse_distance = (np.sum((xz_camera - xz_prediction)**2))**0.5
	# yaw angle
	vertical_distance = camera_point_3d[2]
	yaw_angle = np.arccos(vertical_distance / hypotenuse_distance)
	# pitch angle
	height = prediction_point_3d[1] - camera_point_3d[1]
	pitch_angle = np.arctan((v**2-(v**4-g*(2*height*(v**2) + g*(camera_point_3d[2]**2))))/(g*camera_point_3d[2]))
	return (yaw_angle/3.14*180,pitch_angle/3.14*180)

def control_shoot(detect_state , yaw_angle ,pitch_angle,shoot_control):
	buf=b'\xAA' + detect_state.to_bytes(length=1,byteorder='big',signed=False) + int((-1)*yaw_angle*100).to_bytes(length=2,byteorder='big',signed=True) + int((-1)*pitch_angle*100).to_bytes(length=2,byteorder='big',signed=True) + shoot_control.to_bytes(length=1,byteorder='big',signed=False)
	checksum = 0x00 # 十六进制
	for i in range(1,7):
		checksum += buf[i]
	checksum &= 0xFF # 都是1，则为1，0xFF为11111111
	buf += checksum.to_bytes(length=1,byteorder='big',signed=False)
	ser.write(buf)
	no_shoot = 1
	buf=b'\xAA' + detect_state.to_bytes(length=1,byteorder='big',signed=False) + int(yaw_angle*100).to_bytes(length=2,byteorder='big',signed=True) + int(pitch_angle*100).to_bytes(length=2,byteorder='big',signed=True) + no_shoot.to_bytes(length=1,byteorder='big',signed=False)
	checksum = 0x00 # 十六进制
	for i in range(1,7):
		checksum += buf[i]
	checksum &= 0xFF # 都是1，则为1，0xFF为11111111
	buf += checksum.to_bytes(length=1,byteorder='big',signed=False)
	ser.write(buf)


def main():
	cap = cv.VideoCapture(1)
	ret = True
	i = 0
	mtx = np.array([[1.03782597e+03,0.00000000e+00,3.40535909e+02],
			[0.00000000e+00,1.03660967e+03,2.52234990e+02],
			[0.00000000e+00,0.00000000e+00,1.00000000e+00]])
	dist = np.array([[-1.16797326e-01,-1.73850970e+00,
			2.96839053e-03,-1.12024427e-03,5.30806455e+01]])
	
	# 云台矫正
	'''
	data_count = ser.inWaiting()
	if data_count != 0 :
		if data_count == 7 :
			recv = ser.read(7)
			print(recv)
			true_yaw_angle = 100
			true_pitch_angle = 100
			tmp_yaw = int.from_bytes(recv[1:3], byteorder='big', signed=True)
			tmp_pitch = int.from_bytes(recv[3:5], byteorder='big', signed=True)
			tmp_shootspeed = int.from_bytes(recv[5:6], byteorder='big', signed=False)
			robot_yaw_angle = tmp_yaw/100.0
			robot_pitch_angle = tmp_pitch/100.0
			robot_shoot_speed = tmp_shootspeed/10.0
			print("yaw-",robot_yaw_angle," pitch-",robot_pitch_angle," shoot-",robot_shoot_speed)
			robot_yaw_angle = true_yaw_angle - robot_yaw_angle
			robot_pitch_angle = true_pitch_angle - robot_pitch_angle
			robot_detect_state = 1
			robot_shoot_control = 1
			buf=b'\xAA' + robot_detect_state.to_bytes(length=1,byteorder='big',signed=False) + int(robot_yaw_angle*100).to_bytes(length=2,byteorder='big',signed=True) + int(robot_pitch_angle*100).to_bytes(length=2,byteorder='big',signed=True) + robot_shoot_control.to_bytes(length=1,byteorder='big',signed=False)
			checksum = 0x00 # 十六进制
			for i in range(1,6):
				checksum += buf[i]
			checksum &= 0xFF # 都是1，则为1，0xFF为11111111
			buf += checksum.to_bytes(length=1,byteorder='big',signed=False)
			ser.write(buf)
		else:
			ser.reset_input_buffer()
		'''
	while ret:
		ret , frame = cap.read()
		if ret :
			frame = cv.resize(frame, (640,480), interpolation = cv.INTER_LINEAR)
			h,  w = frame.shape[:2]
			newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
			frame = cv.undistort(frame, mtx, dist, None, newcameramtx)
			img_BGR = fill_like_flood(frame)
			img_GRAY = cv.cvtColor(img_BGR,cv.COLOR_BGR2GRAY)
			ret_1,img_thresh_1 = cv.threshold(img_GRAY,240,255,cv.THRESH_TOZERO_INV)
			ret_2,img_thresh_2 = cv.threshold(img_thresh_1,100,255,cv.THRESH_BINARY)
			img = cv.bitwise_not(img_thresh_2)
			contours,hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
			cv.drawContours(img,contours,-1,(0,255,0),2)

			for contour in contours:
				lw_rate = 0
				area_rate = 0
				rect = cv.minAreaRect(contour)
				width = max([rect[1][0],rect[1][1]])
				height = min([rect[1][0],rect[1][1]])
				if height != 0:
					lw_rate = width / height
				area = cv.contourArea(contour)
				if width + height != 0:
					area_rate = area / (width + height)

				if 1.5 < lw_rate and lw_rate < 2.0 and area > 300 and area < 700 :
					i += 1
					x_range_1 = int(rect[0][0]) - 30
					x_range_2 = int(rect[0][0]) + 30
					y_range_1 = int(rect[0][1]) - 30 
					y_range_2 = int(rect[0][1]) + 30
					img_classify = img[y_range_1:y_range_2,x_range_1:x_range_2]
					feature = bag_of_image(img_classify)
					print(feature)
					if feature[0] > 1000 and feature[0] < 1500 and feature[1] > 2000 and feature[1] < 3300:
						cv.rectangle(img,(x_range_1,y_range_1),(x_range_2,y_range_2),(0,0,255),2)
						detected_point = [int(0.5*(x_range_1+x_range_2)),int(0.5*(y_range_1+y_range_2))]
						angle = angle_predict(point_predict(detected_point)) # located in [-pi/2,pi/2]
						print(angle)
						control_shoot(1,angle[0],angle[1],1)
						time.sleep(1)
							
		cv.imshow('image',img)
		if cv.waitKey(20)& 0xFF == 'q':
			break

	cap.release()
	cv.destroyAllWindows()

main()