import numpy as np
import pyrealsense2 as rs

import cv2

class RS_Cam() :
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def __call__(self, show_image = 0, get_map = 0):
        frames = self.pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        #if(not depth_frame or not color_frame) :
        #    return 0, 0
        
        #depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.03), cv2.COLORMAP_JET)
        if(show_image == 1):
            cv2.imshow('RS', color_image)
            key = cv2.waitKey(1)
        return color_image
    
    def Stop_Pip(self):
        self.pipeline.stop()
if(__name__ == '__main__'):
    Cam = RS_Cam()
    while(1):
        Cam(1)
