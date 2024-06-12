import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import open3d as o3d

pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow("Color Image", color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break