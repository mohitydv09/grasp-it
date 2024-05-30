import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    cv2.imshow('color', color_image)
    cv2.imshow('depth', depth_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
