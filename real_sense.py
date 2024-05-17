import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Initialize the align object.
align_to = rs.stream.color
align = rs.align(align_to)

## Start streaming
pipeline.start(config)

frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)

color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

depth_info = rs.video_stream_profile(depth_frame.get_profile())
intrinsics = depth_info.get_intrinsics()

intrinsics_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1]])

camera_output = {'color': color_image, 'depth': depth_image, 'intrinsics': intrinsics_matrix}

## Save the output.
np.save('camera_output.npy', camera_output)

pipeline.stop()