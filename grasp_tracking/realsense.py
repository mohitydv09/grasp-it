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

def get_images():
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())/255.0
        color_image = color_image[:, :, ::-1]
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_profile = rs.video_stream_profile(depth_frame.get_profile())
        intrinsics = depth_profile.get_intrinsics()
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.ppx
        cy = intrinsics.ppy

        pipeline.stop()

        return color_image, depth_image, fx, fy, cx, cy
    
if __name__ == '__main__':
    get_images()
    cv2.waitKey(0)
    cv2.destroyAllWindows()