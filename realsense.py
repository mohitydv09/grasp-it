import pyrealsense2 as rs
import numpy as np
import cv2
import threading

class RealSenceStream:
    def __init__(self, visualization=False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.running = True

        self.pipeline.start(self.config)
        self.color_image = None
        self.depth_image = None
        self.intrinsics_matrix = None
        self.visualization = visualization

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())

            depth_info = rs.video_stream_profile(depth_frame.get_profile())
            intrinsics = depth_info.get_intrinsics()
            intrinsics_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                        [0, intrinsics.fy, intrinsics.ppy],
                                        [0, 0, 1]])
            self.intrinsics_matrix = intrinsics_matrix

    def get_frame(self):
        return self.color_image, self.depth_image, self.intrinsics_matrix
    
    def stop(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    rs_stream = RealSenceStream(visualization=True)
    while True:
        color_image, depth_image, intrinsics_matrix = rs_stream.get_frame()
        if color_image is None or depth_image is None:
            continue
        cv2.imshow("color", color_image)
        cv2.imshow("depth", depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    rs_stream.stop()
    cv2.destroyAllWindows()