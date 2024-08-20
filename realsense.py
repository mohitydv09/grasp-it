import pyrealsense2 as rs
import numpy as np
import threading
import cv2

# Constants for RealSense device serial numbers
D405_SERIAL_NUMBER = '128422270081'
D405_SERIAL_NUMBER2 = '126122270307'

class RealSense:
    def __init__(self, depth=False, device_serial_number=None, visualization=False) -> None:
        """
        Initializes the RealSense camera interface.

        Parameters:
        - depth (bool): If True, enables depth stream alongside the color stream.
        - device_serial_number (str): The serial number of the RealSense device to connect to.
        - visualization (bool): If True, displays the camera feed in a separate window.
        """
        # Initialize instance variables
        self._depth = depth
        self._device_serial_number = device_serial_number
        self._visualization = visualization
        self._running = True

        self.color_frame = None
        self.depth_frame = None
        self.intrinsics = None              ## Initialized in the _initialize_pipeline function
        self.distortion_coeffs = None       ## Initialized in the _initialize_pipeline function
        self.distortion_model = None        ## Initialized in the _initialize_pipeline function

        # Initialize the camera pipeline
        self._initialize_pipeline()

        # Start a new thread to continuously update frames from the camera
        self._update_thread = threading.Thread(target=self._update, args=())
        self._update_thread.daemon = True
        self._update_thread.start()

        # Start a visualization thread if visualization is enabled
        if self._visualization:
            self._visualization_thread = threading.Thread(target=self._visualize, args=())
            self._visualization_thread.daemon = True
            self._visualization_thread.start()

    def _initialize_pipeline(self) -> None:
        """
        Initializes the RealSense pipeline and configures the streams.
        """
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # If a specific device serial number is provided, enable the device
        if self._device_serial_number:
            self._config.enable_device(self._device_serial_number)

        # Enable the color stream
        self._config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Enable the depth stream if required
        if self._depth:
            self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self._align = rs.align(rs.stream.color)  # Align depth to color stream

        # Start the pipeline with the configured streams
        self._pipeline.start(self._config)

        ## Set the Intrinsics and Distortion Coefficients
        rs_intrinsics = self._pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([[rs_intrinsics.fx, 0, rs_intrinsics.ppx,],[0, rs_intrinsics.fy, rs_intrinsics.ppy,],[0,0,1]])
        self.distortion_coeffs = np.array(rs_intrinsics.coeffs)
        self.distortion_model = rs_intrinsics.model.name

    def _update(self) -> None:
        """
        Continuously updates the frames from the RealSense camera.
        """
        while self._running:
            frames = self._pipeline.wait_for_frames()

            # Process depth and color frames if depth is enabled
            if self._depth:
                aligned_frames = self._align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()

            if not color_frame:
                continue
            
            # Convert depth frame to a NumPy array if depth is enabled
            if self._depth:
                self.depth_frame = np.asanyarray(depth_frame.get_data())
                self.intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                # print(self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.ppx, self.intrinsics.ppy)

            # Convert color frame to a NumPy array
            self.color_frame = np.asanyarray(color_frame.get_data(), dtype=np.uint8)


    def _visualize(self) -> None:
        """
        Continuously displays the RealSense camera feed in a window.
        """
        while self._running:
            if self.color_frame is not None:
                cv2.imshow('RealSense feed: ', self.color_frame)
            else:
                # Show a black screen if no color frame is available
                cv2.imshow('RealSense feed: ', np.zeros((480, 640, 3), dtype=np.uint8))
            cv2.waitKey(10)
        
        cv2.destroyAllWindows()

    def get_color_frame(self) -> np.ndarray:
        """
        Returns the latest color frame from the RealSense camera.

        Returns:
        - np.ndarray: The latest color frame.
        """
        return self.color_frame
    
    def get_depth_frame(self) -> np.ndarray:
        """
        Returns the latest depth frame from the RealSense camera.

        Returns:
        - np.ndarray: The latest depth frame.
        """
        return self.depth_frame
    
    def get_intrinsics(self) -> rs.intrinsics:
        """
        Returns the intrinsics of the RealSense camera.

        Returns:
        - rs.intrinsics: The intrinsics of the RealSense camera.
        """
        return self.intrinsics

    def stop(self) -> None:
        """
        Stops the RealSense camera interface and associated threads.
        """
        self._running = False
        self._update_thread.join()

        if self._visualization:
            self._visualization_thread.join()

        self._pipeline.stop()

def print_realsense_devices():
    """
    Prints all available RealSense devices connected to the system.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    for device in devices:
        print(device)

if __name__ == "__main__":
    realsense = RealSense(depth=False,visualization=True)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        realsense.stop()
        print("RealSense stopped")