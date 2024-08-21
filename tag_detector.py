import cv2
import numpy as np
import threading
import pyrealsense2 as rs # type: ignore
import open3d as o3d # type: ignore

import helper

class RealSense:
    def __init__(self, device_serial_number=None, visualization=False, depth=False) -> None:
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


class ArUcoDetector(RealSense):
    """
    Extends the RealSense class to detect and estimate the pose of ArUco markers.

    Inherits:
    - RealSense: Inherits functionality for interfacing with the RealSense camera.
    """
    def __init__(self, tag_id:int=5, aruco_size:float = 0.05, device_serial_number:str=None, visualization:bool=False, depth:bool=False) -> None:
        """
        Initializes the ArUcoDetector class.

        Parameters:
        - tag_id (int): The ID of the ArUco marker to detect. Defaults to 5.
        - aruco_size (float): The size of the ArUco marker in meters. Defaults to 0.05.
        - device_serial_number (str): The serial number of the RealSense device to connect to. Defaults to None.
        - visualization (bool): If True, displays the camera feed in a separate window. Defaults to False.
        - depth (bool): If True, enables the depth stream alongside the color stream. Defaults to False.
        """
        super().__init__(device_serial_number, visualization, depth)
        self._aruco_size = aruco_size
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._tag_id = tag_id

    def get_aruco_pose(self) -> tuple[list[3], list[3]]:
        """
        Detects the pose of the specified ArUco marker.

        Returns:
        - tuple[list[3], list[3]]: A tuple containing the rotation vector and translation vector of the detected marker.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(self.color_frame, self._aruco_dict, parameters=self._aruco_params)
        index_of_interest = np.where(ids == self._tag_id)
        if ids is None:
            return None, None
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self._aruco_size, self.intrinsics, self.distortion_coeffs)

        rvec = rvecs[index_of_interest][0]
        tvec = tvecs[index_of_interest][0]

        return rvec, tvec

    def get_target_pose_with_curr_eff(self, curr_eef_pose : list[6], pre_grasp_distance : float = 0.25) -> list[6]:
        """
        Calculates the target pose of the robot's end effector relative to the detected ArUco marker.

        Parameters:
        - curr_eef_pose (list[6]): The current pose of the end effector (EEF) in the form [x, y, z, rx, ry, rz].
        - pre_grasp_distance (float): The distance to maintain from the target object before grasping. Defaults to 0.25.

        Returns:
        - list[6]: The target pose of the end effector relative to the detected marker.
        """
        ## Transfromation Matrix from World to Eff.
        T_w2eef = helper.make_matrix_from_tvec_and_rvec(curr_eef_pose[0:3], curr_eef_pose[3:])

        ## Transformation Matrix from Eff to Camera.
        trans_vector_eef2cam = np.array([-0.01, -0.08, 0.01])   ## 8 cm in Y and 1 cm in Z
        T_eef2cam = helper.make_matrix_from_tvec_axis_angle(trans_vector_eef2cam, 'x', -np.pi/12) ## 15 Degree

        T_w2cam = T_w2eef @ T_eef2cam

        rvec, tvec = self.get_aruco_pose()
        if rvec is None or tvec is None:
            return None

        T_cam2tag = helper.make_matrix_from_tvec_and_rvec(tvec, rvec)
        T_cam2tag[0:3, 0:3] = T_cam2tag[0:3, 0:3] @ o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi, 0, 0]))

        T_w2tag = T_w2cam @ T_cam2tag

        ## Make the Final Target Pose.
        T_w2target = np.eye(4)
        T_w2target[0:3, 0:3] = T_w2tag[0:3, 0:3]

        approach_vector = T_w2target[0:3, 2]
        norm_approach_vector = approach_vector/np.linalg.norm(approach_vector)
        translate_vector = - pre_grasp_distance * norm_approach_vector

        T_w2target[0:3, 3] = T_w2tag[0:3, 3] + translate_vector

        robot_target = T_w2target[0:3,3].tolist() + cv2.Rodrigues(T_w2target[0:3,0:3])[0].reshape(-1).tolist()

        return robot_target


if __name__=="__main__":
    aruco_detector = ArUcoDetector(tag_id=5, aruco_size=0.05, device_serial_number=None, visualization=True, depth=False)

    while True:
        try:
            if aruco_detector.color_frame is None:
                continue
            robot_target_pose = aruco_detector.get_target_pose_with_curr_eff([0.5,0.0,0.0,0.0,0.0,0.0])
            print(robot_target_pose)
        except KeyboardInterrupt:
            break
    aruco_detector.stop()
    cv2.destroyAllWindows()