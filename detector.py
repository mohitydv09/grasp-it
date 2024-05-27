import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import open3d as o3d

class Detector:
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
        self.distortion_matrix = None
        self.visualization = visualization

        self.aruco_size = 0.05
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

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

    def get_aruco_pose(self):
        ## Detect Aruco Marker in the image
        if self.color_image is None:
            # print("No Image Detected")
            return None,None
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(self.color_image, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.aruco_size, self.intrinsics_matrix, self.distortion_matrix)
        # print(f"Detected {len(corners)} markers in frame, with {len(rejectedImgPoints)} rejected frames.")

        if self.visualization:
            ## Draw the detected markers
            self.color_image = cv2.aruco.drawDetectedMarkers(self.color_image, corners, ids)
            for i in range(len(rvecs)):
                self.color_image = cv2.drawFrameAxes(self.color_image, self.intrinsics_matrix, self.distortion_matrix,
                                                      rvecs[i], tvecs[i], length=0.01, thickness=1) ## Last parameter is the size of the axis.

            ## Display the image
            cv2.imshow("Image with Detection : ", self.color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if rvecs is None:
            # print("No Aruco Marker Detected")
            return None, None
        if len(rvecs) == 0:
            # print("No Aruco Marker Detected")
            return None, None
        if len(rvecs) > 1:
            # print("More than one Aruco Marker Detected")
            return None, None
        return rvecs, tvecs

    def get_target_pose(self, curr_eef_pose, pre_grasp_distance=0.3, require_matrices=False):
        ## Transfromation Matrix from World to Eff.
        T_w2eef = self.make_matrix(curr_eef_pose[0:3], np.array(curr_eef_pose[3:]))

        ## Transformation Matrix from Eff to Camera.
        trans_vector_eef2cam = np.array([0, -0.1, 0.02])
        T_eef2cam = self.make_matrix_from_angle(trans_vector_eef2cam, 'x', -np.pi/12)

        T_w2cam = T_w2eef @ T_eef2cam

        rvecs, tvecs = self.get_aruco_pose()
        if rvecs is None:
            # print("Marker Detection Error")
            return None, None, None, None, None
        elif len(rvecs) > 1:
            # print("More than one Marker Detected")
            return None, None, None, None, None
        T_cam2tag = self.make_matrix(tvecs[0][0], rvecs[0][0])
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
        if require_matrices:
            return robot_target, T_w2eef, T_w2cam, T_w2tag, T_w2target
        return robot_target, None, None, None, None

    def make_matrix(self, tvec, rvec):
        T = np.eye(4)
        T[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
        T[0:3, 3] = tvec
        return T
    
    def make_matrix_from_angle(self, tvec, axis, angle):
        T = np.eye(4)
        if axis == 'x':
            axis_vector = np.array([angle, 0, 0])
        elif axis == 'y':
            axis_vector = np.array([0, angle, 0])
        elif axis == 'z':
            axis_vector = np.array([0, 0, angle])
        T[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(axis_vector)
        T[0:3, 3] = tvec
        return T

class Open3dVisualizer:
    def __init__(self, detector, robot):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.robot = robot
        self.detector = detector
        self.running = True


        self.curr_eff_pose = self.robot.get_eff_pose()
        robot_target, T_w2eef, T_w2cam, T_w2tag, T_w2target = self.detector.get_target_pose(self.curr_eff_pose, require_matrices=True)
        self.robot_target = robot_target
        self.T_w2eef = T_w2eef
        self.T_w2cam = T_w2cam
        self.T_w2tag = T_w2tag
        self.T_w2target = T_w2target

        self.base_frame = self.create_base_frame()
        self.eef_frame = self.create_eef_frame()

        self.vis.add_geometry(self.base_frame)
        self.vis.add_geometry(self.eef_frame)

        # for geom in self.init_geometries():
        #     self.vis.add_geometry(geom)

        # self.thread = threading.Thread(target=self.update, args=())
        # self.thread.daemon = False
        # self.thread.start()

    def init_geometries(self):
        # if self.curr_eff_pose is None or self.T_w2cam is None or self.T_w2tag is None or self.T_w2target is None:
        #     return None, None, None, None, None, None
        base_frame = self.create_base_frame()
        grid = self.create_grid()
        eff_frame = self.create_eef_frame()
        camera_frame = self.create_camera_frame()
        tag_frame = self.tag_frame()
        target_frame = self.target_frame()
        return base_frame, grid, eff_frame, camera_frame, tag_frame, target_frame

    def create_base_frame(self):
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])  
        return base_frame
    
    def create_eef_frame(self):
        eff_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=self.curr_eff_pose[0:3])
        rot_matrix = cv2.Rodrigues(np.array(self.curr_eff_pose[3:]))[0]
        eff_frame.rotate(rot_matrix, center=self.curr_eff_pose[0:3])
        return eff_frame

    def create_camera_frame(self):
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.T_w2cam[0:3,3])
        camera_frame.rotate(self.T_w2cam[0:3,0:3])
        return camera_frame
    
    def tag_frame(self):
        tag_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.T_w2tag[0:3,3])
        tag_frame.rotate(self.T_w2tag[0:3,0:3])
        return tag_frame
    
    def target_frame(self):
        target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=self.T_w2target[0:3,3])
        target_frame.rotate(self.T_w2target[0:3,0:3])
        return target_frame

    def create_grid(self,size=10, step=1):
        lines = []
        points = []

        # Create grid points and lines in the X-Y plane
        for i in range(-size, size + 1, step):
            points.append([i, -size, 0])
            points.append([i, size, 0])
            points.append([-size, i, 0])
            points.append([size, i, 0])
            
            lines.append([len(points) - 4, len(points) - 3])
            lines.append([len(points) - 2, len(points) - 1])
        
        # Create grid points and lines in the X-Z plane
        for i in range(-size, size + 1, step):
            points.append([i, 0, -size])
            points.append([i, 0, size])
            points.append([-size, 0, i])
            points.append([size, 0, i])
            
            lines.append([len(points) - 4, len(points) - 3])
            lines.append([len(points) - 2, len(points) - 1])
        
        # Create grid points and lines in the Y-Z plane
        for i in range(-size, size + 1, step):
            points.append([0, i, -size])
            points.append([0, i, size])
            points.append([0, -size, i])
            points.append([0, size, i])
            
            lines.append([len(points) - 4, len(points) - 3])
            lines.append([len(points) - 2, len(points) - 1])

        # Convert to numpy arrays
        points = np.array(points)
        lines = np.array(lines)

        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set color for the grid lines
        colors = [[0.8, 0.8, 0.8] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def update(self):
        self.curr_eff_pose = self.robot.get_eff_pose()
        robot_target, T_w2eef, T_w2cam, T_w2tag, T_w2target = self.detector.get_target_pose(self.curr_eff_pose,
                                                                                                require_matrices=True)
        self.robot_target = robot_target
        self.T_w2eef = T_w2eef
        self.T_w2cam = T_w2cam
        self.T_w2tag = T_w2tag
        self.T_w2target = T_w2target

        self.base_frame = self.create_base_frame()
        self.eef_frame = self.create_eef_frame()
        
        self.vis.update_geometry(self.eef_frame)
        # for geom in self.init_geometries():
        #     self.vis.update_geometry(geom)
        self.vis.poll_events()
        self.vis.update_renderer()

    def stop(self):
        self.vis.destroy_window()
        self.running = False
        # self.thread.join()

if __name__=="__main__":
    rs_stream = Detector(visualization=True)
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