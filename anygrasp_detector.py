import os
import argparse
import numpy as np
import open3d as o3d
import threading
import cv2
from PIL import Image
from graspnetAPI import GraspGroup
import time
import typing_extensions
from tracker import AnyGraspTracker

from detector import Detector
detector = Detector()

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
# parser.add_argument('--filter', type=str, default='oneeuro', help='Filter to smooth grasp parameters(rotation, width, depth). [oneeuro/kalman/none]')
# parser.add_argument('--debug', action='store_true', help='Enable visualization')
# cfgs = parser.parse_args()

cfgs = {
    "checkpoint_path": "log/checkpoint_tracking.tar",
    "debug": True,
}

class AnyGrasp:
    def __init__(self, cfgs:dict):
        self.cfgs = cfgs
        self.AnyGraspTracker = AnyGraspTracker(cfgs)
        self.AnyGraspTracker.load_net()

        self.first_run = True

        self.thread = threading.Thread(target=self.get_grasp)
        self.thread.start()


    def create_point_cloud_from_depth_image(self, depth, camera, organized=True):
        assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
        xmap = np.arange(camera.width)
        ymap = np.arange(camera.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / camera.scale
        points_x = (xmap - camera.cx) * points_z / camera.fx
        points_y = (ymap - camera.cy) * points_z / camera.fy
        points = np.stack([points_x, points_y, points_z], axis=-1)
        if not organized:
            points = points.reshape([-1, 3])
        return points
    
    def get_data(self):
        colors = detector.color_image
        colors = colors[:, :, ::-1] / 255.0
        depths = detector.depth_image

        intrinsics_matrix = detector.intrinsics_matrix
        fx = intrinsics_matrix[0, 0]
        fy = intrinsics_matrix[1, 1]
        cx = intrinsics_matrix[0, 2]
        cy = intrinsics_matrix[1, 2]

        width, height = depths.shape[1], depths.shape[0]

        scale = 10000.0

        camera = CameraInfo(width, height, fx, fy, cx, cy, scale)

        # get point cloud
        points = self.create_point_cloud_from_depth_image(depths, camera)
        mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.5)
        points = points[mask]
        colors = colors[mask]

        return points, colors

    def get_grasp(self, curr_eef_pose):
        grasp_ids = [0]

        # for i in range(10000):
        #     # get prediction
        points, colors = self.get_data()
        target_gg, curr_gg, target_grasp_ids, corres_preds = self.anygrasp_tracker.update(points, colors, grasp_ids)

        if self.first_run:
            # select grasps on objects to track for the 1st frame
            grasp_mask_x = ((curr_gg.translations[:,0]>-0.18) & (curr_gg.translations[:,0]<0.18))
            grasp_mask_y = ((curr_gg.translations[:,1]>-0.12) & (curr_gg.translations[:,1]<0.12))
            grasp_mask_z = ((curr_gg.translations[:,2]>0.2) & (curr_gg.translations[:,2]<0.99))
            grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][:30:6]

            target_gg = curr_gg[grasp_ids]
            self.first_run = False
        else:
            grasp_ids = target_grasp_ids

        first_grasp = np.array([0])
        target_gg = target_gg[first_grasp]

        target_matrix = np.zeros((4,4))
        target_matrix[0:3, 3] = target_gg.translations[0]
        target_matrix[0:3, 0:3] = target_gg.rotations[0]

        final_target_pose,_,_,_,_ = self.get_target_pose(curr_eef_pose, target_matrix)

        return final_target_pose

    def get_target_pose(self, curr_eef_pose, T_cam2tag, pre_grasp_distance=0.25, require_matrices=False):
        ## Transfromation Matrix from World to Eff.
        T_w2eef = self.make_matrix(curr_eef_pose[0:3], np.array(curr_eef_pose[3:]))

        ## Transformation Matrix from Eff to Camera.
        trans_vector_eef2cam = np.array([-0.01, -0.08, 0.01])   ## 8 cm in Y and 1 cm in Z
        T_eef2cam = self.make_matrix_from_angle(trans_vector_eef2cam, 'x', -np.pi/12) ## 15 Degree

        T_w2cam = T_w2eef @ T_eef2cam

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
    

class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        points = points.reshape([-1, 3])
    return points

def get_data():
    colors = detector.color_image
    colors = colors[:, :, ::-1] / 255.0
    depths = detector.depth_image

    intrinsics_matrix = detector.intrinsics_matrix
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    width, height = depths.shape[1], depths.shape[0]

    scale = 10000.0

    camera = CameraInfo(width, height, fx, fy, cx, cy, scale)

    # get point cloud
    points = create_point_cloud_from_depth_image(depths, camera)
    mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.5)
    points = points[mask]
    colors = colors[mask]

    return points, colors

def demo():
    # intialization
    anygrasp_tracker = AnyGraspTracker(cfgs)
    anygrasp_tracker.load_net()

    grasp_ids = [0]
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=720, width=1280)

    for i in range(10000):
        # get prediction
        points, colors = get_data()
        target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, grasp_ids)

        if i == 0:
            # select grasps on objects to track for the 1st frame
            grasp_mask_x = ((curr_gg.translations[:,0]>-0.18) & (curr_gg.translations[:,0]<0.18))
            grasp_mask_y = ((curr_gg.translations[:,1]>-0.12) & (curr_gg.translations[:,1]<0.12))
            grasp_mask_z = ((curr_gg.translations[:,2]>0.2) & (curr_gg.translations[:,2]<0.99))
            grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][:30:6]

            target_gg = curr_gg[grasp_ids]
        else:
            grasp_ids = target_grasp_ids
        print(i, target_grasp_ids)

        first_grasp = np.array([0])
        target_gg = target_gg[first_grasp]

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.transform(trans_mat)
            grippers = target_gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            vis.add_geometry(cloud)
            for gripper in grippers:
                vis.add_geometry(gripper)
            vis.poll_events()
            vis.remove_geometry(cloud)
            for gripper in grippers:
                vis.remove_geometry(gripper)

        return target_gg

if __name__ == "__main__":
    while detector.color_image is None or detector.depth_image is None:
        pass
    demo()