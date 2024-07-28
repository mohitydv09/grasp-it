import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from graspnetAPI import GraspGroup
import time

from tracker import AnyGraspTracker

from detector import Detector
detector = Detector()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--filter', type=str, default='oneeuro', help='Filter to smooth grasp parameters(rotation, width, depth). [oneeuro/kalman/none]')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
cfgs = parser.parse_args()

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

def get_data(data_dir, index):
    # load image
    # colors = np.array(Image.open(os.path.join(data_dir, 'color_%03d.png'%index)), dtype=np.float32) / 255.0
    # depths = np.load(os.path.join(data_dir, 'depth_%03d.npy'%index))

    # set camera intrinsics
    # width, height = depths.shape[1], depths.shape[0]
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000.0
    colors = detector.color_image
    colors = colors[:, :, ::-1] / 255.0
    depths = detector.depth_image

    intrinsics_matrix = detector.intrinsics_matrix
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    # colors, depths, fx, fy, cx, cy = get_images()
    width, height = depths.shape[1], depths.shape[0]

    scale = 10000.0

    camera = CameraInfo(width, height, fx, fy, cx, cy, scale)

    # get point cloud
    points = create_point_cloud_from_depth_image(depths, camera)
    mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.5)
    points = points[mask]
    colors = colors[mask]

    return points, colors

def demo(data_dir_list, indices):
    # intialization
    anygrasp_tracker = AnyGraspTracker(cfgs)
    anygrasp_tracker.load_net()

    grasp_ids = [0]
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=720, width=1280)
    start_time = time.time()
    for i in range(len(indices)):
        # get prediction
        points, colors = get_data(data_dir_list, indices[i])
        target_gg, curr_gg, target_grasp_ids, corres_preds = anygrasp_tracker.update(points, colors, grasp_ids)
        # print("Target gg: ", target_gg)
        # print("Curr gg: ", type(curr_gg))
        # print("Grasp IDS: ", type(target_grasp_ids))
        # print("Corres Preds: ", type(corres_preds))
        # print("length of target gg: ", len(target_gg))
        # print("length of curr gg: ", len(curr_gg))
        # print("length of target grasp ids: ", len(target_grasp_ids))
        # print("length of corres preds: ", len(corres_preds))

        if i == 0:
            # select grasps on objects to track for the 1st frame
            grasp_mask_x = ((curr_gg.translations[:,0]>-0.18) & (curr_gg.translations[:,0]<0.18))
            grasp_mask_y = ((curr_gg.translations[:,1]>-0.12) & (curr_gg.translations[:,1]<0.12))
            grasp_mask_z = ((curr_gg.translations[:,2]>0.2) & (curr_gg.translations[:,2]<0.99))
            grasp_ids = np.where(grasp_mask_x & grasp_mask_y & grasp_mask_z)[0][:30:6]
            #### Pick Only One Grasp ID
            # print("Grasp IDS: ", grasp_ids)
            # print("Type of Grasp IDS: ", type(grasp_ids))

            target_gg = curr_gg[grasp_ids]
        else:
            grasp_ids = target_grasp_ids
        print(i, target_grasp_ids)
        print("Type of curr gg: ", type(curr_gg))
        print("Type of grasp ids: ", type(grasp_ids))   
        print("Type of target gg: ", type(target_gg))
        first_grasp = np.array([0])
        print("first grasp: ", first_grasp, type(first_grasp))
        target_gg = target_gg[first_grasp]
        print("target gg: ", target_gg)

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
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

if __name__ == "__main__":
    data_dir = "example_data"
    data_dir_list = [x for x in range(10000)]
    while detector.color_image is None or detector.depth_image is None:
        pass
    demo(data_dir, data_dir_list)