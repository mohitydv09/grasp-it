import open3d as o3d
import numpy as np
import cv2``
import rtde_receive
import rtde_control

class Open3DVisualizer:
    def __init__(self):
        pass

    def create_grid(self,size=10, step=0.1):
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



# Add the base Frame.
base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
grid = create_grid(size=1, step=1)

# Add the Frame for the the Eff.
LIGHTNING_IP = "192.168.0.102"
reciever = rtde_receive.RTDEReceiveInterface(LIGHTNING_IP)
curr_eff_pose = reciever.getActualTCPPose()

## Make the End Effector Frame.
eff_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=curr_eff_pose[0:3])
rotation_of_eff = cv2.Rodrigues(np.array(curr_eff_pose[3:]))[0]
eff_frame.rotate(rotation_of_eff, center = curr_eff_pose[0:3])

## Make the Transformation Matrix.
T_w2eff = np.eye(4)
T_w2eff[0:3, 0:3] = rotation_of_eff
T_w2eff[0:3, 3] = curr_eff_pose[0:3]

# Add the Gripper Mesh
path_to_gripper_file = 'model.obj'
gripper_mesh = o3d.io.read_triangle_mesh(path_to_gripper_file)
gripper_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0] for i in range(len(gripper_mesh.vertices))]))
gripper_mesh.vertices = o3d.utility.Vector3dVector(np.array(gripper_mesh.vertices) * 0.001)
gripper_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi/2,0,0])) , center = [0,0,0])
gripper_mesh.rotate(rotation_of_eff, center = [0,0,0])
gripper_mesh.translate(curr_eff_pose[0:3])

## Make the Transformation Matrix fro Eff to Camera.
T_eff2cam = np.eye(4)
T_eff2cam[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(np.array([-np.pi/12, 0, 0]))
T_eff2cam[0:3, 3] = np.array([0, -0.1, 0.02])
T_w3cam = T_w2eff @ T_eff2cam

## Add the camera frame.
# camera_Visual = o3d.geometry.LineSet.create_camera_visualization(320,240)
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=T_w3cam[0:3,3])
camera_frame.rotate(T_w3cam[0:3,0:3])

## Detect the Tag.
rvecs, tvecs = get_aruco_pose()
rvecs = rvecs[0][0]
tvecs = tvecs[0][0]
print("rvecs: ", rvecs)
print("tvecs: ", tvecs)

## Make the Transformation Matrix from Camera to Tag.
T_cam2tag = np.eye(4)
T_cam2tag[0:3, 0:3] = cv2.Rodrigues(rvecs)[0]

## Rotate by 180degree about x axis.
T_cam2tag[0:3, 0:3] = T_cam2tag[0:3, 0:3] @ o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi, 0, 0]))
T_cam2tag[0:3, 3] = tvecs

## Make World to Tag Transformation Matrix.
T_w2tag = T_w3cam @ T_cam2tag

## Make the Tag Frame.
tag_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=T_w2tag[0:3,3])
tag_frame.rotate(T_w2tag[0:3,0:3])

## Make the target Location Transformation Vector.
T_w2target = np.eye(4)
T_w2target[0:3, 0:3] = T_w2tag[0:3,0:3]

approach_vector = T_w2target[0:3, 2]
norm_approach_vector = approach_vector/np.linalg.norm(approach_vector)
translate_vector = - 0.3 * norm_approach_vector

T_w2target[0:3, 3] = T_w2tag[0:3, 3] + translate_vector

## Make the target Location Coordinate Frame.
target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=T_w2target[0:3,3])
target_frame.rotate(T_w2target[0:3,0:3])

## Convert to robot input.
print(cv2.Rodrigues(T_w2target[0:3,0:3])[0])
robot_target = T_w2target[0:3,3].tolist() + cv2.Rodrigues(T_w2target[0:3,0:3])[0].reshape(-1).tolist()
print(robot_target)

controller = rtde_control.RTDEControlInterface(LIGHTNING_IP)
controller.moveL(robot_target,0.1,0.1)

# Plot the Visualization
o3d.visualization.draw_geometries([base_frame, grid, eff_frame, gripper_mesh,
                                    camera_frame, tag_frame, target_frame])

# viz = o3d.visualization.Visualizer()
# viz.create_window()

# viz.add_geometry(base_frame)
# viz.add_geometry(grid)
# viz.add_geometry(eff_frame)
# viz.add_geometry(gripper_mesh)
# viz.add_geometry(camera_frame)
# viz.add_geometry(tag_frame)
# viz.add_geometry(target_frame)

# try:
#     while True:
#         viz.update_geometry(base_frame)
#         viz.update_geometry(grid)
#         viz.poll_events()
#         viz.update_renderer()
# except KeyboardInterrupt:
#     print("Exiting...")

# viz.destroy_window()