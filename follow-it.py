import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import queue
import time
import rtde_receive
import rtde_control
from scipy.spatial.transform import Rotation as R

## Connect to the robot. Setup RTDE receive and control interface.
LIGHTNING_IP = "192.168.0.102"
reciever = rtde_receive.RTDEReceiveInterface(LIGHTNING_IP)
controller = rtde_control.RTDEControlInterface(LIGHTNING_IP)

# while True:
#     time.sleep(0.1)
#     print(reciever.getActualTCPPose())


# def move_to_zero_angle():
#     current_eff_ps = reciever.getActualTCPPose()
#     print("Current Pose: ", current_eff_ps)
#     current_eff_ps[3] += 3.00
#     current_eff_ps[4] += 0.09
#     current_eff_ps[5] += -0.73
    # controller.moveL(current_eff_ps,0.1,0.1)

controller.moveL([-0.400303615706161, 0.10037057867020892, 0.9578918260330468,
                   -1.4658670169242356, -1.642197804985105, 1.0122059450126883],0.1,0.1)

## Configure the colow stream.
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

def get_position(q):
    ## Start streaming
    pipeline.start(config)
    while True:
        time.sleep(0.1)
        print("Current Queue Size: ", q.qsize())
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        profile = rs.video_stream_profile(color_frame.get_profile())
        intrinsics = profile.get_intrinsics()
        intrinsics_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                        [0, intrinsics.fy, intrinsics.ppy],
                                        [0, 0, 1]])
        
        ## Detect the Aruco Marker.
        dict = cv2.aruco.DICT_6X6_50
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict)
        aruco_parameter = cv2.aruco.DetectorParameters()

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=aruco_parameter)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, intrinsics_matrix, None)
            ## Convert Tvecs to rotation matrix.
            rot_mat = cv2.Rodrigues(rvecs[0])[0]
            rot_mat_x_180 = np.array([[1,0,0],
                                    [0,-1,0],
                                    [0,0,-1]])
            
            rot_mat_z_90 = np.array([[0,-1,0],
                                    [1,0,0],
                                    [0,0,1]])
            
            rot_mat = rot_mat_z_90 @ rot_mat_x_180 @ rot_mat

            ## Form the composite Matrix.
            composite_matrix = np.zeros((4,4))
            composite_matrix[:3,:3] = rot_mat
            composite_matrix[:3,3] = tvecs[0][0]
            composite_matrix[3,3] = 1

            ## Matrix to transform this to the EFF frame.
            cam2gripper = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.96592583, -0.25881905, 0.10176896],
                [0.0, 0.25881905, 0.96592583, 0.00656339],
                [0.0, 0.0, 0.0, 1.0]
])

            rotation_matrix_x_15 = np.array([[1,0,0,0],
                            [0,np.cos(np.pi/12),-np.sin(np.pi/12),0],
                            [0,np.sin(np.pi/12),np.cos(np.pi/12),0],
                            [0,0,0,1]])
            translate_matrix = np.array([[1,0,0,0],
                            [0,1,0,0.1],
                            [0,0,1,-0.02],
                            [0,0,0,1]])
            camera_to_gripper =  rotation_matrix_x_15 @ translate_matrix

            transformed_tag = np.linalg.inv(camera_to_gripper) @ composite_matrix

            ## Get Position.
            eff_pos = transformed_tag[:3,3].reshape(1,3)
            eff_rot = cv2.Rodrigues(transformed_tag[:3,:3])[0].reshape(1,3)
            final_pose = np.concatenate((eff_pos, eff_rot), axis=1).tolist()

            print("Final Pose: ", final_pose)


            x0 = tvecs[0][0][0]
            y0 = tvecs[0][0][1]
            ## Draw markers on image.
            for i in range(len(rvecs)):
                color_image = cv2.drawFrameAxes(color_image, intrinsics_matrix, None, rvecs[i], tvecs[i], 0.01)
            q.put((x0, y0))
            
        ## Display the image
        cv2.imshow("image", color_image)
        
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    q.put((None, None))
    pipeline.stop()
    cv2.destroyAllWindows()
    return x0, y0


def move_robot(q):
    while True:
        x0, y0 = q.get()
        if x0 is None:
            print("None object in Queue. Exiting")
            break
        current_eff_ps = reciever.getActualTCPPose()
        current_eff_ps[1] += x0
        current_eff_ps[2] -= y0
        controller.moveL(current_eff_ps,0.1,0.1)

q = queue.LifoQueue()

def main():
    camera_thread = threading.Thread(target=get_position, args=(q,))
    robot_thread = threading.Thread(target=move_robot, args=(q,))

    camera_thread.start()
    robot_thread.start()

    camera_thread.join()
    robot_thread.join()

    print("Done")

# get_position(q)
# move_to_zero_angle()
# main()