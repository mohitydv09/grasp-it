import rtde_control
import rtde_receive
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import open3d as o3d
from gripper import RobotiqGripper

lightning_ip = "192.168.0.102"

lightning_reciever = rtde_receive.RTDEReceiveInterface(lightning_ip)
lightning_control = rtde_control.RTDEControlInterface(lightning_ip)

current_eff_pose = lightning_reciever.getActualTCPPose()

## Start RealSense Stream.

pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
i = 0
while True:
    i+=1
    print(i)
    if i % 200 == 0:
        print("Current Eff Pose: ", current_eff_pose)
        current_eff_pose = lightning_reciever.getActualTCPPose()
        current_eff_pose[2] += 0.05
        lightning_control.moveL(current_eff_pose, 0.1, 0.1)
    if i == 800:
        break
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow("Color Image", color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




current_eff_pose[2] += 0.1
current_eff_pose[1] += 0.1

lightning_control.moveL(current_eff_pose, 0.1, 0.1)

gripper = RobotiqGripper()
gripper.connect(lightning_ip, 63352)
gripper.activate()
gripper.set_enable(True)

gripper.set(200) # 0 for open, 255 for close

gripper.set(100)