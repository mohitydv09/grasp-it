import time
import numpy as np

import rtde_receive # type: ignore
import rtde_control # type: ignore
from gripper import RobotiqGripper

from detector import Detector, Open3dVisualizer

class RobotController:
    def __init__(self, lightning_ip = None,thunder_ip = None,need_control = False, need_gripper = False):
        self.ip = lightning_ip
        self.lightning_home = [-np.pi , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]
        if need_gripper:
            self.gripper = RobotiqGripper()
            self.gripper.connect(lightning_ip, 63352)
            self.gripper.activate()
            self.gripper.set_enable(True)
        if need_control:
            self.controller = rtde_control.RTDEControlInterface(lightning_ip)
        else :
            self.controller = None
        self.reciever = rtde_receive.RTDEReceiveInterface(lightning_ip)

    def get_eff_pose(self):
        return self.reciever.getActualTCPPose()
    
    def get_joint_angles(self):
        return self.reciever.getActualQ()
    
    def freeDrive(self):
        self.controller.teachMode()
        while True:
            user_input = input("Enter 'DONE' to Exit Free Drive Mode")
            if user_input == "DONE":
                break
        self.controller.endTeachMode()

    def go_to_home(self):
        self.controller.moveJ(self.lightning_home, 0.1, 0.1)

    def delta_pose(self, pose):
        curr_pose = self.get_eff_pose()
        position_distance= np.linalg.norm(np.array(pose[0:3]) - np.array(curr_pose[0:3]))
        orientation_distance = np.linalg.norm(np.array(pose[3:6]) - np.array(curr_pose[3:6])) ## Not Implemted for now.

        return position_distance


SPEED = 0.1
ACCELERATION = 0.1
DT = 0.3
LOOKAHEAD_TIME = 0.2
GAIN = 2000
LIGHTNING_HOME = [-np.pi , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]
LIGHTNING_HOME_EEF = [-0.2207, 0.1328, 0.5887, -1.2139, -1.2086, 1.206]

## Start RealSense Stream.
robot = RobotController(lightning_ip="192.168.0.102", need_gripper=True,need_control=True)
detector = Detector(visualization=False)

## Wait to start Recieving Data.
while True:
    color_image, depth_image, intrinsics_matrix = detector.get_frame()
    if color_image is not None:
        print("Camera Initialized")
        break

## Implement the Search mode.
# robot.freeDrive()
# input("Press Enter to Start Searching for Object")

## Move to First Location using MoveL
curr_eef_pose = robot.get_eff_pose()
first_target_pose = detector.get_target_pose(curr_eef_pose)[0]
robot.controller.moveL(first_target_pose,SPEED,ACCELERATION)
print("Moved to Initial Location")

start_time = time.time()
try:
    while True:
        color_image, depth_image, intrinsics_matrix = detector.get_frame()
        if color_image is not None:
            curr_eef_pose = robot.get_eff_pose()
            final_target_pose = detector.get_target_pose(curr_eef_pose)
            if final_target_pose[0] is not None:
                delta_pose = robot.delta_pose(final_target_pose[0])
                if delta_pose > 0.02:
                    ## Reset Timer.
                    start_time = time.time()
                    print("Following Target")
                    # print("Timer Reset")
                else:
                    print("Target Not Moving")

                if time.time() - start_time > 5:
                    robot.controller.servoStop(ACCELERATION)
                    print("No Pose Change for 5 Seconds, Grasping Object")
                    ## Get Grasp Position.
                    grasp_pose = detector.get_target_pose(curr_eef_pose, pre_grasp_distance=0.15)
                    robot.controller.moveL(grasp_pose[0], SPEED, ACCELERATION)
                    print("Closing Gripper")
                    robot.gripper.set(100)
                    time.sleep(0.1)
                    print("Moving to Home")
                    robot.go_to_home()
                    break
                else:
                    robot.controller.servoL(final_target_pose[0], SPEED, ACCELERATION, DT, LOOKAHEAD_TIME, GAIN)
                    time.sleep(0.01)
            else:
                print("No Tag in Frame Detected")
                continue
        else:
            print("No Frame from Camera")
            continue
    robot.controller.stopScript()
    detector.stop()
    print("Script Ended After Grasping Object")
        
except KeyboardInterrupt:
    print("Exiting")
    robot.controller.servoStop(ACCELERATION)
    print("Moving to Home")
    robot.controller.moveJ(LIGHTNING_HOME, SPEED, ACCELERATION)
    robot.controller.stopScript()
    detector.stop()
    print("Script Ended Gracefully after Keyboard Interrupt")