import time
import numpy as np

import rtde_receive # type: ignore
import rtde_control # type: ignore

from detector import Detector, Open3dVisualizer

class RobotController:
    def __init__(self, lightning_ip = None,thunder_ip = None,need_control = False):
        self.ip = lightning_ip
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


SPEED = 0.1
ACCELERATION = 0.1
dT = 0.2
LOOKAHEAD_TIME = 0.2
GAIN = 2000
lIGHTNING_HOME = [-np.pi , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]

## Start RealSense Stream.
robot = RobotController(lightning_ip="192.168.0.102", need_control=True)
detector = Detector(visualization=False)

## Run Free Drive
# robot.freeDrive()

## Wait to start Recieving Data.
while True:
    color_image, depth_image, intrinsics_matrix = detector.get_frame()
    if color_image is not None:
        break

## Move to First Location using MoveL
curr_eef_pose = robot.get_eff_pose()
first_target_pose = detector.get_target_pose(curr_eef_pose)[0]
robot.controller.moveL(first_target_pose,SPEED,ACCELERATION)

def pose_is_diffrent(final_pose, curr_pose):
    return True
    diff = 0
    for i in range(6):
        diff += abs(final_pose[i] - curr_pose[i])
    if diff > 0.1:
        return True
    return False

try:
    while True:
        color_image, depth_image, intrinsics_matrix = detector.get_frame()
        if color_image is not None:
            curr_eef_pose = robot.get_eff_pose()
            final_target_pose = detector.get_target_pose(curr_eef_pose)
            if final_target_pose[0] is not None:
                if pose_is_diffrent(final_target_pose[0], curr_eef_pose):
                    robot.controller.servoL(final_target_pose[0], SPEED, ACCELERATION, dT, LOOKAHEAD_TIME, GAIN)
                    time.sleep(0.01)
                else:
                    robot.controller.servoL(curr_eef_pose, SPEED, ACCELERATION, dT, LOOKAHEAD_TIME, GAIN)
            else:
                print("No Tag in Frame Detected")
                continue
        else:
            print("No Frame from Camera")
            continue
        
except KeyboardInterrupt:
    print("Exiting")
    robot.controller.servoStop(ACCELERATION)
    robot.controller.moveJ(lIGHTNING_HOME, SPEED, ACCELERATION)
    robot.controller.stopScript()
    detector.stop()