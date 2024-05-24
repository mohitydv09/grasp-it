import threading
import queue
import time
import cv2

import rtde_receive # type: ignore
import rtde_control # type: ignore

from detector import Detector

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


## Start RealSense Stream.
robot = RobotController(lightning_ip="192.168.0.102", need_control=True)
detector = Detector(visualization=False)

while True:
    color_image, depth_image, intrinsics_matrix = detector.get_frame()
    if color_image is not None:
        curr_eef_pose = robot.get_eff_pose()
        final_target_pose = detector.get_target_pose(curr_eef_pose)
        if final_target_pose[0] is not None:
            robot.controller.moveL(final_target_pose[0],0.1,0.1)
            time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.stop()