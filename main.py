import time
import copy
import numpy as np

import scripts.utils as utils
from scripts.robot import RobotController
from scripts.detector import ArUcoDetector, RealSense
from scripts.planner import VAMP

## Robot Motion Constants.
SPEED = 0.3
ACCELERATION = 0.2
DT = 0.2
LOOKAHEAD_TIME = 0.2
GAIN = 500

tag_detector = ArUcoDetector(visualization=True)
print("Detector Initialized")
lightning = RobotController('lightning', need_control=True, need_gripper=True)
print("Lightning Initialized")
thunder = RobotController('thunder', need_control=False, need_gripper=False)
print("Thunder Initialized")
vamp_panner = VAMP()

## Wait for Camera to Initialize and get the first frame.
while True:
    frame = tag_detector.color_frame
    if frame is not None:
        break
print("Camera Initialized")

## Move to first location using MoveL.
curr_eef_pose = lightning.reciever.getActualTCPPose()
first_target_pose = tag_detector.get_target_pose_with_curr_eff(curr_eef_pose, pre_grasp_distance=0.25)
input("Press Enter to go to Initial Target")
lightning.controller.moveL(first_target_pose, SPEED, ACCELERATION)

## Offset Constacts
OFFSET = 0.00
OFFSET_MAX = 0.15
OFFSET_MIN = 0.00
STEP_SIZE = 0.002
PREV_TARGET = first_target_pose

## Stability Constants
DISTANCE_THRESHOLD = 0.009
OBJECT_STABLE = 0.0
OBJECT_STABLE_MIN = 0.00 
OBJECT_STABLE_MAX = 0.11 ## 11 Centimeter Movement.
STABLE_COUNTER = 0

print("Entering Follow Mode")
while True:
    try:
        ## Get Target Pose in EEF Frame.
        raw_target_eff_pose = tag_detector.get_target_pose_with_curr_eff(lightning.reciever.getActualTCPPose(), pre_grasp_distance=0.25-OBJECT_STABLE)

        ## Item can be grasped.
        if OBJECT_STABLE == OBJECT_STABLE_MAX:
            STABLE_COUNTER += 1
            if STABLE_COUNTER > 30:
                ## Grasp the object.
                lightning.controller.servoStop( a = 1.0)
                print("Object seems stable. Grasping the Object")
                lightning.grasp_object()
                print("Object Grasped")
                print("Moving to Home")
                lightning.go_home()
                break
            pass
        else:
            STABLE_COUNTER = 0

        ## Marker Not Found.
        if raw_target_eff_pose is None:
            print("Marker Not Found, Moving back.")
            OFFSET = min(OFFSET + STEP_SIZE, OFFSET_MAX)
            OBJECT_STABLE = max(OBJECT_STABLE - 0.004, OBJECT_STABLE_MIN)
            target_eff_pose = utils.move_pose_back(PREV_TARGET, distance = OFFSET)
        else:
            print("Marker Visible, following the target.")
            ## Stability Module.
            if utils.get_distance_between_poses(PREV_TARGET, raw_target_eff_pose) < DISTANCE_THRESHOLD - (OBJECT_STABLE/20):
                OBJECT_STABLE = min(OBJECT_STABLE + 0.001, OBJECT_STABLE_MAX)
            else:
                OBJECT_STABLE = max(OBJECT_STABLE - 0.004, OBJECT_STABLE_MIN)
            PREV_TARGET = raw_target_eff_pose
            OFFSET = max(OFFSET - STEP_SIZE, OFFSET_MIN)
            target_eff_pose = utils.move_pose_back(raw_target_eff_pose, distance = OFFSET)

        ## Collition Module.
        while True:
            if lightning.controller.getInverseKinematicsHasSolution(target_eff_pose, lightning.home):
                target_joint_config = lightning.controller.getInverseKinematics(target_eff_pose, lightning.home)
            else:
                # print("No IK Solution Found")
                break

            lightning_joint_config = copy.deepcopy(target_joint_config)
            lightning_joint_config[0] += 0.5*np.pi
            thunder_joint_config = thunder.reciever.getActualQ()
            thunder_joint_config[0] += 1.5*np.pi
            pose_validity = vamp_panner.pose_is_valid("lightning", lightning_joint_config, thunder_joint_config, fos=1.5)
            if pose_validity:
                break
            else:
                print("Pose is in Collision")
                OFFSET = min(OFFSET + STEP_SIZE, OFFSET_MAX)
                OBJECT_STABLE = max(OBJECT_STABLE - 0.004, OBJECT_STABLE_MIN)
                if (OFFSET == OFFSET_MAX):
                    print("No Valid Pose Found, Trying to get new target")
                    break
                if raw_target_eff_pose is None:
                    target_eff_pose = utils.move_pose_back(PREV_TARGET, distance = OFFSET)
                else:
                    target_eff_pose = utils.move_pose_back(raw_target_eff_pose, distance = OFFSET)

        lightning.controller.servoJ(target_joint_config, SPEED, ACCELERATION, DT, LOOKAHEAD_TIME, GAIN)

    except KeyboardInterrupt:
        break

print("Reached Home")
print("Exiting Follow Mode")
lightning.controller.servoStop( a = 1) 
tag_detector.stop()
