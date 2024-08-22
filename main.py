import time
import numpy as np
import utils
import sys
import copy
from collections import deque
from robot import RobotController
from tag_detector import ArUcoDetector, RealSense
from planner import VAMP


SPEED = 0.1
ACCELERATION = 0.1
DT = 0.3
LOOKAHEAD_TIME = 0.2
GAIN = 200

PREV_TARGETS = deque(maxlen=2)

tag_detector = ArUcoDetector(visualization=True)
lightning = RobotController('lightning', need_control=True, need_gripper=False)
thunder = RobotController('thunder', need_control=False, need_gripper=False)
vamp_panner = VAMP()
# pose_estimator = PoseEstimator()

## Wait for Camera to Initialize and get the first frame.
while True:
    frame = tag_detector.color_frame
    if frame is not None:
        break
print("Camera Initialized")

# lightning.go_home()

## Move to first location using MoveL.
curr_eef_pose = lightning.reciever.getActualTCPPose()
first_target_pose = tag_detector.get_target_pose_with_curr_eff(curr_eef_pose)
input("Press Enter to Move to First Target")
lightning.controller.moveL(first_target_pose, 0.1, 0.1)

input("Press Enter to Start Following the Target")
OFFSET = 0.00
OFFSET_MAX = 0.15
OFFSET_MIN = 0.00
STEP_SIZE = 0.005
while True:
    try:
        print("Offset: {}".format(OFFSET))
        raw_target_eff_pose = tag_detector.get_target_pose_with_curr_eff(lightning.reciever.getActualTCPPose(), pre_grasp_distance=0.25)

        ## Marker was not found in the frame.
        if raw_target_eff_pose is None:
            print("Marker Not Found at {}".format(time.time()))
            OFFSET = min(OFFSET + STEP_SIZE, OFFSET_MAX)
            target_eff_pose = utils.move_pose_back(PREV_TARGETS[-1], distance = OFFSET)
        else:
            PREV_TARGETS.append(raw_target_eff_pose)
            OFFSET = max(OFFSET - STEP_SIZE, OFFSET_MIN)
            target_eff_pose = utils.move_pose_back(raw_target_eff_pose, distance = OFFSET)

        ## Collition Module.
        while True:
            try:
                target_joint_config = lightning.controller.getInverseKinematics(target_eff_pose, lightning.home)
            except:
                print("No IK Solution Found")
                break
            # if lightning.controller.getInverseKinematicsHasSolution(target_eff_pose, lightning.home):
            #     target_joint_config = lightning.controller.getInverseKinematics(target_eff_pose, lightning.home)
            # else:
            #     print("No IK Solution Found")
            #     break
            #     # target_eff_pose = utils.jitter_pose(target_eff_pose, distance = 0.01)
            #     # continue

            lightning_joint_config = copy.deepcopy(target_joint_config)
            lightning_joint_config[0] += 0.5*np.pi
            thunder_joint_config = thunder.reciever.getActualQ()
            thunder_joint_config[0] += 1.5*np.pi
            pose_validity = vamp_panner.pose_is_valid("lightning", lightning_joint_config, thunder_joint_config, fos=1.5)
            if pose_validity:
                # OFFSET = max(OFFSET - 0.01, 0.00)
                break
            else:
                print("Pose is in Collision")
                OFFSET = min(OFFSET + STEP_SIZE, OFFSET_MAX)
                if (OFFSET == OFFSET_MAX):
                    print("No Valid Pose Found, Trying to get new target")
                    break
                    # print("No Valid Pose Found")
                    # lightning.controller.servoStop()
                    # tag_detector.stop()
                    # sys.exit()
                if raw_target_eff_pose is None:
                    target_eff_pose = utils.move_pose_back(PREV_TARGETS[-1], distance = OFFSET)
                else:
                    target_eff_pose = utils.move_pose_back(raw_target_eff_pose, distance = OFFSET)

        lightning.controller.servoJ(target_joint_config, SPEED, ACCELERATION, DT, LOOKAHEAD_TIME, GAIN)

    except KeyboardInterrupt:
        break

print("Exiting Follow Mode")
tag_detector.stop()
