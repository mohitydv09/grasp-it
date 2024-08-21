import time

import utils
from robot import RobotController
from tag_detector import ArUcoDetector, RealSense
from planner import VAMP

SPEED = 0.1
ACCELERATION = 0.1
DT = 0.3
LOOKAHEAD_TIME = 0.2
GAIN = 200

tag_detector = ArUcoDetector(visualization=False)
lightning = RobotController('lightning', need_control=True, need_gripper=False)
vamp_panner = VAMP()

## Wait for Camera to Initialize and get the first frame.
while True:
    frame = tag_detector.color_frame
    if frame is not None:
        break
print("Camera Initialized")

## Move to first location using MoveL.
curr_eef_pose = lightning.reciever.getActualTCPPose()
first_target_pose = tag_detector.get_target_pose_with_curr_eff(curr_eef_pose)

## Move to the first target.
input("Press Enter to Move to First Target")
lightning.controller.moveL(first_target_pose, 0.1, 0.1)

## Start the Following Mode.
while True:
    try:
        time_start = time.time()
        target_TCPPose = tag_detector.get_target_pose_with_curr_eff(lightning.reciever.getActualTCPPose())
        pose_time = time.time()-time_start
        print("Time to get Target Pose: ", pose_time)
        target_joint_config = lightning.controller.getInverseKinematics(target_TCPPose)
        ik_time = time.time()-time_start-pose_time
        print("Time to get Inverse Kinematics: ", ik_time)

        ## Use VAMP Planner.

        path = vamp_panner.get_path("lightning", lightning.reciever.getActualQ(), target_joint_config, [0.0,0.0,0.0,0.0,0.0,0.0])
        planning_time = time.time()-time_start-pose_time-ik_time
        print("Time to Plan Path: ", planning_time)
        next = path[1] if len(path) > 1 else path[0]
        print("Length of Path: ", len(path))
        lightning.controller.servoJ(next, SPEED, ACCELERATION, DT, LOOKAHEAD_TIME, GAIN)
        # time.sleep(1)
        # lightning.controller.servoL(next, SPEED, ACCELERATION, DT, LOOKAHEAD_TIME, GAIN)
        # for waypoint in path:
        #     lightning.controller.servoL(next, SPEED, ACCELERATION, DT, LOOKAHEAD_TIME, GAIN)
    except KeyboardInterrupt:
        break

print("Exiting Follow Mode")
tag_detector.stop()
