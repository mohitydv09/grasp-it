import vamp

import rtde_receive # type: ignore
import rtde_control # type: ignore
from gripper import RobotiqGripper

import numpy as np


class RobotController:
    def __init__(self, arm, need_control = False, need_gripper = False):
        if arm == 'lightning':
            self.ip = "192.168.0.102"
            self.home = [-np.pi , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]
        elif arm == 'thunder':
            self.ip = "192.168.0.101"
            self.home = [-np.pi , -np.pi*13/18, np.pi*13/18, -np.pi, -np.pi/2, 0.0]
        else:
            print("Invalid Arm Name. Please enter 'lightning' or 'thunder'.")
            return
    
        if need_gripper:
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.ip, 63352)
            self.gripper.activate()
            self.gripper.set_enable(True)
        if need_control:
            self.controller = rtde_control.RTDEControlInterface(self.ip)
        else :
            self.controller = None
        self.reciever = rtde_receive.RTDEReceiveInterface(self.ip)

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
        self.controller.moveJ(self.home, 0.1, 0.1)

    def delta_pose(self, pose):
        curr_pose = self.get_eff_pose()
        position_distance= np.linalg.norm(np.array(pose[0:3]) - np.array(curr_pose[0:3]))
        orientation_distance = np.linalg.norm(np.array(pose[3:6]) - np.array(curr_pose[3:6])) ## Not Implemted for now.
        return position_distance
    
class VampPlanner:
    def __init__(self, arm, visualization = False):
        self.arm = arm
        self.visualization = visualization
        self.robot = 'ur5'
        self.robot_urdf_path = 'vamp/resources/ur5/ur5.urdf'
        self.planning_algorithm = 'rrtc'

        self.robot_module, self.panner_module,\
        self.plan_settings, self.simp_settings =  \
        vamp.configure_robot_and_planner_with_kwargs(self.robot,
                                                    self.planning_algorithm)
        if self.visualization:
            self.enviroment, self.visualization = self.create_env()
        else:
            self.enviroment = self.create_env()

    def create_env(self, other_arm_spheres = None):
        env = vamp.Environment()
        if self.visualization:
            sim = vamp.Simulation()

        ## Add Table

        ## Wall

        ## Add other robot spheres.
        for sphere in other_arm_spheres:
            sphere_translated_to_position\
                 = vamp.Sphere([sphere.x, sphere.y + 0.74350, sphere.z], sphere.r)
            env.add_sphere(sphere_translated_to_position)
            if self.visualization:
                sim.add_sphere(sphere_translated_to_position)
        if self.visualization:
            return env, sim
        else:
            return env

class VampSimulation(VampPlanner):
    def __init__(self):
        super().__init__(robot, planner)
        self.sim = vamp.Simulation()

lightning = RobotController('lightning', need_gripper=False, need_control=True)
thunder = RobotController('thunder', need_gripper=False, need_control=False)

lightning_vamp_planner = VampPlanner(lightning, planner)