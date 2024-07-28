import vamp
from vamp import pybullet_interface as vpb

import pybullet as pb # type: ignore
from pybullet_utils.bullet_client import BulletClient # type: ignore
from threading import Thread
import time

import rtde_receive # type: ignore
import rtde_control # type: ignore
from gripper import RobotiqGripper

import numpy as np

class PyBulletSim:
    def __init__(self, urdf_path, lightning = None, thunder = None, visualize = True):
        self.urdf_path = urdf_path
        self.visualize = visualize
        self.lightning_client = lightning
        self.thunder_client = thunder
        self.client = BulletClient(connection_mode=pb.GUI if visualize else pb.DIRECT)
        self.client.setGravity(0, 0, -9.81)
        self.lightning_id = self.load_lightning()
        self.thunder_id = self.load_thunder()
        self.build_env()
        self.set_camera_view([1, 0.36, 2], [0, 0.36, 1])

        self.running = True
        self.simulation_thread = Thread(target=self.run_simulation)
        self.simulation_thread.start()

    def build_env(self):
        ## Wall 
        wall_center = [0,1,0.8544]
        wall_orientation = [0, 0, 0]
        wall_dimensions = [1, 2, 0.05]
        self.add_cuboid(wall_dimensions, wall_center, wall_orientation, color = [1, 1, 1, 1])

        ## Table
        table_center = [-0.71, 1 , 1]
        table_orientation = [0, 0, 0]
        table_dimensions = [0.1, 2, 1]
        self.add_cuboid(table_dimensions, table_center, table_orientation, color = [1/2, 1/4, 0, 1])

    def add_sphere(self, radius, position, color = [1, 0, 0, 1]):
        visual_shape_id = self.client.createVisualShape(pb.GEOM_SPHERE, radius=radius, rgbaColor=color)
        self.client.createMultiBody(baseMass=0,baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def add_cuboid(self, half_extents, position, rot_xyzw, color = [1, 0, 0, 1]):
        visual_shape_id = self.client.createVisualShape(pb.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        self.client.createMultiBody(baseMass=0,baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=rot_xyzw)

    def set_camera_view(self, camera_position = [1,1,1], camera_target=[0,0,0]):
        dx = camera_position[0] - camera_target[0]
        dy = camera_position[1] - camera_target[1]
        dz = camera_position[2] - camera_target[2]

        import math

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw = math.atan2(dz, dx)
        pitch = math.atan2(math.sqrt(dz * dz + dx * dx), dy) + math.pi

        self.client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw = math.degrees(yaw),
            cameraPitch = math.degrees(pitch),
            cameraTargetPosition=camera_target
        )

    def load_lightning(self):
        lightning_id = self.client.loadURDF(self.urdf_path, 
                                            basePosition = [0.0, 0.0, 0.0],
                                            baseOrientation = [0.0, 0.0, 0.0, 1.0],
                                            useFixedBase=True,
                                            flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_USE_SELF_COLLISION,
                                            )
        return lightning_id

    def load_thunder(self):
        thunder_id = self.client.loadURDF(self.urdf_path, 
                                          ## Hand Cal : y = 736.6
                                          ## -0.01261909 -0.74350812  0.00501703
                                            basePosition = [0.016, 0.725, 0.00501703],
                                            baseOrientation = [0.0, 0.0, 0.0, 1.0],
                                            useFixedBase=True,
                                            flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_USE_SELF_COLLISION,
                                            )
        return thunder_id

    def set_joint_positions(self, lightning_joint_config, thunder_joint_config):
        for i, joint_angle in enumerate(lightning_joint_config):
            self.client.resetJointState(self.lightning_id, i + 1, joint_angle) ## Robot Joint mapping is +1
        for i, joint_angle in enumerate(thunder_joint_config):
            self.client.resetJointState(self.thunder_id, i + 1, joint_angle)

    def run_simulation(self):
        while self.running:
            lightning_current_joint_config = self.lightning_client.get_joint_angles()
            thunder_current_joint_config = self.thunder_client.get_joint_angles()

            ## Modigy Joint Angle to match the simulation.
            lightning_current_joint_config[0] += np.pi/2
            thunder_current_joint_config[0] -= np.pi/2

            self.set_joint_positions(lightning_current_joint_config, thunder_current_joint_config)
            self.client.stepSimulation()
            time.sleep(1/240)

    def stop(self):
        self.running = False
        self.simulation_thread.join()
        self.client.disconnect()

    def animate_path(self, path):
        ## Stop the simulation.
        self.running = False
        while True:
            try:
                for pose in path:
                    pose = pose.to_list()
                    # pose[0] += np.pi/2
                    thunder_pose = self.thunder_client.get_joint_angles()
                    thunder_pose[0] -= np.pi/2
                    self.set_joint_positions(pose, thunder_pose)
                    self.client.stepSimulation()
                    time.sleep(1/120)
            except KeyboardInterrupt:
                break
        self.running = True

class RobotController:
    def __init__(self, arm, need_control = False, need_gripper = False):
        self.name = arm
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
    def __init__(self, arm, other_arm):
        self.arm = arm
        self.other_arm = other_arm
        self.robot = 'ur5'
        self.robot_urdf_path = 'vamp/resources/ur5/ur5.urdf'
        self.planning_algorithm = 'rrtc'

        self.robot_module, self.planner_module,\
        self.plan_settings, self.simp_settings =  \
        vamp.configure_robot_and_planner_with_kwargs(self.robot,
                                                    self.planning_algorithm)
        # self.enviroment = self.create_env()

    def create_env(self):
        if self.arm.name == 'lightning':
            env = vamp.Environment()

            # Add Table
            table_center = [-0.71, 1 , 1]
            table_orientation = [0, 0, 0]
            table_dimensions = [0.1, 2, 1]
            table = vamp.Cuboid(table_center, table_orientation, table_dimensions)
            env.add_cuboid(table)

            # Wall
            wall_center = [0,1,0.8544]
            wall_orientation = [0, 0, 0]
            wall_dimensions = [1, 2, 0.05]
            wall = vamp.Cuboid(wall_center, wall_orientation, wall_dimensions)
            env.add_cuboid(wall)

            ## Add thunder robot spheres.
            thunder_config = self.other_arm.get_joint_angles()
            thunder_config[0] += 1.5*np.pi
            thunder_spheres = self.robot_module.fk(thunder_config)

            for sphere in thunder_spheres:
                sphere_translated_to_position\
                    = vamp.Sphere([sphere.x + 0.016, sphere.y + 0.725, sphere.z + 0.005], sphere.r)
                env.add_sphere(sphere_translated_to_position)

            # Add the end-effector attachment.
            camera_module_lightning = vamp.Attachment([0, -0.08, 0.05], [0, 0, 0, 1])
            camera_module_lightning.add_spheres([vamp.Sphere([0, 0, 0], 0.06)])
            env.attach(camera_module_lightning)

        return env

    def generate_path(self, goal):
        start = self.arm.get_joint_angles()
        start[0] += np.pi/2

        curr_environment = self.create_env()
        planner = self.planner_module(start, goal, curr_environment, self.plan_settings)
        if not planner.solved:
            print("Planner not solved. Exiting.")
            return None
        simplify = self.robot_module.simplify(planner.path, curr_environment, self.simp_settings)

        simplified_path = simplify.path
        simplified_path.interpolate(self.robot_module.resolution())

        return simplified_path
    
    def pose_is_valid(self):
        curr_pose = self.arm.get_joint_angles()
        curr_pose[0] += np.pi/2

        curr_environment = self.create_env()
        return self.robot_module.validate(curr_pose, curr_environment)

## Initialize the Robot Controllers
lightning = RobotController('lightning', need_gripper=False, need_control=False)
thunder = RobotController('thunder', need_gripper=False, need_control=False)

# Activate Simulation
# sim = PyBulletSim('vamp/resources/ur5/ur5.urdf', lightning, thunder)

# ## Initialize the Vamp Planner
# lightning_vamp_planner = VampPlanner(lightning, thunder)

# start_time = time.time()
# counter = 0
# while True:
#     print(lightning_vamp_planner.pose_is_valid())
#     counter += 1
#     fps = counter/(time.time() - start_time)
#     print(f"FPS: {fps}")

## Old goal 
# lightning_goal = [-2.8318, -0.6182, -2.60389,
#                 -0.02335, 1.67462, -0.09153]

# ## Near the Table.
# lightning_goal = [-3.436719, -1.2075, -1.892,
#                     0.0010856, 1.864951, -0.0248]

# lightning_goal[0] += np.pi/2  # get_from_rtde + pi/2 for the first joint


sim = PyBulletSim('vamp/resources/ur5/ur5.urdf', lightning, thunder)