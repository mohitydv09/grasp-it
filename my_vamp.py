import vamp
from vamp import pybullet_interface as vpb
import numpy as np
import time
import matplotlib.pyplot as plt
from urdfpy import URDF
# import vamp.build

robot_urdf_path = 'vamp/resources/ur5/ur5.urdf'
sim = vpb.PyBulletSimulator(robot_urdf_path, vamp.ROBOT_JOINTS['ur5'], True)

robot_ = URDF.load(robot_urdf_path)
for joint in robot_.joints:
    print(joint.name)

for joint in vamp.ROBOT_JOINTS['ur5']:
    print(joint)

lightning_module, planner_module, plan_settings, simp_settings = vamp.configure_robot_and_planner_with_kwargs('ur5', 'rrtc')
#lightning_start = [-np.pi/2 , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]
lightning_start = [-2.6247716585742396, -0.9443705838969727, -1.6287304162979126,
                  -0.5083511632731934, 1.0703270435333252, -0.38782006898988897]
lightning_start[0] += np.pi/2  #

# get_from_rtde + pi/2 for the first joint
# lightning_goal = [0.0, 0.1, 2.5, 0.0, 0.3, 3.5] # Save from the scene.
lightning_goal = [-2.831836525593893, -0.6182095569423218, -2.6038901805877686,
                -0.023354844456054735, 1.6746294498443604, -0.09153873125185186]
lightning_goal[0] += np.pi/2  # get_from_rtde + pi/2 for the first joint

## Build the environment with only sphere of thunder.
thunder_module = vamp.configure_robot_and_planner_with_kwargs('ur5', 'rrtc')[0]
## Make config.
# thunder_config = [-np.pi , -np.pi*13/18, np.pi*13/18, -np.pi, -np.pi/2, 0.0] ## Get from thunder + 1.5 Pi for the first joint
thunder_config = [-2.241622273121969, -0.9576506775668641, 0.8925235907184046,
                 -2.9204904041686, -0.804232422505514, -1.0714810530291956]
# thunder_config = [0,0,0,0,0,0]
thunder_config[0] += 1.5*np.pi 

## Get the spheres.
thunder_spheres = thunder_module.fk(thunder_config)
## Add the spheres to the PyBullet and Enviroment

def make_camera_sphere(sphere_wrist, sphere_right, sphere_left):
    right_vector = np.array([sphere_right.x - sphere_wrist.x, sphere_right.y - sphere_wrist.y, sphere_right.z - sphere_wrist.z])
    left_vector = np.array([sphere_left.x - sphere_wrist.x, sphere_left.y - sphere_wrist.y, sphere_left.z - sphere_wrist.z])
    cross_vector = np.cross(right_vector, left_vector)
    cross_vector_norm = cross_vector / np.linalg.norm(cross_vector)
    displacement = 0.05
    radius = sphere_wrist.r
    camera_pos = np.array([sphere_wrist.x, sphere_wrist.y, sphere_wrist.z]) + displacement * cross_vector_norm
    camera_sphere = vamp.Sphere(camera_pos, radius)
    return camera_sphere


# ## Print Info.
# for i,sphere in enumerate(thunder_spheres):
#     print(type(sphere))
#     print(f"Sphere {i}: {sphere.x:.4f}, {sphere.y:.4f}, {sphere.z:.4f}, {sphere.r:.4f}")



lightning_env = vamp.Environment()
# camera_sphere = make_camera_sphere(thunder_spheres[21], thunder_spheres[25], thunder_spheres[35])
# thunder_spheres.append(camera_sphere)
for sphere in thunder_spheres:
    ## Translate the spheres to the desired position.
    sphere_translated_to_position = vamp.Sphere([sphere.x, sphere.y + 0.74350, sphere.z], sphere.r)
    lightning_env.add_sphere(sphere_translated_to_position)
    sim.add_sphere(sphere.r, [sphere.x, sphere.y + 0.74350, sphere.z])

sim.client.addUserDebugLine([0, 0, 0], [0.5, 0, 0], [1, 0, 0], 3)
sim.client.addUserDebugLine([0, 0, 0], [0, 0.5, 0], [0, 1, 0], 3)
sim.client.addUserDebugLine([0, 0, 0], [0, 0, 0.5], [0, 0, 1], 3)

## Plan the path.
planner = planner_module(lightning_start, lightning_goal, lightning_env, plan_settings)
simplify = lightning_module.simplify(planner.path, lightning_env, simp_settings)

simplified_path = simplify.path
simplified_path.interpolate(lightning_module.resolution())

print(type(simplified_path))

print(f"Path Length: {len(simplified_path)}")
# path_numpy = np.zeros((len(simplified_path), 6))
# for i, path in enumerate(simplified_path):
#     path_numpy[i] = path.to_list()

# np.save('lightning_path.npy', path_numpy)

sim.animate(simplified_path)


# curr_config = robot_module.Configuration().to_list()
# curr_config = [-np.pi/2 , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]
# #         self.start = [0 , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]

# curr_config = [+np.pi/2 , -np.pi*13/18, np.pi*13/18, -np.pi, -np.pi/2, 0.0]
# collision_spheres = robot_module.fk(curr_config)

# sim.add_cuboid(half_extents = [0.1, 2.0, 1.0],
#                position = [-1.0, 0.0, 0.0],
#                rot_xyzw = [0, 0, 0, 1],
#                name = 'table',
#                color = 'brown')

## Add Axis to the PyBullet


# while True:
#     sim.stepSimulation()
# sim.animate()

# class myPlanner:
#     def __init__(self, robot : str = 'ur5', planner : str = 'rrtc'):
#         self.robot = robot
#         self.planner = planner
#         self.problem = myProblem()
#         robot_module, planner_module, plan_settings, simp_settings = self.configure_robot_and_planner_with_kwargs()
#         self.robot_module = robot_module
#         self.planner_module = planner_module
#         self.plan_settings = plan_settings
#         self.simp_settings = simp_settings

#     def configure_robot_and_planner_with_kwargs(self, **kwargs):
#         return vamp.configure_robot_and_planner_with_kwargs(self.robot, self.planner, **kwargs)
    
#     def find_path(self):
#         raw_path = self.planner_module(self.problem.start, self.problem.goal, self.problem.env, self.plan_settings)
#         if raw_path.solved:
#             print("Path Found.")
#             return raw_path
#         else:
#             print("Path not found.")
#             return raw_path
        
#     def simplify_path(self, path):
#         simplified_path = self.robot_module.simplify(path, self.problem.env, self.simp_settings)
#         path = simplified_path.path
#         path.interpolate(self.robot_module.resolution())
#         return path


# class myProblem:
#     def __init__(self):
#         self.start = [0 , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]  ## Lightning
#         # self.start = [-np.pi , -np.pi*13/18, np.pi*13/18, -np.pi, -np.pi/2, 0.0] ## THunder
#         self.goal = [0.0, 0.1, 2.5, 0.0, 0.3, 3.5]
#         self.valid = True
#         self.env = self.build_env()

#     def build_env(self):
#         env = vamp.Environment()
#         rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
#         traslate_matrix = np.array([0, 0.74350, 10])
#         robot2config = [0.0, 0.0, 1.57, 0.0, 0.0, 0.0]
#         robot2module = vamp.configure_robot_and_planner_with_kwargs('ur5', 'rrtc')[0]
#         robot_spheres = robot2module.fk(robot2config)
#         for sphere in robot_spheres:
#             curr_pos = np.array([sphere.x, sphere.y, sphere.z])
#             new_pos = traslate_matrix + (rot_matrix @ curr_pos)
#             new_sphere = vamp.Sphere(new_pos, sphere.r)
#             env.add_sphere(new_sphere)
#         return env

# class Visualizer:
#     def __init__(self, robot_urdf_path : str = 'resources/ur5/ur5.urdf'):
#         self.robot_urdf_path = robot_urdf_path
#         self.sim = vpb.PyBulletSimulator(self.robot_urdf_path, vamp.ROBOT_JOINTS['ur5'], True)
#         self.build_sim()

#     def build_sim(self):
#         robot2config = [0 , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]
#         robot2module = vamp.configure_robot_and_planner_with_kwargs('ur5', 'rrtc')[0]
#         robot_spheres = robot2module.fk(robot2config)
#         self.sim.client.addUserDebugLine([0, 0, 0], [0.5, 0, 0], [1, 0, 0], 3)
#         self.sim.client.addUserDebugLine([0, 0, 0], [0, 0.5, 0], [0, 1, 0], 3)
#         self.sim.client.addUserDebugLine([0, 0, 0], [0, 0, 0.5], [0, 0, 1], 3)
#         # rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
#         # traslate_matrix = np.array([0, 0, 0])
#         for sphere in robot_spheres:
#             # curr_pos = np.array([sphere.x, sphere.y, sphere.z])
#             # new_pos = traslate_matrix + (rot_matrix @ curr_pos)
#             self.sim.add_sphere(sphere.r, [sphere.x, sphere.y, sphere.z])

#     def start_animate(self, path):
#         self.sim.animate(path)

# ## Initialize the planner
# planner = myPlanner()
# problem = myProblem()

# ## Find the path
# raw_path = planner.find_path()
# simplified_path = planner.simplify_path(raw_path.path)

# ## Visualize the path
# visualizer = Visualizer()
# visualizer.start_animate(simplified_path)


# ## Make the Enviroment items.

# ## Table Vamp
# table_vamp = vamp.Cuboid([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [2.0, 1.0, 0.5])
# wall_vamp = vamp.Cuboid([0.0, 0.0, 0.0], [np.pi/2, 0.0, 0.0], [2.0, 2.0, 0.1])

# ## Table PyBullet
# sim.add_cuboid(half_extents = [2.0, 1.0, 0.5],
#                position = [0.0, 0.0, -1.0],
#                rot_xyzw = [0.0, 0.0, 0.0, 1.0],
#                name = 'table')

# sim.add_cuboid(half_extents = [2.0, 2.0, 0.1],
#                 position = [0.0, 0.0, 0.0],
#                 rot_xyzw = [0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)],
#                 name = 'wall')

# ## Add the Other robot
# robot2config = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# robot2module = vamp.configure_robot_and_planner_with_kwargs('ur5', 'rrtc')[0]
# robot_spheres = robot2module.fk(robot2config)

# def vamp2pybullet(robot_spheres):
#     for sphere in robot_spheres:
#         sim.add_sphere(sphere.r, [sphere.x, sphere.y, sphere.z])