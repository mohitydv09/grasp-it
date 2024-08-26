import vamp

## Dual arm offsets.
X_DISPLACEMENT = 0.016
Y_DISPLACEMENT = 0.710
Z_DISPLACEMENT = 0.005

## Table offsets.
TABLE_CENTER = [-0.70, 0 , 1]
TABLE_ORIENTATION = [0, 0, 0]
TABLE_DIMENSIONS = [0.1, 3, 2]

class VAMP:
    def __init__(self):
        self.robot = vamp.ur5
        self._x_displacement = X_DISPLACEMENT
        self._y_displacement = Y_DISPLACEMENT
        self._z_displacement = Z_DISPLACEMENT
        self._table_center = TABLE_CENTER
        self._table_orientation = TABLE_ORIENTATION
        self._table_dimensions = TABLE_DIMENSIONS

    def create_env(self, arm_name:str, other_arm_config:list[6], fos:float = 2.0, model_camera:bool=False) -> vamp.Environment:
        env = vamp.Environment()

        ## Add Table
        table = vamp.Cuboid(self._table_center, self._table_orientation, self._table_dimensions)
        env.add_cuboid(table)

        ## Arm Displacements
        x_displacement = -self._x_displacement if arm_name == "thunder" else self._x_displacement
        y_displacement = -self._y_displacement if arm_name == "thunder" else self._y_displacement
        z_displacement = -self._z_displacement if arm_name == "thunder" else self._z_displacement

        other_arm_spheres = self.robot.fk(other_arm_config)

        for sphere in other_arm_spheres:
            sphere_translated_to_position\
                = vamp.Sphere([sphere.x + x_displacement, sphere.y + y_displacement, sphere.z + z_displacement], fos*sphere.r)
            env.add_sphere(sphere_translated_to_position)
        
        # if model_camera:
        #     ## Self Camera Module.
        #     camera_module = vamp.Attachment([[0, -0.08, 0.05], [0, 0, 0, 1]]) ## Pos, Orientation
        #     camera_module.add_sphere(vamp.Sphere([0, 0, 0], 0.06))   ## 0.06 is attachment Radius
        #     env.attach(camera_module)

        #     ## Other arm's Camera Module.


        return env

    def pose_is_valid(self, arm:str, lightning_config:list[6], thunder_config:list[6], fos:float=2.0) -> bool:
        if arm == "lightning":
            curr_environment = self.create_env("lightning", thunder_config)
            return self.robot.validate(lightning_config, curr_environment)
        else: ## arm == "Thunder"
            curr_environment = self.create_env("thunder", lightning_config)
            return self.robot.validate(thunder_config, curr_environment)
        
    def get_path(self, arm:str, start_config:list[6], goal_config:list[6], other_arm_config:list[6]) -> list[list[6]]:
        environment = self.create_env(arm, other_arm_config)
        
        ## Plan Path
        rrtc_settings = vamp.RRTCSettings()
        planning_result = self.robot.rrtc(start_config, goal_config, environment,rrtc_settings)
        if not planning_result.solved:
            print("Path not found")
            return None
        
        ## Simplify Path.
        simplify_settings = vamp.SimplifySettings()
        simplify_result = self.robot.simplify(planning_result.path, environment, simplify_settings)
        simplified_path = simplify_result.path
        simplified_path.interpolate(self.robot.resolution())
        path_numpy = simplified_path.numpy()
        path_list = path_numpy.tolist()
        return path_list

if __name__=="__main__":
    print(vamp.ur5)