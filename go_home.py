
from scripts.robot import RobotController

lightning = RobotController('lightning', need_control=True, need_gripper=False)
lightning.go_home()