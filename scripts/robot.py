import numpy as np
import rtde_control # type: ignore
import rtde_receive # type: ignore
from scripts.gripper import RobotiqGripper

THUNDER_IP = '192.168.0.101'
LIGHTNING_IP = '192.168.0.102'
THUNDER_HOME = [-np.pi , -np.pi*13/18, np.pi*13/18, -np.pi, -np.pi/2, 0.0]
LIGHTNING_HOME = [-np.pi , -np.pi*5/18, -np.pi*13/18, 0.0, np.pi/2, 0.0]


class RobotController:
    def __init__(self, arm : str, need_control: bool = False, need_gripper: bool = False):
        self._ip = THUNDER_IP if arm == 'thunder' else LIGHTNING_IP
        self.home = THUNDER_HOME if arm == 'thunder' else LIGHTNING_HOME
        self.gripper = self._init_gripper() if need_gripper else None
        self.reciever = rtde_receive.RTDEReceiveInterface(self._ip)
        self.controller = rtde_control.RTDEControlInterface(self._ip) if need_control else None

    def _init_gripper(self) -> RobotiqGripper:
        gripper = RobotiqGripper()
        gripper.connect(self._ip, 63352)
        gripper.activate()
        gripper.set_enable(True)
        return gripper

    def get_eff_pose(self) -> list[6]:
        return self.reciever.getActualTCPPose()
    
    def get_joint_angles(self) -> list[6]:
        return self.reciever.getActualQ()
    
    def freeDrive(self):
        self.controller.teachMode()
        while True:
            user_input = input("Enter 'DONE' to Exit Free Drive Mode")
            if user_input == "DONE":
                break
        self.controller.endTeachMode()

    def go_home(self):
        self.controller.moveJ(self.home, 0.1, 0.1)

if __name__=="__main__":
    robot = RobotController('thunder', False, False)

