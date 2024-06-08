import rtde_receive # type: ignore
import rtde_control # type: ignore

robot_reciever = rtde_receive.RTDEReceiveInterface("192.168.0.102")    
robot_controler = rtde_control.RTDEControlInterface("192.168.0.102")

print("FK  : ", robot_controler.getForwardKinematics())
print("IK : ", robot_controler.getInverseKinematics(x=robot_reciever.getActualTCPPose()))
print("Joints : ", robot_reciever.getActualQ())
print("Pose : ", robot_reciever.getActualTCPPose())