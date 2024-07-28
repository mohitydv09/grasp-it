from urdfpy import URDF

ur5 = URDF.load("/home/rpmdt05/Code/grasp_it/vamp/resources/ur5/ur5.urdf")

for joint in ur5.joints:
    print(joint.name)
# 

import rtde_control
import rtde_receive

lightning_ip = "192.168.0.102"

# controller = rtde_control.RTDEControlInterface(lightning_ip)
reciever = rtde_receive.RTDEReceiveInterface(lightning_ip)

print(reciever.getActualQ())