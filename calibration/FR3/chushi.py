import panda_py
from panda_py import libfranka
import logging

# Panda hostname/IP and Desk login information of your robot
hostname = '192.168.1.100'

gripper = libfranka.Gripper(hostname)
# panda-py is chatty, activate information log level

logging.basicConfig(level=logging.INFO)
print("1")
panda = panda_py.Panda(hostname)
print("2")
panda.set_default_behavior()
print("2.1")
pose = panda.get_pose()
print("Current Pose:", pose)

# safe_position = [0, -0.785, 0, -2.356, 0, 1.571, 0]  # 这是 Panda 机器人标准 home 位置
# print("2.2")
# panda.move_to_joint_position(safe_position)
# print("2.3")
# panda.move_to_start()
# print("3")

# pose = panda.get_pose()
# pose[2,3] -= .1
# q = panda_py.ik(pose)
# panda.move_to_joint_position(q)

# print ("pose: ", pose)



# panda.move_to_start()
# panda.get_state()
#
# print(panda.get_state())
#
# pose = panda.get_pose()
# print(pose)
# pose[2,3] -= .3
# panda.move_to_pose(pose)
# print(panda.get_pose())
# print(panda.get_position())

panda.move_to_start()
gripper.move(0.08, 0.3)  # 抓取
