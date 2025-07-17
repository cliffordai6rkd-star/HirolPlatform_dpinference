import panda_py
from panda_py import libfranka
import logging
from panda_py import constants
import numpy as np
from matplotlib import pyplot as plt
from utilis.IK_solver import IK_solve
# 初始化机械臂和夹爪
hostname = '192.168.1.100'
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

panda.move_to_start()
gripper.move(0.0, 0.2)
pose_start = panda.get_pose()

pose_zhong = np.array(
[[ 0.61534204 , 0.39490714,  0.68220416 , 0.06858511],
 [ 0.77874987, -0.4385905 , -0.44853876 , 0.01186015],
 [ 0.1220771 ,  0.80727116 ,-0.57741706 , 0.30176098],
 [ 0.  ,        0.  ,        0.   ,       1.        ]]

)

pose1 = np.array(
[[ 1,  0, 0,  0.38611272],
 [ 0, -1,  0 , 0.01496891],
 [ 0, 0, -1,  0.22228209],
 [ 0., 0., 0., 1. ]]
)

pose = np.array(
[[ 1,  0, 0,  0.38611272],
 [ 0, -1,  0 ,0.01496891],
 [ 0, 0, -1,  0.27228209],
 [ 0., 0., 0., 1. ]]
)

panda.move_to_pose(pose1)

q_0 = panda.q

q_1 = IK_solve(pose, q_0)
print(q_1)

panda.move_to_joint_position(q_1)


# panda.move_to_pose(posebai)
# panda.move_to_pose(posebai)
# panda.move_to_pose(posebai)

# panda.move_to_pose(posebai1)
# panda.move_to_pose(posebai1)
# panda.move_to_pose(posebai1)