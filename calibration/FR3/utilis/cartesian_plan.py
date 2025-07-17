import roboticstoolbox as rtb
import panda_py 
from panda_py import controllers, Panda, libfranka
import qpsolvers as qp
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import SE3
import numpy as np
from roboticstoolbox.tools.trajectory import ctraj

panda = panda_py.Panda('192.168.1.102')

# 初始化roboticstoolbox模型
panda_rtb = rtb.models.Panda()
# 当前姿态（由机器人当前关节角 q 得到）
T_start = panda_rtb.fkine(panda.q)
print("当前姿态：", T_start)

# 目标姿态
T_end = panda_rtb.fkine([ 0.10718473, -0.18535104, -0.15528882 ,-2.37565005, -0.01084687 , 2.32120423,0.72854127]) 
print("目标姿态：\n", T_end)
print('-----------------------------------')
n_points = 100  # 设置轨迹点数
traj = ctraj(T_start, T_end,100)  # 生成10个点的轨迹
for i in range(len(traj)):
    print(traj[i])

# #运动规划与waypoints - 使用ctraj可以创建中间waypoints，例如: 
# # 创建预抓取位置(略高于目标)到最终抓取位置的轨迹
# pre_grasp = target_pose * SE3(0, 0, 0.1)  # 在z轴上偏移0.1米
# grasp_traj = ctraj(pre_grasp, target_pose, 20)  # 生成20个点的轨迹
# #安全轨迹规划 - 可以通过多段ctraj连接，避开工作台等障碍物: 
# # 从当前位置到安全高度
# traj1 = ctraj(current_pose, safe_height_pose, 10)
# # 从安全高度到目标上方
# traj2 = ctraj(safe_height_pose, pre_grasp_pose, 10)
# # 从目标上方到抓取位置
# traj3 = ctraj(pre_grasp_pose, grasp_pose, 10)
# # 组合轨迹
# full_traj = traj1 + traj2 + traj3

