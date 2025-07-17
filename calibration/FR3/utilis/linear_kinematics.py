# --- coding: utf-8 ---
# @Time    : 4/16/25 1:24 AM        # 文件创建时间
# @Author  : htLiang
# @Email   : ryzeliang@163.com
import panda_py
from panda_py import controllers, Panda, libfranka
import qpsolvers as qp
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from roboticstoolbox.tools.trajectory import ctraj



panda = Panda("192.168.1.102")




# 初始化机器人和控制器
ctrl = controllers.IntegratedVelocity()
panda.move_to_start()
panda.start_controller(ctrl)

# 初始化roboticstoolbox模型
panda_rtb = rtb.models.Panda()

# 当前末端执行器位姿
T_current = panda_rtb.fkine(panda.q)

# 目标位姿
T_target = panda_rtb.fkine(panda.q) * sm.SE3(0.3, 0.2, 0.3)

# 生成笛卡尔轨迹 - 创建50个点的轨迹
traj = ctraj(T_current, T_target, 10)
print(traj.A)

# 关节数量
n = 7

# 控制循环
# with panda.create_context(frequency=20) as ctx:
#     # 轨迹索引
#     traj_idx = 0

#     while ctx.ok() and traj_idx < len(traj):
#         # 获取当前轨迹点作为目标
#         Tep = traj[traj_idx]

#         # 当前末端执行器位姿
#         Te = panda_rtb.fkine(panda.q)

#         # 计算误差变换
#         eTep = Te.inv() * Tep

#         # 计算所需的末端执行器空间速度
#         v, arrived = rtb.p_servo(Te, Tep, 1.0)

#         # 二次规划求解器参数设置
#         Y = 0.01
#         Q = np.eye(n + 6)
#         Q[:n, :n] *= Y

#         e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
#         Q[n:, n:] = (1 / e) * np.eye(6)

#         # 等式约束
#         Aeq = np.c_[panda_rtb.jacobe(panda.q), np.eye(6)]
#         beq = v.reshape((6,))

#         # 目标函数线性分量
#         c = np.r_[-panda_rtb.jacobm().reshape((n,)), np.zeros(6)]

#         # 速度边界
#         lb = -np.r_[panda_rtb.qdlim[:n], 10 * np.ones(6)]
#         ub = np.r_[panda_rtb.qdlim[:n], 10 * np.ones(6)]

#         # 求解关节速度
#         qd = qp.solve_qp(Q, c, None, None, Aeq, beq, lb=lb, ub=ub, solver='daqp')

#         # 应用关节速度到机械臂
#         ctrl.set_control(qd[:n])

#         # 如果到达当前点，则移动到下一个轨迹点
#         if arrived:
#             traj_idx += 1