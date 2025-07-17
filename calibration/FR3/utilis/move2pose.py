import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import panda_py
from panda_py import controllers
import qpsolvers as qp
    
def move2pose(target_pose, frequency=20, robot_hostname='192.168.1.102',threshold=1e-3):
    """
    将Panda机器人移动到指定的目标位姿
    
    参数:
        target_pose: 目标位姿，sm.SE3类型
        robot_hostname: 机器人主机名，如果为None则从命令行获取
        frequency: 控制循环频率，默认20Hz
    
    返回:
        bool: 是否成功到达目标位置
    """
    
    # 初始化机器人硬件和控制器
    panda = panda_py.Panda(robot_hostname)
    # panda.move_to_start()
    ctrl = controllers.IntegratedVelocity()

    panda.start_controller(ctrl)
    
    # 初始化roboticstoolbox模型
    panda_rtb = rtb.models.Panda()
    
    # 设置期望的末端执行器位姿
    if isinstance(target_pose, sm.SE3):
        Tep = target_pose
    else:
        # 如果提供的不是SE3对象，则基于当前位置计算目标位姿
        Tep = panda_rtb.fkine(panda.q) * target_pose
    
    # Panda机器人控制的关节数
    n = 7
    
    arrived = False
    
    with panda.create_context(frequency=frequency) as ctx:
        while ctx.ok() and not arrived:
            # Panda末端执行器的当前位姿
            Te = panda_rtb.fkine(panda.q)
            
            # 从当前末端执行器位姿到期望位姿的变换
            eTep = Te.inv() * Tep
            
            # 空间误差
            e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
            
            # 计算机器人接近目标所需的末端执行器空间速度
            # 增益设置为1.0
            v, arrived = rtb.p_servo(Te, Tep, 0.6, threshold=threshold)
            
            # 控制最小化的增益项(lambda)
            Y = 0.01
            
            # 目标函数的二次项
            Q = np.eye(n + 6)
            
            # Q的关节速度分量
            Q[:n, :n] *= Y
            
            # Q的松弛分量
            Q[n:, n:] = (1 / e) * np.eye(6)
            
            # 等式约束
            Aeq = np.c_[panda_rtb.jacobe(panda.q), np.eye(6)]
            beq = v.reshape((6,))
            
            # 目标函数的线性分量：可操作性雅可比
            c = np.r_[-panda_rtb.jacobm().reshape((n,)), np.zeros(6)]
            
            # 关节速度和松弛变量的上下界
            lb = -np.r_[panda_rtb.qdlim[:n], 10 * np.ones(6)]
            ub = np.r_[panda_rtb.qdlim[:n], 10 * np.ones(6)]
            
            # 求解关节速度dq
            qd = qp.solve_qp(Q, c, None, None, Aeq, beq, lb=lb, ub=ub, solver='daqp')
            
            # 将关节速度应用到Panda机器人
            ctrl.set_control(qd[:n])
    
    return arrived

if __name__ == '__main__':
    # 使用示例:
    panda_rtb = rtb.models.Panda()
    start_q = [1.93522609e-03, -7.83874706e-01, 3.44093733e-04,
               -2.35621553e+00, -4.98195832e-03, 1.57430503e+00, 7.78670807e-01]
    target = panda_rtb.fkine(start_q)

    # 方式1：直接指定目标位姿
    # target = s
    move2pose(target)

# # 方式2：基于当前位置的相对位姿
# relative_move = sm.SE3(0.3, 0.2, 0.3)
# move_to_pose(relative_move, "robot-hostname")