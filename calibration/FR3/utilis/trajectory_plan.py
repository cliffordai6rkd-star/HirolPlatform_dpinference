# @Time    : 4/16/25 12:24 AM        # 文件创建时间
# @Author  : htLiang
# @Email   : haotianliang10@gmail.com
# @File    : trajectory_plan.py

# STEP:
# 1. 计算目标位姿IK
# 2. 计算笛卡尔轨迹
# 3. 计算关节轨迹
# 4. 控制器参数优化
# 5. 控制器运动
# Motivation:
# panda_py的cartesian控制器运动精度不高，所以需要自定义一个运动控制器，实现点到点的控制,以及轨迹规划。

import numpy as np
import panda_py
from panda_py import controllers, Panda, libfranka
import qpsolvers as qp
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from roboticstoolbox.tools.trajectory import ctraj,jtraj
import time
from utilis.iks import minkowski_ik,kdl_ik, analytical_ik, qp_ik
from utilis.move2pose import move2pose

# 获取当前脚本所在目录
project_root = Path(__file__).parent.resolve()

def IK_solve(target_pose, current_joints):
    """
    计算逆运动学
    :param panda: Panda对象
    :param target_pose: 目标位姿
    :param current_joints: 当前关节角度
    :return: 关节角度 np.array

    添加了异常检测,防止某种ik失败，提升鲁棒性
    """
    # 获取当前关节配置
    # 不需要在这里获取，因为已经传入了current_joints参数
    
    # 找到模型路径
    # xml_path = "/home/ryze/hirol/TicTacToe/franka_fr3/fr3_with_hand.xml"
    # urdf_file = "/home/ryze/hirol/TicTacToe/franka_fr3/fr3_franka_hand.urdf"
    xml_path = str(project_root / "assets" / "franka_fr3" / "fr3_with_hand.xml")
    urdf_file = str(project_root / "assets" / "franka_fr3" / "fr3_franka_hand.urdf")
    

    # 尝试各种IK方法

    try:
        # 尝试QP IK
        solution = qp_ik(target_pose, current_joints)
        if solution is not None:
            return solution
    except Exception as e:
        print(f"QP IK failed: {e}")

    try:
        # 尝试解析解IK 
        solution = analytical_ik(target_pose, current_joints)
        if solution is not None:
            return solution
    except Exception as e:
        print(f"Analytical IK failed: {e}")
    
    try:
        # 尝试Minkowski IK 
        if xml_path:
            solution = minkowski_ik(target_pose, xml_path, current_joints)
            if solution is not None:
                return solution
    except Exception as e:
        print(f"Minkowski IK failed: {e}")
    
    try:
        # 尝试KDL IK 
        if urdf_file:
            solution = kdl_ik(target_pose, urdf_file, current_joints)
            if solution is not None:
                return solution
    except Exception as e:
        print(f"KDL IK failed: {e}")
    

    
    # 如果所有方法都失败，返回None
    print("All IK methods failed to find a solution")
    return None




def calculate_cartesian_trajectory(init_joints, target_joints, num_points=10):
    """
    计算从初始位置到目标位置的线性轨迹
    :param init_joints: 初始关节角度 (由IK得到)
    :param target_joints: 目标关节角度
    :param num_points: 轨迹点数
    :return: 轨迹点 se3格式 后面要拼接轨迹,拼接以后再返回array
    """
    # 计算初始和目标位姿
    panda_rtb = rtb.models.Panda()
    init_pos = panda_rtb.fkine(init_joints)
    target_pos = panda_rtb.fkine(target_joints)
    # 计算笛卡尔轨迹
    traj = ctraj(init_pos, target_pos, num_points)

    return traj

def calculate_joint_trajectory(panda, cartesian_trajectory):
    """
    计算关节轨迹
    :param panda: Panda对象
    :param cartesian_trajectory: 笛卡尔轨迹
    :return: 关节轨迹Array
    
    提升鲁棒性：跳过IK失败的中间点，但确保最后一个点正确
    """
    joint_trajectory = []
    skipped_points = 0
    total_points = len(cartesian_trajectory)
    
    for i, pose in enumerate(cartesian_trajectory):
        # 检查是否为最后一个点
        is_last_point = (i == total_points - 1)
        
        # 计算逆运动学
        q = IK_solve(pose, panda.q)
        
        if q is None:
            if is_last_point:
                print("最后一个点的逆运动学计算失败，无法生成有效轨迹")
                return None
            else:
                # 中间点失败可以跳过
                print(f"第 {i+1}/{total_points} 个点的逆运动学计算失败，已跳过")
                skipped_points += 1
                continue

        # TODO 检查关节空间连续性 (如果不是第一个点)
        if joint_trajectory and len(joint_trajectory) > 0:
            prev_joints = np.array(joint_trajectory[-1])
            current_joints = np.array(q)
            joint_diff = np.abs(current_joints - prev_joints)
            
            # 如果关节差异过大，尝试找到更连续的解
            if np.max(joint_diff) > 0.5:  # 阈值可以根据需要调整
                print(f"点 {i+1} 关节空间不连续，尝试替代解...")
                
                # 尝试使用优化方法找到更连续的解
                # 这里可以添加更复杂的优化逻辑
                
                # 简单处理：重新计算，直到满足条件
                # try:
                #     while np.max(joint_diff) > 0.5:
                #         print(f"点 {i+1} 关节空间不连续，最大差异: {np.max(joint_diff):.4f}，重新求解...")
                #         # 尝试使用当前解作为初始值，寻找更连续的解
                #         q = IK_solve(pose, current_joints)
                #         if q is None:
                #             print("重新求解失败，使用原始解")
                #             break
                # except Exception as e:
                #     print(f"重新求解失败: {e}")

                # 如果关节差异过大，尝试迭代解决，但设置最大尝试次数
                # MAX_ATTEMPTS = 5  # 最大尝试次数，防止无限循环
                # attempt_count = 0
                
                # while np.max(joint_diff) > 0.5 and attempt_count < MAX_ATTEMPTS:
                #     print(f"点 {i+1} 关节空间不连续，最大差异: {np.max(joint_diff):.4f}，第 {attempt_count+1}/{MAX_ATTEMPTS} 次尝试重新求解...")
                    
                #     # 尝试使用当前解作为初始值，寻找更连续的解
                #     if hasattr(panda, 'set_joint_positions'):
                #         panda.set_joint_positions(current_joints)
                    
                #     # 重新计算IK
                #     new_q = IK_solve(panda, pose)
                    
                #     # 恢复原始关节位置
                #     if original_joints is not None and hasattr(panda, 'set_joint_positions'):
                #         panda.set_joint_positions(original_joints)
                    
                #     if new_q is None:
                #         print(f"重新求解尝试 {attempt_count+1} 失败，使用原始解")
                #         break
                    
                #     # 更新当前解和差异
                #     current_joints = np.array(new_q)
                #     joint_diff = np.abs(current_joints - prev_joints)
                #     q = new_q  # 更新q为新解
                    
                #     attempt_count += 1
                
                # # 如果尝试次数达到上限但仍不连续
                # if attempt_count == MAX_ATTEMPTS and np.max(joint_diff) > 0.5:
                #     print(f"达到最大尝试次数 ({MAX_ATTEMPTS})，使用最佳可得解，最终关节差异: {np.max(joint_diff):.4f}")           
        
        # 添加有效的关节角度到轨迹中
        joint_trajectory.append(q)
    
    if skipped_points > 0:
        print(f"注意：轨迹中有 {skipped_points}/{total_points} 个点被跳过")
    
    # 检查是否还有有效点
    if len(joint_trajectory) == 0:
        print("所有点都失败了，无法生成轨迹")
        return None
    
    return np.array(joint_trajectory)


import numpy as np
import time
from panda_py import controllers

def move2start(panda):
    """
    缓慢、平滑地移动到初始位置
    :param panda: Panda对象
    """
    print("[INFO] 正在缓慢移动至初始位置...")

    # 当前关节角度
    current_joints = panda.q

    # 目标初始关节角度
    start_q = [1.93522609e-03, -7.83874706e-01, 3.44093733e-04,
               -2.35621553e+00, -4.98195832e-03, 1.57430503e+00, 7.78670807e-01]

    # 生成笛卡尔轨迹（更密集）
    traj_cartesian = calculate_cartesian_trajectory(current_joints, start_q, num_points=200)

    # 关节轨迹计算（包含IK + 连续性处理）
    traj_joint = calculate_joint_trajectory(panda, traj_cartesian)
    # traj_joint = 
    if traj_joint is None:
        print("[ERROR] 关节轨迹计算失败，移动终止。")
        return
    # -----------------------方案1--controller---------------------------
    
    # # 控制器参数设置（软一些）
    # stiffness = np.array([150., 150., 150., 100., 50., 30., 20.])
    # damping = np.array([50., 50., 50., 30., 20., 20., 10.])
    # filter_coeff = 0.9
    # controller = controllers.JointPosition(
    #     stiffness=stiffness, damping=damping, filter_coeff=filter_coeff
    # )

    # panda.start_controller(controller)

    # # 控制上下文：低频率+长时间
    # with panda.create_context(frequency=30, max_runtime=200) as ctx:
    #     for joint_pos in traj_joint:
    #         if not ctx.ok():
    #             print("[WARNING] 安全机制触发，提前停止。")
    #             break
    #         controller.set_control(joint_pos)

    # print("[INFO] 初始位姿移动完成。")
    # time.sleep(2)

    # -------------------------方案2--move2jointposition--------------------
    # for joint_pos in traj_joint:
    #     panda.move_to_joint_position(joint_pos,speed_factor=0.4)  # 设置速度因子
    #     # time.sleep(0.1)
    # print("[INFO] 初始位姿移动完成。")
    # time.sleep(2)

    # -------------------------方案3--IntegratedVelocity-------------------- 
    for traj_c in traj_cartesian:
        move2pose(traj_c, frequency=30)
    print("[INFO] 初始位姿移动完成。")
        


def move_cartesian_trajectory(panda):

    
    return None
  
  
