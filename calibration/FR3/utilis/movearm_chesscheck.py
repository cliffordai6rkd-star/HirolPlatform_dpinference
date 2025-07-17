import panda_py
from panda_py import libfranka
import numpy as np
import time
import matplotlib.pyplot as plt
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R
from panda_py import controllers
import qpsolvers as qp
import roboticstoolbox as rtb
from panda_py import libfranka
import logging
import torch
from utilis.move2pose import move2pose
from utilis.IK_solver import IK_solve
import sys

# Panda hostname/IP and Desk login information of your robot
hostname = '192.168.1.102'
username = 'franka'
password = 'franka123'
# 从相机坐标系到机械臂坐标系的变换矩阵
# 修正后的变换矩阵定义（添加缺失的逗号）
trans = [[-0.00196906 ,-0.99948439 , 0.03204814  ,0.49130777],
 [-0.99982274 , 0.0025678  , 0.01865203 ,-0.20790817],
 [-0.01872471, -0.03200573 ,-0.99931227 , 0.84799009],
 [ 0.         , 0.  ,        0.    ,      1.        ]]
x_temp = 0.00
y_temp = -0.0
z_temp = -0.00

panda_rtb = rtb.models.Panda()

joint_speed_factor = 0.2
cart_speed_factor = 0.2
impedance = [
    [150, 0, 0, 0, 0, 0],
    [0, 150, 0, 0, 0, 0],
    [0, 0, 150, 0, 0, 0],
    [0, 0, 0, 80, 0, 0],
    [0, 0, 0, 0, 80, 0],
    [0, 0, 0, 0, 0, 80]
]
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)
# panda-py is chatty, activate information log level

logging.basicConfig(level=logging.INFO)


def initial():
    gripper.move(0.04, 0.02)
    # Tar = panda_rtb.fkine([-2.15540842e-02, -1.25595494e+00, -1.30071604e-02 ,-2.53076737e+00,
    #                                 1.39331738e-04,  1.32695355e+00 , 7.85398163e-01])
    # move2pose(Tar)
    # panda.move_to_joint_position([-2.15540842e-02, -1.25595494e+00, -1.30071604e-02 ,-2.53076737e+00,
    #                                 1.39331738e-04,  1.32695355e+00 , 7.85398163e-01],speed_factor=0.4)
    panda.move_to_joint_position([ 9.04551228e-02 ,-8.19963613e-01, -1.53815292e-03 ,-2.63172359e+00,
  2.89953182e-03,  1.81390849e+00  ,9.23664549e-01],speed_factor=0.1)
    return panda.get_pose()


def place_chess(loca_chess, loca_move, grab_angle=0.0):
    """
    抓取并移动棋子
    
    参数:
        loca_chess: 棋子的3D坐标(米)
        loca_move: 目标位置的3D坐标(米)
        grab_angle: 抓取角度(弧度)，相对于机器人基座x轴的旋转角度
    
    返回:
        是否成功
    """
    chess_arm = trans @ np.append(loca_chess, 1)  # 齐次坐标变换
    move_arm = trans @ np.append(loca_move, 1)
    
    # 提取前三个元素（x,y,z）并转换为list类型
    target_pose = chess_arm[:3].tolist()
    release_pose = move_arm[:3].tolist()
    release_pose[2] = 0.24
    T_0 = panda.get_pose()
    print(T_0)
    print(target_pose)
    print(release_pose)
    tar_pose = T_0.copy()

    # safe point
    panda.move_to_joint_position([-0.02011124, -0.50858959 ,-0.09952221, -2.55542135, -0.0331041 ,  2.15538415,
  0.68253017],speed_factor=0.1)

    # 调整抓取姿态，根据grab_angle参数旋转夹爪方向
    # 创建绕Z轴旋转的旋转矩阵
    rot_z = R.from_euler('z', grab_angle).as_matrix()
    
    # 获取当前末端姿态的旋转部分
    current_rotation = tar_pose[0:3, 0:3]
    
    # 应用Z轴旋转到当前姿态
    new_rotation = current_rotation @ rot_z
    
    # 更新tar_pose的旋转部分
    tar_pose[0:3, 0:3] = new_rotation

    """
    此处需要进行误差补偿
    tar_pose[0, 3]、tar_pose[1, 3]、tar_pose[2, 3]分别为xyz坐标
    """
    tar_pose[0, 3] = target_pose[0] + x_temp
    tar_pose[1, 3] = target_pose[1] + y_temp
    tar_pose[2, 3] = 0.218 + 0.06
    
    joints = IK_solve(tar_pose,panda.q)
    if not np.any(joints):
        return False
    else:
        # 预抓取点
        panda.move_to_joint_position(joints,speed_factor=0.1)
    
    """
    此处需要进行抓取深度调整
    """
    tar_pose[2, 3] = 0.218+ z_temp
    
    joints = IK_solve(tar_pose,panda.q)
    if not np.any(joints):
        return False
    else:
        # 抓取点
        panda.move_to_joint_position(joints,speed_factor=0.1)

    gripper.grasp(0.02, 0.02, 20, 0.04, 0.04)
    tar_pose[2, 3] = 0.218 + 0.06

    joints = IK_solve(tar_pose,panda.q)
    if (not np.any(joints)):
        return False
    else:
        # 中间路点1
        panda.move_to_joint_position(joints,speed_factor=0.1)
        # 判断是否夹住物体
        state = gripper.read_once()
        if state.width < 0.021:
            print(state.width)
            print("[✗] 抓取失败，夹爪未检测到物体")
            sys.exit(1)  # 非0退出表示异常退出
    
    """
    此处需要进行误差补偿
    tar_pose[0, 3]、tar_pose[1, 3]、tar_pose[2, 3]分别为xyz坐标
    """
    tar_pose[0, 3] = release_pose[0] + x_temp-0.003
    tar_pose[1, 3] = release_pose[1] + y_temp
    tar_pose[2, 3] = release_pose[2] + 0.06
    
    joints = IK_solve(tar_pose,panda.q)
    if not np.any(joints):
        return False
    else:
        # 中间路点2（预释放点）
        panda.move_to_joint_position(joints,speed_factor=0.1)
    
    """
    此处需要进行释放深度调整
    """
    tar_pose[2, 3] = release_pose[2]
    joints = IK_solve(tar_pose,panda.q)
    if not np.any(joints):
        return False
    else:
        panda.move_to_joint_position(joints,speed_factor=0.1)
    gripper.move(0.03, 0.02)
    tar_pose[2, 3] = release_pose[2] + 0.06 
    joints = IK_solve(tar_pose,panda.q)
    if not np.any(joints):
        return False
    else:
        panda.move_to_joint_position(joints,speed_factor=0.1)
        # gripper.move(0.04, 0.02)
    # panda.move_to_start(speed_factor=0.4)
        Tar = panda_rtb.fkine([-2.15540842e-02, -1.25595494e+00, -1.30071604e-02 ,-2.53076737e+00,
         1.39331738e-04,  1.32695355e+00 , 7.85398163e-01])
        
    # panda.move_to_joint_position([-2.15540842e-02, -1.25595494e+00, -1.30071604e-02 ,-2.53076737e+00,
    #                             1.39331738e-04,  1.32695355e+00 , 7.85398163e-01],speed_factor=0.1)
    panda.move_to_joint_position([ 9.04551228e-02 ,-8.19963613e-01, -1.53815292e-03 ,-2.63172359e+00,
                2.89953182e-03,  1.81390849e+00  ,9.23664549e-01],speed_factor=0.1)
    gripper.move(0.04, 0.02)
        # move2pose(Tar)
    return True



