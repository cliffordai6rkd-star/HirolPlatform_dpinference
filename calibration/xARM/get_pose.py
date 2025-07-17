import os
import sys
import time
import numpy as np
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from euler_to_matrix import euler_to_matrix, robot_pose_to_matrix
from xarm.wrapper import XArmAPI


def pose_to_matrix(pose_data):
    """
    将xArm的位姿数据转换为4x4齐次变换矩阵
    
    参数:
        pose_data: 包含位置和欧拉角的列表 [x, y, z, roll, pitch, yaw]，
                  其中位置单位为毫米，角度单位为度
    
    返回:
        4x4的齐次变换矩阵
    """
    # 提取位置和欧拉角
    x, y, z = pose_data[0], pose_data[1], pose_data[2]
    roll, pitch, yaw = pose_data[3], pose_data[4], pose_data[5]
    
    # 将角度转换为弧度
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # 计算旋转矩阵的元素
    # 根据ZYX欧拉角顺序 (yaw->pitch->roll)
    
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    
    # 构建旋转矩阵 R = Rz(yaw) * Ry(pitch) * Rx(roll)
    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr
    
    # 构建4x4齐次变换矩阵
    matrix = np.array([
        [r11, r12, r13, x],
        [r21, r22, r23, y],
        [r31, r32, r33, z],
        [0, 0, 0, 1]
    ])
    
    return matrix


# left 192.168.11.11
# right 192.168.11.12
ip = '192.168.11.12'
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
# 正常模式
arm.set_mode(0)
arm.set_state(state=0)

# 访问pose

ret , pose = arm.get_position()
print('pose:', pose)

# 转换为4x4矩阵
if ret == 0:  # 确保返回正常
    transform_matrix = pose_to_matrix(pose)
    print("\n4x4变换矩阵:")
    print(transform_matrix)
    
    # 格式化打印以便更清晰地查看
    print("\n格式化变换矩阵:")
    for row in transform_matrix:
        print("[{:10.6f}, {:10.6f}, {:10.6f}, {:10.6f}]".format(*row))
else:
    print("获取机器人位姿失败，错误代码:", ret)


matrix = robot_pose_to_matrix(pose)
print("机器人末端位姿的4×4变换矩阵（米）:")
print(matrix)