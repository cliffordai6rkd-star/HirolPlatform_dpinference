'''
Description: 位姿转换工具函数
Version: V1.0
Author: Gaojing
Date: 2025-05-08
Copyright (C) 2024-2050 Corenetic Technology Inc All rights reserved.
'''

import numpy as np
import math


def euler_to_matrix(position, euler_angles, is_degrees=True,to_meters=True):
    """
    将位置和欧拉角转换为4×4变换矩阵
    
    参数:
        position: [x, y, z] 格式的位置向量，默认单位为毫米
        euler_angles: [roll, pitch, yaw] 格式的欧拉角
        is_degrees: 指示欧拉角是否为角度单位（True）或弧度单位（False）
        to_meters: 是否将位置从毫米转换为米（True）或保持原始单位（False）
    
    返回:
        4×4的numpy数组表示变换矩阵，平移部分根据to_meters参数决定单位
    """
    # 提取位置
    px, py, pz = position
    
    # 如果需要，将毫米转换为米
    if to_meters:
        px /= 1000.0
        py /= 1000.0
        pz /= 1000.0
    
    # 提取欧拉角
    roll, pitch, yaw = euler_angles
    
    # 如果输入是角度，转换为弧度
    if is_degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    
    # 计算三角函数值
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    # 构建旋转矩阵 (ZYX顺序: yaw -> pitch -> roll)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    
    # 创建4×4变换矩阵
    matrix = np.array([
        [r00, r01, r02, px],
        [r10, r11, r12, py],
        [r20, r21, r22, pz],
        [0, 0, 0, 1]
    ])
    
    return matrix

def robot_pose_to_matrix(pose_data, is_degrees=True, to_meters=True):
    """
    将机器人API返回的欧拉角位姿直接转换为4×4变换矩阵
    
    参数:
        pose_data: 机器人位姿列表 [x, y, z, roll, pitch, yaw]
        is_degrees: 指示欧拉角是否为角度单位（True）或弧度单位（False）
    
    返回:
        4×4的numpy数组表示变换矩阵
    """
    position = pose_data[0:3]
    euler_angles = pose_data[3:6]
    return euler_to_matrix(position, euler_angles, is_degrees, to_meters)


# # 在您的代码中导入
# from transform_utils import euler_to_matrix, robot_pose_to_matrix

# # 示例1: 在获取机器人末端位姿后直接转换
# success, end_pose = robot.get_position()  # [x, y, z, roll, pitch, yaw]
# if success == 0:
#     transform_matrix = robot_euler_pose_to_matrix(end_pose)
#     print("机器人末端位姿的4×4变换矩阵:")
#     print(transform_matrix)

# # 示例2: 手动提供位置和欧拉角
# position = [100.0, 200.0, 300.0]  # 单位：毫米
# euler_angles = [30.0, 45.0, 60.0]  # 单位：度
# matrix = euler_to_matrix(position, euler_angles)
# print("指定位置和欧拉角的4×4变换矩阵:")
# print(matrix)