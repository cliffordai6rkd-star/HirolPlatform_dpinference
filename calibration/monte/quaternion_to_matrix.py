'''
Description: 位姿转换工具函数
Version: V1.0
Author: Gaojing、Haotian 
Date: 2025-05-06
Copyright (C) 2024-2050 Corenetic Technology Inc All rights reserved.
'''

import numpy as np

def quaternion_to_matrix(quaternion, position=None):
    """
    将四元数和位置向量转换为4×4变换矩阵
    
    参数:
        quaternion: 可以是以下格式之一:
            - [qw, qx, qy, qz] 格式的四元数列表或数组
            - 如果是完整位姿 [x, y, z, qw, qx, qy, qz]，则position参数可以省略
        position: [x, y, z] 格式的位置向量列表或数组，如果quaternion包含位置则可省略
    
    返回:
        4×4的numpy数组表示变换矩阵
    """
    # 处理不同的输入格式
    if position is None:
        if len(quaternion) == 7:  # 假设格式为 [x, y, z, qw, qx, qy, qz]
            px, py, pz = quaternion[0:3]
            qw, qx, qy, qz = quaternion[3:7]
        else:
            raise ValueError("如果未提供position参数，quaternion必须是长度为7的完整位姿")
    else:
        if len(quaternion) == 4:  # 格式为 [qw, qx, qy, qz]
            qw, qx, qy, qz = quaternion
            px, py, pz = position
        else:
            raise ValueError("quaternion参数应为长度为4的四元数")
    
    # 计算旋转矩阵的元素
    r00 = 1 - 2 * (qy**2 + qz**2)
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
    
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 1 - 2 * (qx**2 + qz**2)
    r12 = 2 * (qy * qz - qw * qx)
    
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx**2 + qy**2)
    
    # 创建4×4变换矩阵
    matrix = np.array([
        [r00, r01, r02, px],
        [r10, r11, r12, py],
        [r20, r21, r22, pz],
        [0, 0, 0, 1]
    ])
    
    return matrix

def robot_pose_to_matrix(end_pose):
    """
    将机器人API返回的末端位姿直接转换为4×4变换矩阵
    
    参数:
        end_pose: 机器人API返回的位姿列表 [x, y, z, qw, qx, qy, qz]
        
    返回:
        4×4的numpy数组表示变换矩阵
    """
    return quaternion_to_matrix(end_pose)



# # 在您的代码中导入
# from transform_utils import quaternion_to_matrix, robot_pose_to_matrix

# # 示例1: 在获取机器人末端位姿后直接转换
# success, end_pose = robot.get_arm_end_pose(component_type)
# if success:
#     transform_matrix = robot_pose_to_matrix(end_pose)
#     print("机器人末端位姿的4×4变换矩阵:")
#     print(transform_matrix)

# # 示例2: 手动提供四元数和位置
# position = [0.1, 0.2, 0.3]
# quaternion = [1.0, 0.0, 0.0, 0.0]  # 四元数 [qw, qx, qy, qz]
# matrix = quaternion_to_matrix(quaternion, position)
# print("指定四元数和位置的4×4变换矩阵:")
# print(matrix)