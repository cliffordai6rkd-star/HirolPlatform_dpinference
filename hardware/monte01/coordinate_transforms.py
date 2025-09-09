#!/usr/bin/env python3
"""Monte01机器人坐标系转换常量和工具函数.

该模块定义了Monte01双臂机器人中XARM与CORENETIC坐标系之间的转换常量和函数。
用于确保训练数据和推理数据在同一坐标系下进行处理。
"""

import numpy as np
from typing import Union, List


# XARM到CORENETIC坐标系转换常量
# 该常量用于将XARM原始关节角度转换到CORENETIC坐标系
JP_XARM2CORENETIC = np.array([0.0, 1.5708, -1.5708, -3.1416, -1.5708, 0.0, -0.785])

def xarm_to_corenetic(joint_positions: Union[np.ndarray, List[float]]) -> np.ndarray:
    """将XARM坐标系的关节角度转换到CORENETIC坐标系.
    
    Args:
        joint_positions: XARM坐标系的关节角度 (7维)
        
    Returns:
        CORENETIC坐标系的关节角度 (7维)
    """
    positions = np.array(joint_positions)
    if len(positions) != 7:
        raise ValueError(f"期望7维关节角度，实际得到{len(positions)}维")
    
    return positions + JP_XARM2CORENETIC

def corenetic_to_xarm(joint_positions: Union[np.ndarray, List[float]]) -> np.ndarray:
    """将CORENETIC坐标系的关节角度转换到XARM坐标系.
    
    Args:
        joint_positions: CORENETIC坐标系的关节角度 (7维)
        
    Returns:
        XARM坐标系的关节角度 (7维)
    """
    positions = np.array(joint_positions)
    if len(positions) != 7:
        raise ValueError(f"期望7维关节角度，实际得到{len(positions)}维")
    
    return positions - JP_XARM2CORENETIC

def dual_arm_corenetic_to_xarm(dual_arm_positions: Union[np.ndarray, List[float]]) -> np.ndarray:
    """将双臂CORENETIC坐标系转换到XARM坐标系.
    
    Args:
        dual_arm_positions: 16维双臂位置 [左臂7关节, 左夹爪, 右臂7关节, 右夹爪]
        
    Returns:
        转换后的16维双臂位置 (关节部分转换，夹爪部分保持不变)
    """
    positions = np.array(dual_arm_positions)
    if len(positions) != 16:
        raise ValueError(f"期望16维双臂位置，实际得到{len(positions)}维")
    
    result = positions.copy()
    
    # 转换左臂关节 (索引0-6)
    result[0:7] = corenetic_to_xarm(positions[0:7])
    # 左夹爪 (索引7) 保持不变
    
    # 转换右臂关节 (索引8-14)
    result[8:15] = corenetic_to_xarm(positions[8:15])
    # 右夹爪 (索引15) 保持不变
    
    return result

def dual_arm_xarm_to_corenetic(dual_arm_positions: Union[np.ndarray, List[float]]) -> np.ndarray:
    """将双臂XARM坐标系转换到CORENETIC坐标系.
    
    Args:
        dual_arm_positions: 16维双臂位置 [左臂7关节, 左夹爪, 右臂7关节, 右夹爪]
        
    Returns:
        转换后的16维双臂位置 (关节部分转换，夹爪部分保持不变)
    """
    positions = np.array(dual_arm_positions)
    if len(positions) != 16:
        raise ValueError(f"期望16维双臂位置，实际得到{len(positions)}维")
    
    result = positions.copy()
    
    # 转换左臂关节 (索引0-6)
    result[0:7] = xarm_to_corenetic(positions[0:7])
    # 左夹爪 (索引7) 保持不变
    
    # 转换右臂关节 (索引8-14)  
    result[8:15] = xarm_to_corenetic(positions[8:15])
    # 右夹爪 (索引15) 保持不变
    
    return result