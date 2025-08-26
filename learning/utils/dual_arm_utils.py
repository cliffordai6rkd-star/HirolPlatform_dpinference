#!/usr/bin/env python3
"""
双臂机器人工具类（简化版）

提供双臂机器人的基本状态管理功能。
专门为Monte01双臂机器人设计（16 DOF）。
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import glog as log


@dataclass
class DualArmState:
    """双臂状态数据结构"""
    left_arm: np.ndarray   # 左臂状态 (8 DOF: 7关节 + 1夹爪)
    right_arm: np.ndarray  # 右臂状态 (8 DOF: 7关节 + 1夹爪) 
    timestamp: float       # 时间戳
    
    def to_combined_array(self) -> np.ndarray:
        """转换为16维组合数组"""
        return np.concatenate([self.left_arm, self.right_arm])
    
    @classmethod
    def from_combined_array(cls, combined: np.ndarray, timestamp: float = None):
        """从16维组合数组创建"""
        if len(combined) != 16:
            raise ValueError(f"组合数组应为16维，实际为{len(combined)}维")
        
        return cls(
            left_arm=combined[:8],
            right_arm=combined[8:16], 
            timestamp=timestamp or time.time()
        )


class SimpleDualArmCoordinator:
    """简化的双臂协调器 - 主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化双臂协调器
        
        Args:
            config: 双臂控制配置（可选）
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # 统计信息（简化版）
        self.total_steps = 0
        
        log.info("✅ 简化双臂协调控制器初始化完成")
    
    def process_dual_arm_action(self, combined_action: np.ndarray, 
                               left_pose: Optional[np.ndarray] = None,
                               right_pose: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        处理双臂动作（简化版 - 无碰撞检测）
        
        Args:
            combined_action: 16维组合动作
            left_pose: 左臂末端位姿 (可选，未使用)
            right_pose: 右臂末端位姿 (可选，未使用)
            
        Returns:
            Tuple[np.ndarray, Dict]: (处理后的动作, 状态信息)
        """
        # 创建双臂状态
        dual_arm_state = DualArmState.from_combined_array(combined_action)
        
        # 简化的状态信息
        info = {
            'collision_risk': False,  # 始终为False（无碰撞检测）
            'workspace_violation': False,  # 始终为False（无工作空间检查）
            'sync_adjusted': False,  # 始终为False（无同步调整）
            'warnings': []
        }
        
        # 更新统计
        self.total_steps += 1
        
        # 直接返回原始动作（不做任何修改）
        return dual_arm_state.to_combined_array(), info
    
    def split_dual_arm_action(self, combined_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离双臂动作
        
        Args:
            combined_action: 16维组合动作
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (左臂8维动作, 右臂8维动作)
        """
        if len(combined_action) != 16:
            raise ValueError(f"动作维度错误：期望16，实际{len(combined_action)}")
        
        return combined_action[:8], combined_action[8:16]
    
    def combine_dual_arm_state(self, left_state: np.ndarray, right_state: np.ndarray) -> np.ndarray:
        """
        组合双臂状态
        
        Args:
            left_state: 左臂8维状态
            right_state: 右臂8维状态
            
        Returns:
            np.ndarray: 16维组合状态
        """
        if len(left_state) != 8 or len(right_state) != 8:
            raise ValueError(f"状态维度错误：左臂{len(left_state)}，右臂{len(right_state)}，都应为8维")
        
        return np.concatenate([left_state, right_state])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取协调器统计信息（简化版）"""
        if self.total_steps == 0:
            return {"message": "无统计数据"}
        
        return {
            "总步数": self.total_steps,
            "简化模式": "已启用（无碰撞检测）"
        }
    
    def print_statistics(self):
        """打印协调器统计信息"""
        stats = self.get_statistics()
        
        log.info("📊 双臂机器人统计摘要（简化版）:")
        for key, value in stats.items():
            log.info(f"   {key}: {value}")


def create_dual_arm_coordinator(config: Dict[str, Any] = None) -> SimpleDualArmCoordinator:
    """
    工厂函数：创建简化的双臂协调器
    
    Args:
        config: 双臂控制配置
        
    Returns:
        SimpleDualArmCoordinator: 简化的双臂协调器实例
    """
    return SimpleDualArmCoordinator(config)