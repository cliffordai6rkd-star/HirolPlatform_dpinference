#!/usr/bin/env python3
"""
动作时间聚合器 - 用于平滑机器人动作序列
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, Any, List
import glog as log


class ActionAggregator:
    """动作时间聚合器 - 专门处理动作序列的平滑和聚合"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动作聚合器
        
        Args:
            config: 动作聚合配置
        """
        self.config = config
        
        # 聚合参数
        self.window_size = config.get('window_size', 3)
        self.method = config.get('method', 'ema')  # ema/mean/weighted
        self.ema_alpha = config.get('ema_alpha', 0.4)
        
        # 【新增】机器人类型自适应
        self.robot_type = config.get('robot_type', 'fr3')
        if self.robot_type == 'monte01':
            # Monte01双臂：14关节 + 2夹爪 = 16维
            self.joint_dim = 14
            self.gripper_dim = 2
        else:
            # FR3单臂：7关节 + 1夹爪 = 8维
            self.joint_dim = 7
            self.gripper_dim = 1
        
        # 加权聚合权重
        default_weights = [0.5, 0.3, 0.2]
        self.weights = np.array(config.get('weights', default_weights[:self.window_size]))
        self.weights = self.weights / np.sum(self.weights)
        
        # 动作历史缓存
        self.action_history = deque(maxlen=self.window_size)
        self.smoothed_action = None
        
        # 分离处理参数
        self.joint_smoothing = config.get('joint_smoothing_enabled', True)
        self.gripper_smoothing = config.get('gripper_smoothing_enabled', False)
        self.joint_ema_alpha = config.get('joint_ema_alpha', self.ema_alpha)
        self.gripper_ema_alpha = config.get('gripper_ema_alpha', 0.8)
        
        log.info("⏱️ 动作聚合器初始化完成")
        self._log_config()
    
    def _log_config(self):
        """记录配置信息"""
        log.info(f"   - 机器人类型: {self.robot_type} (关节:{self.joint_dim}, 夹爪:{self.gripper_dim})")
        log.info(f"   - 窗口大小: {self.window_size}")
        log.info(f"   - 聚合方法: {self.method}")
        log.info(f"   - 关节平滑: {'启用' if self.joint_smoothing else '禁用'} (α={self.joint_ema_alpha})")
        log.info(f"   - 夹爪平滑: {'启用' if self.gripper_smoothing else '禁用'} (α={self.gripper_ema_alpha})")
    
    def process_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        处理并平滑动作
        
        Args:
            raw_action: 原始动作
            
        Returns:
            np.ndarray: 平滑后的动作
        """
        # 确保输入是numpy数组
        if torch.is_tensor(raw_action):
            raw_action = raw_action.cpu().numpy()
        raw_action = np.asarray(raw_action).flatten()
        
        # 添加到历史
        self.action_history.append(raw_action.copy())
        
        # 首次调用直接返回
        if len(self.action_history) == 1:
            self.smoothed_action = raw_action.copy()
            return self.smoothed_action
        
        if self.robot_type == 'monte01':
            # Monte01特殊结构：[左臂7+左夹爪1+右臂7+右夹爪1]
            left_arm_joints = raw_action[0:7]      # 左臂关节
            left_gripper = raw_action[7]           # 左夹爪
            right_arm_joints = raw_action[8:15]    # 右臂关节  
            right_gripper = raw_action[15]         # 右夹爪

            # 组合为处理格式
            joint_action = np.concatenate([left_arm_joints, right_arm_joints])  # 14维关节
            gripper_action = np.array([left_gripper, right_gripper])            # 2维夹爪
        else:
            # FR3结构：[关节1-7, 夹爪]
            joint_action = raw_action[:7]
            gripper_action = raw_action[7] if len(raw_action) > 7 else None
        
        # 处理关节动作
        smoothed_joint = self._process_joint(joint_action)
        
        # 处理夹爪动作
        if gripper_action is not None:
            smoothed_gripper = self._process_gripper(gripper_action)
            self.smoothed_action = np.append(smoothed_joint, smoothed_gripper)
        else:
            self.smoothed_action = smoothed_joint
        
        return self.smoothed_action
    
    def _process_joint(self, current_joint: np.ndarray) -> np.ndarray:
        """处理关节动作"""
        if not self.joint_smoothing:
            return current_joint.copy()
        
        joint_history = [action[:7] for action in self.action_history]
        
        if self.method == 'ema':
            return self._ema_smooth(current_joint, joint_history, self.joint_ema_alpha)
        elif self.method == 'mean':
            return np.mean(np.array(joint_history), axis=0)
        elif self.method == 'weighted':
            return self._weighted_smooth(joint_history)
        else:
            return current_joint.copy()
    
    def _process_gripper(self, current_gripper: float) -> float:
        """处理夹爪动作"""
        if not self.gripper_smoothing:
            return current_gripper
        
        gripper_history = [
            action[7] if len(action) > 7 else 0.0 
            for action in self.action_history
        ]
        
        if self.method == 'ema':
            if self.smoothed_action is not None and len(self.smoothed_action) > 7:
                prev = self.smoothed_action[7]
            else:
                prev = gripper_history[-2] if len(gripper_history) > 1 else current_gripper
            return self.gripper_ema_alpha * current_gripper + (1 - self.gripper_ema_alpha) * prev
        elif self.method == 'mean':
            return np.mean(gripper_history)
        else:
            return current_gripper
    
    def _ema_smooth(self, current: np.ndarray, history: List[np.ndarray], alpha: float) -> np.ndarray:
        """指数移动平均"""
        if len(history) == 1:
            return current.copy()
        
        if self.smoothed_action is not None:
            previous = self.smoothed_action[:7]
        else:
            previous = history[-2]
        
        return alpha * current + (1 - alpha) * previous
    
    def _weighted_smooth(self, history: List[np.ndarray]) -> np.ndarray:
        """加权平滑"""
        weights = self.weights[:len(history)]
        weights = weights / np.sum(weights)
        
        weighted_sum = np.zeros_like(history[0])
        for i, action in enumerate(reversed(history)):
            weighted_sum += weights[i] * action
        
        return weighted_sum
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'history_length': len(self.action_history),
            'method': self.method,
            'window_size': self.window_size,
            'joint_smoothing': self.joint_smoothing,
            'gripper_smoothing': self.gripper_smoothing
        }
    
    def reset(self):
        """重置聚合器状态"""
        self.action_history.clear()
        self.smoothed_action = None
        log.info("🔄 动作聚合器已重置")