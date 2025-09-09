#!/usr/bin/env python3

import torch
import numpy as np
import pickle
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import glog as log


class PolicyInference(ABC):
    """通用推理算法基类，抽象出推理算法接口"""
    
    def __init__(self, ckpt_dir, policy_config):
        """
        初始化推理器
        
        Args:
            ckpt_dir: 检查点目录路径
            policy_config: 策略配置字典
        """
        self.ckpt_dir = ckpt_dir
        self.policy_config = policy_config
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"✅ 推理设备初始化: {self.device}")
        
        # 加载数据集统计信息
        self.load_dataset_stats()
        
        # 初始化策略
        self.init_policy()
        
        # 加载模型
        self.load_policy()
        
    def load_dataset_stats(self):
        """加载数据集统计信息用于归一化"""
        stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        
        # 如果没有找到dataset_stats.pkl，尝试找dataset_stats_bac.pkl
        if not os.path.exists(stats_path):
            stats_path = os.path.join(self.ckpt_dir, 'dataset_stats_bac.pkl')
            
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"找不到数据集统计文件: {stats_path}")
            
        with open(stats_path, 'rb') as f:
            self.dataset_stats = pickle.load(f)
            
        log.info(f"✅ 加载数据集统计信息: {stats_path}")
        log.info(f"   - 统计信息键: {list(self.dataset_stats.keys())}")
        
    @abstractmethod
    def init_policy(self):
        """初始化策略模型（抽象方法，子类需要实现）"""
        pass
        
    @abstractmethod
    def load_policy(self):
        """加载训练好的策略模型（抽象方法，子类需要实现）"""
        pass
        
    @abstractmethod
    def forward_policy(self, state, images):
        """
        策略前向推理（抽象方法，子类需要实现）
        
        Args:
            state: 状态张量
            images: 图像张量
            
        Returns:
            actions: 预测的动作
        """
        pass
        
    def normalize_data(self, data, stats, key):
        """数据归一化"""
        mean = stats[f'{key}_mean']
        std = stats[f'{key}_std']
        return (data - mean) / std
        
    def denormalize_data(self, data, stats, key):
        """数据反归一化"""
        mean = stats[f'{key}_mean']
        std = stats[f'{key}_std']
        return data * std + mean
        
    def predict(self, state, images):
        """
        进行动作预测
        
        Args:
            state: 状态向量
            images: 图像数据
            
        Returns:
            predicted_actions: 预测的动作序列
        """
        with torch.no_grad():
            # 预处理状态
            state = np.array(state)
            state_normalized = self.normalize_data(state, self.dataset_stats, 'qpos')
            state_tensor = torch.from_numpy(state_normalized).float().to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # 添加batch维度
            
            # 策略推理
            actions = self.forward_policy(state_tensor, images)
            
            # 反归一化动作
            actions_np = actions.cpu().numpy().squeeze(0)  # 移除batch维度
            actions_denorm = self.denormalize_data(actions_np, self.dataset_stats, 'action')
            
            return actions_denorm


class ACTInference(PolicyInference):
    """ACT算法特定的推理器"""
    
    def __init__(self, ckpt_dir, state_dim, camera_names, **kwargs):
        """
        初始化ACT推理器
        
        Args:
            ckpt_dir: 检查点目录路径
            state_dim: 状态维度
            camera_names: 相机名称列表
            **kwargs: 其他ACT参数
        """
        # ACT策略配置 - 确保数值参数为正确类型
        policy_config = {
            'lr': float(kwargs.get('lr', 1e-5)),
            'num_queries': int(kwargs.get('num_queries', 100)),  # chunk_size
            'kl_weight': float(kwargs.get('kl_weight', 10)),
            'hidden_dim': int(kwargs.get('hidden_dim', 512)),
            'dim_feedforward': int(kwargs.get('dim_feedforward', 3200)),
            'lr_backbone': float(kwargs.get('lr_backbone', 1e-5)),
            'backbone': kwargs.get('backbone', 'resnet18'),
            'enc_layers': int(kwargs.get('enc_layers', 4)),
            'dec_layers': int(kwargs.get('dec_layers', 7)),
            'nheads': int(kwargs.get('nheads', 8)),
            'camera_names': camera_names,
            'vq': kwargs.get('vq', False),
            'vq_class': kwargs.get('vq_class', None),
            'vq_dim': kwargs.get('vq_dim', None),
            'action_dim': int(state_dim),
            'state_dim': int(state_dim),  # 添加state_dim别名
            'no_encoder': kwargs.get('no_encoder', False),
            # 必需的DETR参数
            'ckpt_dir': ckpt_dir,
            'policy_class': kwargs.get('policy_class', 'ACT'),
            'task_name': kwargs.get('task_name', 'default_task'),
            'seed': int(kwargs.get('seed', 0)),
            'num_epochs': int(kwargs.get('num_epochs', 1)),
        }
        
        super().__init__(ckpt_dir, policy_config)
        
    def init_policy(self):
        """初始化ACT策略"""
        from learning.policies.act.policy import ACTPolicy
        
        self.policy = ACTPolicy(self.policy_config)
        self.policy.to(self.device)
        self.policy.eval()
        
        log.info(f"✅ ACT策略初始化完成")
        log.info(f"   - State维度: {self.policy_config['action_dim']}")
        log.info(f"   - Action块大小: {self.policy_config['num_queries']}")
        log.info(f"   - 相机: {self.policy_config['camera_names']}")
        
    def load_policy(self):
        """加载ACT策略模型"""
        policy_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
        
        # 如果没有找到policy_best.ckpt，尝试找policy_best_bac.ckpt
        if not os.path.exists(policy_path):
            policy_path = os.path.join(self.ckpt_dir, 'policy_best_bac.ckpt')
            
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"找不到策略文件: {policy_path}")
            
        checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint)
        
        log.info(f"✅ 加载ACT策略模型: {policy_path}")
        
    def forward_policy(self, state, images):
        """ACT策略前向推理"""
        return self.policy(state, images)


# class PPOInference(PolicyInference):
#     """PPO算法特定的推理器（示例，待实现）"""
    
#     def __init__(self, ckpt_dir, state_dim, camera_names, **kwargs):
#         # PPO策略配置
#         policy_config = {
#             'state_dim': state_dim,
#             'action_dim': state_dim,
#             'camera_names': camera_names,
#             'hidden_dim': kwargs.get('hidden_dim', 256),
#             # 其他PPO特定参数...
#         }
        
#         super().__init__(ckpt_dir, policy_config)
        
#     def init_policy(self):
#         """初始化PPO策略"""
#         # TODO: 实现PPO策略初始化
#         # from ppo_policy import PPOPolicy
#         # self.policy = PPOPolicy(self.policy_config)
#         raise NotImplementedError("PPO推理器待实现")
        
#     def load_policy(self):
#         """加载PPO策略模型"""
#         # TODO: 实现PPO模型加载
#         raise NotImplementedError("PPO推理器待实现")
        
#     def forward_policy(self, state, images):
#         """PPO策略前向推理"""
#         # TODO: 实现PPO前向推理
#         raise NotImplementedError("PPO推理器待实现")