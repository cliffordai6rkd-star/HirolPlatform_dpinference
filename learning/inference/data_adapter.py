#!/usr/bin/env python3
"""数据适配器：机器人数据与学习数据格式转换器.

该模块提供HIROLRobotPlatform机器人数据格式与学习算法数据格式之间的转换功能。
支持多种机器人平台和相机配置的数据适配。
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
import cv2

import glog as log
from hardware.base.utils import RobotJointState


class RobotDataAdapter:
    """机器人数据与学习数据格式转换器.
    
    提供HIROLRobotPlatform与学习算法之间的数据格式转换功能，
    包括关节状态、相机数据和动作指令的双向转换。
    """
    
    def __init__(self, robot_type: str = "generic"):
        """初始化数据适配器.
        
        Args:
            robot_type: 机器人类型，用于特定的数据格式适配
        """
        self.robot_type = robot_type
        log.info(f"✅ 数据适配器初始化: {robot_type}")
    
    @staticmethod
    def robot_state_to_numpy(joint_state: RobotJointState) -> np.ndarray:
        """将RobotJointState转换为numpy数组.
        
        Args:
            joint_state: HIROLRobotPlatform的关节状态对象
            
        Returns:
            np.ndarray: 关节位置的numpy数组
            
        Raises:
            ValueError: 当关节状态数据无效时抛出
        """
        if joint_state._positions is None:
            raise ValueError("关节位置数据为空")
        
        if isinstance(joint_state._positions, (list, tuple)):
            positions = np.array(joint_state._positions, dtype=np.float32)
        elif isinstance(joint_state._positions, np.ndarray):
            positions = joint_state._positions.astype(np.float32)
        else:
            raise ValueError(f"不支持的关节位置数据类型: {type(joint_state._positions)}")
        
        log.debug(f"关节状态转换: {positions.shape} -> numpy数组")
        return positions
    
    @staticmethod
    def numpy_to_robot_actions(actions: np.ndarray) -> List[float]:
        """将numpy动作数组转换为机器人关节指令.
        
        Args:
            actions: 学习算法输出的动作数组
            
        Returns:
            List[float]: 机器人关节指令列表
            
        Raises:
            ValueError: 当动作数据格式无效时抛出
        """
        if not isinstance(actions, np.ndarray):
            raise ValueError(f"动作数据必须是numpy数组，当前类型: {type(actions)}")
        
        # 确保数据是一维的
        if actions.ndim > 1:
            if actions.shape[0] == 1:
                actions = actions.squeeze(0)
            else:
                # 如果是动作序列，取第一个动作
                actions = actions[0]
                log.warning(f"⚠️ 检测到动作序列，使用第一个动作: {actions.shape}")
        
        # 转换为Python浮点数列表
        action_list = actions.astype(float).tolist()
        
        log.debug(f"动作转换: {len(action_list)}个关节指令")
        return action_list
    
    @staticmethod
    def camera_dict_to_tensor(
        camera_data: Dict[str, np.ndarray], 
        normalize: bool = True,
        target_size: Optional[tuple] = None
    ) -> torch.Tensor:
        """将相机数据字典转换为tensor.
        
        Args:
            camera_data: 相机名称到图像数据的映射
            normalize: 是否进行ImageNet归一化
            target_size: 目标图像尺寸 (height, width)
            
        Returns:
            torch.Tensor: 形状为(batch, num_cameras, channels, height, width)的tensor
            
        Raises:
            ValueError: 当相机数据格式无效时抛出
        """
        if not camera_data:
            raise ValueError("相机数据为空")
        
        # 按相机名称排序以确保一致性
        camera_names = sorted(camera_data.keys())
        processed_images = []
        
        for cam_name in camera_names:
            image = camera_data[cam_name]
            
            # 验证图像数据
            if not isinstance(image, np.ndarray):
                raise ValueError(f"相机 {cam_name} 数据必须是numpy数组")
            
            # 处理图像尺寸
            if target_size is not None:
                if image.shape[:2] != target_size:
                    image = cv2.resize(image, (target_size[1], target_size[0]))
                    log.debug(f"相机 {cam_name} 图像调整至: {target_size}")
            
            # 转换为float并归一化到[0,1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # 确保RGB通道顺序 (H, W, C)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 转换为 (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            else:
                raise ValueError(f"相机 {cam_name} 图像必须是3通道RGB格式")
            
            processed_images.append(image)
        
        # 堆叠为 (num_cameras, C, H, W)
        images_array = np.stack(processed_images, axis=0)
        
        # 转换为tensor
        images_tensor = torch.from_numpy(images_array).float()
        
        # ImageNet归一化
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images_tensor = (images_tensor - mean) / std
        
        # 添加batch维度: (1, num_cameras, C, H, W)
        images_tensor = images_tensor.unsqueeze(0)
        
        log.debug(f"相机数据转换: {len(camera_names)}个相机 -> {images_tensor.shape}")
        return images_tensor
    
    def validate_robot_state(self, joint_state: RobotJointState) -> bool:
        """验证机器人状态数据的有效性.
        
        Args:
            joint_state: 机器人关节状态
            
        Returns:
            bool: 数据是否有效
        """
        if joint_state._positions is None:
            log.error("❌ 关节位置数据为空")
            return False
        
        try:
            positions = self.robot_state_to_numpy(joint_state)
            if len(positions) == 0:
                log.error("❌ 关节位置数组为空")
                return False
            
            # 检查数值是否有效
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                log.error("❌ 关节位置包含无效数值")
                return False
                
            return True
        except Exception as e:
            log.error(f"❌ 机器人状态验证失败: {str(e)}")
            return False
    
    def validate_camera_data(self, camera_data: Dict[str, np.ndarray]) -> bool:
        """验证相机数据的有效性.
        
        Args:
            camera_data: 相机数据字典
            
        Returns:
            bool: 数据是否有效
        """
        if not camera_data:
            log.error("❌ 相机数据为空")
            return False
        
        for cam_name, image in camera_data.items():
            if not isinstance(image, np.ndarray):
                log.error(f"❌ 相机 {cam_name} 数据类型错误")
                return False
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                log.error(f"❌ 相机 {cam_name} 图像格式错误: {image.shape}")
                return False
        
        return True


class FR3DataAdapter(RobotDataAdapter):
    """FR3机器人专用数据适配器."""
    
    def __init__(self):
        super().__init__("fr3")
        self.expected_dof = 7  # FR3机器人7自由度
    
    def validate_robot_state(self, joint_state: RobotJointState) -> bool:
        """FR3特定的状态验证."""
        if not super().validate_robot_state(joint_state):
            return False
        
        positions = self.robot_state_to_numpy(joint_state)
        if len(positions) != self.expected_dof:
            log.error(f"❌ FR3关节数量错误: 期望{self.expected_dof}, 实际{len(positions)}")
            return False
        
        return True


class Monte01DataAdapter(RobotDataAdapter):
    """Monte01双臂机器人专用数据适配器 (16 DOF)."""
    
    def __init__(self):
        super().__init__("monte01")
        self.expected_dof = 16  # Monte01双臂: (7+1) × 2 = 16自由度
        self.left_arm_indices = slice(0, 8)   # 左臂: 0-7 (7关节+1夹爪)
        self.right_arm_indices = slice(8, 16) # 右臂: 8-15 (7关节+1夹爪)
        self.gripper_range = (0.0, 0.074)    # Monte01夹爪范围(米)
        
        # 导入坐标系转换函数
        try:
            from hardware.monte01.coordinate_transforms import dual_arm_corenetic_to_xarm, dual_arm_xarm_to_corenetic
            self.corenetic_to_xarm = dual_arm_corenetic_to_xarm
            self.xarm_to_corenetic = dual_arm_xarm_to_corenetic
            self.coordinate_transform_enabled = True
            log.info(f"✅ Monte01双臂数据适配器初始化 (16 DOF, 坐标系转换已启用)")
        except ImportError as e:
            log.warning(f"⚠️ 坐标系转换模块导入失败: {e}, 将禁用坐标系转换")
            self.coordinate_transform_enabled = False
            log.info(f"✅ Monte01双臂数据适配器初始化 (16 DOF, 坐标系转换已禁用)")
    
    def validate_robot_state(self, joint_state: RobotJointState) -> bool:
        """Monte01双臂状态验证."""
        if not super().validate_robot_state(joint_state):
            return False
        
        positions = self.robot_state_to_numpy(joint_state)
        
        # 验证转换后的维度
        if len(positions) != self.expected_dof:
            log.error(f"❌ Monte01关节数量错误: 期望{self.expected_dof}, 实际{len(positions)}")
            return False
        
        # 验证双臂状态合理性（使用16维数据）
        left_state, right_state = self.split_dual_arm_state(positions)
        
        # 验证左臂状态
        if not self._validate_single_arm_state(left_state, "左臂"):
            return False
            
        # 验证右臂状态
        if not self._validate_single_arm_state(right_state, "右臂"):
            return False
        
        return True
    
    def robot_state_to_numpy(self, joint_state: RobotJointState) -> np.ndarray:
        """Monte01专用：将RobotJointState转换为numpy数组.
        
        关键修复：保持CORENETIC坐标系用于模型输入，确保与训练数据一致
        推理引擎已在_get_monte01_dual_arm_state()中处理了14维->16维转换
        
        Args:
            joint_state: HIROLRobotPlatform的关节状态对象
            
        Returns:
            np.ndarray: CORENETIC坐标系的关节位置数组 (用于ACT模型输入)
        """
        # 使用父类方法获取CORENETIC坐标系的位置
        corenetic_positions = super().robot_state_to_numpy(joint_state)
        
        # 关键修复：保持CORENETIC坐标系不转换，确保与训练数据一致
        log.debug(f"💡 状态保持CORENETIC坐标系用于模型输入: 左臂{corenetic_positions[:7].round(3)}, 右臂{corenetic_positions[8:15].round(3)}")
        return corenetic_positions  # 直接返回CORENETIC坐标系状态
    
    def split_dual_arm_state(self, state: np.ndarray) -> tuple:
        """分离双臂状态.
        
        Args:
            state: 16维双臂状态
            
        Returns:
            tuple: (左臂8维状态, 右臂8维状态)
        """
        if len(state) != self.expected_dof:
            raise ValueError(f"状态维度错误: 期望{self.expected_dof}, 实际{len(state)}")
        
        left_state = state[self.left_arm_indices]
        right_state = state[self.right_arm_indices]
        
        return left_state, right_state
    
    def action_to_robot_command(self, action: np.ndarray) -> np.ndarray:
        """将模型输出的动作转换为机器人命令.
        
        模型输出的是CORENETIC坐标系的动作，需要转换为XARM坐标系。
        转换后的XARM动作将直接被xarm7_arm.py转换回XARM硬件格式执行。
        
        Args:
            action: CORENETIC坐标系的16维动作 (模型输出)
            
        Returns:
            XARM坐标系的16维机器人命令 (给xarm7_arm.py)
        """
        if len(action) != self.expected_dof:
            raise ValueError(f"动作维度错误: 期望{self.expected_dof}, 实际{len(action)}")
        
        return action
    
    def combine_dual_arm_state(self, left_state: np.ndarray, right_state: np.ndarray) -> np.ndarray:
        """组合双臂状态.
        
        Args:
            left_state: 左臂8维状态
            right_state: 右臂8维状态
            
        Returns:
            np.ndarray: 16维双臂状态
        """
        if len(left_state) != 8 or len(right_state) != 8:
            raise ValueError(f"单臂状态维度错误: 左臂{len(left_state)}, 右臂{len(right_state)}, 都应为8维")
        
        return np.concatenate([left_state, right_state])
    
    def _validate_single_arm_state(self, arm_state: np.ndarray, arm_name: str) -> bool:
        """验证单个手臂状态."""
        if len(arm_state) != 8:
            log.error(f"❌ {arm_name}状态维度错误: 期望8, 实际{len(arm_state)}")
            return False
        
        
        # 验证夹爪位置范围
        gripper_position = arm_state[7]
        if not (self.gripper_range[0] <= gripper_position <= self.gripper_range[1]):
            log.warning(f"⚠️ {arm_name}夹爪位置超出范围 {self.gripper_range}: {gripper_position}")
        
        return True
    
    def validate_camera_data(self, camera_data: Dict[str, np.ndarray]) -> bool:
        """Monte01三相机数据验证."""
        expected_cameras = ['left_ee_cam', 'right_ee_cam', 'third_person_cam']
        
        # 基本相机验证
        if not super().validate_camera_data(camera_data):
            return False
        
        # 检查Monte01特定的相机配置
        missing_cameras = [cam for cam in expected_cameras if cam not in camera_data]
        if missing_cameras:
            log.warning(f"⚠️ Monte01缺少相机: {missing_cameras}")
        
        return True


def create_data_adapter(robot_type: str) -> RobotDataAdapter:
    """数据适配器工厂函数.
    
    Args:
        robot_type: 机器人类型
        
    Returns:
        RobotDataAdapter: 对应的数据适配器实例
        
    Raises:
        ValueError: 当机器人类型不支持时抛出
    """
    adapters = {
        "fr3": FR3DataAdapter,
        "monte01": Monte01DataAdapter,
        "generic": RobotDataAdapter,
    }
    
    if robot_type not in adapters:
        raise ValueError(f"不支持的机器人类型: {robot_type}")
    
    return adapters[robot_type]()