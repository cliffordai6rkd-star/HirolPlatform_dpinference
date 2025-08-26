#!/usr/bin/env python3
"""数据适配器测试."""

import unittest
import numpy as np
from unittest.mock import Mock

from hardware.base.utils import RobotJointState
from learning.inference.data_adapter import (
    RobotDataAdapter, 
    FR3DataAdapter, 
    Monte01DataAdapter,
    create_data_adapter
)


class TestRobotDataAdapter(unittest.TestCase):
    """RobotDataAdapter基类测试."""
    
    def setUp(self):
        """测试设置."""
        self.adapter = RobotDataAdapter("test_robot")
    
    def test_robot_state_to_numpy_conversion(self):
        """测试RobotJointState到numpy的转换准确性."""
        # 创建测试关节状态
        joint_state = RobotJointState()
        test_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        joint_state._positions = test_positions
        
        # 转换
        result = self.adapter.robot_state_to_numpy(joint_state)
        
        # 验证
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, test_positions)
    
    def test_robot_state_to_numpy_with_numpy_input(self):
        """测试numpy输入的处理."""
        joint_state = RobotJointState()
        test_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        joint_state._positions = test_positions
        
        result = self.adapter.robot_state_to_numpy(joint_state)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, test_positions)
    
    def test_robot_state_to_numpy_invalid_input(self):
        """测试无效输入的错误处理."""
        joint_state = RobotJointState()
        joint_state._positions = None
        
        with self.assertRaises(ValueError):
            self.adapter.robot_state_to_numpy(joint_state)
    
    def test_numpy_to_robot_actions_conversion(self):
        """测试numpy到机器人指令的转换准确性."""
        test_actions = np.array([0.5, -0.3, 0.8, 0.0, -0.2, 0.1, 0.4], dtype=np.float32)
        
        result = self.adapter.numpy_to_robot_actions(test_actions)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(test_actions))
        for i, action in enumerate(result):
            self.assertIsInstance(action, float)
            self.assertAlmostEqual(action, test_actions[i], places=5)
    
    def test_numpy_to_robot_actions_multidimensional(self):
        """测试多维输入的处理."""
        # 测试2D数组 (batch_size=1)
        test_actions = np.array([[0.5, -0.3, 0.8]], dtype=np.float32)
        
        result = self.adapter.numpy_to_robot_actions(test_actions)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
    
    def test_numpy_to_robot_actions_invalid_input(self):
        """测试无效输入的错误处理."""
        with self.assertRaises(ValueError):
            self.adapter.numpy_to_robot_actions([0.1, 0.2, 0.3])  # 不是numpy数组
    
    def test_camera_dict_to_tensor_conversion(self):
        """测试相机数据到tensor的转换格式正确性."""
        # 创建测试相机数据
        camera_data = {
            "front_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        result = self.adapter.camera_dict_to_tensor(camera_data)
        
        # 验证tensor格式
        self.assertEqual(result.shape, (1, 2, 3, 480, 640))  # (batch, num_cameras, channels, height, width)
        self.assertTrue(str(result.dtype) in ['torch.float32', 'torch.float64'])  # torch.float tensor
    
    def test_camera_dict_to_tensor_target_size(self):
        """测试图像尺寸调整."""
        camera_data = {
            "camera1": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        }
        target_size = (480, 640)
        
        result = self.adapter.camera_dict_to_tensor(camera_data, target_size=target_size)
        
        self.assertEqual(result.shape, (1, 1, 3, 480, 640))
    
    def test_camera_dict_to_tensor_invalid_input(self):
        """测试无效相机数据的错误处理."""
        # 空字典
        with self.assertRaises(ValueError):
            self.adapter.camera_dict_to_tensor({})
        
        # 错误的图像格式
        invalid_data = {"camera1": np.random.rand(100, 100)}  # 缺少通道维度
        with self.assertRaises(ValueError):
            self.adapter.camera_dict_to_tensor(invalid_data)
    
    def test_validate_robot_state_valid(self):
        """测试有效机器人状态验证."""
        joint_state = RobotJointState()
        joint_state._positions = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result = self.adapter.validate_robot_state(joint_state)
        
        self.assertTrue(result)
    
    def test_validate_robot_state_invalid(self):
        """测试无效机器人状态验证."""
        # 空位置
        joint_state = RobotJointState()
        joint_state._positions = None
        
        result = self.adapter.validate_robot_state(joint_state)
        
        self.assertFalse(result)
        
        # 包含NaN
        joint_state._positions = [0.1, np.nan, 0.3]
        
        result = self.adapter.validate_robot_state(joint_state)
        
        self.assertFalse(result)
    
    def test_validate_camera_data_valid(self):
        """测试有效相机数据验证."""
        camera_data = {
            "camera1": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        result = self.adapter.validate_camera_data(camera_data)
        
        self.assertTrue(result)
    
    def test_validate_camera_data_invalid(self):
        """测试无效相机数据验证."""
        # 空字典
        result = self.adapter.validate_camera_data({})
        self.assertFalse(result)
        
        # 错误的数据类型
        invalid_data = {"camera1": [1, 2, 3]}
        result = self.adapter.validate_camera_data(invalid_data)
        self.assertFalse(result)
        
        # 错误的图像格式
        invalid_image = {"camera1": np.random.rand(100, 100)}  # 缺少通道
        result = self.adapter.validate_camera_data(invalid_image)
        self.assertFalse(result)


class TestFR3DataAdapter(unittest.TestCase):
    """FR3DataAdapter测试."""
    
    def setUp(self):
        """测试设置."""
        self.adapter = FR3DataAdapter()
    
    def test_fr3_specific_validation(self):
        """测试FR3特定的状态验证."""
        # 正确的7自由度
        joint_state = RobotJointState()
        joint_state._positions = [0.0] * 7
        
        result = self.adapter.validate_robot_state(joint_state)
        
        self.assertTrue(result)
        
        # 错误的自由度数量
        joint_state._positions = [0.0] * 6
        
        result = self.adapter.validate_robot_state(joint_state)
        
        self.assertFalse(result)


class TestMonte01DataAdapter(unittest.TestCase):
    """Monte01DataAdapter测试."""
    
    def setUp(self):
        """测试设置."""
        self.adapter = Monte01DataAdapter()
    
    def test_monte01_specific_validation(self):
        """测试Monte01特定的状态验证."""
        # 正确的7自由度
        joint_state = RobotJointState()
        joint_state._positions = [0.0] * 7
        
        result = self.adapter.validate_robot_state(joint_state)
        
        self.assertTrue(result)


class TestDataAdapterFactory(unittest.TestCase):
    """数据适配器工厂测试."""
    
    def test_create_fr3_adapter(self):
        """测试创建FR3适配器."""
        adapter = create_data_adapter("fr3")
        
        self.assertIsInstance(adapter, FR3DataAdapter)
    
    def test_create_monte01_adapter(self):
        """测试创建Monte01适配器."""
        adapter = create_data_adapter("monte01")
        
        self.assertIsInstance(adapter, Monte01DataAdapter)
    
    def test_create_generic_adapter(self):
        """测试创建通用适配器."""
        adapter = create_data_adapter("generic")
        
        self.assertIsInstance(adapter, RobotDataAdapter)
    
    def test_create_unsupported_adapter(self):
        """测试创建不支持的适配器."""
        with self.assertRaises(ValueError):
            create_data_adapter("unsupported_robot")


if __name__ == "__main__":
    unittest.main()