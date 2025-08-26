#!/usr/bin/env python3
"""ACT集成测试."""

import unittest
import tempfile
import os
import pickle
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from learning.inference.policy_inference import ACTInference, PolicyInference
from learning.inference.data_adapter import RobotDataAdapter
from factory.components.learning_inference_factory import LearningInferenceFactory
from hardware.base.utils import RobotJointState


class TestACTIntegration(unittest.TestCase):
    """ACT推理集成测试."""
    
    def setUp(self):
        """测试设置."""
        # 创建临时检查点目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟的数据集统计文件
        self.mock_stats = {
            "qpos_mean": np.array([0.0] * 7),
            "qpos_std": np.array([1.0] * 7),
            "action_mean": np.array([0.0] * 7),
            "action_std": np.array([1.0] * 7)
        }
        
        dataset_stats_path = os.path.join(self.temp_dir, "dataset_stats.pkl")
        with open(dataset_stats_path, "wb") as f:
            pickle.dump(self.mock_stats, f)
        
        # 创建模拟的策略文件
        policy_path = os.path.join(self.temp_dir, "policy_best.ckpt")
        with open(policy_path, "wb") as f:
            pickle.dump({"model_state": "mock"}, f)
    
    def tearDown(self):
        """清理测试环境."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('learning.policies.act.policy.ACTPolicy')
    @patch('torch.load')
    def test_act_inference_initialization(self, mock_torch_load, mock_act_policy):
        """测试ACT推理器初始化."""
        # 配置mocks
        mock_policy = Mock()
        mock_act_policy.return_value = mock_policy
        mock_torch_load.return_value = {"state_dict": "mock"}
        
        # 创建ACT推理器
        act_inference = ACTInference(
            ckpt_dir=self.temp_dir,
            state_dim=7,
            camera_names=["front_camera"]
        )
        
        # 验证初始化
        self.assertEqual(act_inference.ckpt_dir, self.temp_dir)
        self.assertEqual(act_inference.policy_config["action_dim"], 7)
        self.assertEqual(act_inference.policy_config["camera_names"], ["front_camera"])
        
        # 验证策略创建
        mock_act_policy.assert_called_once()
        mock_policy.to.assert_called_once()
        mock_policy.eval.assert_called_once()
    
    @patch('learning.policies.act.policy.ACTPolicy')
    @patch('torch.load')
    def test_act_inference_prediction_pipeline(self, mock_torch_load, mock_act_policy):
        """测试ACT策略完整预测流程."""
        # 配置mocks
        mock_policy = Mock()
        mock_act_policy.return_value = mock_policy
        mock_torch_load.return_value = {"state_dict": "mock"}
        
        # 模拟策略预测输出
        mock_prediction = Mock()
        mock_prediction.cpu.return_value.numpy.return_value.squeeze.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        mock_policy.return_value = mock_prediction
        
        # 创建ACT推理器
        act_inference = ACTInference(
            ckpt_dir=self.temp_dir,
            state_dim=7,
            camera_names=["front_camera"]
        )
        
        # 准备测试数据
        test_state = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        test_images = {
            "front_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        # 执行预测
        with patch('learning.inference.data_adapter.RobotDataAdapter.camera_dict_to_tensor') as mock_tensor_conv:
            mock_tensor_conv.return_value = Mock()  # 模拟tensor
            
            result = act_inference.predict(test_state, test_images)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 7)
        
        # 验证策略被调用
        mock_policy.assert_called_once()
    
    def test_act_inference_checkpoint_loading_robustness(self):
        """测试检查点加载的健壮性."""
        # 测试备份文件回退
        # 删除主文件，创建备份文件
        main_policy_path = os.path.join(self.temp_dir, "policy_best.ckpt")
        backup_policy_path = os.path.join(self.temp_dir, "policy_best_bac.ckpt")
        
        os.remove(main_policy_path)
        with open(backup_policy_path, "wb") as f:
            pickle.dump({"backup_model": "mock"}, f)
        
        with patch('learning.policies.act.policy.ACTPolicy'):
            with patch('torch.load') as mock_torch_load:
                mock_torch_load.return_value = {"backup_state": "mock"}
                
                # 应该能够成功加载备份文件
                act_inference = ACTInference(
                    ckpt_dir=self.temp_dir,
                    state_dim=7,
                    camera_names=["front_camera"]
                )
                
                # 验证加载了备份文件
                mock_torch_load.assert_called_with(backup_policy_path, map_location=act_inference.device, weights_only=False)
    
    def test_act_inference_missing_checkpoint_error(self):
        """测试检查点文件缺失时的错误处理."""
        # 删除所有检查点文件
        os.remove(os.path.join(self.temp_dir, "policy_best.ckpt"))
        
        with self.assertRaises(FileNotFoundError):
            ACTInference(
                ckpt_dir=self.temp_dir,
                state_dim=7,
                camera_names=["front_camera"]
            )
    
    def test_dataset_stats_loading_robustness(self):
        """测试数据集统计文件加载的健壮性."""
        # 删除主统计文件，创建备份文件
        main_stats_path = os.path.join(self.temp_dir, "dataset_stats.pkl")
        backup_stats_path = os.path.join(self.temp_dir, "dataset_stats_bac.pkl")
        
        os.remove(main_stats_path)
        with open(backup_stats_path, "wb") as f:
            pickle.dump(self.mock_stats, f)
        
        with patch('learning.policies.act.policy.ACTPolicy'):
            with patch('torch.load'):
                # 应该能够成功加载备份统计文件
                act_inference = ACTInference(
                    ckpt_dir=self.temp_dir,
                    state_dim=7,
                    camera_names=["front_camera"]
                )
                
                # 验证统计信息加载成功
                self.assertIsNotNone(act_inference.dataset_stats)
    
    @patch('learning.policies.act.policy.ACTPolicy')
    @patch('torch.load')
    def test_prediction_output_format(self, mock_torch_load, mock_act_policy):
        """测试预测输出格式符合预期."""
        # 配置mocks
        mock_policy = Mock()
        mock_act_policy.return_value = mock_policy
        mock_torch_load.return_value = {"state_dict": "mock"}
        
        # 模拟不同维度的策略输出
        test_cases = [
            # (模拟输出, 期望结果长度)
            (np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]), 7),  # 带batch维度
            (np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), 7),     # 无batch维度
        ]
        
        for mock_output, expected_length in test_cases:
            with self.subTest(output_shape=mock_output.shape):
                # 重置mock
                mock_policy.reset_mock()
                
                # 配置mock返回值 - 模拟tensor输出
                mock_prediction = Mock()
                if len(mock_output.shape) == 1:
                    # 一维数组，添加batch维度进行模拟
                    tensor_output = mock_output.reshape(1, -1)
                else:
                    # 二维数组，直接使用
                    tensor_output = mock_output
                mock_prediction.cpu.return_value.numpy.return_value = tensor_output
                mock_policy.return_value = mock_prediction
                
                # 创建推理器
                act_inference = ACTInference(
                    ckpt_dir=self.temp_dir,
                    state_dim=7,
                    camera_names=["front_camera"]
                )
                
                # 执行预测
                test_state = np.array([0.0] * 7)
                test_images = {"front_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
                
                with patch('learning.inference.data_adapter.RobotDataAdapter.camera_dict_to_tensor'):
                    result = act_inference.predict(test_state, test_images)
                
                # 验证输出格式
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(len(result), expected_length)
                
                # 验证所有值都是有效数值
                self.assertFalse(np.any(np.isnan(result)))
                self.assertFalse(np.any(np.isinf(result)))
    
    def test_data_adapter_integration(self):
        """测试数据适配器集成."""
        adapter = RobotDataAdapter("test")
        
        # 测试关节状态转换
        joint_state = RobotJointState()
        joint_state._positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        state_array = adapter.robot_state_to_numpy(joint_state)
        
        self.assertIsInstance(state_array, np.ndarray)
        self.assertEqual(len(state_array), 7)
        
        # 测试相机数据转换
        camera_data = {
            "front_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        image_tensor = adapter.camera_dict_to_tensor(camera_data)
        
        self.assertEqual(image_tensor.shape, (1, 1, 3, 480, 640))
        
        # 测试动作转换
        mock_actions = np.array([0.5, -0.3, 0.8, 0.0, -0.2, 0.1, 0.4])
        action_commands = adapter.numpy_to_robot_actions(mock_actions)
        
        self.assertIsInstance(action_commands, list)
        self.assertEqual(len(action_commands), 7)
        for action in action_commands:
            self.assertIsInstance(action, float)


if __name__ == "__main__":
    unittest.main()