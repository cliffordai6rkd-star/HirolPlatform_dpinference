#!/usr/bin/env python3
"""学习推理工厂测试."""

import unittest
import tempfile
import os
import pickle
from unittest.mock import Mock, patch, MagicMock

from factory.components.learning_inference_factory import (
    LearningInferenceFactory,
    create_act_inference
)
from learning.inference.policy_inference import ACTInference
from learning.inference.data_adapter import FR3DataAdapter


class TestLearningInferenceFactory(unittest.TestCase):
    """LearningInferenceFactory测试."""
    
    def setUp(self):
        """测试设置."""
        # 创建临时检查点目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建必要的文件
        self.dataset_stats_path = os.path.join(self.temp_dir, "dataset_stats.pkl")
        self.policy_path = os.path.join(self.temp_dir, "policy_best.ckpt")
        
        # 创建模拟的数据集统计文件
        mock_stats = {
            "qpos_mean": [0.0] * 7,
            "qpos_std": [1.0] * 7,
            "action_mean": [0.0] * 7,
            "action_std": [1.0] * 7
        }
        with open(self.dataset_stats_path, "wb") as f:
            pickle.dump(mock_stats, f)
        
        # 创建模拟的策略文件
        with open(self.policy_path, "wb") as f:
            pickle.dump({"model_state": "mock"}, f)
    
    def tearDown(self):
        """清理测试环境."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_supported_algorithms(self):
        """测试获取支持的算法列表."""
        algorithms = LearningInferenceFactory.get_supported_algorithms()
        
        self.assertIn("ACT", algorithms)
        self.assertIn("act", algorithms)
        self.assertIsInstance(algorithms, list)
    
    def test_get_supported_robots(self):
        """测试获取支持的机器人类型列表."""
        robots = LearningInferenceFactory.get_supported_robots()
        
        self.assertIn("fr3", robots)
        self.assertIn("monte01", robots)
        self.assertIn("generic", robots)
        self.assertIsInstance(robots, list)
    
    def test_validate_checkpoint_directory_valid(self):
        """测试有效检查点目录验证."""
        result = LearningInferenceFactory.validate_checkpoint_directory(
            self.temp_dir, "ACT"
        )
        
        self.assertTrue(result)
    
    def test_validate_checkpoint_directory_missing_stats(self):
        """测试缺少统计文件的检查点目录."""
        os.remove(self.dataset_stats_path)
        
        result = LearningInferenceFactory.validate_checkpoint_directory(
            self.temp_dir, "ACT"
        )
        
        self.assertFalse(result)
    
    def test_validate_checkpoint_directory_missing_policy(self):
        """测试缺少策略文件的检查点目录."""
        os.remove(self.policy_path)
        
        result = LearningInferenceFactory.validate_checkpoint_directory(
            self.temp_dir, "ACT"
        )
        
        self.assertFalse(result)
    
    def test_validate_checkpoint_directory_nonexistent(self):
        """测试不存在的检查点目录."""
        result = LearningInferenceFactory.validate_checkpoint_directory(
            "/nonexistent/path", "ACT"
        )
        
        self.assertFalse(result)
    
    @patch('learning.inference.policy_inference.ACTInference')
    def test_create_inference_engine_act(self, mock_act_class):
        """测试创建ACT推理引擎."""
        # 配置mock
        mock_inference = Mock()
        mock_act_class.return_value = mock_inference
        
        config = {
            "state_dim": 7,
            "camera_names": ["front_camera"],
            "num_queries": 100,
            "kl_weight": 10
        }
        
        result = LearningInferenceFactory.create_inference_engine(
            "ACT", self.temp_dir, config
        )
        
        # 验证
        self.assertEqual(result, mock_inference)
        mock_act_class.assert_called_once_with(
            ckpt_dir=self.temp_dir,
            state_dim=7,
            camera_names=["front_camera"],
            num_queries=100,
            kl_weight=10
        )
    
    def test_create_inference_engine_unsupported_algorithm(self):
        """测试创建不支持的算法引擎."""
        config = {"state_dim": 7, "camera_names": ["front_camera"]}
        
        with self.assertRaises(ValueError) as cm:
            LearningInferenceFactory.create_inference_engine(
                "UNSUPPORTED", self.temp_dir, config
            )
        
        self.assertIn("不支持的学习算法", str(cm.exception))
    
    def test_create_inference_engine_nonexistent_checkpoint(self):
        """测试检查点目录不存在时的错误处理."""
        config = {"state_dim": 7, "camera_names": ["front_camera"]}
        
        with self.assertRaises(FileNotFoundError):
            LearningInferenceFactory.create_inference_engine(
                "ACT", "/nonexistent/path", config
            )
    
    def test_create_inference_engine_invalid_config(self):
        """测试无效配置的错误处理."""
        # 缺少必需参数
        invalid_config = {"state_dim": 7}  # 缺少camera_names
        
        with self.assertRaises(ValueError) as cm:
            LearningInferenceFactory.create_inference_engine(
                "ACT", self.temp_dir, invalid_config
            )
        
        self.assertIn("缺少必需参数", str(cm.exception))
    
    def test_create_data_adapter_fr3(self):
        """测试创建FR3数据适配器."""
        adapter = LearningInferenceFactory.create_data_adapter("fr3")
        
        self.assertIsInstance(adapter, FR3DataAdapter)
    
    def test_create_data_adapter_unsupported(self):
        """测试创建不支持的数据适配器."""
        with self.assertRaises(ValueError) as cm:
            LearningInferenceFactory.create_data_adapter("unsupported")
        
        self.assertIn("不支持的机器人类型", str(cm.exception))
    
    @patch('learning.inference.policy_inference.ACTInference')
    def test_create_learning_pipeline(self, mock_act_class):
        """测试创建完整的学习推理流水线."""
        # 配置mock
        mock_inference = Mock()
        mock_act_class.return_value = mock_inference
        
        config = {
            "state_dim": 7,
            "camera_names": ["front_camera"]
        }
        
        inference_engine, data_adapter = LearningInferenceFactory.create_learning_pipeline(
            "ACT", "fr3", self.temp_dir, config
        )
        
        # 验证
        self.assertEqual(inference_engine, mock_inference)
        self.assertIsInstance(data_adapter, FR3DataAdapter)
    
    def test_validate_config_valid_act(self):
        """测试有效的ACT配置验证."""
        valid_config = {
            "state_dim": 7,
            "camera_names": ["front_camera", "wrist_camera"]
        }
        
        # 应该不抛出异常
        LearningInferenceFactory._validate_config("ACT", valid_config)
    
    def test_validate_config_invalid_act(self):
        """测试无效的ACT配置验证."""
        # 缺少必需参数
        invalid_config = {"state_dim": 7}
        
        with self.assertRaises(ValueError):
            LearningInferenceFactory._validate_config("ACT", invalid_config)
        
        # 错误的参数类型
        invalid_config = {
            "state_dim": "seven",  # 应该是int
            "camera_names": ["front_camera"]
        }
        
        with self.assertRaises(ValueError):
            LearningInferenceFactory._validate_config("ACT", invalid_config)


class TestCreateACTInference(unittest.TestCase):
    """create_act_inference便捷函数测试."""
    
    def setUp(self):
        """测试设置."""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建必要的文件
        dataset_stats = {
            "qpos_mean": [0.0] * 7,
            "qpos_std": [1.0] * 7,
            "action_mean": [0.0] * 7,
            "action_std": [1.0] * 7
        }
        with open(os.path.join(self.temp_dir, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(dataset_stats, f)
        
        with open(os.path.join(self.temp_dir, "policy_best.ckpt"), "wb") as f:
            pickle.dump({"model": "mock"}, f)
    
    def tearDown(self):
        """清理测试环境."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('learning.inference.policy_inference.ACTInference')
    def test_create_act_inference_convenience_function(self, mock_act_class):
        """测试便捷的ACT推理创建函数."""
        mock_inference = Mock()
        mock_act_class.return_value = mock_inference
        
        inference_engine, data_adapter = create_act_inference(
            ckpt_dir=self.temp_dir,
            state_dim=7,
            camera_names=["front_camera"],
            robot_type="fr3",
            num_queries=50
        )
        
        # 验证
        self.assertEqual(inference_engine, mock_inference)
        self.assertIsInstance(data_adapter, FR3DataAdapter)
        
        # 验证ACT被正确调用
        mock_act_class.assert_called_once()
        call_args = mock_act_class.call_args
        self.assertEqual(call_args[1]["ckpt_dir"], self.temp_dir)
        self.assertEqual(call_args[1]["state_dim"], 7)
        self.assertEqual(call_args[1]["camera_names"], ["front_camera"])
        self.assertEqual(call_args[1]["num_queries"], 50)


if __name__ == "__main__":
    unittest.main()