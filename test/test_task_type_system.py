#!/usr/bin/env python3
"""
任务类型系统单元测试
测试TaskType枚举、工厂方法和策略系统
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入被测试的模块
import sys
sys.path.append('/workspace')

from factory.tasks.inferences_tasks.utils.task_types import (
    TaskType, TaskTypeFactory, TaskConfigurationError
)
from factory.tasks.inferences_tasks.utils.gripper_strategies import (
    GripperStrategyFactory, GraspReleaseStrategy,
    LiquidTransferStrategy, SolidTransferStrategy
)


class TestTaskType(unittest.TestCase):
    """测试TaskType枚举"""

    def test_task_type_values(self):
        """测试任务类型枚举值"""
        self.assertEqual(TaskType.PEG_IN_HOLE.value, "peg_in_hole")
        self.assertEqual(TaskType.BLOCK_STACKING.value, "block_stacking")
        self.assertEqual(TaskType.LIQUID_TRANSFER.value, "liquid_transfer")
        self.assertEqual(TaskType.SOLID_TRANSFER.value, "solid_transfer")

    def test_display_names(self):
        """测试显示名称"""
        self.assertEqual(TaskType.PEG_IN_HOLE.display_name, "插孔任务")
        self.assertEqual(TaskType.BLOCK_STACKING.display_name, "叠方块任务")
        self.assertEqual(TaskType.LIQUID_TRANSFER.display_name, "倒水任务")
        self.assertEqual(TaskType.SOLID_TRANSFER.display_name, "固体转移任务")

    def test_from_checkpoint_peg_in_hole(self):
        """测试从checkpoint路径推断插孔任务"""
        test_paths = [
            "learning/ckpts/fr3_peg_in_hole_0914",
            "learning/ckpts/fr3_pih_0918_48ep",
            "learning/ckpts/monte01_peg_insertion"
        ]

        for path in test_paths:
            with self.subTest(path=path):
                task_type = TaskType.from_checkpoint(path)
                self.assertEqual(task_type, TaskType.PEG_IN_HOLE)

    def test_from_checkpoint_block_stacking(self):
        """测试从checkpoint路径推断叠方块任务"""
        test_paths = [
            "learning/ckpts/fr3_block_stacking_0915",
            "learning/ckpts/fr3_bs_0916_50ep",
            "learning/ckpts/fr3_blockstack_latest",
            "learning/ckpts/fr3_lego_tower"
        ]

        for path in test_paths:
            with self.subTest(path=path):
                task_type = TaskType.from_checkpoint(path)
                self.assertEqual(task_type, TaskType.BLOCK_STACKING)

    def test_from_checkpoint_liquid_transfer(self):
        """测试从checkpoint路径推断倒水任务"""
        test_paths = [
            "learning/ckpts/fr3_liquid_transfer_0920",
            "learning/ckpts/pour_water_v2",
            "learning/ckpts/liquid_pouring_task"
        ]

        for path in test_paths:
            with self.subTest(path=path):
                task_type = TaskType.from_checkpoint(path)
                self.assertEqual(task_type, TaskType.LIQUID_TRANSFER)

    def test_from_checkpoint_default(self):
        """测试未知路径返回默认任务类型"""
        unknown_paths = [
            "learning/ckpts/unknown_task",
            "learning/ckpts/random_model"
        ]

        for path in unknown_paths:
            with self.subTest(path=path):
                task_type = TaskType.from_checkpoint(path)
                self.assertEqual(task_type, TaskType.SOLID_TRANSFER)

    def test_from_checkpoint_empty_path(self):
        """测试空路径抛出异常"""
        with self.assertRaises(ValueError):
            TaskType.from_checkpoint("")

    def test_from_config_explicit_type(self):
        """测试从配置中显式获取任务类型"""
        config = {"task_type": "peg_in_hole"}
        task_type = TaskType.from_config(config)
        self.assertEqual(task_type, TaskType.PEG_IN_HOLE)

    def test_from_config_invalid_explicit_type(self):
        """测试无效的显式任务类型回退到checkpoint推断"""
        config = {
            "task_type": "invalid_type",
            "checkpoint_path": "learning/ckpts/fr3_bs_0916"
        }
        task_type = TaskType.from_config(config)
        self.assertEqual(task_type, TaskType.BLOCK_STACKING)

    def test_from_config_checkpoint_fallback(self):
        """测试从checkpoint路径推断任务类型"""
        config = {"checkpoint_path": "learning/ckpts/fr3_liquid_transfer_0920"}
        task_type = TaskType.from_config(config)
        self.assertEqual(task_type, TaskType.LIQUID_TRANSFER)


class TestTaskTypeFactory(unittest.TestCase):
    """测试TaskTypeFactory工厂类"""

    def test_get_default_max_steps(self):
        """测试获取默认最大步数"""
        expected_steps = {
            TaskType.PEG_IN_HOLE: 600,
            TaskType.BLOCK_STACKING: 550,
            TaskType.LIQUID_TRANSFER: 800,
            TaskType.SOLID_TRANSFER: 500
        }

        for task_type, expected in expected_steps.items():
            with self.subTest(task_type=task_type):
                steps = TaskTypeFactory.get_default_max_steps(task_type)
                self.assertEqual(steps, expected)

    def test_get_task_characteristics(self):
        """测试获取任务特征"""
        # 测试插孔任务特征
        pih_chars = TaskTypeFactory.get_task_characteristics(TaskType.PEG_IN_HOLE)
        self.assertTrue(pih_chars['requires_precision'])
        self.assertEqual(pih_chars['gripper_control_type'], 'act_driven')
        self.assertTrue(pih_chars['completion_detection'])
        self.assertTrue(pih_chars['force_open_after_completion'])

        # 测试倒水任务特征
        liquid_chars = TaskTypeFactory.get_task_characteristics(TaskType.LIQUID_TRANSFER)
        self.assertFalse(liquid_chars['requires_precision'])
        self.assertEqual(liquid_chars['gripper_control_type'], 'hold_after_close')
        self.assertFalse(liquid_chars['completion_detection'])
        self.assertFalse(liquid_chars['force_open_after_completion'])

    def test_validate_task_config_valid(self):
        """测试有效配置验证"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")
            os.makedirs(checkpoint_path)

            config = {
                "checkpoint_path": checkpoint_path,
                "gripper_postprocess": {"enabled": True}
            }

            # 应该不抛出异常
            result = TaskTypeFactory.validate_task_config(TaskType.PEG_IN_HOLE, config)
            self.assertTrue(result)

    def test_validate_task_config_missing_checkpoint(self):
        """测试缺少checkpoint路径的配置"""
        config = {"gripper_postprocess": {"enabled": True}}

        with self.assertRaises(TaskConfigurationError):
            TaskTypeFactory.validate_task_config(TaskType.PEG_IN_HOLE, config)

    def test_validate_task_config_nonexistent_checkpoint(self):
        """测试不存在的checkpoint路径"""
        config = {
            "checkpoint_path": "/nonexistent/path",
            "gripper_postprocess": {"enabled": True}
        }

        with self.assertRaises(TaskConfigurationError):
            TaskTypeFactory.validate_task_config(TaskType.PEG_IN_HOLE, config)


class TestGripperStrategies(unittest.TestCase):
    """测试夹爪策略系统"""

    def test_strategy_factory_creation(self):
        """测试策略工厂创建"""
        config = {"min_holding_steps": 30}

        # 测试创建抓放策略
        grasp_strategy = GripperStrategyFactory.create("peg_in_hole", config)
        self.assertIsInstance(grasp_strategy, GraspReleaseStrategy)

        # 测试创建倒水策略
        liquid_strategy = GripperStrategyFactory.create("liquid_transfer", config)
        self.assertIsInstance(liquid_strategy, LiquidTransferStrategy)

        # 测试创建固体转移策略
        solid_strategy = GripperStrategyFactory.create("solid_transfer", config)
        self.assertIsInstance(solid_strategy, SolidTransferStrategy)

    def test_strategy_factory_unknown_type(self):
        """测试未知任务类型抛出异常"""
        config = {}

        with self.assertRaises(ValueError):
            GripperStrategyFactory.create("unknown_type", config)

    def test_grasp_release_strategy_max_steps(self):
        """测试抓放策略的最大步数"""
        config = {"max_step_nums": 600}
        strategy = GraspReleaseStrategy(config)
        self.assertEqual(strategy.get_max_steps(), 600)

        # 测试默认值
        config_default = {}
        strategy_default = GraspReleaseStrategy(config_default)
        self.assertEqual(strategy_default.get_max_steps(), 550)  # 默认值

    def test_liquid_transfer_strategy_behavior(self):
        """测试倒水策略行为"""
        config = {"max_step_nums": 800}
        strategy = LiquidTransferStrategy(config)

        # 测试最大步数
        self.assertEqual(strategy.get_max_steps(), 800)

        # 测试初始状态
        self.assertEqual(strategy.get_state(), 'OPEN')

        # 测试不会自动打开
        context = {}
        should_open = strategy.should_open(0.9, context)
        self.assertFalse(should_open)

    def test_solid_transfer_strategy_behavior(self):
        """测试固体转移策略行为"""
        config = {"max_step_nums": 500}
        strategy = SolidTransferStrategy(config)

        # 测试最大步数
        self.assertEqual(strategy.get_max_steps(), 500)

        # 测试初始状态为保持闭合
        self.assertEqual(strategy.get_state(), 'HOLDING')

        # 测试始终不闭合（已经闭合）
        context = {}
        should_close = strategy.should_close(0.1, context)
        self.assertFalse(should_close)

        # 测试始终不打开
        should_open = strategy.should_open(0.9, context)
        self.assertFalse(should_open)

    def test_strategy_reset(self):
        """测试策略重置"""
        config = {}
        strategy = GraspReleaseStrategy(config)

        # 改变状态
        strategy.state = 'HOLDING'

        # 重置
        strategy.reset()

        # 验证状态已重置
        self.assertEqual(strategy.get_state(), 'OPEN')

    def test_strategy_debug_info(self):
        """测试策略调试信息"""
        config = {}
        strategy = GraspReleaseStrategy(config)

        debug_info = strategy.get_debug_info()

        self.assertIn('strategy', debug_info)
        self.assertIn('state', debug_info)
        self.assertEqual(debug_info['strategy'], 'GraspReleaseStrategy')
        self.assertEqual(debug_info['state'], 'OPEN')


if __name__ == '__main__':
    unittest.main()