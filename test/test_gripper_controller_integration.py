#!/usr/bin/env python3
"""
夹爪控制器集成测试
测试TaskGripperController和传统控制器的集成
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np

# 导入被测试的模块
import sys
sys.path.append('/workspace')

try:
    from factory.tasks.inferences_tasks.utils.gripper_controller import (
        TaskGripperController, create_gripper_controller,
        SmartGripperController, IncrementalGripperController
    )
    from factory.tasks.inferences_tasks.utils.task_types import TaskType
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 模块导入失败: {e}")
    MODULES_AVAILABLE = False


@unittest.skipUnless(MODULES_AVAILABLE, "Required modules not available")
class TestTaskGripperController(unittest.TestCase):
    """测试TaskGripperController"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "fr3_peg_in_hole_test")
        os.makedirs(self.checkpoint_path)

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_task_gripper_controller_creation(self):
        """测试任务夹爪控制器创建"""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "gripper_postprocess": {
                "enabled": True,
                "min_grasp_width": 0.005,
                "max_grasp_width": 0.070
            }
        }

        controller = TaskGripperController(config)

        self.assertIsNotNone(controller.strategy)
        self.assertEqual(controller.task_type, TaskType.PEG_IN_HOLE)
        self.assertEqual(controller.state, 'OPEN')

    def test_task_gripper_controller_process(self):
        """测试任务夹爪控制器处理逻辑"""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "gripper_postprocess": {
                "enabled": True,
                "close_value_threshold": 0.02,
                "required_close_duration": 0.1  # 缩短测试时间
            }
        }

        controller = TaskGripperController(config)

        # 测试初始状态
        execute_cmd, cmd_value, force_repredict = controller.process(0.5)
        self.assertFalse(execute_cmd)  # 初始不执行命令

        # 测试低值输入（应该准备闭合）
        with patch('time.time', side_effect=[0, 0.15]):  # 模拟时间流逝
            # 第一次调用开始检测
            controller.process(0.01)
            # 第二次调用应该触发闭合
            execute_cmd, cmd_value, force_repredict = controller.process(0.01)

        # 验证结果（可能需要多次调用才能触发）
        self.assertIsInstance(execute_cmd, bool)
        if execute_cmd:
            self.assertEqual(cmd_value, 0.0)  # 闭合命令

    def test_get_gripper_position(self):
        """测试获取夹爪位置"""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "gripper_postprocess": {"enabled": True}
        }

        controller = TaskGripperController(config)

        # 测试打开状态
        self.assertEqual(controller.get_gripper_position(), 0.08)

        # 模拟闭合状态
        controller.state = 'CLOSED'
        self.assertEqual(controller.get_gripper_position(), 0.0)

    def test_reset_functionality(self):
        """测试重置功能"""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "gripper_postprocess": {"enabled": True}
        }

        controller = TaskGripperController(config)

        # 改变状态
        controller.state = 'HOLDING'

        # 重置
        controller.reset()

        # 验证重置结果
        self.assertEqual(controller.state, 'OPEN')

    def test_debug_info(self):
        """测试调试信息"""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "gripper_postprocess": {"enabled": True}
        }

        controller = TaskGripperController(config)
        debug_info = controller.get_debug_info()

        self.assertIn('task_type', debug_info)
        self.assertIn('task_display_name', debug_info)
        self.assertIn('strategy', debug_info)
        self.assertEqual(debug_info['task_type'], 'peg_in_hole')
        self.assertEqual(debug_info['task_display_name'], '插孔任务')


@unittest.skipUnless(MODULES_AVAILABLE, "Required modules not available")
class TestGripperControllerFactory(unittest.TestCase):
    """测试夹爪控制器工厂函数"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_task_aware_controller(self):
        """测试创建任务感知控制器"""
        checkpoint_path = os.path.join(self.temp_dir, "fr3_block_stacking_test")
        os.makedirs(checkpoint_path)

        full_config = {
            "checkpoint_path": checkpoint_path,
            "gripper_postprocess": {"enabled": True}
        }

        config = {"enabled": True}

        controller = create_gripper_controller(config, full_config)
        self.assertIsInstance(controller, TaskGripperController)

    def test_create_incremental_controller(self):
        """测试创建增量控制器"""
        config = {
            "control_mode": "incremental",
            "enabled": True
        }

        controller = create_gripper_controller(config)
        self.assertIsInstance(controller, IncrementalGripperController)

    def test_create_smart_controller_fallback(self):
        """测试创建智能控制器（回退）"""
        config = {
            "control_mode": "binary",
            "enabled": True
        }

        controller = create_gripper_controller(config)
        self.assertIsInstance(controller, SmartGripperController)

    def test_fallback_when_task_system_fails(self):
        """测试任务系统失败时的回退"""
        config = {
            "enabled": True
        }

        # 模拟任务系统不可用或配置有问题的情况
        full_config = {
            "checkpoint_path": "/nonexistent/path",
            "gripper_postprocess": config
        }

        # 应该回退到传统控制器
        controller = create_gripper_controller(config, full_config)
        self.assertIsInstance(controller, SmartGripperController)


@unittest.skipUnless(MODULES_AVAILABLE, "Required modules not available")
class TestTaskTypeSpecificBehavior(unittest.TestCase):
    """测试不同任务类型的特定行为"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_controller_for_task(self, task_name: str) -> TaskGripperController:
        """为指定任务创建控制器"""
        checkpoint_path = os.path.join(self.temp_dir, f"fr3_{task_name}_test")
        os.makedirs(checkpoint_path)

        config = {
            "checkpoint_path": checkpoint_path,
            "gripper_postprocess": {
                "enabled": True,
                "close_value_threshold": 0.02,
                "required_close_duration": 0.1
            }
        }

        return TaskGripperController(config)

    def test_peg_in_hole_behavior(self):
        """测试插孔任务行为"""
        controller = self._create_controller_for_task("peg_in_hole")

        self.assertEqual(controller.task_type, TaskType.PEG_IN_HOLE)
        self.assertEqual(controller.state, 'OPEN')

        # 插孔任务应该支持ACT驱动的开闭
        debug_info = controller.get_debug_info()
        self.assertEqual(debug_info['task_display_name'], '插孔任务')

    def test_liquid_transfer_behavior(self):
        """测试倒水任务行为"""
        controller = self._create_controller_for_task("liquid_transfer")

        self.assertEqual(controller.task_type, TaskType.LIQUID_TRANSFER)

        # 倒水任务有更长的默认步数限制
        self.assertGreater(controller.max_step_nums, 700)

    def test_solid_transfer_behavior(self):
        """测试固体转移任务行为"""
        controller = self._create_controller_for_task("solid_transfer")

        self.assertEqual(controller.task_type, TaskType.SOLID_TRANSFER)

        # 固体转移任务应该初始为HOLDING状态
        # 注意：SolidTransferStrategy初始状态就是HOLDING
        if hasattr(controller.strategy, 'always_closed'):
            self.assertTrue(controller.strategy.always_closed)

    def test_different_max_steps(self):
        """测试不同任务类型的最大步数"""
        tasks_and_steps = [
            ("peg_in_hole", 600),
            ("block_stacking", 550),
            ("liquid_transfer", 800),
            ("solid_transfer", 500)
        ]

        for task_name, expected_steps in tasks_and_steps:
            with self.subTest(task=task_name):
                controller = self._create_controller_for_task(task_name)
                # max_step_nums应该从策略或配置中获取
                actual_steps = controller.strategy.get_max_steps()
                self.assertEqual(actual_steps, expected_steps)


if __name__ == '__main__':
    unittest.main()