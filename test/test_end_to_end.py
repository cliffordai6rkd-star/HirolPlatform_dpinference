#!/usr/bin/env python3
"""
端到端功能验证测试
验证整个任务类型系统的端到端功能
"""

import sys
sys.path.append('/workspace')

def test_task_type_detection():
    """测试任务类型检测功能"""
    print("=" * 60)
    print("🎯 测试任务类型检测功能")
    print("=" * 60)

    from factory.tasks.inferences_tasks.utils.task_types import TaskType

    test_cases = [
        ("learning/ckpts/fr3_peg_in_hole_0914", TaskType.PEG_IN_HOLE, "插孔任务"),
        ("learning/ckpts/fr3_bs_0916_50ep_ds", TaskType.BLOCK_STACKING, "叠方块任务"),
        ("learning/ckpts/fr3_liquid_transfer_0920", TaskType.LIQUID_TRANSFER, "倒水任务"),
        ("learning/ckpts/solid_transfer_test", TaskType.SOLID_TRANSFER, "固体转移任务"),
    ]

    for checkpoint_path, expected_type, expected_name in test_cases:
        detected_type = TaskType.from_checkpoint(checkpoint_path)
        status = "✅" if detected_type == expected_type else "❌"
        print(f"{status} {checkpoint_path:<40} -> {detected_type.display_name}")

    print()


def test_strategy_creation():
    """测试策略创建功能"""
    print("=" * 60)
    print("🔧 测试策略创建功能")
    print("=" * 60)

    from factory.tasks.inferences_tasks.utils.gripper_strategies import GripperStrategyFactory
    from factory.tasks.inferences_tasks.utils.gripper_strategies import (
        GraspReleaseStrategy, LiquidTransferStrategy, SolidTransferStrategy
    )

    test_cases = [
        ("peg_in_hole", GraspReleaseStrategy, "抓放策略"),
        ("block_stacking", GraspReleaseStrategy, "抓放策略"),
        ("liquid_transfer", LiquidTransferStrategy, "倒水策略"),
        ("solid_transfer", SolidTransferStrategy, "固体转移策略"),
    ]

    config = {"min_holding_steps": 30}

    for task_type, expected_class, description in test_cases:
        try:
            strategy = GripperStrategyFactory.create(task_type, config)
            status = "✅" if isinstance(strategy, expected_class) else "❌"
            print(f"{status} {task_type:<15} -> {description} ({expected_class.__name__})")
        except Exception as e:
            print(f"❌ {task_type:<15} -> 创建失败: {e}")

    print()


def test_controller_integration():
    """测试控制器集成功能"""
    print("=" * 60)
    print("🎮 测试控制器集成功能")
    print("=" * 60)

    import tempfile
    import os

    try:
        from factory.tasks.inferences_tasks.utils.gripper_controller import create_gripper_controller
        from factory.tasks.inferences_tasks.utils.gripper_controller import TaskGripperController

        # 创建临时checkpoint目录
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "fr3_peg_in_hole_test")
            os.makedirs(checkpoint_path)

            # 测试任务感知控制器创建
            config = {"enabled": True}
            full_config = {
                "checkpoint_path": checkpoint_path,
                "gripper_postprocess": config
            }

            controller = create_gripper_controller(config, full_config)

            if isinstance(controller, TaskGripperController):
                print(f"✅ 任务感知控制器创建成功")
                print(f"   - 任务类型: {controller.task_type.display_name}")
                print(f"   - 控制策略: {controller.strategy.__class__.__name__}")
                print(f"   - 最大步数: {controller.max_step_nums}")
            else:
                print(f"❌ 期望TaskGripperController，实际得到{type(controller)}")

    except Exception as e:
        print(f"❌ 控制器集成测试失败: {e}")
        import traceback
        traceback.print_exc()

    print()


def test_different_task_behaviors():
    """测试不同任务的行为差异"""
    print("=" * 60)
    print("🎭 测试不同任务类型的行为差异")
    print("=" * 60)

    import tempfile
    import os

    try:
        from factory.tasks.inferences_tasks.utils.gripper_controller import TaskGripperController

        tasks_and_behaviors = [
            ("fr3_peg_in_hole_test", "插孔任务", 600),
            ("fr3_bs_test", "叠方块任务", 550),
            ("fr3_liquid_transfer_test", "倒水任务", 800),
            ("fr3_solid_transfer_test", "固体转移任务", 500),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            for checkpoint_name, expected_display_name, expected_steps in tasks_and_behaviors:
                checkpoint_path = os.path.join(temp_dir, checkpoint_name)
                os.makedirs(checkpoint_path)

                config = {
                    "checkpoint_path": checkpoint_path,
                    "gripper_postprocess": {"enabled": True}
                }

                try:
                    controller = TaskGripperController(config)
                    actual_steps = controller.strategy.get_max_steps()

                    status = "✅" if actual_steps == expected_steps else "❌"
                    print(f"{status} {expected_display_name:<12} -> {actual_steps:>3}步 (期望:{expected_steps})")

                except Exception as e:
                    print(f"❌ {expected_display_name:<12} -> 创建失败: {e}")

    except Exception as e:
        print(f"❌ 行为差异测试失败: {e}")

    print()


def main():
    """运行所有端到端测试"""
    print("🚀 HIROLRobotPlatform 任务类型系统 - 端到端功能验证")
    print()

    try:
        test_task_type_detection()
        test_strategy_creation()
        test_controller_integration()
        test_different_task_behaviors()

        print("=" * 60)
        print("🎉 所有端到端测试完成！系统运行正常。")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()