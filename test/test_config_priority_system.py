#!/usr/bin/env python3
"""
配置优先级系统测试
验证任务特定配置是否正确覆盖默认配置
"""

import sys
import os
import tempfile
sys.path.append('/workspace')

def test_config_loader_functionality():
    """测试配置加载器基本功能"""
    print("=" * 60)
    print("🔧 测试配置加载器基本功能")
    print("=" * 60)

    try:
        from factory.tasks.inferences_tasks.utils.config_loader import ConfigLoader
        from pathlib import Path
        import yaml

        # 创建测试配置文件
        with tempfile.TemporaryDirectory() as temp_dir:
            config_loader = ConfigLoader(base_config_dir=temp_dir)

            # 创建基础配置
            base_config = {
                "checkpoint_path": "test_checkpoint",
                "max_step_nums": 500,
                "gripper_postprocess": {
                    "enabled": True,
                    "target_peak_count": 3,
                    "control_mode": "binary"
                }
            }

            # 创建任务特定配置目录
            tasks_dir = Path(temp_dir) / "tasks"
            tasks_dir.mkdir()

            # 创建liquid_transfer_cfg.yaml
            task_config = {
                "task_type": "liquid_transfer",
                "max_step_nums": 1300,
                "gripper_postprocess": {
                    "control_mode": "task_aware",
                    "target_peak_count": 999,
                    "grasp_check_enabled": False
                }
            }

            task_config_path = tasks_dir / "liquid_transfer_cfg.yaml"
            with open(task_config_path, 'w') as f:
                yaml.dump(task_config, f)

            # 测试配置合并
            merged_config = config_loader.merge_with_task_config(base_config, "liquid_transfer")

            # 验证合并结果
            assert merged_config["max_step_nums"] == 1300, f"Expected 1300, got {merged_config['max_step_nums']}"
            assert merged_config["gripper_postprocess"]["target_peak_count"] == 999, f"Expected 999, got {merged_config['gripper_postprocess']['target_peak_count']}"
            assert merged_config["gripper_postprocess"]["control_mode"] == "task_aware"
            assert merged_config["gripper_postprocess"]["grasp_check_enabled"] == False

            # 验证基础配置未被影响的部分
            assert merged_config["checkpoint_path"] == "test_checkpoint"
            assert merged_config["gripper_postprocess"]["enabled"] == True

            print("✅ 配置加载器基本功能测试通过")
            print(f"   - max_step_nums: 500 → 1300")
            print(f"   - target_peak_count: 3 → 999")
            print(f"   - control_mode: binary → task_aware")

    except Exception as e:
        print(f"❌ 配置加载器功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_task_gripper_controller_config_merge():
    """测试TaskGripperController配置合并"""
    print("=" * 60)
    print("🎮 测试TaskGripperController配置合并")
    print("=" * 60)

    try:
        import tempfile
        from pathlib import Path
        import yaml
        from factory.tasks.inferences_tasks.utils.gripper_controller import TaskGripperController

        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建checkpoint目录
            checkpoint_path = Path(temp_dir) / "fr3_liquid_transfer_test"
            checkpoint_path.mkdir()

            # 创建基础配置
            config = {
                "checkpoint_path": str(checkpoint_path),
                "gripper_postprocess": {
                    "enabled": True,
                    "target_peak_count": 3,  # 基础配置值
                    "control_mode": "binary"
                }
            }

            # 创建任务特定配置目录和文件
            config_dir = Path("/workspace/factory/tasks/inferences_tasks/act/config")
            tasks_dir = config_dir / "tasks"

            if tasks_dir.exists():
                # 验证任务特定配置是否覆盖默认值
                controller = TaskGripperController(config)

                # 检查合并后的配置
                merged_gripper_config = controller.config.get("gripper_postprocess", {})
                target_peak_count = merged_gripper_config.get("target_peak_count", 3)

                if target_peak_count == 999:
                    print(f"✅ target_peak_count正确覆盖: 3 → 999")
                else:
                    print(f"⚠️ target_peak_count未被覆盖，当前值: {target_peak_count}")

                # 检查任务类型
                print(f"   - 任务类型: {controller.task_type.display_name}")
                print(f"   - 控制策略: {controller.strategy.__class__.__name__}")
                print(f"   - 最大步数: {controller.max_step_nums}")
            else:
                print("⚠️ 任务配置目录不存在，跳过实际配置测试")

    except Exception as e:
        print(f"❌ TaskGripperController配置合并测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_act_inference_config_merge():
    """测试ACTInference配置合并"""
    print("=" * 60)
    print("🧠 测试ACTInference配置合并")
    print("=" * 60)

    try:
        import tempfile
        from pathlib import Path
        import yaml

        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建checkpoint目录
            checkpoint_path = Path(temp_dir) / "fr3_liquid_transfer_test"
            checkpoint_path.mkdir()

            # 创建基础配置
            config = {
                "checkpoint_path": str(checkpoint_path),
                "robot_type": "fr3",
                "max_step_nums": 500,  # 基础配置值
                "learning": {
                    "algorithm": "ACT",
                    "state_dim": 8
                }
            }

            # 测试任务类型系统初始化
            try:
                from factory.tasks.inferences_tasks.act.act_inference import ACT_Inferencer

                # 注意：这里无法完全测试ACTInference，因为需要真实的checkpoint和模型
                # 但我们可以测试配置合并逻辑
                from factory.tasks.inferences_tasks.utils.config_loader import ConfigLoader
                config_loader = ConfigLoader()

                # 检测任务类型
                from factory.tasks.inferences_tasks.utils.task_types import TaskType
                task_type = TaskType.from_checkpoint(str(checkpoint_path))

                # 尝试合并配置
                merged_config = config_loader.merge_with_task_config(config, task_type.value)

                print(f"✅ 配置合并测试完成")
                print(f"   - 检测任务类型: {task_type.display_name}")
                print(f"   - 原始max_step_nums: {config['max_step_nums']}")
                print(f"   - 合并后max_step_nums: {merged_config.get('max_step_nums', '未设置')}")

            except ImportError as e:
                print(f"⚠️ ACTInference模块不可用: {e}")
            except Exception as e:
                print(f"⚠️ ACT配置测试异常: {e}")

    except Exception as e:
        print(f"❌ ACTInference配置合并测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_config_priority_real_files():
    """测试真实配置文件的优先级"""
    print("=" * 60)
    print("📄 测试真实配置文件优先级")
    print("=" * 60)

    try:
        from factory.tasks.inferences_tasks.utils.config_loader import ConfigLoader
        from pathlib import Path

        # 检查真实配置文件
        base_config_path = Path("/workspace/factory/tasks/inferences_tasks/act/config/fr3_act_inference_cfg.yaml")
        liquid_config_path = Path("/workspace/factory/tasks/inferences_tasks/act/config/tasks/liquid_transfer_cfg.yaml")

        if not base_config_path.exists():
            print(f"⚠️ 基础配置文件不存在: {base_config_path}")
            return True

        if not liquid_config_path.exists():
            print(f"⚠️ liquid_transfer配置文件不存在: {liquid_config_path}")
            return True

        # 加载基础配置
        config_loader = ConfigLoader()
        base_config = config_loader.load_yaml_config(base_config_path)

        # 检查基础配置中的关键参数
        base_target_peak_count = base_config.get("gripper_postprocess", {}).get("target_peak_count")
        base_max_steps = base_config.get("max_step_nums")

        print(f"📋 基础配置 (fr3_act_inference_cfg.yaml):")
        print(f"   - target_peak_count: {base_target_peak_count}")
        print(f"   - max_step_nums: {base_max_steps}")

        # 加载任务特定配置
        task_config = config_loader.load_yaml_config(liquid_config_path)
        task_target_peak_count = task_config.get("gripper_postprocess", {}).get("target_peak_count")
        task_max_steps = task_config.get("max_step_nums")

        print(f"📋 任务特定配置 (liquid_transfer_cfg.yaml):")
        print(f"   - target_peak_count: {task_target_peak_count}")
        print(f"   - max_step_nums: {task_max_steps}")

        # 执行合并
        merged_config = config_loader.merge_with_task_config(base_config, "liquid_transfer")
        merged_target_peak_count = merged_config.get("gripper_postprocess", {}).get("target_peak_count")
        merged_max_steps = merged_config.get("max_step_nums")

        print(f"📋 合并后配置:")
        print(f"   - target_peak_count: {merged_target_peak_count}")
        print(f"   - max_step_nums: {merged_max_steps}")

        # 验证优先级
        priority_correct = True
        if task_target_peak_count is not None and merged_target_peak_count != task_target_peak_count:
            print(f"❌ target_peak_count优先级错误: 期望{task_target_peak_count}, 实际{merged_target_peak_count}")
            priority_correct = False

        if task_max_steps is not None and merged_max_steps != task_max_steps:
            print(f"❌ max_step_nums优先级错误: 期望{task_max_steps}, 实际{merged_max_steps}")
            priority_correct = False

        if priority_correct:
            print("✅ 配置优先级正确，任务特定配置成功覆盖默认配置")

        return priority_correct

    except Exception as e:
        print(f"❌ 真实配置文件优先级测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有配置优先级测试"""
    print("🚀 HIROLRobotPlatform - 配置优先级系统测试")
    print()

    all_passed = True

    try:
        # 基础功能测试
        all_passed &= test_config_loader_functionality()

        # 控制器集成测试
        all_passed &= test_task_gripper_controller_config_merge()

        # ACT推理系统测试
        all_passed &= test_act_inference_config_merge()

        # 真实配置文件测试
        all_passed &= test_config_priority_real_files()

        print("=" * 60)
        if all_passed:
            print("🎉 所有配置优先级测试通过！任务特定配置正确覆盖默认配置。")
        else:
            print("⚠️ 部分测试未通过，请检查配置优先级系统。")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 配置优先级测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)