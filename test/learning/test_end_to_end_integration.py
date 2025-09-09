#!/usr/bin/env python3
"""端到端集成测试验证脚本."""

import os
import sys
import numpy as np
import tempfile
import pickle
from pathlib import Path

# 添加HIROLRobotPlatform到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_mock_checkpoint_directory():
    """创建模拟的检查点目录用于测试."""
    temp_dir = tempfile.mkdtemp()
    
    # 创建模拟的数据集统计文件
    mock_stats = {
        "qpos_mean": np.array([0.0] * 7),
        "qpos_std": np.array([1.0] * 7),
        "action_mean": np.array([0.0] * 7),
        "action_std": np.array([1.0] * 7)
    }
    
    with open(os.path.join(temp_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(mock_stats, f)
    
    # 创建模拟的策略文件
    with open(os.path.join(temp_dir, "policy_best.ckpt"), "wb") as f:
        pickle.dump({"model_state": "mock"}, f)
    
    return temp_dir


def test_learning_module_imports():
    """测试学习模块导入."""
    print("🔄 测试学习模块导入...")
    
    try:
        # 测试核心模块导入
        from learning.inference.policy_inference import PolicyInference, ACTInference
        from learning.inference.data_adapter import RobotDataAdapter, FR3DataAdapter
        from factory.components.learning_inference_factory import LearningInferenceFactory
        
        print("✅ 学习模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 学习模块导入失败: {e}")
        return False


def test_data_adapter_functionality():
    """测试数据适配器功能."""
    print("🔄 测试数据适配器功能...")
    
    try:
        from learning.inference.data_adapter import RobotDataAdapter, create_data_adapter
        from hardware.base.utils import RobotJointState
        
        # 测试创建适配器
        adapter = create_data_adapter("generic")
        print("  ✅ 数据适配器创建成功")
        
        # 测试关节状态转换
        joint_state = RobotJointState()
        joint_state._positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        state_array = adapter.robot_state_to_numpy(joint_state)
        assert isinstance(state_array, np.ndarray)
        assert len(state_array) == 7
        print("  ✅ 关节状态转换成功")
        
        # 测试相机数据转换
        camera_data = {
            "front_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        image_tensor = adapter.camera_dict_to_tensor(camera_data)
        assert image_tensor.shape == (1, 1, 3, 480, 640)
        print("  ✅ 相机数据转换成功")
        
        # 测试动作转换
        mock_actions = np.array([0.5, -0.3, 0.8, 0.0, -0.2, 0.1, 0.4])
        action_commands = adapter.numpy_to_robot_actions(mock_actions)
        assert isinstance(action_commands, list)
        assert len(action_commands) == 7
        print("  ✅ 动作转换成功")
        
        return True
    except Exception as e:
        print(f"❌ 数据适配器测试失败: {e}")
        return False


def test_learning_factory_functionality():
    """测试学习推理工厂功能."""
    print("🔄 测试学习推理工厂功能...")
    
    try:
        from factory.components.learning_inference_factory import LearningInferenceFactory
        
        # 测试获取支持的算法
        algorithms = LearningInferenceFactory.get_supported_algorithms()
        assert "ACT" in algorithms
        print(f"  ✅ 支持的算法: {algorithms}")
        
        # 测试获取支持的机器人
        robots = LearningInferenceFactory.get_supported_robots()
        assert "fr3" in robots
        print(f"  ✅ 支持的机器人: {robots}")
        
        # 测试数据适配器创建
        adapter = LearningInferenceFactory.create_data_adapter("fr3")
        print("  ✅ 数据适配器工厂创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 学习推理工厂测试失败: {e}")
        return False


def test_checkpoint_validation():
    """测试检查点验证功能."""
    print("🔄 测试检查点验证功能...")
    
    try:
        from factory.components.learning_inference_factory import LearningInferenceFactory
        
        # 创建模拟检查点目录
        temp_dir = create_mock_checkpoint_directory()
        
        # 测试有效检查点验证
        is_valid = LearningInferenceFactory.validate_checkpoint_directory(temp_dir, "ACT")
        assert is_valid
        print("  ✅ 检查点目录验证成功")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"❌ 检查点验证测试失败: {e}")
        return False


def test_arm_base_learning_interface():
    """测试ArmBase学习接口."""
    print("🔄 测试ArmBase学习接口...")
    
    try:
        from hardware.base.arm import ArmBase
        
        # 检查学习推理方法是否存在
        arm_methods = dir(ArmBase)
        required_methods = [
            'set_learning_inference',
            'get_learning_prediction', 
            'execute_learned_action_sequence',
            'run_learning_control_loop',
            'is_learning_enabled',
            'disable_learning_inference'
        ]
        
        for method in required_methods:
            assert method in arm_methods, f"缺少方法: {method}"
        
        print("  ✅ ArmBase学习接口方法验证成功")
        
        return True
    except Exception as e:
        print(f"❌ ArmBase学习接口测试失败: {e}")
        return False


def test_robot_factory_learning_integration():
    """测试RobotFactory学习集成."""
    print("🔄 测试RobotFactory学习集成...")
    
    try:
        from factory.components.robot_factory import RobotFactory
        
        # 检查学习推理方法是否存在
        factory_methods = dir(RobotFactory)
        required_methods = [
            '_initialize_learning_components',
            'setup_learning_inference',
            'get_learning_prediction',
            'execute_learned_actions',
            'run_learning_control_loop',
            'is_learning_enabled',
            'disable_learning_inference'
        ]
        
        for method in required_methods:
            assert method in factory_methods, f"缺少方法: {method}"
        
        print("  ✅ RobotFactory学习集成方法验证成功")
        
        return True
    except Exception as e:
        print(f"❌ RobotFactory学习集成测试失败: {e}")
        return False


def test_configuration_files():
    """测试配置文件存在性."""
    print("🔄 测试配置文件...")
    
    try:
        # 检查配置文件是否存在
        config_files = [
            "learning/config/act_inference_cfg.yaml",
            "learning/config/learning_factory_cfg.yaml", 
            "controller/config/learning_inference_controller_cfg.yaml"
        ]
        
        base_path = Path(__file__).parent.parent.parent
        
        for config_file in config_files:
            file_path = base_path / config_file
            assert file_path.exists(), f"配置文件不存在: {config_file}"
        
        print("  ✅ 配置文件验证成功")
        
        return True
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False


def run_integration_tests():
    """运行所有集成测试."""
    print("🚀 开始运行ACT集成测试...\n")
    
    tests = [
        ("模块导入测试", test_learning_module_imports),
        ("数据适配器功能测试", test_data_adapter_functionality),
        ("学习推理工厂测试", test_learning_factory_functionality),
        ("检查点验证测试", test_checkpoint_validation),
        ("ArmBase学习接口测试", test_arm_base_learning_interface),
        ("RobotFactory学习集成测试", test_robot_factory_learning_integration),
        ("配置文件测试", test_configuration_files),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"🎯 集成测试结果:")
    print(f"   ✅ 通过: {passed}")
    print(f"   ❌ 失败: {failed}")
    print(f"   📊 总计: {passed + failed}")
    
    if failed == 0:
        print("🎉 所有集成测试通过！ACT模块已成功集成到HIROLRobotPlatform")
        return True
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)