#!/usr/bin/env python3
"""
ACT推理脚本集成测试

测试推理脚本的核心组件是否能正确导入和初始化
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# 添加项目根路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试核心模块导入
        from factory.components.robot_factory import RobotFactory
        from factory.components.learning_inference_factory import LearningInferenceFactory
        from learning.inference.data_adapter import create_data_adapter
        from hardware.base.utils import RobotJointState
        print("  ✅ 核心模块导入成功")
        
        # 测试ACT相关模块导入
        from learning.inference.policy_inference import ACTInference
        from learning.policies.act.policy import ACTPolicy
        print("  ✅ ACT模块导入成功")
        
        return True
    except ImportError as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def test_data_adapter_creation():
    """测试数据适配器创建"""
    print("🔍 测试数据适配器创建...")
    
    try:
        from learning.inference.data_adapter import create_data_adapter
        
        # 测试创建不同类型的适配器
        adapters = {}
        for robot_type in ['fr3', 'monte01', 'generic']:
            adapter = create_data_adapter(robot_type)
            adapters[robot_type] = adapter
            print(f"  ✅ {robot_type}数据适配器创建成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 数据适配器创建失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("🔍 测试配置文件加载...")
    
    config_path = Path(__file__).parent / "configs" / "fr3_inference_config.yaml"
    
    if not config_path.exists():
        print(f"  ⚠️ 配置文件不存在: {config_path}")
        return False
    
    try:
        from hardware.base.utils import dynamic_load_yaml
        config = dynamic_load_yaml(str(config_path))
        
        # 检查必要的配置项
        required_keys = ['robot', 'learning', 'control_frequency']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"  ❌ 缺少必要配置项: {missing_keys}")
            return False
        
        print("  ✅ 配置文件加载成功")
        print(f"    - 机器人类型: {config.get('robot')}")
        print(f"    - 状态维度: {config.get('learning', {}).get('state_dim')}")
        print(f"    - 相机数量: {len(config.get('learning', {}).get('camera_names', []))}")
        
        return True
    except Exception as e:
        print(f"  ❌ 配置文件加载失败: {e}")
        return False

def test_safety_checker():
    """测试安全检查器"""
    print("🔍 测试安全检查器...")
    
    try:
        from hardware.base.safety_checker import SafetyChecker
        
        checker = SafetyChecker()
        
        # 测试关节命令检查
        import numpy as np
        
        # FR3的有效关节位置
        valid_positions = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
        
        # 先更新当前状态
        checker.update_state(joint_positions=valid_positions)
        
        # 测试安全的目标位置
        target_positions = valid_positions + 0.1  # 小幅移动
        is_safe, message = checker.check_joint_command(target_positions)
        
        if is_safe:
            print("  ✅ 安全检查器测试通过")
        else:
            print(f"  ❌ 安全检查器测试失败: {message}")
            
        return is_safe
    except Exception as e:
        print(f"  ❌ 安全检查器测试失败: {e}")
        return False

def test_mock_inference_setup():
    """测试模拟推理设置"""
    print("🔍 测试模拟推理设置...")
    
    try:
        # 创建临时目录模拟检查点
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建模拟文件
            policy_path = os.path.join(temp_dir, "policy_best.ckpt")
            stats_path = os.path.join(temp_dir, "dataset_stats.pkl")
            
            # 创建模拟检查点文件
            import torch
            import pickle
            
            # 创建简单的模拟状态字典
            dummy_state_dict = {
                'model.action_head.weight': torch.randn(8, 512),
                'model.action_head.bias': torch.randn(8)
            }
            torch.save(dummy_state_dict, policy_path)
            
            # 创建模拟数据集统计
            dummy_stats = {
                'qpos_mean': [0.0] * 8,
                'qpos_std': [1.0] * 8,
                'action_mean': [0.0] * 8,
                'action_std': [1.0] * 8
            }
            with open(stats_path, 'wb') as f:
                pickle.dump(dummy_stats, f)
            
            # 验证文件创建
            from factory.components.learning_inference_factory import LearningInferenceFactory
            
            is_valid = LearningInferenceFactory.validate_checkpoint_directory(temp_dir, "ACT")
            
            if is_valid:
                print("  ✅ 模拟推理设置测试通过")
            else:
                print("  ❌ 检查点验证失败")
            
            return is_valid
            
    except Exception as e:
        print(f"  ❌ 模拟推理设置失败: {e}")
        return False

def main():
    """运行所有集成测试"""
    print("🚀 开始ACT推理脚本集成测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("数据适配器创建", test_data_adapter_creation),
        ("配置文件加载", test_config_loading),
        ("安全检查器", test_safety_checker),
        ("模拟推理设置", test_mock_inference_setup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有集成测试通过！推理脚本准备就绪。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关组件。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)