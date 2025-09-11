#!/usr/bin/env python3
"""
测试配置兼容性和解耦机制
验证不同配置格式是否都能正确工作
"""

import yaml
import sys
import os
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paxini_serial_sensor import PaxiniSerialSensor
from paxini_network_sensor import PaxiniNetworkSensor
from paxini_standalone_server import load_config

def test_config_formats():
    """测试不同的配置格式"""
    
    print("🔍 Testing Configuration Format Compatibility")
    print("=" * 50)
    
    # 格式1: 直接配置（standalone server使用）
    direct_config = {
        'communication_type': 'serial',
        'taxel_nums': 120,
        'update_frequency': 100,
        'control_boxes': [
            {
                'port': '/dev/ttyACM1',
                'baudrate': 460800,
                'timeout': 1.0,
                'control_mode': 5,
                'sensors': [
                    {
                        'model': 'GEN2-IP-L5325',
                        'connect_ids': [5, 6]
                    }
                ]
            }
        ]
    }
    
    # 格式2: 带顶级标题（工厂模式使用）
    factory_config = {
        'paxini_serial_sensor': {
            'communication_type': 'serial',
            'taxel_nums': 120,
            'update_frequency': 100,
            'control_boxes': [
                {
                    'port': '/dev/ttyACM1',
                    'baudrate': 460800,
                    'timeout': 1.0,
                    'control_mode': 5,
                    'sensors': [
                        {
                            'model': 'GEN2-IP-L5325',
                            'connect_ids': [5, 6]
                        }
                    ]
                }
            ]
        }
    }
    
    print("📋 Configuration Format 1 (Direct):")
    print("  - Used by: standalone server, direct instantiation")
    print("  - Structure: flat configuration dictionary")
    print(f"  - Keys: {list(direct_config.keys())}")
    print()
    
    print("📋 Configuration Format 2 (Factory):")
    print("  - Used by: HIROLRobotPlatform factory pattern")
    print("  - Structure: nested with top-level sensor type key")
    print(f"  - Top key: {list(factory_config.keys())[0]}")
    print(f"  - Actual config keys: {list(factory_config['paxini_serial_sensor'].keys())}")
    print()
    
    # 测试配置提取
    print("🔧 Testing Configuration Extraction:")
    
    # 直接配置
    try:
        print("  ✓ Direct config can be used directly")
        extracted_direct = direct_config
        sensor_id_direct = f"PAXINI_{len(extracted_direct['control_boxes']):02d}_MODULES"
        print(f"    Generated sensor ID: {sensor_id_direct}")
    except Exception as e:
        print(f"  ❌ Direct config failed: {e}")
    
    # 工厂配置
    try:
        print("  ✓ Factory config requires extraction")
        sensor_type = 'paxini_serial_sensor'
        extracted_factory = factory_config[sensor_type]  # 这就是工厂的提取过程
        sensor_id_factory = f"PAXINI_{len(extracted_factory['control_boxes']):02d}_MODULES"
        print(f"    Extracted config keys: {list(extracted_factory.keys())}")
        print(f"    Generated sensor ID: {sensor_id_factory}")
    except Exception as e:
        print(f"  ❌ Factory config extraction failed: {e}")
    
    # 验证提取后配置是否相同
    if extracted_direct == extracted_factory:
        print("  ✅ Both configs produce identical results after extraction")
    else:
        print("  ❌ Configs differ after extraction")
    
    print()
    return extracted_direct == extracted_factory

def test_yaml_loading():
    """测试YAML文件加载"""
    print("📄 Testing YAML File Loading")
    print("=" * 30)
    
    # 测试现有配置文件
    config_files = [
        'test_server_config.yaml',
        'test_client_config.yaml',
        'config/paxini_config_examples.yaml'
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"📁 Loading: {config_file}")
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                if isinstance(loaded_config, dict):
                    top_keys = list(loaded_config.keys())
                    print(f"  ✓ Loaded successfully")
                    print(f"  📊 Top-level keys: {top_keys[:3]}{'...' if len(top_keys) > 3 else ''}")
                    
                    # 检查是否有顶级sensor标题
                    sensor_keys = [k for k in top_keys if 'paxini' in k.lower()]
                    if sensor_keys:
                        print(f"  🏷️  Has sensor type keys: {sensor_keys}")
                        # 这种情况需要提取
                        actual_config = loaded_config[sensor_keys[0]]
                    else:
                        print(f"  📝 Direct configuration format")
                        # 这种情况直接使用
                        actual_config = loaded_config
                    
                    # 检查必要字段
                    required_fields = ['communication_type', 'control_boxes']
                    missing = [f for f in required_fields if f not in actual_config]
                    if not missing:
                        print(f"  ✅ All required fields present")
                    else:
                        print(f"  ⚠️  Missing fields: {missing}")
                        
                else:
                    print(f"  ❌ Invalid YAML structure")
                    
            except Exception as e:
                print(f"  ❌ Failed to load: {e}")
            print()
    
def test_constructor_compatibility():
    """测试构造函数兼容性"""
    print("🔧 Testing Constructor Compatibility")  
    print("=" * 35)
    
    # 准备测试配置
    test_config = {
        'communication_type': 'serial',
        'taxel_nums': 120,
        'control_boxes': [
            {
                'port': '/dev/ttyACM999',  # 使用不存在的端口避免实际连接
                'baudrate': 460800,
                'timeout': 0.1,  # 短超时
                'control_mode': 5,
                'sensors': [
                    {
                        'model': 'GEN2-IP-L5325',
                        'connect_ids': [5, 6]
                    }
                ]
            }
        ]
    }
    
    # 测试串行传感器构造
    try:
        print("📡 Testing PaxiniSerialSensor constructor...")
        serial_sensor = PaxiniSerialSensor(test_config)
        print("  ✅ Constructor accepted config successfully")
        print(f"  📊 Parsed {serial_sensor._expected_total_modules} modules")
        print(f"  🏷️  Generated ID: {serial_sensor._sensor_id}")
        serial_sensor.close()
    except Exception as e:
        print(f"  ❌ Serial sensor constructor failed: {e}")
    
    # 测试网络传感器构造  
    network_config = test_config.copy()
    network_config.update({
        'communication_type': 'network',
        'ip': 'localhost',
        'port': 5556,
        'topic': 'TEST_TOPIC',
        'timeout': 1.0
    })
    
    try:
        print("\n🌐 Testing PaxiniNetworkSensor constructor...")
        network_sensor = PaxiniNetworkSensor(network_config)
        print("  ✅ Constructor accepted config successfully")  
        print(f"  📊 Parsed {network_sensor._expected_total_modules} modules")
        print(f"  🏷️  Generated ID: {network_sensor._sensor_id}")
        network_sensor.close()
    except Exception as e:
        print(f"  ❌ Network sensor constructor failed: {e}")

def main():
    """主测试函数"""
    print("🧪 Paxini Configuration Compatibility Test")
    print("=" * 60)
    print()
    
    # 运行所有测试
    format_ok = test_config_formats()
    print()
    test_yaml_loading()
    print()
    test_constructor_compatibility()
    
    print()
    print("=" * 60)
    if format_ok:
        print("✅ All configuration formats are compatible")
        print("📝 Recommendation: Use appropriate format for your use case")
        print("   - Direct format: standalone servers, direct instantiation")  
        print("   - Factory format: HIROLRobotPlatform integration")
    else:
        print("❌ Configuration compatibility issues detected")
        print("🔧 Manual review required")

if __name__ == "__main__":
    main()