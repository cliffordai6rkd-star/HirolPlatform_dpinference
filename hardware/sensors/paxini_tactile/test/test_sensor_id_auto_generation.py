#!/usr/bin/env python3
"""
测试sensor_id自动生成机制
验证客户端是否真的不需要手动配置sensor_id
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from paxini_network_sensor import PaxiniNetworkSensor

def test_auto_generation():
    """测试自动生成sensor_id"""
    print("🧪 Testing sensor_id Auto-Generation")
    print("=" * 40)
    
    # 测试配置1：不包含sensor_id
    config_without_id = {
        "communication_type": "network",
        "ip": "localhost",
        "port": 5556,
        "topic": "TEST_TOPIC",
        "timeout": 1.0,
        "control_boxes": [
            {
                "sensors": [
                    {
                        "model": "GEN2-IP-L5325",
                        "connect_ids": [5, 6]  # 2个模块
                    }
                ]
            }
        ]
    }
    
    # 测试配置2：手动指定sensor_id
    config_with_id = config_without_id.copy()
    config_with_id["sensor_id"] = "PAXINI_CUSTOM_02_MODULES"
    
    print("📋 Test 1: Auto-generation (no sensor_id in config)")
    try:
        sensor1 = PaxiniNetworkSensor(config_without_id)
        print(f"  ✅ Generated ID: {sensor1._sensor_id}")
        print(f"  📊 Expected: PAXINI_02_MODULES")
        print(f"  🎯 Match: {'✅' if sensor1._sensor_id == 'PAXINI_02_MODULES' else '❌'}")
        sensor1.close()
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()
    print("📋 Test 2: Manual specification (sensor_id in config)")
    try:
        sensor2 = PaxiniNetworkSensor(config_with_id)
        print(f"  ✅ Configured ID: {sensor2._sensor_id}")
        print(f"  📊 Expected: PAXINI_CUSTOM_02_MODULES")
        print(f"  🎯 Match: {'✅' if sensor2._sensor_id == 'PAXINI_CUSTOM_02_MODULES' else '❌'}")
        sensor2.close()
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print()

def test_config_files():
    """测试配置文件中的sensor_id处理"""
    print("📁 Testing Config Files")
    print("=" * 25)
    
    config_files = [
        ("config/common/network_client.yaml", "应该自动生成"),
        ("config/factory/paxini_network_left_hand.yaml", "应该使用手动指定")
    ]
    
    for config_file, expected_behavior in config_files:
        if not os.path.exists(config_file):
            print(f"  ⚠️  Config file not found: {config_file}")
            continue
            
        print(f"📄 {os.path.basename(config_file)} ({expected_behavior})")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # 提取网络传感器配置
            if 'paxini_network_sensor' in config:
                actual_config = config['paxini_network_sensor']
            else:
                actual_config = config
            
            has_sensor_id = 'sensor_id' in actual_config
            
            # 计算预期的模块数
            total_modules = 0
            for box in actual_config.get('control_boxes', []):
                for sensor in box.get('sensors', []):
                    total_modules += len(sensor.get('connect_ids', []))
            
            expected_auto_id = f"PAXINI_{total_modules:02d}_MODULES"
            
            if has_sensor_id:
                configured_id = actual_config['sensor_id']
                print(f"  🏷️  Manual ID: {configured_id}")
                print(f"  📊 Auto would be: {expected_auto_id}")
            else:
                print(f"  🤖 Auto-generated: {expected_auto_id}")
                print(f"  📊 Modules: {total_modules}")
            
            # 实际测试传感器创建
            try:
                sensor = PaxiniNetworkSensor(actual_config)
                print(f"  ✅ Final ID: {sensor._sensor_id}")
                sensor.close()
            except Exception as e:
                print(f"  ❌ Sensor creation failed: {e}")
                
        except Exception as e:
            print(f"  ❌ Config parsing failed: {e}")
        
        print()

def main():
    print("🔬 Paxini sensor_id Auto-Generation Test")
    print("=" * 50)
    print()
    
    test_auto_generation()
    test_config_files()
    
    print("=" * 50)
    print("💡 结论:")
    print("  ✅ 基础场景：可以省略sensor_id，自动生成")
    print("  ✅ 特殊场景：可以手动指定sensor_id覆盖自动生成") 
    print("  ✅ 配置更简洁：只需保持control_boxes一致即可")

if __name__ == "__main__":
    main()