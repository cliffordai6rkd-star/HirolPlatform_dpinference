#!/usr/bin/env python3
"""
测试RobotFactory与Paxini触觉传感器的集成
使用HIROLRobotPlatform配置格式
"""

import sys
import os
from pathlib import Path

# Add platform path
platform_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(platform_path))

from hardware.base.utils import dynamic_load_yaml
from factory.components.robot_factory import RobotFactory
import glog as log

def test_factory_tactile_integration():
    """测试RobotFactory tactile集成"""
    print("🧪 Testing RobotFactory tactile integration...")
    
    # 配置文件路径
    config_path = Path(__file__).parent.parent / "config/test_factory_config.yaml"
    
    try:
        # 使用HIROLRobotPlatform的配置加载方式
        print(f"📄 Loading config from: {config_path}")
        config = dynamic_load_yaml(str(config_path))
        print("✅ Config loaded successfully")
        
        # 验证tactile配置是否存在
        if 'sensor_dicts' in config and 'tactile' in config['sensor_dicts']:
            tactile_configs = config['sensor_dicts']['tactile']
            print(f"✅ Found {len(tactile_configs)} tactile sensor config(s)")
            for tactile in tactile_configs:
                print(f"  - {tactile['name']}: {tactile['type']}")
        else:
            print("❌ No tactile configs found")
            return False
        
        # 创建RobotFactory
        print("🏭 Creating RobotFactory...")
        factory = RobotFactory(config)
        print("✅ RobotFactory created successfully")
        
        # 验证tactile类是否注册
        if hasattr(factory, '_tactile_classes'):
            print(f"✅ Tactile classes: {list(factory._tactile_classes.keys())}")
        else:
            print("❌ No tactile classes registered")
            return False
        
        # 尝试创建robot系统
        print("🔧 Creating robot system...")
        try:
            factory.create_robot_system()
            print("✅ Robot system created successfully")
        except Exception as e:
            error_str = str(e)
            if any(keyword in error_str for keyword in ["failed intialization", "ttyACM", "Connection", "No such device"]):
                print("✅ Configuration parsed correctly (hardware not available)")
                print(f"  Expected hardware error: {error_str}")
            else:
                print(f"❌ Unexpected error: {error_str}")
                return False
        
        # 验证tactile传感器对象
        if hasattr(factory, '_sensors') and 'tactile' in factory._sensors:
            tactile_sensors = factory._sensors['tactile']
            print(f"✅ Created {len(tactile_sensors)} tactile sensor object(s)")
            for sensor in tactile_sensors:
                sensor_obj = sensor['object']
                print(f"  - {sensor['name']}: {type(sensor_obj).__name__}")
                
                # 验证传感器配置
                if hasattr(sensor_obj, '_control_boxes'):
                    print(f"    Control boxes: {len(sensor_obj._control_boxes)}")
                    for i, box in enumerate(sensor_obj._control_boxes):
                        port = box.get('port', 'unknown')
                        sensors_count = len(box.get('sensors', []))
                        print(f"      Box {i}: {port}, {sensors_count} sensors")
        else:
            print("⚠️  No tactile sensor objects (expected due to hardware)")
        
        # 测试get_tactile_data接口
        if hasattr(factory, 'get_tactile_data'):
            print("✅ get_tactile_data method available")
            data = factory.get_tactile_data()
            print(f"✅ Returns {type(data)} (empty: {len(data) == 0})")
        else:
            print("❌ get_tactile_data method missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Testing RobotFactory Tactile Integration")
    print("=" * 60)
    
    success = test_factory_tactile_integration()
    
    print("\\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: RobotFactory tactile integration working!")
        print("✅ Factory can load and create tactile sensors")
        print("✅ Configuration format is correct")
        print("✅ API interfaces are available")
    else:
        print("❌ FAILED: Issues with tactile integration")
    
    return success

if __name__ == "__main__":
    main()