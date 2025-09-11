#!/usr/bin/env python3
"""
测试Paxini网络客户端
从运行中的server接收触觉数据并显示统计信息
"""

import yaml
import time
import numpy as np
import sys
import os
import argparse

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paxini_network_sensor import PaxiniNetworkSensor

def main():
    parser = argparse.ArgumentParser(description="Test Paxini Network Client")
    parser.add_argument("--config", default="test_client_config.yaml",
                       help="Configuration file path")
    args = parser.parse_args()
    
    print("🔌 Starting Paxini Network Client Test")
    
    # 加载客户端配置
    with open(args.config, 'r') as f:
        file_config = yaml.safe_load(f)
    
    # 处理嵌套配置结构
    if 'paxini_network_sensor' in file_config:
        config = file_config['paxini_network_sensor']
    else:
        config = file_config
    
    print(f"📄 Client config loaded:")
    print(f"  📍 Server: {config['ip']}:{config['port']}")
    print(f"  📡 Topic: {config['topic']}")
    print(f"  🏷️  Sensor ID: {config.get('sensor_id', 'auto-generated')}")
    
    # 创建网络传感器
    sensor = PaxiniNetworkSensor(config)
    
    try:
        # 初始化传感器
        if not sensor.initialize():
            print("❌ Failed to initialize network sensor")
            return
        
        print("✅ Network sensor initialized successfully")
        
        # 测试数据接收
        print("\n🔄 Starting data reception test (10 seconds)...")
        start_time = time.time()
        last_stats_time = start_time
        frame_count = 0
        
        while time.time() - start_time < 10.0:
            # 读取触觉数据
            success, tactile_data, timestamp = sensor.read_tactile_data()
            
            if success and tactile_data is not None:
                frame_count += 1
                
                # 每2秒显示统计信息
                if time.time() - last_stats_time >= 2.0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    
                    print(f"📊 Stats after {elapsed:.1f}s:")
                    print(f"  📈 Frame rate: {fps:.1f} FPS")
                    print(f"  📐 Data shape: {tactile_data.shape}")
                    print(f"  📏 Data range: [{tactile_data.min():.1f}, {tactile_data.max():.1f}]")
                    print(f"  🕒 Timestamp: {timestamp:.3f}")
                    print(f"  📊 Non-zero values: {np.count_nonzero(tactile_data)}")
                    print()
                    
                    last_stats_time = time.time()
            
            time.sleep(0.01)  # 100Hz采样
        
        # 最终统计
        total_time = time.time() - start_time
        final_fps = frame_count / total_time
        
        print(f"🎯 Final Results:")
        print(f"  ⏱️  Total time: {total_time:.1f}s")
        print(f"  📦 Total frames: {frame_count}")
        print(f"  📈 Average FPS: {final_fps:.1f}")
        print(f"  🎭 Expected FPS: 100 (server)")
        
        # 测试传感器信息
        print(f"\n📋 Sensor Info:")
        sensor_info = sensor.get_sensor_info()
        print(f"  🔗 Communication: {sensor_info['communication_type']}")
        print(f"  📍 Network: {sensor_info['ip']}:{sensor_info['port']}")
        print(f"  📡 Topic: {sensor_info['topic']}")
        print(f"  🏷️  Sensor ID: {sensor_info['sensor_id']}")
        print(f"  📊 Total modules: {sensor_info['total_modules']}")
        print(f"  🔄 Is receiving: {sensor_info['is_receiving']}")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        # 清理
        sensor.close()
        print("✅ Network sensor closed")

if __name__ == "__main__":
    main()