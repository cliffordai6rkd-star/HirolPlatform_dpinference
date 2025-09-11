#!/usr/bin/env python3
"""
Test script for PaxiniSerialSensor connection and functionality.

Usage:
    python test_paxini_connection.py [config_file]

If no config file is provided, will use default configuration.
"""

import os
import sys
import time
import argparse
import numpy as np

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paxini_serial_sensor import PaxiniSerialSensor


def create_default_config():
    """Create default test configuration"""
    return {
        "communication_type": "serial",
        "taxel_nums": 120,
        "update_frequency": 100,
        "type": "paxini_tactile",
        
        "control_boxes": [
            {
                "port": "/dev/ttyACM0",
                "baudrate": 460800,
                "timeout": 1.0,
                "control_mode": 5,
                "sensors": [
                    {
                        "model": "GEN2-IP-L5325",
                        "connect_ids": [5, 6]
                    }
                ]
            }
        ]
    }


def test_basic_connection(config):
    """Test basic sensor connection and initialization"""
    print("=" * 50)
    print("Testing Paxini Serial Sensor Connection")
    print("=" * 50)
    
    try:
        # Initialize sensor
        print("Initializing sensor...")
        sensor = PaxiniSerialSensor(config)
        
        if not sensor.is_sensor_connected():
            print("❌ Failed to connect to sensor")
            print("  Check if:")
            print("  - Sensors are properly connected to CN5, CN6")  
            print("  - Sensors are powered on")
            print("  - No other process is using the serial port")
            sensor.close()
            return False
            
        print("✅ Sensor connected successfully")
        
        # Print sensor information
        info = sensor.get_sensor_info()
        print("\nSensor Information:")
        print(f"  Type: {info['sensor_type']}")
        print(f"  Connected: {info['is_connected']}")
        print(f"  Update frequency: {info['update_frequency']} Hz")
        print(f"  Control boxes: {info['control_boxes_count']}")
        print(f"  Total modules: {info['total_modules']}")
        
        # Print control box details
        print("\nControl Box Details:")
        for box in info['box_summary']:
            print(f"  Box {box['box_id']}: {box['port']}")
            print(f"    Mode: {box['control_mode']}")
            print(f"    Models: {box['sensor_models']}")
            print(f"    Modules: {box['total_modules']}")
        
        # Test data reading
        print("\nTesting data acquisition...")
        time.sleep(2.0)  # Wait for data acquisition to start
        
        success, data, timestamp = sensor.read_tactile_data()
        
        if success and data is not None:
            print("✅ Data acquisition working")
            print(f"  Data shape: {data.shape}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
            print(f"  Mean force per module: {np.mean(data, axis=(1,2))}")
        else:
            print("❌ Data acquisition failed")
            
        # Test state retrieval
        state = sensor.get_tactile_state()
        print(f"\nTactile State:")
        print(f"  Connected: {state._is_connected}")
        print(f"  Modules: {state._n_modules}")
        print(f"  Module IDs: {state._module_ids}")
        
        # Clean up
        sensor.close()
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_data_streaming(config, duration=10):
    """Test continuous data streaming"""
    print("=" * 50)
    print(f"Testing Data Streaming ({duration}s)")
    print("=" * 50)
    
    try:
        sensor = PaxiniSerialSensor(config)
        
        if not sensor.is_sensor_connected():
            print("❌ Sensor not connected for streaming")
            sensor.close()
            return False
        
        print("Starting data streaming test...")
        start_time = time.time()
        data_count = 0
        
        while time.time() - start_time < duration:
            success, data, timestamp = sensor.read_tactile_data()
            print(data.shape)
            
            if success and data is not None:
                data_count += 1
                
                # Print statistics every 2 seconds
                if data_count % 200 == 0:  # Assuming ~100Hz
                    elapsed = time.time() - start_time
                    rate = data_count / elapsed
                    max_force = np.max(np.abs(data))
                    avg_force = np.mean(np.abs(data))
                    
                    print(f"  Time: {elapsed:.1f}s | "
                          f"Rate: {rate:.1f}Hz | "
                          f"Max: {max_force:.1f} | "
                          f"Avg: {avg_force:.2f}")
            
            time.sleep(0.01)  # Small delay
        
        elapsed = time.time() - start_time
        actual_rate = data_count / elapsed
        
        print(f"\nStreaming Results:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Data packets: {data_count}")
        print(f"  Average rate: {actual_rate:.1f}Hz")
        print(f"  Expected rate: {config['update_frequency']}Hz")
        
        sensor.close()
        
        if actual_rate > config['update_frequency'] * 0.8:  # 80% of expected
            print("✅ Streaming test passed")
            return True
        else:
            print("⚠️  Streaming rate lower than expected")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Paxini Serial Sensor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--port", default="/dev/ttyACM1", help="Serial port")
    parser.add_argument("--streaming", action="store_true", help="Run streaming test")
    parser.add_argument("--duration", type=int, default=20, help="Streaming test duration")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Handle factory config format (with top-level sensor type key)
        if 'paxini_serial_sensor' in loaded_config:
            config = loaded_config['paxini_serial_sensor']
            print(f"🏭 Factory config detected, extracted sensor config")
        else:
            config = loaded_config
            print(f"📄 Direct config format detected")
    else:
        config = create_default_config()
        # Override port if specified
        config["control_boxes"][0]["port"] = args.port
    
    print(f"Using configuration:")
    print(f"  Port: {config['control_boxes'][0]['port']}")
    print(f"  Sensors: {[s['model'] for s in config['control_boxes'][0]['sensors']]}")
    print(f"  Connect IDs: {[s['connect_ids'] for s in config['control_boxes'][0]['sensors']]}")
    
    # Run basic test
    success = test_basic_connection(config)
    
    if success and args.streaming:
        # Run streaming test if requested
        test_data_streaming(config, args.duration)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())