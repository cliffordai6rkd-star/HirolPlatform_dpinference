#!/usr/bin/env python3

import pyspacemouse
import time

def test_device_detection():
    """Test detection and initialization of multiple 3D mouse devices"""
    
    print("=== Testing Device Detection ===")
    
    # List all available devices
    devices = pyspacemouse.list_devices()
    print(f"Found {len(devices)} 3D mouse device types: {devices}")
    
    if len(devices) == 0:
        print("No devices found!")
        return
    
    # Try to open multiple instances
    device1 = None
    device2 = None
    
    try:
        print("\n=== Opening Device 1 (DeviceNumber=0) ===")
        device1 = pyspacemouse.open(device=devices[0], DeviceNumber=0)
        if device1:
            print(f"Device 1 opened successfully")
            print(f"Device 1 connected: {device1.connected}")
        else:
            print("Device 1 failed to open")
            
    except Exception as e:
        print(f"Error opening device 1: {e}")
    
    try:
        print("\n=== Opening Device 2 (DeviceNumber=1) ===")
        device2 = pyspacemouse.open(device=devices[0], DeviceNumber=6)
        if device2:
            print(f"Device 2 opened successfully")
            print(f"Device 2 connected: {device2.connected}")
        else:
            print("Device 2 failed to open")
            
    except Exception as e:
        print(f"Error opening device 2: {e}")
    
    # Test reading from both devices
    print("\n=== Testing Reading ===")
    print("Move the 3D mice to test input...")
    
    for i in range(50):  # Test for 5 seconds
        try:
            if device1:
                state1 = device1.read()
                if state1 and (abs(state1.x) > 0.01 or abs(state1.y) > 0.01 or abs(state1.z) > 0.01):
                    print(f"Device 1: x={state1.x:.3f}, y={state1.y:.3f}, z={state1.z:.3f}")
            
            if device2:
                state2 = device2.read()
                if state2 and (abs(state2.x) > 0.01 or abs(state2.y) > 0.01 or abs(state2.z) > 0.01):
                    print(f"Device 2: x={state2.x:.3f}, y={state2.y:.3f}, z={state2.z:.3f}")
                    
        except Exception as e:
            print(f"Error reading: {e}")
            
        time.sleep(0.1)
    
    # Cleanup
    print("\n=== Cleanup ===")
    try:
        if device1:
            device1.close()
            print("Device 1 closed")
    except Exception as e:
        print(f"Error closing device 1: {e}")
        
    try:
        if device2:
            device2.close()
            print("Device 2 closed")
    except Exception as e:
        print(f"Error closing device 2: {e}")
        
    try:
        pyspacemouse.close()
        print("Module-level close called")
    except Exception as e:
        print(f"Error in module-level close: {e}")

if __name__ == "__main__":
    test_device_detection()