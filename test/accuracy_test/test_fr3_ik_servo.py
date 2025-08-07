#!/usr/bin/env python3
"""
Test script for FR3 IK servo control
"""
import numpy as np
import time
from fr3_interface import FR3Interface

def test_ik_servo():
    """Test IK servo control functionality"""
    
    # Initialize interface
    print("Initializing FR3 interface...")
    interface = FR3Interface()
    
    try:
        # Get current pose
        current_pose = interface.get_current_pose()
        print(f"Current pose: {current_pose}")
        
        # Move to start position first
        print("\nMoving to start position...")
        interface.move_to_start()
        time.sleep(1.5)
        
        # Test 1: Small movement in X direction
        print("\nTest 1: Move +5cm in X direction")
        target_pose = {
            'x': current_pose['x'] + 0.05,  # Move 5cm in X
            'y': current_pose['y'],
            'z': current_pose['z'],
            'qx': current_pose['qx'],
            'qy': current_pose['qy'],
            'qz': current_pose['qz'],
            'qw': current_pose['qw']
        }
        
        print(f"Target pose: {target_pose}")
        interface.move_to_pose_servo(target_pose)
        time.sleep(2)
        
        # Get new pose and check
        new_pose = interface.get_current_pose()
        print(f"New pose after movement: {new_pose}")
        print(f"X difference: {new_pose['x'] - target_pose['x']:.4f} m")
        
        # Test 2: Move in Y direction
        print("\nTest 2: Move +3cm in Y direction")
        target_pose['y'] += 0.03  # Move 3cm in Y
        
        interface.move_to_pose_servo(target_pose)
        time.sleep(2)
        
        final_pose = interface.get_current_pose()
        print(f"Final pose: {final_pose}")
        

        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    
    finally:
        # Close interface
        interface.close()


if __name__ == "__main__":
    test_ik_servo()