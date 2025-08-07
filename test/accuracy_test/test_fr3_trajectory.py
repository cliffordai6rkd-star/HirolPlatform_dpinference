#!/usr/bin/env python3
"""
Test script for FR3 trajectory control
"""
import numpy as np
import time
from fr3_interface import FR3Interface

def test_trajectory_control():
    """Test trajectory control functionality"""
    
    # Initialize interface
    print("Initializing FR3 interface...")
    interface = FR3Interface()
    
    try:
        # Move to start position first
        print("\nMoving to start position...")
        interface.move_to_start()
        time.sleep(2.0)
        
        # Get current pose
        current_pose = interface.get_current_pose()
        print(f"Current pose: {current_pose}")
        
        # Test 1: Small trajectory movement
        print("\n=== Test 1: Small trajectory (5cm in X) ===")
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
        print("Executing trajectory...")
        start_time = time.time()
        interface.move_to_pose_traj(target_pose, finish_time=None)
        execution_time = time.time() - start_time
        print(f"Trajectory execution time: {execution_time:.2f} seconds")
        
        # Check final position
        time.sleep(0.5)
        final_pose = interface.get_current_pose()
        error_x = abs(final_pose['x'] - target_pose['x'])
        print(f"Final pose: {final_pose}")
        print(f"Position error in X: {error_x*1000:.2f} mm")
        
        # Test 2: Multi-axis movement
        print("\n=== Test 2: Multi-axis trajectory ===")
        target_pose = {
            'x': current_pose['x'] - 0.03,  # Move back 3cm in X
            'y': current_pose['y'] + 0.04,  # Move 4cm in Y
            'z': current_pose['z'] + 0.02,  # Move up 2cm in Z
            'qx': current_pose['qx'],
            'qy': current_pose['qy'],
            'qz': current_pose['qz'],
            'qw': current_pose['qw']
        }
        
        print(f"Target pose: {target_pose}")
        print("Executing trajectory...")
        start_time = time.time()
        interface.move_to_pose_traj(target_pose, finish_time=3.0)
        execution_time = time.time() - start_time
        print(f"Trajectory execution time: {execution_time:.2f} seconds")
        
        # Check final position
        time.sleep(0.5)
        final_pose = interface.get_current_pose()
        error_x = abs(final_pose['x'] - target_pose['x'])
        error_y = abs(final_pose['y'] - target_pose['y'])
        error_z = abs(final_pose['z'] - target_pose['z'])
        total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        print(f"Final pose: {final_pose}")
        print(f"Position errors - X: {error_x*1000:.2f} mm, Y: {error_y*1000:.2f} mm, Z: {error_z*1000:.2f} mm")
        print(f"Total position error: {total_error*1000:.2f} mm")
        
        # Test 3: Fast trajectory
        print("\n=== Test 3: Fast trajectory (1 second) ===")
        target_pose = {
            'x': current_pose['x'],
            'y': current_pose['y'],
            'z': current_pose['z'],
            'qx': current_pose['qx'],
            'qy': current_pose['qy'],
            'qz': current_pose['qz'],
            'qw': current_pose['qw']
        }
        
        print("Returning to original position...")
        start_time = time.time()
        interface.move_to_pose_traj(target_pose, finish_time=1.0)
        execution_time = time.time() - start_time
        print(f"Trajectory execution time: {execution_time:.2f} seconds")
        
        print("\nAll trajectory tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    
    finally:
        # Close interface
        interface.close()


if __name__ == "__main__":
    test_trajectory_control()