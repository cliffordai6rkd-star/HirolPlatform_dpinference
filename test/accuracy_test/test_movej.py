"""
Test script for FR3 movej functionality
"""
import numpy as np
import time
from fr3_interface import FR3Interface

def test_movej():
    """Test move_to_joint functionality"""
    # Initialize interface
    interface = FR3Interface()
    
    try:
        # Get current pose
        current_pose = interface.get_current_pose()
        print(f"Current pose: {current_pose}")
        
        # Move to start position
        print("\nMoving to start position...")
        interface.move_to_start()
        time.sleep(2)  # Wait for robot to stabilize
        
        # Test joint trajectory (movej)
        print("\n=== Testing Joint Trajectory (movej) ===")
        
        # Get current joint positions
        current_joints = interface._fr3_arm.get_joint_states()._positions
        print(f"Current joint positions: {current_joints}")
        
        # Define target joint positions (small movement from current)
        target_joints = current_joints.copy()
        target_joints[0] += 0.3  # Rotate base joint by 0.3 rad
        target_joints[1] -= 0.2  # Adjust shoulder joint
        target_joints[3] -= 0.2  # Adjust elbow joint
        
        print(f"Target joint positions: {target_joints}")
        
        # Execute joint trajectory
        print("\nExecuting joint trajectory with 3 second duration...")
        start_time = time.time()
        interface.move_to_joint(target_joints, finish_time=3.0)
        execution_time = time.time() - start_time
        print(f"Execution completed in {execution_time:.2f} seconds")
        
        # Verify final position
        final_joints = interface._fr3_arm.get_joint_states()._positions
        print(f"\nFinal joint positions: {final_joints}")
        joint_error = np.linalg.norm(final_joints - target_joints)
        print(f"Joint position error: {joint_error:.6f} rad")
        
        if joint_error < 0.01:  # 0.01 rad tolerance
            print("✓ Joint trajectory test PASSED")
        else:
            print("✗ Joint trajectory test FAILED - error too large")
        
        # Wait before returning to start
        print("\nWaiting 2 seconds before returning to start...")
        time.sleep(2)
        
        # Return to start position
        print("Returning to start position...")
        interface.move_to_start()
        time.sleep(2)
        
        # Test with auto-calculated time
        print("\n=== Testing Joint Trajectory with Auto Time ===")
        
        # Larger movement for auto time test
        target_joints2 = current_joints.copy()
        target_joints2[0] -= 0.5
        target_joints2[2] += 0.4
        target_joints2[4] -= 0.3
        
        print(f"Target joint positions: {target_joints2}")
        print("Executing with auto-calculated time...")
        
        start_time = time.time()
        interface.move_to_joint(target_joints2)  # No finish_time specified
        execution_time = time.time() - start_time
        print(f"Auto-calculated execution time: {execution_time:.2f} seconds")
        
        # Verify final position
        final_joints2 = interface._fr3_arm.get_joint_states()._positions
        joint_error2 = np.linalg.norm(final_joints2 - target_joints2)
        print(f"Joint position error: {joint_error2:.6f} rad")
        
        if joint_error2 < 0.01:
            print("✓ Auto-time trajectory test PASSED")
        else:
            print("✗ Auto-time trajectory test FAILED")
        
        print("\n=== All tests completed ===")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise
    finally:
        interface.close()


if __name__ == "__main__":
    test_movej()