#!/usr/bin/env python3
"""
Unit test for trajectory functionality
"""
import sys
import os
import numpy as np
import threading
import time

# Add the HIROLRobotPlatform to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
platform_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, platform_dir)

from hardware.base.utils import Buffer, TrajectoryState
from trajectory.cartesian_trajectory import CartessianTrajectory
import yaml

def test_trajectory_buffer():
    """Test trajectory generation and buffer operations"""
    
    print("Loading trajectory configuration...")
    config_path = os.path.join(platform_dir, 'trajectory/config/cartesian_polynomial_traj_cfg.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create buffer and trajectory generator
    buffer = Buffer(config["buffer"]["size"], config["buffer"]["dim"])
    lock = threading.Lock()
    cart_traj = CartessianTrajectory(config["cart_polynomial"], buffer, lock)
    
    # Test trajectory
    start = np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])  # x,y,z,qx,qy,qz,qw
    end = np.array([0.6, 0.1, 0.35, 0, 0, 0, 1])   # Move 10cm in X, 10cm in Y, 5cm in Z
    
    target = TrajectoryState()
    target._zero_order_values = np.vstack((start, end))
    target._first_order_values = np.zeros((2, 7))
    target._second_order_values = np.zeros((2, 7))
    
    print(f"Start pose: {start}")
    print(f"End pose: {end}")
    print(f"Trajectory dt: {cart_traj.dt} seconds")
    
    # Start trajectory generation in thread
    print("\nStarting trajectory generation...")
    traj_thread = threading.Thread(target=cart_traj.plan_trajectory, args=(target, 2.0))
    traj_thread.start()
    
    # Consume points from buffer
    points_consumed = []
    while True:
        lock.acquire()
        success, point = buffer.pop_data()
        buffer_size = buffer.size()
        lock.release()
        
        if success:
            points_consumed.append(point)
            print(f"Point {len(points_consumed)}: pos=[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}], "
                  f"buffer_size={buffer_size}")
        else:
            if cart_traj.trajectory_idle and buffer_size == 0:
                break
            time.sleep(0.001)
    
    # Wait for thread to complete
    traj_thread.join()
    
    print(f"\nTrajectory generation completed!")
    print(f"Total points generated: {len(points_consumed)}")
    print(f"First point: {points_consumed[0]}")
    print(f"Last point: {points_consumed[-1]}")
    print(f"Expected end: {end}")
    
    # Verify trajectory
    position_error = np.linalg.norm(points_consumed[-1][:3] - end[:3])
    print(f"\nFinal position error: {position_error*1000:.2f} mm")
    
    if position_error < 0.001:  # Less than 1mm error
        print("✓ Trajectory test PASSED")
    else:
        print("✗ Trajectory test FAILED")
        

if __name__ == "__main__":
    test_trajectory_buffer()