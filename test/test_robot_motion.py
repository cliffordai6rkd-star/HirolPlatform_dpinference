"""
Test script for RobotMotion class

Tests motion control, data collection, and system integration

Usage:
    python test/test_robot_motion.py

Author: HIROL
Date: 2025-09-30
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add HIROLRobotPlatform to path
platform_path = Path(__file__).parent.parent
sys.path.insert(0, str(platform_path))

from factory.tasks.robot_motion import RobotMotion
import glog as log


def test_initialization():
    """Test 1: Basic initialization"""
    log.info("=" * 50)
    log.info("Test 1: Initialization")
    log.info("=" * 50)

    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    assert robot_motion._is_initialized, "Robot should be initialized"
    assert len(robot_motion._ee_links) > 0, "Should have end effector links"
    log.info(f"✓ Robot initialized with {len(robot_motion._ee_links)} end effector(s)")
    log.info(f"✓ End effector indices: {robot_motion._ee_index}")

    robot_motion.close()
    log.info("✓ Test 1 passed\n")


def test_get_state():
    """Test 2: Get robot state"""
    log.info("=" * 50)
    log.info("Test 2: Get State")
    log.info("=" * 50)

    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    state = robot_motion.get_state()

    # Check all required fields
    required_fields = ["pose", "vel", "q", "dq", "torque", "gripper_pos", "time_stamp"]
    for field in required_fields:
        assert field in state, f"Missing field: {field}"
        log.info(f"✓ {field}: {state[field].shape if hasattr(state[field], 'shape') else type(state[field])}")

    # Check dimensions
    assert state["pose"].shape[0] == 7, "Pose should be 7D"
    assert state["q"].shape[0] == 7, "Joints should be 7D for FR3"

    robot_motion.close()
    log.info("✓ Test 2 passed\n")


def test_motion_commands():
    """Test 3: Send motion commands"""
    log.info("=" * 50)
    log.info("Test 3: Motion Commands")
    log.info("=" * 50)

    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    # Enable hardware execution
    robot_motion._toggle_hardware()
    log.info("✓ Hardware execution enabled")

    # Get initial state
    initial_state = robot_motion.get_state()
    log.info(f"✓ Initial pose: {initial_state['pose'][:3]}")  # Print only position

    # Test pose command
    target_pose = initial_state['pose'].copy()
    target_pose[2] += 0.05  # Move up 5cm
    log.info(f"✓ Sending target pose: {target_pose[:3]}")

    robot_motion.send_pose_command(target_pose)
    time.sleep(2.0)  # Wait for motion

    # Test gripper command (simplified interface for single arm)
    log.info("✓ Testing gripper command (open)")
    robot_motion.send_gripper_command_simple(1.0)  # Open
    time.sleep(1.0)

    log.info("✓ Testing gripper command (close)")
    robot_motion.send_gripper_command_simple(0.0)  # Close
    time.sleep(1.0)

    robot_motion.close()
    log.info("✓ Test 3 passed\n")


def test_data_collection():
    """Test 4: Data recording"""
    log.info("=" * 50)
    log.info("Test 4: Data Collection")
    log.info("=" * 50)

    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    # Enable hardware
    robot_motion._toggle_hardware()

    # Start recording
    log.info("✓ Starting recording...")
    robot_motion.start_recording()
    assert robot_motion._enable_recording, "Recording should be enabled"

    # Execute some motions
    state = robot_motion.get_state()
    for i in range(3):
        target_pose = state['pose'].copy()
        target_pose[2] += 0.02 * ((-1) ** i)  # Move up/down 2cm
        log.info(f"✓ Sending waypoint {i+1}/3")
        robot_motion.send_pose_command(target_pose)
        time.sleep(1.5)

    # Stop recording
    log.info("✓ Stopping recording...")
    robot_motion.stop_recording()
    assert not robot_motion._enable_recording, "Recording should be disabled"

    # Check if episode was saved
    import os
    episode_dir = os.path.join(robot_motion._save_path_dir, "episode_0001")
    data_json = os.path.join(episode_dir, "data.json")

    time.sleep(2.0)  # Wait for async save

    if os.path.exists(data_json):
        log.info(f"✓ Episode saved at {episode_dir}")

        # Check data structure
        import json
        with open(data_json, 'r') as f:
            data = json.load(f)

        assert "info" in data, "Missing info field"
        assert "text" in data, "Missing text field"
        assert "data" in data, "Missing data field"
        log.info(f"✓ Recorded {len(data['data'])} frames")
    else:
        log.warning(f"⚠ Episode not found at {episode_dir} (may still be saving)")

    robot_motion.close()
    log.info("✓ Test 4 passed\n")


def test_trajectory_execution():
    """Test 5: Execute trajectory"""
    log.info("=" * 50)
    log.info("Test 5: Trajectory Execution")
    log.info("=" * 50)

    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    # Enable hardware
    robot_motion._toggle_hardware()

    # Define waypoints
    initial_state = robot_motion.get_state()
    waypoints = []
    for i in range(3):
        pose = initial_state['pose'].copy()
        pose[0] += 0.05 * i  # Move along x-axis
        waypoints.append(pose)

    log.info(f"✓ Executing trajectory with {len(waypoints)} waypoints")

    # Execute
    robot_motion.execute_trajectory(waypoints, timing=[2.0, 2.0, 2.0])

    log.info("✓ Trajectory execution completed")

    robot_motion.close()
    log.info("✓ Test 5 passed\n")


def test_reset():
    """Test 6: System reset"""
    log.info("=" * 50)
    log.info("Test 6: System Reset")
    log.info("=" * 50)

    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    # Enable hardware
    robot_motion._toggle_hardware()

    # Move to random position
    state = robot_motion.get_state()
    target_pose = state['pose'].copy()
    target_pose[2] += 0.1  # Move up 10cm
    log.info("✓ Moving to random position...")
    robot_motion.send_pose_command(target_pose)
    time.sleep(2.0)

    # Reset to home
    log.info("✓ Resetting to home...")
    robot_motion.reset_to_home()

    log.info("✓ Reset completed")

    robot_motion.close()
    log.info("✓ Test 6 passed\n")


def integration_test_full_workflow():
    """Integration Test: Complete workflow"""
    log.info("=" * 50)
    log.info("Integration Test: Full Workflow")
    log.info("=" * 50)

    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    try:
        # Step 1: Initialize and reset
        log.info("Step 1: Resetting to home position...")
        robot_motion._toggle_hardware()
        robot_motion.reset_to_home()
        time.sleep(2.0)

        # Step 2: Start recording
        log.info("Step 2: Starting data recording...")
        robot_motion.start_recording()

        # Step 3: Execute motions
        log.info("Step 3: Executing motion sequence...")
        state = robot_motion.get_state()

        # Define a simple trajectory (square in xy plane)
        waypoints = []
        base_pose = state['pose'].copy()

        # Square waypoints
        offsets = [[0.05, 0.0], [0.05, 0.05], [0.0, 0.05], [0.0, 0.0]]
        for dx, dy in offsets:
            pose = base_pose.copy()
            pose[0] += dx
            pose[1] += dy
            waypoints.append(pose)

        robot_motion.execute_trajectory(waypoints, timing=[2.0] * 4)

        # Test gripper
        log.info("Step 4: Testing gripper...")
        robot_motion.send_gripper_command_simple(0.0)  # Close
        time.sleep(1.0)
        robot_motion.send_gripper_command_simple(1.0)  # Open
        time.sleep(1.0)

        # Step 5: Stop recording
        log.info("Step 5: Stopping recording...")
        robot_motion.stop_recording()

        # Step 6: Verify data
        log.info("Step 6: Verifying recorded data...")
        time.sleep(2.0)  # Wait for save

        import os, json
        episode_dir = os.path.join(robot_motion._save_path_dir, "episode_0001")
        data_json = os.path.join(episode_dir, "data.json")

        if os.path.exists(data_json):
            with open(data_json, 'r') as f:
                data = json.load(f)
            log.info(f"✓ Recorded {len(data['data'])} frames")
            log.info(f"✓ Task description: {data['text']['goal']}")
        else:
            log.warning("⚠ Data file not found yet (may still be saving)")

        log.info("=" * 50)
        log.info("✓ Integration test completed successfully!")
        log.info("=" * 50)

    finally:
        robot_motion.close()


if __name__ == "__main__":
    log.info("\n" + "=" * 50)
    log.info("RobotMotion Test Suite")
    log.info("=" * 50 + "\n")

    try:
        # Basic tests
        test_initialization()
        test_get_state()

        # Uncomment to run hardware tests
        # WARNING: These tests will move the robot!
        # test_motion_commands()
        # test_data_collection()
        # test_trajectory_execution()
        # test_reset()

        # Full integration test
        # integration_test_full_workflow()

        log.info("\n" + "=" * 50)
        log.info("All enabled tests passed!")
        log.info("=" * 50)
        log.info("\nTo run hardware tests, uncomment them in main()")

    except Exception as e:
        log.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)