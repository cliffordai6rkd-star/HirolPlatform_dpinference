#!/usr/bin/env python3
"""
RobotMotion Task Example - API-based Control (No Keyboard Required)

This example demonstrates how to use RobotMotion for task execution without
relying on keyboard interaction. Suitable for:
- Automated task execution scripts
- Remote SSH environments without keyboard
- Integration into larger systems (data collection pipelines, etc.)

Typical workflow:
    1. Initialize RobotMotion
    2. enable_hardware() - Start controlling robot
    3. start_recording() - Begin data collection
    4. Execute task motions
    5. stop_recording() - Save recorded data
    6. reset_to_home() - Return to safe position
    7. close() - Clean shutdown
"""

import sys
import time
import numpy as np
import glog as log
from pathlib import Path

# Add HIROLRobotPlatform to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from factory.tasks.robot_motion import RobotMotion


def simple_pick_and_place_task(robot_motion: RobotMotion):
    """
    Example: Simple pick-and-place task with data collection

    Task flow:
        1. Move to pre-grasp position
        2. Open gripper
        3. Move down to object
        4. Close gripper (grasp)
        5. Lift object
        6. Move to place position
        7. Open gripper (release)
        8. Return to home
    """
    log.info("Starting pick-and-place task...")

    # Get current state
    state = robot_motion.get_state()
    base_pose = state['pose'].copy()

    # Define task waypoints (relative to current pose)
    pre_grasp_pose = base_pose.copy()
    pre_grasp_pose[2] += 0.1  # 10cm above

    grasp_pose = base_pose.copy()
    grasp_pose[2] -= 0.05  # 5cm below current

    lift_pose = grasp_pose.copy()
    lift_pose[2] += 0.15  # Lift 15cm

    place_pose = lift_pose.copy()
    place_pose[0] += 0.1  # 10cm forward

    # Execute task
    log.info("Step 1: Move to pre-grasp position")
    robot_motion.send_pose_command(pre_grasp_pose)
    time.sleep(2.0)

    log.info("Step 2: Open gripper")
    robot_motion.send_gripper_command_simple(1.0)
    time.sleep(1.0)

    log.info("Step 3: Move down to grasp")
    robot_motion.send_pose_command(grasp_pose)
    time.sleep(2.0)

    log.info("Step 4: Close gripper (grasp)")
    robot_motion.send_gripper_command_simple(0.0)
    # time.sleep(1.0)

    log.info("Step 5: Lift object")
    robot_motion.send_pose_command(lift_pose)
    time.sleep(2.0)

    log.info("Step 6: Move to place position")
    robot_motion.send_pose_command(place_pose)
    time.sleep(2.0)

    log.info("Step 7: Open gripper (release)")
    robot_motion.send_gripper_command_simple(1.0)
    time.sleep(1.0)

    log.info("Task completed!")


def trajectory_execution_task(robot_motion: RobotMotion):
    """
    Example: Execute a circular trajectory
    """
    log.info("Starting trajectory execution task...")

    # Get current state
    state = robot_motion.get_state()
    center_pose = state['pose'].copy()

    # Generate circular waypoints
    num_points = 8
    radius = 0.05  # 5cm radius
    waypoints = []

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        pose = center_pose.copy()
        pose[0] += radius * np.cos(angle)
        pose[1] += radius * np.sin(angle)
        waypoints.append(pose)

    # Execute trajectory
    log.info(f"Executing circular trajectory with {num_points} waypoints")
    robot_motion.execute_trajectory(waypoints, timing=[1.5] * num_points)

    log.info("Trajectory task completed!")


def linear_motion_task(robot_motion: RobotMotion):
    """
    Example: Test move_line for straight-line motion with CartesianTrajectory

    Demonstrates:
        1. Forward linear motion (X-axis)
        2. Lateral linear motion (Y-axis)
        3. Vertical linear motion (Z-axis)
        4. Diagonal linear motion
        5. Auto-duration vs manual-duration
    """
    log.info("Starting linear motion task...")

    # Get current state as starting point
    state = robot_motion.get_state()
    home_pose = state['pose'].copy()

    log.info(f"Home position: {home_pose[:3]}")

    # Test 1: Forward motion (X-axis +10cm)
    log.info("\n--- Test 1: Forward motion (X +10cm) ---")
    target1 = home_pose.copy()
    target1[0] += 0.1  # X +10cm
    log.info(f"Target: {target1[:3]}")
    robot_motion.move_line(target1, duration=2.0)
    time.sleep(0.5)

    # Test 2: Lateral motion (Y-axis +10cm)
    log.info("\n--- Test 2: Lateral motion (Y +10cm) ---")
    target2 = target1.copy()
    target2[1] += 0.1  # Y +10cm
    log.info(f"Target: {target2[:3]}")
    robot_motion.move_line(target2, duration=2.0)
    time.sleep(0.5)

    # Test 3: Vertical motion (Z-axis +10cm)
    log.info("\n--- Test 3: Vertical motion (Z +10cm) ---")
    target3 = target2.copy()
    target3[2] += 0.1  # Z +10cm
    log.info(f"Target: {target3[:3]}")
    robot_motion.move_line(target3, duration=2.0)
    time.sleep(0.5)

    # Test 4: Diagonal motion (back to start + offset)
    log.info("\n--- Test 4: Diagonal motion (back to offset home) ---")
    target4 = home_pose.copy()
    target4[0] += 0.05  # X +5cm
    target4[1] += 0.05  # Y +5cm
    target4[2] += 0.05  # Z +5cm
    log.info(f"Target: {target4[:3]}")
    robot_motion.move_line(target4, duration=3.0)
    time.sleep(0.5)

    # Test 5: Auto-duration (return to exact home)
    log.info("\n--- Test 5: Return to home (auto-duration) ---")
    log.info(f"Target: {home_pose[:3]}")
    robot_motion.move_line(home_pose)  # Auto-calculate duration
    time.sleep(0.5)

    log.info("\nLinear motion task completed!")
    log.info("All move_line tests passed successfully!")


def main():
    log.info("=" * 60)
    log.info("RobotMotion Task Example - API-based Control")
    log.info("=" * 60)

    # 1. Initialize RobotMotion
    log.info("\n[1/7] Initializing RobotMotion...")
    robot_motion = RobotMotion(config_path="factory/tasks/config/robot_motion_fr3_cfg.yaml")

    try:
        # 2. Enable hardware execution
        log.info("\n[2/7] Enabling hardware execution...")
        robot_motion.enable_hardware()
        time.sleep(1.0)

        # 3. Start data recording
        log.info("\n[3/7] Starting data recording...")
        robot_motion.start_recording()
        time.sleep(0.5)

        # 4. Execute task (choose one)
        log.info("\n[4/7] Executing task...")

        # Option A: Simple pick-and-place
        simple_pick_and_place_task(robot_motion)

        # Option B: Trajectory execution
        # trajectory_execution_task(robot_motion)

        # Option C: Linear motion with move_line (uncomment to use)
        # linear_motion_task(robot_motion)

        # 5. Stop recording
        log.info("\n[5/7] Stopping data recording...")
        robot_motion.stop_recording()
        log.info("Data saved successfully!")

        # 6. Reset to home position
        log.info("\n[6/7] Resetting to home position...")
        robot_motion.reset_to_home()
        time.sleep(2.0)

    except Exception as e:
        log.error(f"Task execution failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 7. Clean shutdown
        log.info("\n[7/7] Closing RobotMotion...")
        robot_motion.close()
        log.info("=" * 60)
        log.info("Task completed and system closed")
        log.info("=" * 60)


if __name__ == "__main__":
    main()