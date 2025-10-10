"""
RobotMotion Usage Example

Demonstrates basic motion control and data collection workflow

Usage:
    python examples/robot_motion_example.py

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


def main():
    log.info("=" * 60)
    log.info("RobotMotion Example - Motion Planning with Data Collection")
    log.info("=" * 60)

    # Initialize RobotMotion
    config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
    robot_motion = RobotMotion(config_path)

    log.info("\nKeyboard controls:")
    log.info("  'h': Toggle hardware execution")
    log.info("  'r': Toggle data recording")
    log.info("  'o': Reset to home position")
    log.info("  'q': Quit\n")

    try:
        # Example 1: Simple motion control
        log.info("Example 1: Simple Motion Control")
        log.info("-" * 60)

        # Get current state
        state = robot_motion.get_state()
        log.info(f"Current pose: {state['pose'][:3]}")  # x, y, z
        log.info(f"Current joints: {state['q']}")
        log.info(f"Gripper position: {state['gripper_pos']}")

        # Enable hardware (or press 'h')
        log.info("\nPress 'h' to enable hardware execution")
        time.sleep(3.0)

        # Send pose command
        target_pose = state['pose'].copy()
        target_pose[2] += 0.05  # Move up 5cm
        log.info(f"Sending target pose: {target_pose[:3]}")
        robot_motion.send_pose_command(target_pose)
        time.sleep(2.0)

        # Control gripper
        log.info("Opening gripper...")
        robot_motion.send_gripper_command_simple(1.0)
        time.sleep(1.0)

        log.info("Closing gripper...")
        robot_motion.send_gripper_command_simple(0.0)
        time.sleep(1.0)

        # Example 2: Data collection
        log.info("\nExample 2: Data Collection")
        log.info("-" * 60)

        # Start recording (or press 'r')
        log.info("Press 'r' to start recording")
        time.sleep(3.0)

        # Execute a simple trajectory
        log.info("Executing trajectory...")
        waypoints = []
        base_pose = robot_motion.get_state()['pose']

        for i in range(4):
            pose = base_pose.copy()
            pose[0] += 0.05 * np.sin(i * np.pi / 2)
            pose[1] += 0.05 * np.cos(i * np.pi / 2)
            waypoints.append(pose)

        robot_motion.execute_trajectory(waypoints, timing=[2.0] * 4)

        # Stop recording (or press 'r' again)
        log.info("Press 'r' to stop recording")
        time.sleep(3.0)

        # Example 3: Reset to home
        log.info("\nExample 3: Reset to Home")
        log.info("-" * 60)

        log.info("Press 'o' to reset to home position")
        time.sleep(3.0)

        log.info("\nExample completed! Press 'q' to quit")

        # Keep running for user interaction
        while robot_motion._main_thread_running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info("\nKeyboard interrupt received")

    finally:
        robot_motion.close()
        log.info("RobotMotion closed")


if __name__ == "__main__":
    main()