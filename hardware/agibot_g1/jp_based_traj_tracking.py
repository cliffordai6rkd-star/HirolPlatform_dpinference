#!/usr/bin/env python3
"""
AgiBot G1 Joint Position Based Trajectory Tracking

This module provides functionality for trajectory tracking control using joint positions
for the AgiBot G1 humanoid robot.
"""

import time
from a2d_sdk.robot import RobotController, RobotDds


# Initialize robot controller and robot interface
robot_controller = RobotController()
robot = RobotDds()


def execute(robot: RobotDds, robot_controller: RobotController, action: dict) -> None:
    """
    Execute trajectory tracking control based on joint position commands.
    
    Args:
        robot: RobotDds instance for robot communication
        robot_controller: RobotController instance for trajectory control
        action: Dictionary containing robot action commands with keys:
            - observation_timestamp: Timestamp in nanoseconds
            - head_joint_states: Head joint positions [yaw, pitch]
            - waist_joint_states: Waist joint positions [pitch, height]
            - arm_joint_states: Current arm joint states (14-DOF)
            - arm_cmd: List of arm joint commands (14-DOF each)
    """
    # Initialize robot state and action containers
    robot_states = {}
    robot_actions = []
    
    # Trajectory control parameters
    robot_link = "base_link"
    trajectory_ref_time = 1.0
    wait_control_time = 0.5
    
    # Extract data from action
    infer_timestamp = action["observation_timestamp"]
    robot_states["head"] = action["head_joint_states"]
    robot_states["waist"] = action["waist_joint_states"]
    robot_states["arm"] = action["arm_joint_states"]
    
    arm_joint_action = action["arm_cmd"]
    
    # Process each trajectory point
    for i in range(len(arm_joint_action)):
        robot_action = {
            "left_arm": {
                "action_data": arm_joint_action[i][:7],  # First 7 joints (left arm)
                "control_type": "ABS_JOINT"
            },
            "right_arm": {
                "action_data": arm_joint_action[i][7:14],  # Last 7 joints (right arm)
                "control_type": "ABS_JOINT"
            }
        }
        robot_actions.append(robot_action)
    
    # Execute trajectory tracking control
    time.sleep(1)
    robot_controller.trajectory_tracking_control(
        infer_timestamp,
        robot_states,
        robot_actions,
        robot_link,
        trajectory_ref_time
    )
    
    # Wait for control completion
    time.sleep(wait_control_time)
    
    # Debug output
    print(f"Infer timestamp: {infer_timestamp}")
    print(f"Robot states: {robot_states}")
    print(f"Robot actions: {robot_actions}")
    
    # Monitor arm joint states
    while True:
        state, _ = robot.arm_joint_states()
        print(f"Arm joint states: {state}")
        # Note: This creates an infinite loop - consider adding a break condition


def main():
    """Main function for testing trajectory tracking."""
    # Create sample action data
    action = {
        "observation_timestamp": int(time.time() * 1e9),  # Current timestamp in nanoseconds
        "head_joint_states": [0.0, 0.0],                  # [yaw, pitch]
        "waist_joint_states": [0.5, 0.3],                 # [pitch, height]
        "arm_joint_states": [0.0] * 14,                   # Current 14-DOF arm states
        "arm_cmd": [[0.1] * 14],                          # Single trajectory point with 14 joints
    }
    
    # Execute trajectory tracking
    execute(robot, robot_controller, action)
    print("Done")


if __name__ == "__main__":
    main()