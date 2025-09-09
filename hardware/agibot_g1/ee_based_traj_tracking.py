#!/usr/bin/env python3
"""
AgiBot G1 End-Effector Based Trajectory Tracking

This module provides functionality for trajectory tracking control using end-effector poses
for the AgiBot G1 humanoid robot.
"""

import time
from typing import Dict, List, Any, Callable, Tuple
from a2d_sdk.robot import RobotController, RobotDds


def poll_state(getter: Callable, length: int, name: str, timeout: float = 2.0, interval: float = 0.1) -> List[float]:
    """
    Poll sensor state until valid data is received.
    
    Args:
        getter: Function to get sensor data (should return tuple of (values, timestamp))
        length: Expected length of the data array
        name: Name of the sensor for logging
        timeout: Timeout in seconds
        interval: Polling interval in seconds
        
    Returns:
        List[float]: Valid sensor data
        
    Raises:
        RuntimeError: If valid data is not received within timeout
    """
    deadline = time.time() + timeout
    last_vals = None
    
    while time.time() < deadline:
        vals, _ = getter()
        last_vals = vals
        
        # Get actual length
        try:
            actual_len = len(vals)
        except Exception:
            actual_len = None
        
        # Check if all elements are numeric
        all_numeric = (actual_len == length) and all(
            _is_number(v) for v in vals
        )
        
        if all_numeric:
            print(f"✅ {name} 就绪")
            return list(vals)
            
        time.sleep(interval)
    
    raise RuntimeError(f"{name} 在 {timeout}s 内未就绪，最后 vals={last_vals!r}")


def _is_number(x) -> bool:
    """
    Check if a value can be converted to float.
    
    Args:
        x: Value to check
        
    Returns:
        bool: True if convertible to float, False otherwise
    """
    try:
        float(x)
        return True
    except Exception:
        return False


def execute(rc: RobotController, action: Dict[str, Any]) -> None:
    """
    Execute end-effector based trajectory tracking control.
    
    Args:
        rc: RobotController instance
        action: Dictionary containing robot action commands with keys:
            - observation_timestamp: Timestamp in nanoseconds
            - head_joint_states: Head joint positions [yaw, pitch]
            - waist_joint_states: Waist joint positions [pitch, height]
            - arm_joint_states: Current arm joint states (14-DOF)
            - arm_cmd: List of delta pose commands (12-DOF each: 6 per arm)
    """
    # Define robot state dictionary
    robot_states = {
        "head": action["head_joint_states"],
        "waist": action["waist_joint_states"],
        "arm": action["arm_joint_states"],
    }
    
    # Define robot action list with delta poses
    robot_actions = [
        {
            "left_arm": {
                "action_data": delta[:6],  # [x, y, z, rx, ry, rz] for left arm
                "control_type": "DELTA_POSE"
            },
            "right_arm": {
                "action_data": delta[6:12],  # [x, y, z, rx, ry, rz] for right arm
                "control_type": "DELTA_POSE"
            },
        }
        for delta in action["arm_cmd"]
    ]
    
    # Execute trajectory tracking control with delta poses
    rc.trajectory_tracking_control(
        infer_timestamp=action["observation_timestamp"],  # Required timestamp
        robot_states=robot_states,                        # Required reference states
        robot_actions=robot_actions,                      # Delta pose commands
        robot_link="base_link",                          # Base coordinate frame
        trajectory_reference_time=1.0,                   # Reference time (smaller = faster)
    )


def main():
    """Main function for end-effector based trajectory tracking."""
    # Initialize robot controller and interface
    rc = RobotController()
    rd = RobotDds()
    
    # Wait for initialization
    time.sleep(2.0)
    
    try:
        # Poll for initial sensor states
        print(">>> 等待传感器数据就绪...")
        head_states = poll_state(rd.head_joint_states, length=2, name="head_joint_states")
        waist_states = poll_state(rd.waist_joint_states, length=2, name="waist_joint_states")
        arm_states = poll_state(rd.arm_joint_states, length=14, name="arm_joint_states")
        
        print(">>> 所有传感器数据就绪，开始下发相对位姿控制")
        
        # Main control loop
        while True:
            ts_ns = int(time.time() * 1e9)
            
            # Define action with delta pose commands
            action = {
                "observation_timestamp": ts_ns,
                "head_joint_states": head_states,
                "waist_joint_states": waist_states,
                "arm_joint_states": arm_states,
                "arm_cmd": [
                    # Delta pose: [left_arm_x, y, z, rx, ry, rz, right_arm_x, y, z, rx, ry, rz]
                    [0.02, 0, 0, 0, 0, 0,  # Left arm: move 2cm in x direction
                     0.02, 0, 0, 0, 0, 0]  # Right arm: move 2cm in x direction
                ],
            }
            
            # Execute trajectory tracking
            execute(rc, action)
            print(f">>> [{time.strftime('%H:%M:%S')}] 相对位姿控制命令已下发")
            
            # Wait until next inference cycle
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n>>> 收到中断，退出。")
    except Exception as e:
        print(f">>> 错误: {e}")
    finally:
        print(">>> 关闭机器人连接...")
        rd.shutdown()


if __name__ == "__main__":
    main()