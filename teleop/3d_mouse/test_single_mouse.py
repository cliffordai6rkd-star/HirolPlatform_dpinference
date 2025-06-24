import os
import time
import numpy as np
from threading import Thread
from hardware.monte01.agent import Agent
from hardware.monte01.arm import Arm
from tools import file_utils
import glog as log
import argparse
import cv2, rclpy
import pyspacemouse
log.setLevel("INFO")

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

class SpaceMouseTeleop:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.control_mode = "left_arm"  # "left_arm", "right_arm"
        self.scale_factor = 0.01  # Scale factor for mouse input
        self.running = True
        self.is_moving = False  # Flag to track if arm is currently moving
        
        # Initialize 3D mouse with device conflict handling
        self.mouse_success = False
        try:
            # Try to open with specific device index or handle multiple devices
            self.mouse_success = pyspacemouse.open(dof_callback=None, button_callback=None)
            log.info("3D mouse initialized successfully")
        except Exception as e:
            log.warning(f"Failed to initialize 3D mouse: {e}")
            # Try alternative initialization methods
            try:
                # Close any existing connections first
                pyspacemouse.close()
                # Try to open again
                self.mouse_success = pyspacemouse.open(dof_callback=None, button_callback=None)
                log.info("3D mouse initialized successfully on retry")
            except Exception as e2:
                log.error(f"Failed to initialize 3D mouse after retry: {e2}")
                self.mouse_success = False
        
    def control_loop(self):
        """Main teleoperation control loop"""
        self.agent.wait_for_ready()
        
        sim = self.agent.sim
        import time
        
        # Wait for the viewer to be initialized
        while sim.viewer is None:
            time.sleep(0.1)
        
        # Wait for the viewer to start running
        while not sim.viewer.is_running():
            time.sleep(0.1)
        
        try:
            arm_left: Arm = self.agent.arm_left()
            arm_right: Arm = self.agent.arm_right()
            
            # Move to start position
            arm_left.move_to_start()
            arm_left.hold_position_for_duration(0.2)
            
            if self.mouse_success:
                log.info("Starting 3D mouse teleoperation...")
                log.info("Button 0: Switch to left arm control")
                log.info("Button 1: Switch to right arm control")
            else:
                log.warning("3D mouse not available, running in simulation-only mode")
                log.info("Press Ctrl+C to exit")
            
            t_prev = -1
            
            while self.running and sim.viewer.is_running():
                try:
                    # Read 3D mouse state only if mouse is available
                    if self.mouse_success:
                        state = pyspacemouse.read()
                    else:
                        # Fallback: create dummy state for testing
                        import time
                        time.sleep(0.1)
                        continue
                    
                    if state.t != t_prev:
                        # Handle button presses for mode switching
                        if state.buttons[0]:  # Button 0 - switch to left arm
                            if self.control_mode != "left_arm":
                                self.control_mode = "left_arm"
                                log.info("Switched to left arm control")
                        
                        if state.buttons[1]:  # Button 1 - switch to right arm
                            if self.control_mode != "right_arm":
                                self.control_mode = "right_arm"
                                log.info("Switched to right arm control")
                        
                        # Get current arm based on control mode
                        current_arm = arm_left if self.control_mode == "left_arm" else arm_right
                        
                        # Apply 3D mouse input if there's significant movement and arm is not currently moving
                        if (abs(state.x) > 0.01 or abs(state.y) > 0.01 or abs(state.z) > 0.01 or
                            abs(state.roll) > 0.01 or abs(state.pitch) > 0.01 or abs(state.yaw) > 0.01) and not self.is_moving:
                            
                            # Get current TCP pose
                            current_pose = current_arm.get_tcp_pose()
                            
                            if current_pose is not None:
                                # Create target pose by applying deltas
                                target_pose = current_pose.copy()
                                
                                # Apply translation (scaled)
                                target_pose[AXIS_X, 3] += state.x * self.scale_factor
                                target_pose[AXIS_Y, 3] += state.y * self.scale_factor
                                target_pose[AXIS_Z, 3] += state.z * self.scale_factor
                                
                                # Apply rotation (scaled)
                                # Note: For full 6DOF control, you would need to properly apply
                                # roll, pitch, yaw rotations to the rotation matrix
                                # For now, we'll keep it simple with translation only
                                
                                # Set moving flag and move arm to target pose (non-blocking)
                                self.is_moving = True
                                success = current_arm.move_to_pose(target_pose, blocking=False)
                                if not success:
                                    log.warning("Failed to move to target pose")
                                    self.is_moving = False
                        
                        
                        # Check if current movement is complete
                        if self.is_moving:
                            current_arm = arm_left if self.control_mode == "left_arm" else arm_right
                            if current_arm.is_trajectory_done():
                                self.is_moving = False
                        
                        t_prev = state.t
                    else:
                        # No significant input, hold current joint positions for all arms
                        arm_left.hold_joint_positions()
                        arm_right.hold_joint_positions()
                    time.sleep(0.01)  # Small delay to prevent excessive updates
                    
                except Exception as e:
                    log.error(f"Error in teleoperation loop: {e}")
                    break
            
        except Exception as e:
            log.error(f"Error in control loop: {e}", exc_info=True)
        finally:
            if self.mouse_success:
                try:
                    pyspacemouse.close()
                    log.info("3D mouse closed successfully")
                except Exception as e:
                    log.warning(f"Error closing 3D mouse: {e}")

def ros2_spin(agent: Agent):
    """ROS2 spinning thread"""
    try:
        rclpy.spin(agent.head_front_camera())
    except KeyboardInterrupt:
        print("偵測到 Ctrl+C，正在關閉節點...")
    finally:
        if rclpy.ok():
            agent.head_front_camera().destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()
        print("程式已乾淨地關閉。")

if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(cur_path, '../../hardware/monte01/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)
    print(f"Configuration loaded: {config}")

    parser = argparse.ArgumentParser(description='3D Mouse teleoperation with option to use real robot.')
    parser.add_argument('--use_real_robot', action='store_true', help='Enable real robot mode.')
    args = parser.parse_args()
    
    agent = Agent(config=config, use_real_robot=args.use_real_robot)
    
    # Create teleoperation controller
    teleop = SpaceMouseTeleop(agent)
    
    if not teleop.mouse_success:
        print("Failed to initialize 3D mouse. Exiting.")
        exit(1)
    
    # Start the control loop in another thread
    control_thread = Thread(target=teleop.control_loop)
    control_thread.start()

    # Start the ROS2 node in another thread
    ros2_thread = Thread(target=ros2_spin, args=(agent,))
    ros2_thread.start()
    
    try:
        ros2_thread.join()
    except KeyboardInterrupt:
        print("收到中斷信號，正在關閉...")
        teleop.running = False
    
    # Wait for threads to finish
    control_thread.join()
    agent.sim_thread.join()

    print("3D Mouse teleoperation finished.")