#!/usr/bin/env python3
"""
Test script for Xbox controller teleoperation with Monte01 robot.

This script demonstrates Xbox controller teleoperation functionality:
- Switch between control modes (None/Left Arm/Right Arm/Chassis)
- Real-time arm control with Xbox controller
- Emergency stop functionality
- Speed adjustments via D-pad
- Gripper control
- Return to home position

Usage:
    python test_xbox_teleop.py [--use_real_robot]
"""

import os
import sys
import time
import argparse
from threading import Thread
import glog as log
import cv2
import rclpy

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from hardware.monte01.agent import Agent
from tools import file_utils
from teleop.gamepads.xbox_teleop import XboxTeleop, ControlMode

log.setLevel("INFO")


def xbox_teleop_demo(agent: Agent):
    """Xbox controller teleoperation demo"""
    print("\n" + "="*50)
    print("Xbox Controller Teleoperation Demo")
    print("="*50)
    
    # Wait for agent to be ready
    agent.wait_for_ready()
    
    # Wait for simulation to start
    sim = agent.sim
    while sim.viewer is None:
        time.sleep(0.1)
    
    while not sim.viewer.is_running():
        time.sleep(0.1)
    
    print("\nInitializing Xbox teleoperation...")
    
    # Create and start Xbox teleop controller
    xbox_teleop = XboxTeleop(agent=agent)
    
    if not xbox_teleop.start():
        log.error("Failed to start Xbox teleoperation")
        return
    
    try:
        print("\n" + "-"*50)
        print("Xbox Controller Controls:")
        print("-"*50)
        print("X Button:        Switch control mode (None/Left Arm/Right Arm/Chassis)")
        print("Left Stick:      Translation control (X-Y for arms, linear for chassis)")
        print("Right Stick:     Rotation control (X-Y rotation for arms, angular for chassis)")
        print("LT/RT:          Z-axis translation (arms only)")
        print("LB/RB:          Z-axis rotation (arms only)")
        print("A + LB/RB:      Gripper control (decrease/increase)")
        print("D-pad Up/Down:   Speed adjustment (translation)")
        print("D-pad Left/Right: Gripper speed adjustment")
        print("Double-click A:  Emergency stop/resume")
        print("Hold Y (1s):     Return to home position")
        print("Ctrl+C:         Exit")
        print("-"*50)
        
        print(f"\nCurrent control mode: {xbox_teleop.control_mode.value}")
        print("Connect your Xbox controller and start controlling!")
        
        # Initialize arms to start position
        try:
            print("\nMoving arms to start position...")
            arm_left = agent.arm_left()
            arm_right = agent.arm_right()
            
            arm_left.move_to_start()
            arm_right.move_to_start()
            
            print("Arms moved to start position.")
            
        except Exception as e:
            log.warning(f"Could not move arms to start position: {e}")
        
        # Main demo loop
        demo_start_time = time.time()
        status_update_interval = 5.0  # seconds
        last_status_update = demo_start_time
        
        while xbox_teleop.running:
            current_time = time.time()
            
            # Print status update periodically
            if current_time - last_status_update >= status_update_interval:
                print(f"\nDemo running for {current_time - demo_start_time:.1f}s")
                print(f"Control mode: {xbox_teleop.control_mode.value}")
                print(f"Emergency stop: {'ACTIVE' if xbox_teleop.emergency_stop else 'Inactive'}")
                print(f"Translation speed: {xbox_teleop.translation_speed:.2f} m/s")
                print(f"Rotation speed: {xbox_teleop.rotation_speed:.2f} rad/s")
                current_gripper_pos = xbox_teleop.left_gripper_position if xbox_teleop.control_mode == ControlMode.LEFT_ARM else xbox_teleop.right_gripper_position
                print(f"Gripper position: {current_gripper_pos:.0f}/800")
                last_status_update = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    except Exception as e:
        log.error(f"Error in Xbox teleop demo: {e}")
    
    finally:
        print("\nStopping Xbox teleoperation...")
        xbox_teleop.stop()
        print("Xbox teleoperation stopped.")


def ros2_camera_thread(agent: Agent):
    """ROS2 camera thread function"""
    try:
        rclpy.spin(agent.head_front_camera())
    except KeyboardInterrupt:
        print("Camera thread interrupted...")
    finally:
        if rclpy.ok():
            agent.head_front_camera().destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Xbox Controller Teleoperation Test for Monte01 Robot'
    )
    parser.add_argument(
        '--use_real_robot', 
        action='store_true',
        help='Enable real robot mode (default: simulation only)'
    )
    parser.add_argument(
        '--no_camera',
        action='store_true', 
        help='Disable camera ROS2 node'
    )
    
    args = parser.parse_args()
    
    print("Xbox Controller Teleoperation Test")
    print("==================================")
    
    if args.use_real_robot:
        print("Mode: Real Robot")
        print("WARNING: Make sure the real robot is properly set up and safe to operate!")
        response = input("Continue with real robot? (y/N): ")
        if response.lower() != 'y':
            print("Aborted by user.")
            return
    else:
        print("Mode: Simulation Only")
    
    # Load robot configuration
    config_file = os.path.join(current_dir, '..', '..', 'hardware', 'monte01', 'config', 'agent.yaml')
    config = file_utils.read_config(config_file)
    print(f"Configuration loaded from: {config_file}")
    
    # Create agent
    print("\nInitializing Monte01 agent...")
    agent = Agent(config=config, use_real_robot=args.use_real_robot)
    
    # Start Xbox teleoperation demo in separate thread
    demo_thread = Thread(target=xbox_teleop_demo, args=(agent,), daemon=True)
    demo_thread.start()
    
    # Start ROS2 camera thread if enabled
    camera_thread = None
    if not args.no_camera:
        print("Starting camera ROS2 node...")
        camera_thread = Thread(target=ros2_camera_thread, args=(agent,), daemon=True)
        camera_thread.start()
    
    try:
        # Wait for demo thread to complete
        demo_thread.join()
        
        # Wait for camera thread if running
        if camera_thread:
            camera_thread.join()
        
        # Wait for simulation thread
        if hasattr(agent, 'sim_thread'):
            agent.sim_thread.join()
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    except Exception as e:
        log.error(f"Error in main: {e}")
    
    finally:
        print("Test completed.")

if __name__ == "__main__":
    main()