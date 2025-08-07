"""
Main entry point for VR teleoperation using Monte01 APIs.
This is the refactored version that uses monte01 hardware APIs with config files.

Author: Refactored for monte01 integration
Date: 2025-01-14
"""
import time
import os
import sys
import yaml
import threading

# Add project paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from teleop.XR.quest3.device.vr_television import VRTelevision
from hardware.monte01.agent import Agent
from tools import file_utils
import glog as log


def load_config():
    """Load configuration from YAML files using the same method as test_sim2real.py."""
    # Load base robot config using file_utils like test_sim2real.py
    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(cur_path, '../../../hardware/monte01/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)
    
    # Load VR-specific teleoperation config and merge
    teleop_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(teleop_config_path, 'r') as f:
        teleop_config = yaml.safe_load(f)
    
    # Add VR-specific configuration to the base config
    if 'vr_teleop' not in config:
        config['vr_teleop'] = {}
    
    config['vr_teleop'].update({
        'movement_speed': teleop_config.get('movement_speed', 0.001),
        'rotation_speed': teleop_config.get('rotation_speed', 0.002),
        'gripper_speed': teleop_config.get('gripper_speed', 0.01),
        'max_movement_speed': teleop_config.get('max_movement_speed', 0.008),
        'min_movement_speed': teleop_config.get('min_movement_speed', 0.0005),
        'max_gripper_speed': teleop_config.get('max_gripper_speed', 0.05),
        'min_gripper_speed': teleop_config.get('min_gripper_speed', 0.001),
        'max_gripper_position': teleop_config.get('max_gripper_position', 0.1),
        'end_speed_step': teleop_config.get('end_speed_step', 0.0005),
        'gripper_speed_step': teleop_config.get('gripper_speed_step', 0.001),
        'homing_speed': teleop_config.get('homing_speed', 0.1),
        'left_tcp_offset': teleop_config.get('left_tcp_offset', {}),
        'left_tcp_load': teleop_config.get('left_tcp_load', {}),
        'right_tcp_offset': teleop_config.get('right_tcp_offset', {}),
        'right_tcp_load': teleop_config.get('right_tcp_load', {})
    })
    
    return config


def main(use_real_robot: bool = False):
    """Main function to run VR teleoperation with Monte01 APIs.
    
    Args:
        use_real_robot: Whether to use real robot hardware or simulation only
    """
    log.info("Starting Monte01 VR Teleoperation...")
    
    # Load configuration from files
    config = load_config()
    
    # Initialize robot agent (disable viewer main loop, we'll handle it in main thread)
    log.info("Initializing robot agent...")
    agent = Agent(config, use_real_robot=use_real_robot, start_viewer_main_loop=False)
    
    # Initialize VR device
    log.info("Initializing VR television interface...")
    vr_device = VRTelevision(robot_interface=agent)
    
    # Connect to VR device
    connected = vr_device.connect()
    if not connected:
        log.error("Failed to connect to VR device.")
        return False
    
    log.info("VR device connected successfully. Use the VR controller to operate the robot.")
    
    # Wait for agent to be ready
    log.info("Waiting for agent to be ready...")
    agent.wait_for_ready()
    
    # Wait for the viewer to be initialized and running
    sim = agent.sim
    log.info("Waiting for simulation viewer to start...")
    while sim.viewer is None or not sim.viewer.is_running():
        time.sleep(0.1)
    log.info("Simulation viewer is now running")
    
    # Move arms to start position
    try:
        log.info("Moving arms to start position...")
        agent.move_to_start_position("both")
        log.info("Arms moved to start position successfully")
    except Exception as e:
        log.warning(f"Failed to move to start position: {e}")
    
    def teleoperation_loop():
        """Teleoperation loop running in separate thread"""
        try:
            # Main teleoperation loop
            log.info("Starting teleoperation loop...")
            loop_count = 0
            while sim.viewer.is_running():
                loop_count += 1
                if loop_count % 50 == 0:  # Log every 5 seconds (50 * 0.1s)
                    log.info(f"Teleoperation loop running... iteration {loop_count}")
                
                # Sync robot state to simulator for visualization
                try:
                    agent.sync_to_simulator()
                except Exception as e:
                    log.warning(f"Sync to simulator failed: {e}")
                
                time.sleep(0.1)  # 10Hz sync rate
                
        except KeyboardInterrupt:
            log.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            log.error(f"Error in teleoperation loop: {e}")
        finally:
            # Cleanup
            log.info("Disconnecting VR device...")
            vr_device.disconnect()
            
            log.info("Stopping all robot motion...")
            agent.stop_all_motion()
            
            log.info("Monte01 VR Teleoperation shut down complete.")

    # Start teleoperation loop in separate thread
    teleop_thread = threading.Thread(target=teleoperation_loop, daemon=True)
    teleop_thread.start()
    
    # Main thread runs simulation viewer loop
    try:
        log.info("Main thread starting simulation viewer loop...")
        # The viewer loop runs in the main thread
        while sim.viewer.is_running():
            with sim.locker:
                sim.viewer.sync()
            time.sleep(sim.viewer_dt)
    except KeyboardInterrupt:
        log.info("Received keyboard interrupt in main thread, shutting down...")
    finally:
        # Ensure cleanup happens
        vr_device.disconnect()
        agent.stop_all_motion()
        log.info("Waiting for teleoperation thread to finish...")
        teleop_thread.join(timeout=5.0)


def get_monte01_vr_teleoperation(use_real_robot: bool = False):
    """Get VR teleoperation instance for external use.
    
    Args:
        use_real_robot: Whether to use real robot hardware
    Returns:
        VRTelevision instance configured with RobotInterface
    """
    config = load_config()
    agent = Agent(config, use_real_robot=use_real_robot, start_viewer_main_loop=False)
    vr = VRTelevision(robot_interface=agent)
    return vr


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monte01 VR Teleoperation')
    parser.add_argument('--real-robot', action='store_true', 
                        help='Use real robot hardware (default: simulation only)')
    
    args = parser.parse_args()
    
    main(use_real_robot=args.real_robot)