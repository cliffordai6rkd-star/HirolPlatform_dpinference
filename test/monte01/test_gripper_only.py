import os
import time
import numpy as np
from threading import Thread
from hardware.monte01.agent import Agent
from hardware.monte01.arm import Arm
from tools import file_utils
import glog as log
import argparse
log.setLevel("INFO")

def test_gripper(agent: Agent):
    """Test gripper functionality without 3D mouse"""
    agent.wait_for_ready()
    
    sim = agent.sim
    
    # Wait for the viewer to be initialized
    while sim.viewer is None:
        time.sleep(0.1)
    
    # Wait for the viewer to start running
    while not sim.viewer.is_running():
        time.sleep(0.1)
    
    try:
        arm_left: Arm = agent.arm_right()
        
        # Move to start position
        arm_left.move_to_start()
        arm_left.hold_position_for_duration(1.0)
        
        log.info("Starting gripper test...")
        
        # Get gripper
        gripper = arm_left.get_gripper()
        if gripper is None:
            log.error("No gripper found!")
            return
        
        # Test gripper movements
        log.info("Testing gripper movements:")
        
        log.info("1. Moving to middle position (400)")
        gripper.gripper_move(0.5)
        arm_left.hold_position_for_duration(2.0)
        
        log.info("2. Opening gripper")
        gripper.gripper_open()
        arm_left.hold_position_for_duration(2.0)
        
        log.info("3. Closing gripper")
        gripper.gripper_close()
        arm_left.hold_position_for_duration(2.0)
        
        log.info("4. Opening gripper again")
        gripper.gripper_open()
        arm_left.hold_position_for_duration(2.0)
        
        log.info("Gripper test completed!")
        
        # Keep simulation running
        while sim.viewer.is_running():
            arm_left.hold_joint_positions()
            time.sleep(0.1)
        
    except Exception as e:
        log.error(f"Error in gripper test: {e}", exc_info=True)

if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(cur_path, '../../hardware/monte01/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)
    print(f"Configuration loaded: {config}")

    parser = argparse.ArgumentParser(description='Gripper test without 3D mouse.')
    parser.add_argument('--use_real_robot', action='store_true', help='Enable real robot mode.')
    args = parser.parse_args()
    
    agent = Agent(config=config, use_real_robot=args.use_real_robot)
    
    # Start the gripper test
    test_thread = Thread(target=test_gripper, args=(agent,))
    test_thread.start()
    
    try:
        test_thread.join()
    except KeyboardInterrupt:
        print("收到中斷信號，正在關閉...")
    
    # Wait for simulation to end
    if hasattr(agent, 'sim_thread'):
        agent.sim_thread.join()

    print("Gripper test finished.")