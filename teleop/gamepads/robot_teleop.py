import os
import sys
import time
import numpy as np
import pygame
import argparse
import glog as log

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from hardware.monte01.agent import Agent
from hardware.monte01.arm import Arm
from tools import file_utils

log.setLevel("INFO")

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

class PS5RobotTeleop:
    def __init__(self, agent: Agent):
        pygame.init()
        pygame.joystick.init()
        
        self.agent = agent
        self.joystick = None
        self.running = False
        
        # Control states
        self.control_enabled = False
        self.current_arm = "left"  # "left" or "right"
        self.movement_scale = 0.02  # 2cm per full stick deflection
        self.rotation_scale = 0.1   # 0.1 rad per full stick deflection
        
        # Movement tracking
        self.last_move_time = 0
        self.move_cooldown = 0.05  # 50ms between moves
        
        # Position holding when no input received
        self.last_input_time = time.time()
        self.input_timeout = 0.1   # 100ms timeout for position holding
        self.is_input_active = False
        self.position_hold_active = False
        
        # Button mappings for robot control
        self.button_actions = {
            0: self.toggle_control,      # X - Enable/disable control
            1: self.emergency_stop,      # Circle - Emergency stop
            2: self.reset_to_start,      # Triangle - Reset to start
            3: self.switch_arm,          # Square - Switch between arms
            4: self.gripper_close,       # L1 - Close gripper
            5: self.gripper_open,        # R1 - Open gripper
            8: self.move_to_home,        # Share - Move to home position
        }
        
        # Axis mappings (stick and trigger controls)
        self.axis_map = {
            0: "Left Stick X",      # Left/right translation
            1: "Left Stick Y",      # Forward/backward translation  
            2: "L2 Trigger",        # Down movement
            3: "Right Stick X",     # Rotation around Z
            4: "Right Stick Y",     # Up/down translation
            5: "R2 Trigger",        # Up movement
        }
        
        # Track previous states
        self.prev_buttons = {}
        self.prev_axes = {}
        
        print("PS5 Robot Teleop initialized")
        print("Controls:")
        print("  X: Toggle control enable/disable")
        print("  Circle: Emergency stop")
        print("  Triangle: Reset to start position")
        print("  Square: Switch between left/right arm")
        print("  L1: Close gripper")
        print("  R1: Open gripper")
        print("  Share: Move to home position")
        print("  Left Stick: X/Y translation")
        print("  Right Stick: Z translation / Z rotation")
        print("  L2/R2 Triggers: Fine Z control")
    
    def connect(self) -> bool:
        if pygame.joystick.get_count() == 0:
            print("No joystick connected!")
            return False
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Connected to: {self.joystick.get_name()}")
        return True
    
    def get_current_arm(self) -> Arm:
        if self.current_arm == "left":
            return self.agent.arm_left()
        else:
            return self.agent.arm_right()
    
    def toggle_control(self):
        self.control_enabled = not self.control_enabled
        status = "ENABLED" if self.control_enabled else "DISABLED" 
        print(f"Control {status}")
        
    def emergency_stop(self):
        self.control_enabled = False
        print("EMERGENCY STOP - Control disabled")
        
    def switch_arm(self):
        self.current_arm = "right" if self.current_arm == "left" else "left"
        print(f"Switched to {self.current_arm} arm")
        
    def reset_to_start(self):
        if not self.control_enabled:
            return
        try:
            arm = self.get_current_arm()
            print(f"Moving {self.current_arm} arm to start position...")
            arm.move_to_start()
        except Exception as e:
            log.error(f"Failed to move to start: {e}")
            
    def move_to_home(self):
        if not self.control_enabled:
            return
        try:
            arm = self.get_current_arm()
            print(f"Moving {self.current_arm} arm to home position...")
            # Get home pose - you may need to define this based on your robot
            arm.move_to_start()  # Using start as home for now
        except Exception as e:
            log.error(f"Failed to move to home: {e}")
            
    def gripper_close(self):
        if not self.control_enabled:
            return
        try:
            arm = self.get_current_arm()
            gripper = arm.get_gripper()
            if gripper:
                print(f"Closing {self.current_arm} gripper")
                gripper.gripper_close()
        except Exception as e:
            log.error(f"Failed to close gripper: {e}")
            
    def gripper_open(self):
        if not self.control_enabled:
            return
        try:
            arm = self.get_current_arm()
            gripper = arm.get_gripper()
            if gripper:
                print(f"Opening {self.current_arm} gripper")
                gripper.gripper_open()
        except Exception as e:
            log.error(f"Failed to open gripper: {e}")
    
    def _hold_current_position(self):
        """Hold current joint positions when no input is received"""
        if not self.agent or not self.control_enabled:
            return
            
        try:
            arm = self.get_current_arm()
            if arm:
                arm.hold_joint_positions()
                log.debug(f"Holding {self.current_arm} arm position")
                    
        except Exception as e:
            log.warning(f"Failed to hold position: {e}")
            
    def process_movement(self):
        if not self.control_enabled:
            # Hold position when control is disabled
            if not self.position_hold_active:
                self.position_hold_active = True
                self._hold_current_position()
            return
            
        current_time = time.time()
        if current_time - self.last_move_time < self.move_cooldown:
            return
            
        # Get current axis values
        left_x = self.joystick.get_axis(0)    # Left stick X
        left_y = -self.joystick.get_axis(1)   # Left stick Y (inverted)
        right_x = self.joystick.get_axis(3)   # Right stick X  
        right_y = -self.joystick.get_axis(4)  # Right stick Y (inverted)
        l2_trigger = (self.joystick.get_axis(2) + 1) / 2  # L2 (0 to 1)
        r2_trigger = (self.joystick.get_axis(5) + 1) / 2  # R2 (0 to 1)
        
        # Apply deadzone
        deadzone = 0.15
        left_x = left_x if abs(left_x) > deadzone else 0
        left_y = left_y if abs(left_y) > deadzone else 0
        right_x = right_x if abs(right_x) > deadzone else 0
        right_y = right_y if abs(right_y) > deadzone else 0
        l2_trigger = l2_trigger if l2_trigger > 0.1 else 0
        r2_trigger = r2_trigger if r2_trigger > 0.1 else 0
        
        # Calculate movement vector
        dx = left_x * self.movement_scale    # X translation
        dy = left_y * self.movement_scale    # Y translation  
        dz = right_y * self.movement_scale   # Z translation (right stick Y)
        
        # Add trigger Z control
        dz += r2_trigger * self.movement_scale * 0.5   # R2 for up
        dz -= l2_trigger * self.movement_scale * 0.5   # L2 for down
        
        # Rotation around Z axis (right stick X)
        drot_z = right_x * self.rotation_scale
        
        # Check if there's any significant input
        has_input = (abs(dx) > 0.001 or abs(dy) > 0.001 or abs(dz) > 0.001 or abs(drot_z) > 0.001)
        
        if has_input:
            self.last_input_time = current_time
            self.is_input_active = True
            self.position_hold_active = False
        else:
            # No significant input, check if we should start holding position
            if current_time - self.last_input_time > self.input_timeout:
                if self.is_input_active or not self.position_hold_active:
                    self.is_input_active = False
                    self.position_hold_active = True
                    self._hold_current_position()
                elif self.position_hold_active:
                    # Continue holding position
                    self._hold_current_position()
            return
        
        # Only move if there's significant input
        if has_input:
            try:
                arm = self.get_current_arm()
                current_pose = arm.get_tcp_pose()
                
                if current_pose is not None:
                    # Create target pose with translation
                    target_pose = current_pose.copy()
                    target_pose[AXIS_X, 3] += dx
                    target_pose[AXIS_Y, 3] += dy  
                    target_pose[AXIS_Z, 3] += dz
                    
                    # Apply rotation around Z axis if needed
                    if abs(drot_z) > 0.001:
                        # Create rotation matrix around Z axis
                        cos_z = np.cos(drot_z)
                        sin_z = np.sin(drot_z)
                        rot_z = np.array([
                            [cos_z, -sin_z, 0],
                            [sin_z, cos_z, 0],
                            [0, 0, 1]
                        ])
                        # Apply rotation to current orientation
                        target_pose[:3, :3] = target_pose[:3, :3] @ rot_z
                    
                    # Move to target pose
                    arm.move_to_pose(target_pose)
                    self.last_move_time = current_time
                    
            except Exception as e:
                log.error(f"Movement failed: {e}")
    
    def start_teleop(self):
        if not self.connect():
            return
            
        # Wait for agent to be ready
        print("Waiting for robot agent to be ready...")
        if not self.agent.wait_for_ready(timeout=30):
            print("Timeout waiting for agent to be ready!")
            return
            
        print("Robot ready! Starting teleop...")
        print("Press X to enable control")
        
        self.running = True
        clock = pygame.time.Clock()
        
        try:
            while self.running:
                pygame.event.pump()
                
                # Process button presses
                for i in range(self.joystick.get_numbuttons()):
                    current_state = self.joystick.get_button(i)
                    prev_state = self.prev_buttons.get(i, False)
                    
                    # Button just pressed
                    if current_state and not prev_state:
                        action = self.button_actions.get(i)
                        if action:
                            action()
                    
                    self.prev_buttons[i] = current_state
                
                # Process movement and position holding
                self.process_movement()
                
                clock.tick(60)  # 60 FPS
                
        except KeyboardInterrupt:
            print("\nExiting teleop...")
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        self.control_enabled = False
        if self.joystick:
            self.joystick.quit()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='PS5 Robot Teleop with sim2real support')
    parser.add_argument('--use_real_robot', action='store_true', 
                        help='Enable real robot mode (default: simulation)')
    args = parser.parse_args()
    
    # Load robot configuration
    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(cur_path, '../../../hardware/monte01/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)
    
    print(f"Robot Mode: {'REAL' if args.use_real_robot else 'SIMULATION'}")
    print(f"Configuration: {config}")
    
    # Create agent with sim/real mode
    agent = Agent(config=config, use_real_robot=args.use_real_robot)
    
    # Create and start teleop
    teleop = PS5RobotTeleop(agent)
    
    try:
        teleop.start_teleop()
    except Exception as e:
        log.error(f"Teleop error: {e}", exc_info=True)
    finally:
        print("Shutting down...")


if __name__ == "__main__":
    main()