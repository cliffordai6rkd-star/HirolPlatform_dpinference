import os
import time
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

class DualMouseTeleop:
    def __init__(self, agent: Agent, enable_rpy_ctrl=False):
        self.agent = agent
        self.scale_factor = 0.05  # Scale factor for mouse input
        self.enable_rpy_ctrl = enable_rpy_ctrl  # Enable roll, pitch, yaw control
        self.running = True
        self.left_is_moving = False   # Flag to track if left arm is currently moving
        self.right_is_moving = False  # Flag to track if right arm is currently moving
        
        # Button state tracking to prevent repeated commands
        self.left_button_states = [False, False]  # Track previous button states for left mouse
        self.right_button_states = [False, False]  # Track previous button states for right mouse
        
        # Gripper position tracking and step size
        self.gripper_step = 0.02  # Step size for gripper movement (0-800 range)
        self.left_gripper_position = 0  # Start at middle position
        self.right_gripper_position = 0  # Start at middle position
        self.gripper_move_frequency = 0.03  # Time interval between gripper moves when button held (30ms)
        self.last_gripper_move_time = {"left": 0, "right": 0}  # Track last move time for each gripper
        
        # Initialize dual 3D mice with device detection
        self.left_mouse = None
        self.right_mouse = None
        
        # Try to detect and initialize multiple space mice
        self._detect_and_initialize_mice()
        
    def _detect_and_initialize_mice(self):
        """Detect and initialize multiple 3D mouse devices"""
        try:
            # Get list of available devices
            devices = pyspacemouse.list_devices()
            log.info(f"Found {len(devices)} 3D mouse device types: {devices}")
            
            if len(devices) == 0:
                log.warning("No 3D mouse devices found")
                return
            
            # Test multiple device numbers to find working pairs
            # Since 6 devices were detected, try different combinations
            device_numbers_to_try = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (2, 3)]
            
            for left_num, right_num in device_numbers_to_try:
                try:
                    log.info(f"Trying DeviceNumber {left_num} for left arm and {right_num} for right arm...")
                    
                    # Try to open left mouse
                    temp_left = pyspacemouse.open(device=devices[0], DeviceNumber=left_num)
                    if not temp_left:
                        log.info(f"DeviceNumber {left_num} failed to open")
                        continue
                    
                    # Try to open right mouse
                    temp_right = pyspacemouse.open(device=devices[0], DeviceNumber=right_num)
                    if not temp_right:
                        log.info(f"DeviceNumber {right_num} failed to open")
                        temp_left.close()
                        continue
                    
                    # Test if they're actually different devices by checking for input
                    log.info(f"Testing if devices {left_num} and {right_num} are different physical devices...")
                    log.info("Please move ONLY the LEFT 3D mouse for 3 seconds...")
                    
                    left_detected = False
                    right_detected = False
                    
                    for _ in range(30):  # 3 seconds test
                        try:
                            state_left = temp_left.read()
                            state_right = temp_right.read()
                            
                            if state_left and (abs(state_left.x) > 0.01 or abs(state_left.y) > 0.01 or abs(state_left.z) > 0.01):
                                left_detected = True
                            if state_right and (abs(state_right.x) > 0.01 or abs(state_right.y) > 0.01 or abs(state_right.z) > 0.01):
                                right_detected = True
                                
                        except Exception as e:
                            pass
                        time.sleep(0.1)
                    
                    # If only left device detected movement, we found a good pair (swap assignment)
                    if left_detected and not right_detected:
                        log.info(f"Found working device pair: Left={right_num}, Right={left_num}")
                        self.left_mouse = temp_right
                        self.right_mouse = temp_left
                        log.info(f"Left mouse (DeviceNumber={right_num}) connected: {self.left_mouse.connected}")
                        log.info(f"Right mouse (DeviceNumber={left_num}) connected: {self.right_mouse.connected}")
                        return
                    else:
                        log.info(f"Devices {left_num} and {right_num} seem to be the same physical device or test failed")
                        temp_left.close()
                        temp_right.close()
                        
                except Exception as e:
                    log.warning(f"Error testing DeviceNumber {left_num},{right_num}: {e}")
            
            # If no good pair found, fall back to simple approach
            log.warning("Could not find distinct device pair, using simple approach...")
            try:
                self.left_mouse = pyspacemouse.open(device=devices[0], DeviceNumber=0)
                self.right_mouse = pyspacemouse.open(device=devices[0], DeviceNumber=1)
                if self.left_mouse:
                    log.info(f"Left mouse (DeviceNumber=0) connected: {self.left_mouse.connected}")
                if self.right_mouse:
                    log.info(f"Right mouse (DeviceNumber=1) connected: {self.right_mouse.connected}")
            except Exception as e:
                log.error(f"Fallback initialization failed: {e}")
                
        except Exception as e:
            log.error(f"Error during device detection: {e}")
        
    def control_loop(self):
        """Main dual teleoperation control loop"""
        self.agent.wait_for_ready()
        
        sim = self.agent.sim
        
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

            arm_right.move_to_start()
            arm_right.hold_position_for_duration(0.2)
            
            log.info("Starting dual 3D mouse teleoperation...")
            if self.left_mouse:
                log.info("Left mouse: Controls left arm")
            if self.right_mouse:
                log.info("Right mouse: Controls right arm")
            if not (self.left_mouse or self.right_mouse):
                log.warning("No 3D mice available, running in simulation-only mode")
                log.info("Press Ctrl+C to exit")
                return
            
            # In real robot mode, don't depend on sim viewer state
            use_real_robot = self.agent.robot is not None
            while self.running and (use_real_robot or sim.viewer.is_running()):
                try:
                    # Control left arm with left mouse
                    if self.left_mouse:
                        self._control_arm_with_mouse(self.left_mouse, arm_left, "left")
                    
                    # Control right arm with right mouse  
                    if self.right_mouse:
                        self._control_arm_with_mouse(self.right_mouse, arm_right, "right")
                    
                    # If only one mouse, let user switch between arms manually
                    if self.left_mouse and not self.right_mouse:
                        # For now, just control left arm with the single mouse
                        pass
                    
                    # Check if movements are complete
                    if self.left_is_moving and arm_left.is_trajectory_done():
                        self.left_is_moving = False
                    
                    if self.right_is_moving and arm_right.is_trajectory_done():
                        self.right_is_moving = False
                    
                    # Hold positions if no movement
                    # if not self.left_is_moving:
                    #     arm_left.hold_joint_positions()
                    # if not self.right_is_moving:
                    #     arm_right.hold_joint_positions()
                        
                    time.sleep(0.01)  # Small delay to prevent excessive updates
                    
                except Exception as e:
                    log.error(f"Error in teleoperation loop: {e}")
                    break
            
        except Exception as e:
            log.error(f"Error in control loop: {e}", exc_info=True)
        finally:
            self._cleanup_mice()

    def _control_arm_with_mouse(self, device, arm: Arm, arm_name):
        """Control a specific arm with a specific mouse device"""
        try:
            # Read from the specific device object
            state = device.read()
            
            if state is None:
                return
            
            # Debug: Log which device is providing input
            if (abs(state.x) > 0.01 or abs(state.y) > 0.01 or abs(state.z) > 0.01):
                log.debug(f"Input from {arm_name} mouse: x={state.x:.3f}, y={state.y:.3f}, z={state.z:.3f}")
            
            # Check for significant movement and arm availability
            is_moving = (self.left_is_moving if arm_name == "left" else self.right_is_moving)
            
            if (abs(state.x) > 0.01 or abs(state.y) > 0.01 or abs(state.z) > 0.01 or
                abs(state.roll) > 0.01 or abs(state.pitch) > 0.01 or abs(state.yaw) > 0.01) and not is_moving:
                
                # Get current TCP pose
                current_pose = arm.get_tcp_pose()
                
                if current_pose is not None:
                    # Create target pose by applying deltas
                    target_pose = current_pose.copy()
                    
                    # Apply translation (scaled)
                    if arm_name == "left":
                        # For left arm, invert x and y axes
                        target_pose[AXIS_X, 3] += -state.x * self.scale_factor
                        target_pose[AXIS_Y, 3] += -state.y * self.scale_factor
                        target_pose[AXIS_Z, 3] += state.z * self.scale_factor
                    else:
                        # For right arm, use normal mapping
                        target_pose[AXIS_X, 3] += state.x * self.scale_factor
                        target_pose[AXIS_Y, 3] += state.y * self.scale_factor
                        target_pose[AXIS_Z, 3] += state.z * self.scale_factor
                    
                    # Apply rotation (scaled) if enabled
                    if self.enable_rpy_ctrl:
                        # Apply roll, pitch, yaw rotations to the rotation matrix
                        import numpy as np
                        from scipy.spatial.transform import Rotation as R
                        
                        # Get current rotation matrix
                        current_rotation = target_pose[:3, :3]
                        
                        # Create rotation deltas (scaled)
                        scale = 0.8
                        roll_delta = state.roll * self.scale_factor * scale  # Reduced scale for rotation
                        pitch_delta = state.pitch * self.scale_factor * scale
                        yaw_delta = state.yaw * self.scale_factor * scale
                        
                        # Create rotation matrix from deltas
                        delta_rotation = R.from_euler('xyz', [roll_delta, pitch_delta, yaw_delta]).as_matrix()
                        
                        # Apply rotation delta to current rotation
                        new_rotation = current_rotation @ delta_rotation
                        target_pose[:3, :3] = new_rotation
                    
                    # Set moving flag and move arm to target pose (non-blocking)
                    if arm_name == "left":
                        self.left_is_moving = True
                    else:
                        self.right_is_moving = True
                    
                    log.info(f"Moving {arm_name} arm based on {arm_name} mouse input")
                    success = arm.move_to_pose(target_pose, blocking=False)
                    if not success:
                        log.warning(f"Failed to move {arm_name} arm to target pose")
                        if arm_name == "left":
                            self.left_is_moving = False
                        else:
                            self.right_is_moving = False
            
            # Handle gripper control with buttons (continuous movement while held)
            if hasattr(state, 'buttons') and len(state.buttons) > 0:
                gripper = arm.get_gripper()
                if gripper:
                    current_position = self.left_gripper_position if arm_name == "left" else self.right_gripper_position
                    current_time = time.time()
                    last_move_time = self.last_gripper_move_time[arm_name]
                    
                    # Check if enough time has passed since last gripper move
                    time_since_last_move = current_time - last_move_time
                    should_move = time_since_last_move >= self.gripper_move_frequency
                    
                    # Button 0 - open gripper continuously while held
                    if len(state.buttons) > 0 and state.buttons[0] and should_move:
                        new_position = min(current_position + self.gripper_step, 1)  # Clamp to max 800
                        if new_position != current_position:  # Only move if position changes
                            gripper.gripper_move(new_position)
                            if arm_name == "left":
                                self.left_gripper_position = new_position
                            else:
                                self.right_gripper_position = new_position
                            self.last_gripper_move_time[arm_name] = current_time
                            log.info(f"{arm_name.capitalize()} gripper opening to position {new_position}")
                    
                    # Button 1 - close gripper continuously while held
                    elif len(state.buttons) > 1 and state.buttons[1] and should_move:
                        new_position = max(current_position - self.gripper_step, 0)  # Clamp to min 0
                        if new_position != current_position:  # Only move if position changes
                            gripper.gripper_move(new_position)
                            if arm_name == "left":
                                self.left_gripper_position = new_position
                            else:
                                self.right_gripper_position = new_position
                            self.last_gripper_move_time[arm_name] = current_time
                            log.info(f"{arm_name.capitalize()} gripper closing to position {new_position}")
                        
        except Exception as e:
            log.warning(f"Error reading from {arm_name} mouse: {e}")

    def _cleanup_mice(self):
        """Clean up mouse connections"""
        try:
            if self.left_mouse:
                self.left_mouse.close()
                self.left_mouse = None
                log.info("Left mouse closed")
        except Exception as e:
            log.warning(f"Error closing left mouse: {e}")
            
        try:
            if self.right_mouse:
                self.right_mouse.close()
                self.right_mouse = None
                log.info("Right mouse closed")
        except Exception as e:
            log.warning(f"Error closing right mouse: {e}")
            
        # Also call module-level close as safety measure
        try:
            pyspacemouse.close()
        except Exception as e:
            log.warning(f"Error calling module-level close: {e}")

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

    parser = argparse.ArgumentParser(description='Dual 3D Mouse teleoperation with option to use real robot.')
    parser.add_argument('--use_real_robot', action='store_true', help='Enable real robot mode.')
    parser.add_argument('--enable-rpy-ctrl', action='store_true', help='Enable roll, pitch, yaw control for both arms.')
    args = parser.parse_args()
    
    agent = Agent(config=config, use_real_robot=args.use_real_robot)
    
    # Create dual teleoperation controller
    teleop = DualMouseTeleop(agent, enable_rpy_ctrl=args.enable_rpy_ctrl)
    
    if not (teleop.left_mouse or teleop.right_mouse):
        print("Failed to initialize any 3D mouse devices. Exiting.")
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

    print("Dual 3D Mouse teleoperation finished.")