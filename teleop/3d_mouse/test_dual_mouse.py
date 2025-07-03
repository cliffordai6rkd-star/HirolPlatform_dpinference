import os
import time
import threading
from threading import Thread
from collections import deque
from hardware.monte01.agent import Agent
from hardware.monte01.arm import Arm
from tools import file_utils
import glog as log
import argparse
import cv2, rclpy
import pyspacemouse
import numpy as np
from scipy.spatial.transform import Rotation
log.setLevel("INFO")

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

class DualMouseTeleop:
    def __init__(self, agent: Agent, enable_rpy_ctrl=False):
        self.agent = agent
        self.scale_factor = 0.01  # Scale factor for mouse input
        self.enable_rpy_ctrl = enable_rpy_ctrl  # Enable roll, pitch, yaw control
        self.running = False
        
        # Arm control state
        self.left_arm_active = False
        self.right_arm_active = False
        self.emergency_stop = False
        
        # Origin poses when teleop starts
        self.left_arm_origin = None
        self.right_arm_origin = None
        
        # Target poses for robot arms - updated by mouse thread
        self.left_arm_target = None
        self.right_arm_target = None
        
        # Command poses for robot arms - sent to robot at higher frequency
        self.left_arm_command = None
        self.right_arm_command = None
        
        # Button state tracking
        self.left_button_states = [False, False]  # Track previous button states for left mouse
        self.right_button_states = [False, False]  # Track previous button states for right mouse
        
        # Gripper states
        self.left_gripper_pos = 0.05
        self.right_gripper_pos = 0.05
        self.gripper_min = 0.0
        self.gripper_max = 1.0
        self.gripper_step = 0.02
        self.gripper_move_frequency = 0.03
        self.last_gripper_move_time = {"left": 0, "right": 0}
        
        # Robot end pose smoothing
        self.position_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }
        self.rotation_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }
        
        # Robot end velocity tracking for prediction
        self.position_velocity = {
            'left': np.zeros(3),
            'right': np.zeros(3)
        }
        self.last_position = {
            'left': np.zeros(3),
            'right': np.zeros(3)
        }
        self.last_timestamp = {
            'left': time.perf_counter(),
            'right': time.perf_counter()
        }
        
        # Previous arm commands for safety checks
        self._previous_left_arm_command = None
        self._previous_right_arm_command = None
        self.max_position_change = 0.01  # 1cm maximum change per control cycle
        self.max_rotation_change = 0.1  # ~5.7 degrees maximum rotation per cycle
        self.max_safe_position_change = 0.05  # 5cm maximum change for safety
        
        # Threading
        self.mouse_thread = None
        self.robot_command_thread = None
        
        # Initialize dual 3D mice with device detection
        self.left_mouse = None
        self.right_mouse = None
        
        # Try to detect and initialize multiple space mice
        self._detect_and_initialize_mice()
        
    def _reset_values_by_side(self, arm_name):
        """Resets values for a specific arm side."""
        current_time = time.perf_counter()
        if arm_name == "left":
            # Clear previous position history
            self.position_history['left'].clear()
            self.rotation_history['left'].clear()
            # Initialize position and rotation with current robot arm pose
            arm_left = self.agent.arm_left()
            pose = arm_left.get_tcp_pose()
            self.position_history['left'].append(pose[:3, 3])
            self.rotation_history['left'].append(pose[:3, :3])
            self.last_position['left'] = pose[:3, 3]
            self.last_timestamp['left'] = current_time
            gripper_left = arm_left.get_gripper()
            self.left_gripper_pos = gripper_left.get_position() if gripper_left else 0.05
            self._previous_left_arm_command = pose.copy()
            self.left_arm_command = self._previous_left_arm_command
            # Reset velocity tracking
            self.position_velocity['left'] = np.zeros(3)
        elif arm_name == "right":
            # Clear previous position history
            self.position_history['right'].clear()
            self.rotation_history['right'].clear()
            # Initialize position and rotation with current robot arm pose
            arm_right = self.agent.arm_right()
            pose = arm_right.get_tcp_pose()
            self.position_history['right'].append(pose[:3, 3])
            self.rotation_history['right'].append(pose[:3, :3])
            self.last_position['right'] = pose[:3, 3]
            self.last_timestamp['right'] = current_time
            gripper_right = arm_right.get_gripper()
            self.right_gripper_pos = gripper_right.get_position() if gripper_right else 0.05
            self._previous_right_arm_command = pose.copy()
            self.right_arm_command = self._previous_right_arm_command
            # Reset velocity tracking
            self.position_velocity['right'] = np.zeros(3)

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
        
    def start(self):
        """Start the teleoperation system"""
        if self.running:
            log.warning("Teleoperation already running")
            return False
            
        try:
            log.info("Starting dual 3D mouse teleoperation system...")
            # Add timeout to avoid infinite waiting
            if not self.agent.wait_for_ready(timeout=30):
                log.error("Agent initialization timed out after 30 seconds")
                return False
            
            # Initialize robot arms
            arm_left = self.agent.arm_left()
            arm_right = self.agent.arm_right()
            
            # Move to start position
            arm_left.move_to_start()
            arm_left.hold_position_for_duration(0.2)
            arm_right.move_to_start() 
            arm_right.hold_position_for_duration(0.2)
            
            # Initialize arm commands with current poses
            self.left_arm_command = arm_left.get_tcp_pose()
            self.right_arm_command = arm_right.get_tcp_pose()
            self._previous_left_arm_command = self.left_arm_command.copy()
            self._previous_right_arm_command = self.right_arm_command.copy()
            
            if not (self.left_mouse or self.right_mouse):
                log.error("No 3D mice available")
                return False
            
            self.running = True
            
            # Start processing threads
            self.mouse_thread = threading.Thread(target=self._mouse_processing_loop)
            self.mouse_thread.daemon = True
            self.mouse_thread.start()
            
            self.robot_command_thread = threading.Thread(target=self._robot_command_loop) 
            self.robot_command_thread.daemon = True
            self.robot_command_thread.start()
            
            log.info("Dual 3D mouse teleoperation started successfully")
            if self.left_mouse:
                log.info("Left mouse: Controls left arm (hold button to activate)")
            if self.right_mouse:
                log.info("Right mouse: Controls right arm (hold button to activate)")
            return True
            
        except Exception as e:
            log.error(f"Failed to start teleoperation: {e}")
            return False

    def stop(self):
        """Stop the teleoperation system"""
        if not self.running:
            return True
            
        try:
            log.info("Stopping dual 3D mouse teleoperation...")
            self.running = False
            
            if self.mouse_thread:
                self.mouse_thread.join(timeout=2.0)
            if self.robot_command_thread:
                self.robot_command_thread.join(timeout=2.0)
                
            self._cleanup_mice()
            log.info("Dual 3D mouse teleoperation stopped successfully")
            return True
            
        except Exception as e:
            log.error(f"Error stopping teleoperation: {e}")
            return False

    def _mouse_processing_loop(self):
        """Thread for processing 3D mouse data at 60Hz"""
        log.info("Mouse processing thread started")
        control_hz = 60
        last_process_time = time.perf_counter()
        
        while self.running:
            try:
                current_time = time.perf_counter()
                elapsed = current_time - last_process_time
                if elapsed < 1 / control_hz:
                    time.sleep(1 / control_hz - elapsed)
                last_process_time = time.perf_counter()
                
                # Process button presses for control functions
                self._process_button_presses()
                # Process mouse movements for robot arm control
                self._process_mouse_movement()
                
            except Exception as e:
                log.error(f"Error in mouse processing loop: {e}")
                time.sleep(0.05)

    def _robot_command_loop(self):
        """Thread for sending smoothed commands to robot at 100Hz"""
        log.info("Robot command thread started")
        control_hz = 100
        last_command_time = time.perf_counter()
        
        while self.running:
            try:
                # Maintain 100Hz command rate
                current_time = time.perf_counter()
                elapsed = current_time - last_command_time
                if elapsed < 1 / control_hz:
                    time.sleep(1 / control_hz - elapsed)
                    
                last_command_time = time.perf_counter()
                
                # Skip if emergency stop is active
                if self.emergency_stop:
                    continue
                    
                # Generate smoothed commands for robot arms
                self._generate_smooth_commands()
                
                # Send commands to robot
                if self.left_arm_active:
                    self._send_arm_command('left', self.left_arm_command)
                    if self.left_arm_command is not None:
                        self._previous_left_arm_command = self.left_arm_command.copy()
                        
                if self.right_arm_active:
                    self._send_arm_command('right', self.right_arm_command)
                    if self.right_arm_command is not None:
                        self._previous_right_arm_command = self.right_arm_command.copy()
                        
            except Exception as e:
                log.error(f"Error in robot command loop: {e}")
                time.sleep(0.01)

    def _process_button_presses(self):
        """Process button presses for gripper control and arm activation"""
        try:
            # Process left mouse buttons
            if self.left_mouse:
                left_state = self.left_mouse.read()
                if left_state and hasattr(left_state, 'buttons') and len(left_state.buttons) > 0:
                    self._handle_gripper_control(left_state.buttons, "left")
                    
            # Process right mouse buttons  
            if self.right_mouse:
                right_state = self.right_mouse.read()
                if right_state and hasattr(right_state, 'buttons') and len(right_state.buttons) > 0:
                    self._handle_gripper_control(right_state.buttons, "right")
                    
        except Exception as e:
            log.warning(f"Error processing button presses: {e}")

    def _handle_gripper_control(self, buttons, arm_name):
        """Handle gripper control with buttons (continuous movement while held)"""
        try:
            if arm_name == "left":
                arm = self.agent.arm_left()
                current_position = self.left_gripper_pos
            else:
                arm = self.agent.arm_right()
                current_position = self.right_gripper_pos
                
            gripper = arm.get_gripper()
            if not gripper:
                return
                
            current_time = time.time()
            last_move_time = self.last_gripper_move_time[arm_name]
            
            # Check if enough time has passed since last gripper move
            time_since_last_move = current_time - last_move_time
            should_move = time_since_last_move >= self.gripper_move_frequency
            
            # Button 0 - open gripper continuously while held
            if len(buttons) > 0 and buttons[0] and should_move:
                new_position = min(current_position + self.gripper_step, self.gripper_max)
                if new_position != current_position:
                    gripper.gripper_move(new_position)
                    if arm_name == "left":
                        self.left_gripper_pos = new_position
                    else:
                        self.right_gripper_pos = new_position
                    self.last_gripper_move_time[arm_name] = current_time
                    log.debug(f"{arm_name.capitalize()} gripper opening to position {new_position}")
            
            # Button 1 - close gripper continuously while held
            elif len(buttons) > 1 and buttons[1] and should_move:
                new_position = max(current_position - self.gripper_step, self.gripper_min)
                if new_position != current_position:
                    gripper.gripper_move(new_position)
                    if arm_name == "left":
                        self.left_gripper_pos = new_position
                    else:
                        self.right_gripper_pos = new_position
                    self.last_gripper_move_time[arm_name] = current_time
                    log.debug(f"{arm_name.capitalize()} gripper closing to position {new_position}")
                    
        except Exception as e:
            log.warning(f"Error controlling {arm_name} gripper: {e}")

    def _process_mouse_movement(self):
        """Process mouse movements to update robot arm targets"""
        if self.emergency_stop:
            return
            
        current_time = time.perf_counter()
        
        # Process left mouse movement
        if self.left_mouse:
            self._process_single_mouse_movement(self.left_mouse, "left", current_time)
            
        # Process right mouse movement
        if self.right_mouse:
            self._process_single_mouse_movement(self.right_mouse, "right", current_time)

    def _process_single_mouse_movement(self, device, arm_name, current_time):
        """Process movement from a single mouse device"""
        try:
            state = device.read()
            if state is None:
                return
                
            # Check for significant movement to activate arm control
            has_movement = (abs(state.x) > 0.01 or abs(state.y) > 0.01 or abs(state.z) > 0.01 or
                          abs(state.roll) > 0.01 or abs(state.pitch) > 0.01 or abs(state.yaw) > 0.01)
            
            if has_movement:
                # Auto-activate arm control when movement detected
                if not (self.left_arm_active if arm_name == "left" else self.right_arm_active):
                    if arm_name == "left":
                        self.left_arm_active = True
                        arm_left = self.agent.arm_left()
                        self.left_arm_origin = arm_left.get_tcp_pose()
                        self._reset_values_by_side("left")
                        log.info("Left arm teleoperation auto-activated")
                    else:
                        self.right_arm_active = True
                        arm_right = self.agent.arm_right()
                        self.right_arm_origin = arm_right.get_tcp_pose()
                        self._reset_values_by_side("right")
                        log.info("Right arm teleoperation auto-activated")
                
                # Process movement only if arm is active
                is_active = self.left_arm_active if arm_name == "left" else self.right_arm_active
                origin = self.left_arm_origin if arm_name == "left" else self.right_arm_origin
                
                if is_active and origin is not None:
                    # Get current TCP pose (in world coordinates)
                    if arm_name == "left":
                        current_pose = self.agent.arm_left().get_tcp_pose()
                    else:
                        current_pose = self.agent.arm_right().get_tcp_pose()
                        
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
                            from scipy.spatial.transform import Rotation as R
                            
                            # Get current rotation matrix
                            current_rotation = target_pose[:3, :3]
                            
                            # Create rotation deltas (scaled)
                            scale = 0.8
                            yaw_scale = 3.0
                            roll_delta = state.roll * self.scale_factor * scale
                            pitch_delta = state.pitch * self.scale_factor * scale
                            yaw_delta = state.yaw * self.scale_factor * scale * yaw_scale
                            
                            # Create rotation matrix from deltas
                            delta_rotation = R.from_euler('xyz', [roll_delta, pitch_delta, yaw_delta]).as_matrix()
                            
                            # Apply rotation delta to current rotation
                            new_rotation = current_rotation @ delta_rotation
                            target_pose[:3, :3] = new_rotation
                        
                        # Add to history for smoothing
                        pos = target_pose[:3, 3]
                        rot = target_pose[:3, :3]
                        
                        # Calculate velocity for prediction
                        if len(self.position_history[arm_name]) > 0:
                            dt = current_time - self.last_timestamp[arm_name]
                            if dt > 0:
                                vel = (pos - self.last_position[arm_name]) / dt
                                alpha = 0.3  # Smoothing factor
                                self.position_velocity[arm_name] = alpha * vel + (1 - alpha) * self.position_velocity[arm_name]
                        
                        self.last_position[arm_name] = pos
                        self.last_timestamp[arm_name] = current_time
                        
                        # Add to smoothing history
                        self.position_history[arm_name].append(pos)
                        self.rotation_history[arm_name].append(rot)
                        
                        # Update target pose
                        if arm_name == "left":
                            self.left_arm_target = target_pose
                        else:
                            self.right_arm_target = target_pose
                            
        except Exception as e:
            log.warning(f"Error processing {arm_name} mouse movement: {e}")

    def _generate_smooth_commands(self):
        """Generate smoothed command poses for robot arms"""
        # Left arm smoothing
        if self.left_arm_active and len(self.position_history['left']) > 0 and self._previous_left_arm_command is not None:
            self._generate_smooth_command_for_arm('left')
            
        # Right arm smoothing  
        if self.right_arm_active and len(self.position_history['right']) > 0 and self._previous_right_arm_command is not None:
            self._generate_smooth_command_for_arm('right')

    def _generate_smooth_command_for_arm(self, arm_name):
        """Generate smoothed command for a specific arm"""
        try:
            # Apply position smoothing with weighted average
            positions = list(self.position_history[arm_name])
            weights = [i + 1 for i in range(len(positions))]
            sum_weights = sum(weights)
            
            # Calculate weighted average position
            avg_pos = np.zeros(3)
            for i, pos in enumerate(positions):
                avg_pos += pos * (weights[i] / sum_weights)
                
            # Apply velocity prediction
            prediction_time = 0.02  # Predict 20ms ahead
            predicted_pos = avg_pos + self.position_velocity[arm_name] * prediction_time
            
            # Apply jitter removal - ignore very small movements
            jitter_threshold = 0.0005  # 0.5mm threshold
            prev_command = self._previous_left_arm_command if arm_name == 'left' else self._previous_right_arm_command
            if np.linalg.norm(predicted_pos - prev_command[:3, 3]) < jitter_threshold:
                predicted_pos = prev_command[:3, 3]
                
            # Smooth rotation using weighted average
            rotations = list(self.rotation_history[arm_name])
            quats = [Rotation.from_matrix(r).as_quat() for r in rotations]
            
            if len(quats) >= 2:
                # Weight toward most recent rotation
                weights = [i + 1 for i in range(len(quats))]
                sum_weights = sum(weights)
                
                # Calculate weighted average quaternion
                avg_quat = np.zeros(4)
                for i, quat in enumerate(quats):
                    avg_quat += quat * (weights[i] / sum_weights)
                avg_quat = avg_quat / np.linalg.norm(avg_quat)
                
                smoothed_rot = Rotation.from_quat(avg_quat).as_matrix()
            else:
                smoothed_rot = rotations[-1]
                
            # Combine smoothed position and rotation into a pose
            smoothed_pose = np.eye(4)
            smoothed_pose[:3, :3] = smoothed_rot
            smoothed_pose[:3, 3] = predicted_pos
            
            # Limit the maximum change in position to avoid large jumps
            pos_diff = smoothed_pose[:3, 3] - prev_command[:3, 3]
            pose_diff_norm = np.linalg.norm(pos_diff)
            
            if pose_diff_norm > self.max_safe_position_change:
                log.error(f"Position change too large: {pose_diff_norm:.4f}, disabling {arm_name} arm for safety")
                if arm_name == 'left':
                    self.left_arm_active = False
                else:
                    self.right_arm_active = False
                return
                
            if pose_diff_norm > self.max_position_change:
                # Scale down the position change
                scale_factor = self.max_position_change / pose_diff_norm
                smoothed_pose[:3, 3] = prev_command[:3, 3] + pos_diff * scale_factor
                log.warning(f"Position change too large: {pose_diff_norm:.4f}, scaling down to {self.max_position_change:.3f}")
                
            # Limit the maximum change in rotation
            prev_rot = Rotation.from_matrix(prev_command[:3, :3])
            current_rot = Rotation.from_matrix(smoothed_pose[:3, :3])
            rot_diff = prev_rot.inv() * current_rot
            angle = np.linalg.norm(rot_diff.as_rotvec())
            
            if angle > self.max_rotation_change:
                # Scale down the rotation change
                scale_factor = self.max_rotation_change / angle
                limited_rot = prev_rot * Rotation.from_rotvec(rot_diff.as_rotvec() * scale_factor)
                smoothed_pose[:3, :3] = limited_rot.as_matrix()
                log.warning(f"Rotation change too large: {angle:.4f}, scaling down to {self.max_rotation_change:.4f}")
                
            # Update command pose
            if arm_name == 'left':
                self.left_arm_command = smoothed_pose
            else:
                self.right_arm_command = smoothed_pose
                
        except Exception as e:
            log.error(f"Error generating smooth command for {arm_name} arm: {e}")

    def _send_arm_command(self, arm_name, command):
        """Send arm command to the robot"""
        try:
            if command is None:
                return
                
            # Convert target pose from world frame to chest_link frame for move_to_pose
            try:
                # Get current transformation from world to chest_link
                if arm_name == "left":
                    arm = self.agent.arm_left()
                else:
                    arm = self.agent.arm_right()
                    
                if arm.trunk is not None:
                    world_to_chest = arm.trunk.get_world_to_chest_transform()
                else:
                    world_to_chest = np.eye(4)
                    log.warning(f"No trunk component available for {arm_name} arm, using identity transform")
                
                # Transform target from world frame to chest_link frame
                # NOTE: world_to_chest is actually chest_to_world based on arm.py:340
                chest_to_world = world_to_chest  # This is actually chest_to_world
                world_to_chest_actual = np.linalg.inv(chest_to_world)
                target_in_chest_frame = world_to_chest_actual @ command
                
                log.debug(f"{arm_name} arm - Target in world frame: {command[:3, 3]}")
                log.debug(f"{arm_name} arm - Target in chest frame: {target_in_chest_frame[:3, 3]}")
                
            except Exception as e:
                log.error(f"Failed to transform target pose for {arm_name} arm: {e}")
                log.warning(f"Using target pose as-is for {arm_name} arm (assuming it's already in chest frame)")
                target_in_chest_frame = command
            
            # Send command to arm
            success = arm.move_to_pose(target_in_chest_frame, blocking=False)
            if not success:
                log.warning(f"Failed to send command to {arm_name} arm")
                
        except Exception as e:
            log.error(f"Error sending command to {arm_name} arm: {e}")


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
    
    # Start the teleoperation system
    if not teleop.start():
        print("Failed to start teleoperation system. Exiting.")
        exit(1)

    # Start the ROS2 node in another thread
    ros2_thread = Thread(target=ros2_spin, args=(agent,))
    ros2_thread.start()
    
    try:
        print("3D Mouse teleoperation running. Move mice to control arms, use buttons for grippers.")
        print("Left mouse controls left arm, right mouse controls right arm.")
        print("Press Ctrl+C to exit...")
        ros2_thread.join()
    except KeyboardInterrupt:
        print("收到中斷信號，正在關閉...")
        teleop.stop()
    
    # Wait for threads to finish
    agent.sim_thread.join()

    print("Dual 3D Mouse teleoperation finished.")