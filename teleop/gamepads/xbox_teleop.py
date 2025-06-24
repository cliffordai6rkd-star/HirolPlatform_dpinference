#!/usr/bin/env python3
"""
Xbox Controller Teleoperation for Monte01 Robot

Xbox按键定义:
• X：切换控制对象，包括None/Left Arm/Right Arm/Chassis，默认是None；每按一下会有震动反馈；
• 左摇杆：末端控制时：以末端坐标系控制机械臂X-Y方向移动。上推是y轴正方向，下推是y轴负方向，右推是x轴正方向；左推是x轴负方向；底盘控制时：上下控制线速度，左右控制角速度；
• LT：以末端坐标控制机械臂Z方向移动，Z的负方向；
• RT：以末端坐标控制机械臂Z方向移动。Z的正方向；
• 右摇杆：末端控制时：以末端坐标系控制机械臂X-Y方向旋转。上推是Ry轴正方向，下推是Ry轴负方向，右推是Rx轴正方向；左推是Rx轴负方向；底盘控制时：右推和左推控制顺时针和逆时针自旋；
• LB：以末端坐标控制机械臂Rz方向旋转，Z的负方向；
• RB：以末端坐标控制机械臂Rz方向旋转。Z的正方向；
• A+LB：夹爪开合程度步进减小
• A+RB：夹爪开合程度步进增大
• 方向键盘：上：末端移动速度步进加速；下：末端移动速度步进减速；左：夹爪开合速度步进减速；右：夹爪开合速度步进加速；
• 双击A键：软急停（震动500ms），再次双击恢复（震动200ms）
• 长按Y键：回到机械臂关节零位；
"""

import pygame
import numpy as np
import time
import threading
from enum import Enum
from typing import Optional
import glog as log

from .controller import Controller


class ControlMode(Enum):
    NONE = "None"
    LEFT_ARM = "Left Arm"
    RIGHT_ARM = "Right Arm"
    CHASSIS = "Chassis"

class XboxTeleop:
    def __init__(self, agent=None):
        self.agent = agent
        self.controller = Controller(controller_type='xbox')
        
        # Control state
        self.control_mode = ControlMode.NONE
        self.running = False
        self.emergency_stop = False
        
        # Speed settings
        self.translation_speed = 0.1  # m/s
        self.rotation_speed = 0.5     # rad/s
        self.gripper_speed = 0.5      # gripper speed
        self.speed_step = 0.02        # speed adjustment step
        
        # Note: Gripper settings moved to initialization section above
        
        # Button press tracking
        self.button_press_times = {}
        self.button_states = {}
        self.double_click_threshold = 0.5  # seconds
        self.long_press_threshold = 1.0    # seconds
        
        # Deadzone for analog sticks
        self.deadzone = 0.15
        
        # Control thread
        self.control_thread = None
        self.control_lock = threading.Lock()
        
        # Movement state tracking (like dual mouse teleop)
        self.left_is_moving = False   # Flag to track if left arm is currently moving
        self.right_is_moving = False  # Flag to track if right arm is currently moving
        
        # Current velocities (for chassis control)
        self.current_chassis_cmd = np.zeros(3)  # [linear_x, linear_y, angular_z]
        
        # Gripper control (like dual mouse teleop)
        self.gripper_step = 15  # Step size for gripper movement (0-800 range)
        self.left_gripper_position = 0  # Start at middle position
        self.right_gripper_position = 0  # Start at middle position
        self.gripper_move_frequency = 0.03  # Time interval between gripper moves when button held (30ms)
        self.last_gripper_move_time = {"left": 0, "right": 0}  # Track last move time for each gripper
        
    def start(self):
        """Start the teleoperation controller with infinite retry"""
        retry_count = 0
        log.info("Attempting to connect Xbox controller...")
        
        try:
            while True:
                pygame.init()
                pygame.joystick.init()
                if self.controller.connect():
                    break
                
                retry_count += 1
                log.warning(f"Failed to connect Xbox controller (attempt {retry_count}). Retrying in 2 seconds...")
                log.info("Please ensure Xbox controller is connected and powered on. Press Ctrl+C to cancel.")
                time.sleep(2)
        except KeyboardInterrupt:
            log.info("Controller connection cancelled by user")
            return False
        
        log.info(f"Xbox controller connected successfully after {retry_count} attempts")
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        # Initialize gripper positions and start holding robot positions
        if self.agent:
            try:
                arm_left = self.agent.arm_left()
                arm_right = self.agent.arm_right()
                if arm_left:
                    arm_left.hold_joint_positions()
                if arm_right:
                    arm_right.hold_joint_positions()
            except Exception as e:
                log.warning(f"Failed to initialize position holding: {e}")
        
        log.info("Xbox teleoperation started")
        log.info(f"Current control mode: {self.control_mode.value}")
        return True
    
    def stop(self):
        """Stop the teleoperation controller"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()
        self.controller.stop()
        log.info("Xbox teleoperation stopped")
    
    def _control_loop(self):
        """Main control loop"""
        clock = pygame.time.Clock()
        
        try:
            while self.running:
                pygame.event.pump()
                
                if not self.controller.is_connected():
                    break
                
                # Process inputs
                self._process_buttons()
                self._process_axes()
                
                # Send commands to robot
                self._send_robot_commands()
                
                # Check if movements are complete and hold positions (like dual mouse teleop)
                self._check_movement_completion_and_hold()
                
                clock.tick(50)  # 50 Hz control loop
                
        except Exception as e:
            log.error(f"Error in control loop: {e}")
        finally:
            # Stop all motion on exit
            self._stop_all_motion()
    
    def _process_buttons(self):
        """Process button presses"""
        current_time = time.time()
        
        # Read all button states
        for i in range(self.controller.joystick.get_numbuttons()):
            current_state = self.controller.joystick.get_button(i)
            prev_state = self.button_states.get(i, False)
            
            # Button just pressed
            if current_state and not prev_state:
                self._handle_button_press(i, current_time)
            
            # Button just released
            elif not current_state and prev_state:
                self._handle_button_release(i, current_time)
            
            # Button held (long press)
            elif current_state and prev_state:
                self._handle_button_held(i, current_time)
            
            self.button_states[i] = current_state
    
    def _handle_button_press(self, button_id, current_time):
        """Handle button press events"""
        button_name = self.controller.button_map.get(button_id, f"Button {button_id}")
        
        # X button: Switch control mode
        if button_name == "X":
            self._switch_control_mode()
            self._vibrate_controller(100)  # Short vibration feedback
        
        # A button: Check for double click (emergency stop)
        elif button_name == "A":
            last_press = self.button_press_times.get('A', 0)
            if current_time - last_press < self.double_click_threshold:
                self._toggle_emergency_stop()
            self.button_press_times['A'] = current_time
        
        # Note: Gripper control is now handled in _process_gripper_control() method
        
        self.button_press_times[button_name] = current_time
    
    def _handle_button_release(self, button_id, current_time):
        """Handle button release events"""
        button_name = self.controller.button_map.get(button_id, f"Button {button_id}")
        
        # Y button: Check if it was a long press (return to zero)
        if button_name == "Y":
            press_time = self.button_press_times.get('Y', current_time)
            if current_time - press_time >= self.long_press_threshold:
                self._return_to_zero_position()
    
    def _handle_button_held(self, button_id, current_time):
        """Handle button held events"""
        pass  # Currently no specific held button actions
    
    def _get_current_twist_for_arm(self, arm_name):
        """Get current twist values for specified arm"""
        twist = np.zeros(6)
        
        if self.control_mode in [ControlMode.LEFT_ARM, ControlMode.RIGHT_ARM]:
            # Read analog inputs
            left_stick_x = self._apply_deadzone(self.controller.joystick.get_axis(0))
            left_stick_y = self._apply_deadzone(-self.controller.joystick.get_axis(1))  # Invert Y
            right_stick_x = self._apply_deadzone(self.controller.joystick.get_axis(2))
            right_stick_y = self._apply_deadzone(-self.controller.joystick.get_axis(3))  # Invert Y
            lt_trigger = (self.controller.joystick.get_axis(5) + 1) / 2  # Convert from [-1,1] to [0,1]
            rt_trigger = (self.controller.joystick.get_axis(4) + 1) / 2  # Convert from [-1,1] to [0,1]
            
            # Process shoulder buttons for rotation
            lb_pressed = self.button_states.get(6, False)  # L1 button
            rb_pressed = self.button_states.get(7, False)  # R1 button
            
            # Arm control mode (swapped X/Y axes for Left Stick)
            twist[0] = left_stick_y * self.translation_speed      # X translation (was left_stick_x)
            twist[1] = left_stick_x * self.translation_speed      # Y translation (was left_stick_y)
            twist[2] = (rt_trigger - lt_trigger) * self.translation_speed  # Z translation
            
            twist[3] = right_stick_x * self.rotation_speed        # X rotation
            twist[4] = right_stick_y * self.rotation_speed        # Y rotation
            twist[5] = (1 if rb_pressed else 0) - (1 if lb_pressed else 0)  # Z rotation
            twist[5] *= self.rotation_speed
            
        return twist
    
    def _process_gripper_control(self):
        """Process gripper control like dual mouse teleop"""
        if not self.agent or self.emergency_stop:
            return
            
        try:
            # Check for A + LB/RB gripper control
            a_pressed = self.button_states.get(0, False)  # A button
            lb_pressed = self.button_states.get(6, False)  # L1 button
            rb_pressed = self.button_states.get(7, False)  # R1 button
            
            if a_pressed and (lb_pressed or rb_pressed):
                current_time = time.time()
                arm_name = "left" if self.control_mode == ControlMode.LEFT_ARM else "right"
                
                if self.control_mode in [ControlMode.LEFT_ARM, ControlMode.RIGHT_ARM]:
                    arm = self.agent.arm_left() if self.control_mode == ControlMode.LEFT_ARM else self.agent.arm_right()
                    gripper = arm.get_gripper() if arm else None
                    
                    if gripper:
                        last_move_time = self.last_gripper_move_time[arm_name]
                        time_since_last_move = current_time - last_move_time
                        should_move = time_since_last_move >= self.gripper_move_frequency
                        
                        if should_move:
                            current_position = self.left_gripper_position if arm_name == "left" else self.right_gripper_position
                            
                            if rb_pressed:  # A + RB: open gripper
                                new_position = min(current_position + self.gripper_step, 800)  # Clamp to max 800
                                if new_position != current_position:
                                    gripper.gripper_move(new_position)
                                    if arm_name == "left":
                                        self.left_gripper_position = new_position
                                    else:
                                        self.right_gripper_position = new_position
                                    self.last_gripper_move_time[arm_name] = current_time
                                    log.info(f"{arm_name.capitalize()} gripper opening to position {new_position}")
                            
                            elif lb_pressed:  # A + LB: close gripper
                                new_position = max(current_position - self.gripper_step, 0)  # Clamp to min 0
                                if new_position != current_position:
                                    gripper.gripper_move(new_position)
                                    if arm_name == "left":
                                        self.left_gripper_position = new_position
                                    else:
                                        self.right_gripper_position = new_position
                                    self.last_gripper_move_time[arm_name] = current_time
                                    log.info(f"{arm_name.capitalize()} gripper closing to position {new_position}")
                                    
        except Exception as e:
            log.warning(f"Error in gripper control: {e}")
        
    def _process_axes(self):
        """Process analog stick and trigger inputs"""
        if self.emergency_stop:
            return
        
        # Process D-pad
        dpad = self.controller.joystick.get_hat(0) if self.controller.joystick.get_numhats() > 0 else (0, 0)
        self._process_dpad(dpad)
        
        # Process gripper control like dual mouse teleop
        self._process_gripper_control()
        
        # For chassis mode, update chassis commands
        if self.control_mode == ControlMode.CHASSIS:
            left_stick_x = self._apply_deadzone(self.controller.joystick.get_axis(0))
            left_stick_y = self._apply_deadzone(-self.controller.joystick.get_axis(1))  # Invert Y
            right_stick_x = self._apply_deadzone(self.controller.joystick.get_axis(2))
            
            with self.control_lock:
                # Swapped X/Y axes for Left Stick in chassis mode too
                self.current_chassis_cmd[0] = left_stick_x * self.translation_speed  # Linear X (was left_stick_y)
                self.current_chassis_cmd[1] = left_stick_y * self.translation_speed  # Linear Y (was left_stick_x)
                self.current_chassis_cmd[2] = right_stick_x * self.rotation_speed    # Angular Z
    
    def _process_dpad(self, dpad):
        """Process D-pad inputs for speed adjustments"""
        dpad_x, dpad_y = dpad
        
        # Up: Increase translation speed
        if dpad_y > 0:
            self.translation_speed = min(self.translation_speed + self.speed_step, 0.5)
            log.info(f"Translation speed increased to {self.translation_speed:.2f} m/s")
        
        # Down: Decrease translation speed
        elif dpad_y < 0:
            self.translation_speed = max(self.translation_speed - self.speed_step, 0.01)
            log.info(f"Translation speed decreased to {self.translation_speed:.2f} m/s")
        
        # Right: Increase gripper speed
        if dpad_x > 0:
            self.gripper_speed = min(self.gripper_speed + self.speed_step, 1.0)
            log.info(f"Gripper speed increased to {self.gripper_speed:.2f}")
        
        # Left: Decrease gripper speed
        elif dpad_x < 0:
            self.gripper_speed = max(self.gripper_speed - self.speed_step, 0.1)
            log.info(f"Gripper speed decreased to {self.gripper_speed:.2f}")
    
    def _apply_deadzone(self, value):
        """Apply deadzone to analog inputs"""
        if abs(value) < self.deadzone:
            return 0.0
        # Scale the remaining range to [0, 1]
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)
    
    def _switch_control_mode(self):
        """Switch between control modes"""
        modes = list(ControlMode)
        current_index = modes.index(self.control_mode)
        next_index = (current_index + 1) % len(modes)
        self.control_mode = modes[next_index]
        
        # Stop all motion when switching modes
        self._stop_all_motion()
        
        log.info(f"Control mode switched to: {self.control_mode.value}")
    
    def _toggle_emergency_stop(self):
        """Toggle emergency stop state"""
        self.emergency_stop = not self.emergency_stop
        
        if self.emergency_stop:
            self._stop_all_motion()
            self._vibrate_controller(500)  # Long vibration for e-stop
            log.warning("EMERGENCY STOP ACTIVATED")
        else:
            self._vibrate_controller(200)  # Short vibration for resume
            log.info("Emergency stop deactivated")
    
    
    def _return_to_zero_position(self):
        """Return arm to zero/home position"""
        if self.control_mode in [ControlMode.LEFT_ARM, ControlMode.RIGHT_ARM] and self.agent:
            try:
                if self.control_mode == ControlMode.LEFT_ARM:
                    arm = self.agent.arm_left()
                else:
                    arm = self.agent.arm_right()
                
                log.info(f"Returning {self.control_mode.value} to home position")
                arm.move_to_start()
                
            except Exception as e:
                log.error(f"Failed to return to home position: {e}")
    
    def _vibrate_controller(self, duration_ms):
        """Provide haptic feedback through controller vibration"""
        try:
            if hasattr(self.controller.joystick, 'rumble'):
                # pygame 2.0+ rumble support
                self.controller.joystick.rumble(0.7, 0.7, duration_ms)
        except:
            pass  # Ignore if rumble not supported
    
    def _stop_all_motion(self):
        """Stop all robot motion and hold current position"""
        with self.control_lock:
            self.current_chassis_cmd.fill(0)
        
        # Stop any ongoing arm movements
        self.left_is_moving = False
        self.right_is_moving = False
        
        # Hold current position after stopping motion
        if self.agent:
            try:
                arm_left = self.agent.arm_left()
                arm_right = self.agent.arm_right()
                if arm_left:
                    arm_left.hold_joint_positions()
                if arm_right:
                    arm_right.hold_joint_positions()
            except Exception as e:
                log.warning(f"Failed to hold position after stopping motion: {e}")
    
    def _check_movement_completion_and_hold(self):
        """Check movement completion and hold positions like dual mouse teleop"""
        if not self.agent:
            return
            
        try:
            arm_left = self.agent.arm_left()
            arm_right = self.agent.arm_right()
            
            # Check if movements are complete (like dual mouse teleop)
            if self.left_is_moving and arm_left and arm_left.is_trajectory_done():
                self.left_is_moving = False
            
            if self.right_is_moving and arm_right and arm_right.is_trajectory_done():
                self.right_is_moving = False
            
            # Hold positions if no movement (like dual mouse teleop)
            if not self.left_is_moving and arm_left:
                arm_left.hold_joint_positions()
            if not self.right_is_moving and arm_right:
                arm_right.hold_joint_positions()
                
        except Exception as e:
            log.warning(f"Failed to check movement completion and hold: {e}")
    
    def _send_robot_commands(self):
        """Send commands to the robot based on current control mode"""
        if not self.agent or self.emergency_stop:
            return
        
        try:
            with self.control_lock:
                if self.control_mode == ControlMode.LEFT_ARM:
                    self._send_arm_command(self.agent.arm_left(), "left")
                
                elif self.control_mode == ControlMode.RIGHT_ARM:
                    self._send_arm_command(self.agent.arm_right(), "right")
                
                elif self.control_mode == ControlMode.CHASSIS:
                    self._send_chassis_command(self.current_chassis_cmd)
                    
        except Exception as e:
            log.error(f"Failed to send robot commands: {e}")
    
    def _send_arm_command(self, arm, arm_name):
        """Send pose command to arm like dual mouse teleop"""
        if not arm:
            return
            
        # Check if arm is already moving (like dual mouse teleop)
        is_moving = (self.left_is_moving if arm_name == "left" else self.right_is_moving)
        
        # Get current input values
        twist = self._get_current_twist_for_arm(arm_name)
        
        # Only move if there's significant input and arm is not already moving
        if (abs(twist[0]) > 0.001 or abs(twist[1]) > 0.001 or abs(twist[2]) > 0.001 or
            abs(twist[3]) > 0.001 or abs(twist[4]) > 0.001 or abs(twist[5]) > 0.001) and not is_moving:
            
            try:
                # Get current pose
                current_pose = arm.get_tcp_pose()
                if current_pose is None:
                    return
                
                # Create target pose by applying deltas (like dual mouse teleop)
                target_pose = current_pose.copy()
                
                # Apply translation (scaled)
                scale_factor = 0.1  # Similar to dual mouse teleop
                target_pose[0, 3] += twist[0] * scale_factor
                target_pose[1, 3] += twist[1] * scale_factor
                target_pose[2, 3] += twist[2] * scale_factor
                
                # Apply rotation if significant
                if abs(twist[3]) > 0.001 or abs(twist[4]) > 0.001 or abs(twist[5]) > 0.001:
                    rotation_scale = 0.1
                    # Simple axis-angle rotation application
                    rx, ry, rz = twist[3] * rotation_scale, twist[4] * rotation_scale, twist[5] * rotation_scale
                    
                    # Create rotation matrices
                    if abs(rx) > 1e-6:
                        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                        target_pose[:3, :3] = target_pose[:3, :3] @ Rx
                    if abs(ry) > 1e-6:
                        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                        target_pose[:3, :3] = target_pose[:3, :3] @ Ry
                    if abs(rz) > 1e-6:
                        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                        target_pose[:3, :3] = target_pose[:3, :3] @ Rz
                
                # Set moving flag and move arm to target pose (non-blocking like dual mouse teleop)
                if arm_name == "left":
                    self.left_is_moving = True
                else:
                    self.right_is_moving = True
                
                success = arm.move_to_pose(target_pose, blocking=False)
                if not success:
                    log.warning(f"Failed to move {arm_name} arm to target pose")
                    if arm_name == "left":
                        self.left_is_moving = False
                    else:
                        self.right_is_moving = False
                        
            except Exception as e:
                log.error(f"Failed to send arm command to {arm_name}: {e}")
                if arm_name == "left":
                    self.left_is_moving = False
                else:
                    self.right_is_moving = False
    
    def _send_chassis_command(self, chassis_cmd):
        """Send velocity command to chassis"""
        if not hasattr(self.agent, 'chassis') or np.allclose(chassis_cmd, 0, atol=1e-6):
            return
        
        try:
            # Send chassis velocity command
            # This would depend on the specific chassis interface
            log.debug(f"Chassis command: linear=[{chassis_cmd[0]:.2f}, {chassis_cmd[1]:.2f}], angular={chassis_cmd[2]:.2f}")
            
        except Exception as e:
            log.error(f"Failed to send chassis command: {e}")
    
    @staticmethod
    def _axis_angle_to_rotation_matrix(axis, angle):
        """Convert axis-angle representation to rotation matrix"""
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        x, y, z = axis
        
        rotation_matrix = np.array([
            [cos_angle + x*x*one_minus_cos,     x*y*one_minus_cos - z*sin_angle,  x*z*one_minus_cos + y*sin_angle],
            [y*x*one_minus_cos + z*sin_angle,  cos_angle + y*y*one_minus_cos,     y*z*one_minus_cos - x*sin_angle],
            [z*x*one_minus_cos - y*sin_angle,  z*y*one_minus_cos + x*sin_angle,   cos_angle + z*z*one_minus_cos]
        ])
        
        return rotation_matrix


def main():
    """Test the Xbox teleoperation controller"""
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)
    
    teleop = XboxTeleop()
    
    if not teleop.start():
        print("Failed to start Xbox teleoperation")
        return
    
    try:
        print("Xbox teleoperation started. Press Ctrl+C to exit.")
        print("Current control mode: None")
        print("Press X to switch control modes: None -> Left Arm -> Right Arm -> Chassis -> None")
        
        while teleop.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        teleop.stop()


if __name__ == "__main__":
    main()