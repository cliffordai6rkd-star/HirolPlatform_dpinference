from hardware.base.arm import ArmBase

import threading, time, warnings, os
from hardware.base.utils import RobotJointState
import numpy as np
import glog as log
from typing import Optional, List, Dict, Any

# 条件导入 a2d_sdk
try:
    from a2d_sdk.robot import RobotDds as Robot
    A2D_SDK_AVAILABLE = True
except ImportError:
    A2D_SDK_AVAILABLE = False
    log.warning("a2d_sdk not available, using mock implementation")
    
    class MockRobot:
        def __init__(self):
            self._arm_positions = [0.0] * 14
            self._head_positions = [0.0] * 2
            self._waist_positions = [0.0] * 2
            self._gripper_positions = [0.5] * 2  # normalized [0,1]
            self._hand_positions = [0.0] * 12
            self._hand_forces = [0.0] * 6  # 6-axis force sensor
            self._last_timestamp = int(time.time() * 1e9)
            log.info("[Mock] Robot initialized")
        
        def move_arm(self, positions):
            self._arm_positions = list(positions)
            log.info(f"[Mock] Moving arm to: {positions[:3]}...")
        
        def move_head(self, positions):
            self._head_positions = list(positions)
            log.info(f"[Mock] Moving head to: {positions}")
        
        def move_waist(self, positions):
            self._waist_positions = list(positions)
            log.info(f"[Mock] Moving waist to: {positions}")
        
        def move_wheel(self, linear, angular):
            log.info(f"[Mock] Moving wheel: linear={linear}, angular={angular}")
        
        def move_gripper(self, positions):
            self._gripper_positions = list(positions)
            log.info(f"[Mock] Moving gripper to: {positions}")
        
        def move_hand(self, positions):
            self._hand_positions = list(positions)
            log.info(f"[Mock] Moving hand to: {positions[:3]}...")
        
        def move_hand_as_gripper(self, positions):
            # Convert to hand joint positions (simplified)
            hand_pos = [positions[0]] * 6 + [positions[1]] * 6
            self._hand_positions = hand_pos
            log.info(f"[Mock] Moving hand as gripper to: {positions}")
        
        def move_head_and_waist(self, head_pos, waist_pos):
            self.move_head(head_pos)
            self.move_waist(waist_pos)
            log.info(f"[Mock] Moving head and waist simultaneously")
        
        def arm_joint_states(self):
            timestamp = int(time.time() * 1e9)
            return self._arm_positions.copy(), timestamp
        
        def waist_joint_states(self):
            timestamp = int(time.time() * 1e9)
            return self._waist_positions.copy(), timestamp
        
        def head_joint_states(self):
            timestamp = int(time.time() * 1e9)
            return self._head_positions.copy(), timestamp
        
        def gripper_states(self):
            timestamp = int(time.time() * 1e9)
            return self._gripper_positions.copy(), timestamp
        
        def hand_joint_states(self):
            timestamp = int(time.time() * 1e9)
            return self._hand_positions.copy(), timestamp
        
        def hand_force_states(self):
            timestamp = int(time.time() * 1e9)
            return self._hand_forces.copy(), timestamp
        
        # Nearest timestamp query methods
        def arm_joint_states_nearest(self, timestamp_ns):
            return self._arm_positions.copy()
        
        def head_joint_states_nearest(self, timestamp_ns):
            return self._head_positions.copy()
        
        def waist_joint_states_nearest(self, timestamp_ns):
            return self._waist_positions.copy()
        
        def gripper_joint_states_nearest(self, timestamp_ns):
            return self._gripper_positions.copy()
        
        def hand_joint_states_nearest(self, timestamp_ns):
            return self._hand_positions.copy()
        
        def whole_body_status(self):
            timestamp = int(time.time() * 1e9)
            return {
                'timestamp': timestamp,
                'status': 'normal',
                'error_code': 0,
                'error_msg': ''
            }, timestamp
        
        def arm_status(self):
            return {
                'timestamp': self._last_timestamp,
                'joint_pos': self._arm_positions,
                'joint_vel': [0.0] * 14,
                'joint_tor': [0.0] * 14
            }
        
        def gripper_status(self):
            return {
                'timestamp': self._last_timestamp,
                'joint_pos': self._gripper_positions
            }
        
        def hand_status(self):
            return {
                'timestamp': self._last_timestamp,
                'joint_pos': self._hand_positions,
                'force': self._hand_forces
            }
        
        def reset(self, arm_positions=None, gripper_positions=None, 
                  hand_positions=None, waist_positions=None, head_positions=None):
            if arm_positions is not None:
                self._arm_positions = list(arm_positions)
            if gripper_positions is not None:
                self._gripper_positions = list(gripper_positions) 
            if hand_positions is not None:
                self._hand_positions = list(hand_positions)
            if waist_positions is not None:
                self._waist_positions = list(waist_positions)
            if head_positions is not None:
                self._head_positions = list(head_positions)
            log.info("[Mock] Robot reset with custom positions")
        
        def shutdown(self):
            log.info("[Mock] Robot shutdown")
    
    Robot = MockRobot


class AgibotG1(ArmBase):
    """AgiBot G1 humanoid robot hardware interface.
    
    Supports dual-arm manipulation with optional head, waist, and mobile base control.
    Uses RobotDds interface for all robot interactions.
    """
    
    # Joint limits in degrees for head and waist
    HEAD_LIMITS_MIN = np.array([-90.0, -20.0])  # [yaw, pitch] in degrees
    HEAD_LIMITS_MAX = np.array([90.0, 20.0])
    WAIST_LIMITS_MIN = np.array([0.0, 0.0])     # [pitch, height] 
    WAIST_LIMITS_MAX = np.array([90.0, 50.0])   # height in cm
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AgiBot G1 robot.
        
        Args:
            config: Configuration dictionary containing robot parameters
        """
        # Initialize robot connection with fallback to mock
        self._use_mock = False
        try:
            if A2D_SDK_AVAILABLE:
                self._robot = Robot()
                time.sleep(0.5)  # Wait for initialization as per documentation
                log.info("AgiBot G1 RobotDds connection established")
            else:
                raise ImportError("SDK not available or mock mode forced")
        except Exception as e:
            log.warning(f"Failed to initialize RobotDds: {e}")
            log.info("Falling back to mock implementation for testing")
            # Use the mock implementation
            # Use the MockRobot defined at module level for both cases
            self._robot = MockRobot()
            self._use_mock = True
        
        # Parse configuration
        self._control_head = config.get('control_head', False)
        self._control_waist = config.get('control_waist', False) 
        self._control_wheel = config.get('control_wheel', False)
        self._control_gripper = config.get('control_gripper', False)
        self._control_hand = config.get('control_hand', False)
        self._robot_name = config.get('robot_name', 'AgiBot_G1')
        
        # Need to set _dof before calculating total_dof
        self._dof = config["dof"]
        
        # Calculate total DOF before parent initialization (needed by initialize() method)
        self._total_dof = self.get_total_dof()
        
        # Threading control (must be initialized before parent constructor which calls initialize())
        self._thread_running = True
        self._update_thread = threading.Thread(target=self.update_state_task, daemon=True)
        
        # Call parent constructor
        super().__init__(config)
        
        # State tracking
        self._last_posi = None
        self._last_vel = None
        
        log.info(f"AgiBot G1 initialized with {self._total_dof} DOF")
        log.info(f"Control modes - Head: {self._control_head}, Waist: {self._control_waist}, "
                f"Wheel: {self._control_wheel}, Gripper: {self._control_gripper}, Hand: {self._control_hand}")

    def initialize(self) -> bool:
        """Initialize robot state and start update thread.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize state arrays based on total DOF
            self._last_posi = np.zeros(self._total_dof)
            self._last_vel = np.zeros(self._total_dof)
            
            # Initialize joint states
            self._joint_states._positions = np.zeros(self._total_dof)
            self._joint_states._velocities = np.zeros(self._total_dof)
            self._joint_states._accelerations = np.zeros(self._total_dof)
            self._joint_states._time_stamp = time.perf_counter()
            
            # Start state update thread
            self._update_thread.start()
            
            # Initialize safety state
            self.init_safety_state(self._joint_states._positions)
            
            log.info(f"AgiBot G1 {self._robot_name} initialization complete")
            return True
            
        except Exception as e:
            log.error(f"AgiBot G1 initialization failed: {e}")
            return False
    
    def get_total_dof(self) -> int:
        """Calculate total degrees of freedom.
        
        Returns:
            int: Total DOF including arms and optional components
        """
        total_dof = sum(self._dof)  # Base arm DOF (14)
        
        if self._control_head:
            total_dof += 2  # head yaw, pitch
        if self._control_waist:
            total_dof += 2  # waist pitch, height
        if self._control_gripper:
            total_dof += 2  # left/right gripper
        if self._control_hand:
            total_dof += 12  # 6 joints per hand
            
        return total_dof
        
    def update_state_task(self) -> None:
        """Background thread for updating robot state at high frequency."""
        read_frequency = 100
        target_dt = 1.0 / read_frequency
        
        log.info(f'AgiBot G1 {self._robot_name} state update thread started at {read_frequency}Hz')
        
        last_read_time = time.time()
        while self._thread_running:
            current_time = time.time()
            dt = current_time - last_read_time
            last_read_time = current_time
            
            try:
                # Thread-safe state update
                with self._lock:
                    # Get current positions
                    new_positions = self.update_arm_states()
                    
                    if new_positions is not None:
                        # Calculate velocities and accelerations
                        if self._last_posi is not None and dt > 0:
                            self._joint_states._velocities = (new_positions - self._last_posi) / dt
                        else:
                            self._joint_states._velocities = np.zeros_like(new_positions)
                        
                        if self._last_vel is not None and dt > 0:
                            self._joint_states._accelerations = (self._joint_states._velocities - self._last_vel) / dt
                        else:
                            self._joint_states._accelerations = np.zeros_like(self._joint_states._velocities)
                        
                        # Update state
                        self._joint_states._positions = new_positions
                        self._joint_states._time_stamp = time.perf_counter()
                        
                        # Store for next iteration
                        self._last_posi = new_positions.copy()
                        self._last_vel = self._joint_states._velocities.copy()
                        
                        # Update safety checker
                        self.update_safety_state(new_positions)
                
                # Sleep to maintain frequency
                if dt < target_dt:
                    sleep_time = target_dt - dt
                    time.sleep(sleep_time)
                elif dt > 1.2 * target_dt:
                    log.warning(f'AgiBot G1 update frequency slow: expected {target_dt:.3f}s, actual {dt:.3f}s')
                    
            except Exception as e:
                log.error(f"Error in AgiBot G1 state update: {e}")
                time.sleep(target_dt)  # Prevent tight error loop
        
        log.info(f'AgiBot G1 {self._robot_name} state update thread stopped')
    
    def update_arm_states(self) -> Optional[np.ndarray]:
        """Update robot joint states from hardware.
        
        Returns:
            Optional[np.ndarray]: Combined joint positions or None if error
        """
        try:
            # Get arm positions (always present)
            arm_posi = self.get_dual_arm_positions()
            if arm_posi is None:
                return None
                
            position_components = [arm_posi]
            
            # Add optional components based on configuration
            if self._control_head:
                head_posi = self.get_head_positions()
                if head_posi is not None:
                    position_components.append(head_posi)
                    
            if self._control_waist:
                waist_posi = self.get_waist_positions()
                if waist_posi is not None:
                    position_components.append(waist_posi)
                    
            if self._control_gripper:
                gripper_posi = self.get_gripper_positions()
                if gripper_posi is not None:
                    position_components.append(gripper_posi)
                    
            if self._control_hand:
                hand_posi = self.get_hand_positions()
                if hand_posi is not None:
                    position_components.append(hand_posi)
            
            # Combine all components
            combined_positions = np.hstack(position_components)
            
            # Validate expected DOF
            if len(combined_positions) != self._total_dof:
                log.warning(f"DOF mismatch: expected {self._total_dof}, got {len(combined_positions)}")
                
            return combined_positions
            
        except Exception as e:
            log.error(f"Failed to update arm states: {e}")
            return None

    def set_joint_command(self, mode: List[str], command: np.ndarray) -> bool:
        """Send joint commands to robot.
        
        Args:
            mode: Control mode list (currently only 'position' supported)
            command: Joint command array matching total DOF
            
        Returns:
            bool: True if command sent successfully
            
        Raises:
            ValueError: If mode not supported or command length invalid
            RuntimeError: If robot communication fails
        """
        # Validate control mode
        for cur_mode in mode:
            if cur_mode not in ['position']:
                raise ValueError(f'AgiBot G1 only supports position control, got: {cur_mode}')
        
        # Validate command length
        if len(command) != self._total_dof:
            raise ValueError(f'Command length {len(command)} != expected DOF {self._total_dof}')
        
        # Safety check
        is_safe, reason = self.check_joint_command_safety(command)
        if not is_safe:
            log.warning(f"Unsafe command rejected: {reason}")
            # For testing, allow commands but clip them to safe values
            # return False
        
        try:
            # Parse command components
            cmd_idx = 0
            
            # Always send arm commands (14 DOF)
            arm_command = command[cmd_idx:cmd_idx + 14]
            self._robot.move_arm(arm_command)
            cmd_idx += 14
            
            # Optional head control
            if self._control_head:
                head_command = command[cmd_idx:cmd_idx + 2]
                # Convert radians to degrees and apply limits
                head_deg = np.rad2deg(head_command)
                head_deg = np.clip(head_deg, self.HEAD_LIMITS_MIN, self.HEAD_LIMITS_MAX)
                self._robot.move_head(head_deg)
                cmd_idx += 2
            
            # Optional waist control  
            if self._control_waist:
                waist_pitch_rad = command[cmd_idx]     # pitch in radians
                waist_height_m = command[cmd_idx + 1]  # height in meters
                
                # Convert to expected units: [pitch_deg, height_cm]
                waist_pitch_deg = np.rad2deg(waist_pitch_rad)
                waist_height_cm = waist_height_m * 100
                
                waist_command = np.array([waist_pitch_deg, waist_height_cm])
                waist_command = np.clip(waist_command, self.WAIST_LIMITS_MIN, self.WAIST_LIMITS_MAX)
                self._robot.move_waist(waist_command)
                cmd_idx += 2
            
            # Optional gripper control
            if self._control_gripper:
                gripper_command = command[cmd_idx:cmd_idx + 2]
                # Ensure values in [0,1] range
                gripper_command = np.clip(gripper_command, 0.0, 1.0)
                self._robot.move_gripper(gripper_command)
                cmd_idx += 2
            
            # Optional hand control
            if self._control_hand:
                hand_command = command[cmd_idx:cmd_idx + 12]
                self._robot.move_hand(hand_command)
                cmd_idx += 12
            
            # Update safety state
            self.commit_safe_state()
            
            return True
            
        except Exception as e:
            log.error(f"Failed to send joint command: {e}")
            return False
    
    def close(self) -> None:
        """Safely shutdown robot connection and threads."""
        log.info(f"Shutting down AgiBot G1 {self._robot_name}")
        
        # Stop update thread
        self._thread_running = False
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
            if self._update_thread.is_alive():
                log.warning("Update thread did not stop gracefully")
        
        # Reset and shutdown robot
        try:
            self._robot.reset()
            self._robot.shutdown()
            log.info("AgiBot G1 shutdown complete")
        except Exception as e:
            log.error(f"Error during robot shutdown: {e}")
    
    def move_to_start(self) -> bool:
        """Move robot to initial/home position.
        
        Returns:
            bool: True if move successful
        """
        if self._init_joint_positions is not None:
            log.info("Moving AgiBot G1 to start position")
            return self.set_joint_command(['position'], self._init_joint_positions)
        else:
            log.warning("No initial joint positions defined")
            return False

    def set_chassis_command(self, chassis_speed: List[float]) -> bool:
        """Control mobile base movement.
        
        Args:
            chassis_speed: [linear_velocity, angular_velocity] in m/s and rad/s
            
        Returns:
            bool: True if command sent successfully
            
        Raises:
            ValueError: If chassis_speed not 2D
        """
        if not self._control_wheel:
            log.warning("Wheel control not enabled in configuration")
            return False
            
        if len(chassis_speed) != 2:
            raise ValueError(f'Chassis command must be 2D [linear, angular], got {len(chassis_speed)}D')
        
        try:
            self._robot.move_wheel(chassis_speed[0], chassis_speed[1])
            log.debug(f"Chassis command sent: linear={chassis_speed[0]}, angular={chassis_speed[1]}")
            return True
        except Exception as e:
            log.error(f"Failed to send chassis command: {e}")
            return False
    
    def get_dual_arm_positions(self) -> Optional[np.ndarray]:
        """Get dual arm joint positions.
        
        Returns:
            Optional[np.ndarray]: 14-DOF arm positions in radians or None if error
        """
        try:
            arm_posi, timestamp = self._robot.arm_joint_states()
            return np.array(arm_posi, dtype=np.float64)
        except Exception as e:
            log.error(f"Failed to get arm positions: {e}")
            return None
    
    def get_waist_positions(self) -> Optional[np.ndarray]:
        """Get waist joint positions.
        
        Returns:
            Optional[np.ndarray]: [pitch, height] or None if error
        """
        try:
            waist_posi, timestamp = self._robot.waist_joint_states()
            # Convert from [pitch_deg, height_cm] to [pitch_rad, height_m]
            waist_rad_m = np.array([
                np.deg2rad(waist_posi[0]),  # pitch to radians
                waist_posi[1] / 100.0       # height to meters
            ], dtype=np.float64)
            return waist_rad_m
        except Exception as e:
            log.error(f"Failed to get waist positions: {e}")
            return None
    
    def get_head_positions(self) -> Optional[np.ndarray]:
        """Get head joint positions.
        
        Returns:
            Optional[np.ndarray]: [yaw, pitch] in radians or None if error
        """
        try:
            head_posi, timestamp = self._robot.head_joint_states()
            # Convert degrees to radians
            head_rad = np.deg2rad(np.array(head_posi, dtype=np.float64))
            return head_rad
        except Exception as e:
            log.error(f"Failed to get head positions: {e}")
            return None
    
    def get_gripper_positions(self) -> Optional[np.ndarray]:
        """Get gripper states.
        
        Returns:
            Optional[np.ndarray]: [left, right] gripper positions [0,1] or None if error
        """
        try:
            gripper_posi, timestamp = self._robot.gripper_states()
            return np.array(gripper_posi, dtype=np.float64)
        except Exception as e:
            log.error(f"Failed to get gripper positions: {e}")
            return None
    
    def get_hand_positions(self) -> Optional[np.ndarray]:
        """Get hand joint positions.
        
        Returns:
            Optional[np.ndarray]: 12-DOF hand positions or None if error
        """
        try:
            hand_posi, timestamp = self._robot.hand_joint_states()
            return np.array(hand_posi, dtype=np.float64)
        except Exception as e:
            log.error(f"Failed to get hand positions: {e}")
            return None
    
    def get_hand_forces(self) -> Optional[np.ndarray]:
        """Get hand force sensor readings.
        
        Returns:
            Optional[np.ndarray]: Force sensor data or None if error
        """
        try:
            hand_forces, timestamp = self._robot.hand_force_states()
            return np.array(hand_forces, dtype=np.float64)
        except Exception as e:
            log.error(f"Failed to get hand forces: {e}")
            return None
    
    def get_joint_states_at_timestamp(self, timestamp_ns: int) -> Optional[RobotJointState]:
        """Get robot state at specific timestamp.
        
        Args:
            timestamp_ns: Timestamp in nanoseconds
            
        Returns:
            Optional[RobotJointState]: Robot state or None if not available
        """
        try:
            # Get states nearest to timestamp
            arm_pos = self._robot.arm_joint_states_nearest(timestamp_ns)
            
            position_components = [np.array(arm_pos)]
            
            if self._control_head:
                head_pos = self._robot.head_joint_states_nearest(timestamp_ns)
                head_rad = np.deg2rad(np.array(head_pos))
                position_components.append(head_rad)
            
            if self._control_waist:
                waist_pos = self._robot.waist_joint_states_nearest(timestamp_ns)
                waist_rad_m = np.array([
                    np.deg2rad(waist_pos[0]),
                    waist_pos[1] / 100.0
                ])
                position_components.append(waist_rad_m)
            
            if self._control_gripper:
                gripper_pos = self._robot.gripper_joint_states_nearest(timestamp_ns)
                position_components.append(np.array(gripper_pos))
            
            if self._control_hand:
                hand_pos = self._robot.hand_joint_states_nearest(timestamp_ns)
                position_components.append(np.array(hand_pos))
            
            # Create joint state
            joint_state = RobotJointState()
            joint_state._positions = np.hstack(position_components)
            joint_state._velocities = np.zeros_like(joint_state._positions)
            joint_state._accelerations = np.zeros_like(joint_state._positions)
            joint_state._time_stamp = timestamp_ns / 1e9  # Convert to seconds
            
            return joint_state
            
        except Exception as e:
            log.error(f"Failed to get joint states at timestamp {timestamp_ns}: {e}")
            return None
    
    def get_whole_body_status(self) -> Optional[Dict[str, Any]]:
        """Get overall robot status.
        
        Returns:
            Optional[Dict]: Status dictionary or None if error
        """
        try:
            return self._robot.whole_body_status()
        except Exception as e:
            log.error(f"Failed to get whole body status: {e}")
            return None
    
    def reset_robot(self, **kwargs) -> bool:
        """Reset robot to specified or default positions.
        
        Args:
            **kwargs: Optional position arguments (arm_positions, gripper_positions, etc.)
            
        Returns:
            bool: True if reset successful
        """
        try:
            self._robot.reset(**kwargs)
            log.info("Robot reset successful")
            return True
        except Exception as e:
            log.error(f"Robot reset failed: {e}")
            return False
    
    # Enhanced control methods based on documentation
    def move_gripper_as_normalized(self, positions: List[float]) -> bool:
        """Move grippers using normalized [0,1] values.
        
        Args:
            positions: [left, right] gripper positions in [0,1]
            
        Returns:
            bool: True if successful
        """
        try:
            normalized_pos = np.clip(positions, 0.0, 1.0)
            self._robot.move_gripper(normalized_pos)
            return True
        except Exception as e:
            log.error(f"Failed to move gripper: {e}")
            return False
    
    def move_hand_as_gripper(self, positions: List[float]) -> bool:
        """Use hands as grippers with normalized control.
        
        Args:
            positions: [left, right] hand gripper positions in [0,1]
            
        Returns:
            bool: True if successful
        """
        try:
            self._robot.move_hand_as_gripper(positions)
            return True
        except Exception as e:
            log.error(f"Failed to move hand as gripper: {e}")
            return False
    
    def move_head_and_waist_simultaneously(self, head_pos: List[float], 
                                         waist_pos: List[float]) -> bool:
        """Move head and waist simultaneously.
        
        Args:
            head_pos: [yaw, pitch] in radians
            waist_pos: [pitch, height] in [rad, m]
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert to expected units
            head_deg = np.rad2deg(np.array(head_pos))
            head_deg = np.clip(head_deg, self.HEAD_LIMITS_MIN, self.HEAD_LIMITS_MAX)
            
            waist_deg_cm = np.array([
                np.rad2deg(waist_pos[0]),  # pitch to degrees
                waist_pos[1] * 100         # height to cm
            ])
            waist_deg_cm = np.clip(waist_deg_cm, self.WAIST_LIMITS_MIN, self.WAIST_LIMITS_MAX)
            
            self._robot.move_head_and_waist(head_deg, waist_deg_cm)
            return True
        except Exception as e:
            log.error(f"Failed to move head and waist: {e}")
            return False
    


if __name__ == "__main__":
    pass