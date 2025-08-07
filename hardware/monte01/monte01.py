"""
Monte01 robot control class

Provides unified control interface for Monte01 dual-arm robot,
following AgibotG1's interface design.
Based on hardware/monte01/config/monte01_cfg.yaml configuration.
"""

import importlib.util
import os
import threading
import time
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np

from hardware.base.arm import ArmBase
from hardware.base.utils import RobotJointState
from .defs import *
import glog as log
from hardware.monte01.trunk import Trunk
# Conditional import of RobotLib SDK
try:
    spec = importlib.util.spec_from_file_location(
        "RobotLib", 
        os.path.abspath(os.path.join(os.path.dirname(__file__), ROBOTLIB_SO_PATH))
    )
    RobotLib_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RobotLib_module)
    RobotLib = RobotLib_module.Robot
    ROBOTLIB_AVAILABLE = True
except (ImportError, AttributeError, OSError) as e:
    ROBOTLIB_AVAILABLE = False
    log.info(f"Warning: RobotLib not available ({e}), using mock implementation")
    
    class MockRobotLib:
        """Mock implementation of RobotLib for fallback when SDK is unavailable"""
        
        def __init__(self, ip: str, a: str, b: str) -> None:
            self._left_arm_positions = [0.0] * 7
            self._right_arm_positions = [0.0] * 7
            log.info("[Mock] RobotLib initialized")
        
        def set_arm_enable(self, component_type: int, enable: int) -> bool:
            arm_name = "Left" if component_type == COM_TYPE_LEFT else "Right"
            log.info(f"[Mock] Set {arm_name} arm enable: {enable}")
            return True
        
        def set_arm_mode(self, component_type: int, mode: int) -> bool:
            arm_name = "Left" if component_type == COM_TYPE_LEFT else "Right"
            log.info(f"[Mock] Set {arm_name} arm mode: {mode}")
            return True
        
        def set_arm_state(self, component_type: int, state: int) -> bool:
            arm_name = "Left" if component_type == COM_TYPE_LEFT else "Right"
            log.info(f"[Mock] Set {arm_name} arm state: {state}")
            return True
        
        def get_arm_servo_angle(self, component_type: int) -> Tuple[bool, List[float]]:
            if component_type == COM_TYPE_LEFT:
                return (True, self._left_arm_positions.copy())
            else:
                return (True, self._right_arm_positions.copy())
        
        def set_arm_servo_angle_j(self, component_type: int, angles: List[float], 
                                 velocity: float = 1.0, acceleration: float = 0, 
                                 sync: int = 1) -> bool:
            arm_name = "Left" if component_type == COM_TYPE_LEFT else "Right"
            if component_type == COM_TYPE_LEFT:
                self._left_arm_positions = list(angles)
            else:
                self._right_arm_positions = list(angles)
            log.info(f"[Mock] Set {arm_name} arm angles: {angles[:3]}...")
            return True
        
        def clean_arm_err_warn_code(self, component_type: int) -> bool:
            arm_name = "Left" if component_type == COM_TYPE_LEFT else "Right"
            log.info(f"[Mock] Clean {arm_name} arm error/warning codes")
            return True
    
    RobotLib = MockRobotLib


class Monte01(ArmBase): #TODO: rename ArmBase to RobotBase, adjust related methods
    """
    Monte01 dual-arm robot unified control interface
    
    Provides AgibotG1-compatible interface for 14DOF dual-arm control.
    Based on RobotLib SDK with Mock mode fallback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Monte01 robot controller
        
        Args:
            config: Configuration dictionary containing 'monte01' section
        """
        
        # DOF configuration: left arm + right arm
        dof_config = config['dof']  # [7, 7]
        if len(dof_config) != 2:
            raise ValueError(f"Expected dof config [left, right], got {dof_config}")
        
        self._left_dof, self._right_dof = dof_config
        self._total_dof = self._left_dof + self._right_dof  # 14
        
        # Network configuration
        self._ip = config['ip']  # "192.168.11.3:50051"
        
        # Subsystem control flags
        self._control_body = config['control_body']      # false
        self._control_chassis = config['control_chassis']  # false
        self._comm_freq = config['comm_freq']

        # Thread control
        self._thread_running = True
        self._update_thread = None
        
        # Initialize RobotLib
        self._robot = None
        
        super().__init__(config)
    
    def initialize(self) -> bool:
        """
        Initialize robot hardware connection and control parameters
        
        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Initialize RobotLib instance
            self._robot = RobotLib(self._ip, "", "")
            self._trunk = Trunk(self._robot)  # Initialize trunk control
            # Clear error and warning codes
            self._robot.clean_arm_err_warn_code(COM_TYPE_LEFT)
            self._robot.clean_arm_err_warn_code(COM_TYPE_RIGHT)
            
            # Enable dual arms
            self._robot.set_arm_enable(COM_TYPE_LEFT, ARM_ENABLE)
            self._robot.set_arm_enable(COM_TYPE_RIGHT, ARM_ENABLE)
            
            # Set control mode to servo motion mode
            self._robot.set_arm_mode(COM_TYPE_LEFT, ARM_MODE_SERVO_MOTION)
            self._robot.set_arm_mode(COM_TYPE_RIGHT, ARM_MODE_SERVO_MOTION)
            
            # Set motion state
            self._robot.set_arm_state(COM_TYPE_LEFT, ARM_STATE_SPORT)
            self._robot.set_arm_state(COM_TYPE_RIGHT, ARM_STATE_SPORT)
            
            # Initialize joint states
            self._last_posi = np.zeros(self._total_dof)
            self._last_vel = np.zeros(self._total_dof)
            
            # Initialize safety checker with current robot state
            success, current_positions = self._get_current_joint_positions()
            if success:
                self.init_safety_state(current_positions)
                log.info(f"Safety checker initialized with current robot state")
                log.info(f"Current joint positions: {current_positions}")
            else:
                # Fallback: initialize with zero position
                self.init_safety_state(np.zeros(self._total_dof))
                log.info(f"Safety checker initialized with zero state (fallback)")
            
            # Start state update thread
            self._update_thread = threading.Thread(target=self.update_state_task)
            self._update_thread.start()
            
            log.info(f"Monte01 robot initialized successfully (IP: {self._ip})")
            
            return True
            
        except Exception as e:
            log.info(f"Warning: Monte01 initialization failed: {e}")
            log.info("Continuing with mock implementation")
            if self._robot is None:
                self._robot = RobotLib(self._ip, "", "")  # This will create MockRobotLib instance
            
            # Initialize safety checker even in mock mode
            success, current_positions = self._get_current_joint_positions()
            if success:
                self.init_safety_state(current_positions)
                log.info(f"Safety checker initialized with current robot state (mock mode)")
            else:
                self.init_safety_state(np.zeros(self._total_dof))
                log.info(f"Safety checker initialized with zero state (mock mode fallback)")
            
            # Display safety configuration
            log.info("\n=== Safety Configuration ===")
            self.print_safety_statistics()
            
            return True
    
    def update_state_task(self) -> None:
      """
      State update thread task
      
      Periodically updates joint states, calculates velocities and accelerations
      """
      log.info(f'Monte01 robot {self._ip} started update thread!')

      read_period = 1.0 / self._comm_freq
      next_update_time = time.perf_counter()
      last_positions = None
      last_velocities = None
      actual_dt = read_period  # 初始化为目标周期

      while self._thread_running:
          try:
              loop_start_time = time.perf_counter()

              # State update
              with self._lock:
                  # 更新关节状态
                  self.update_arm_states()
                  current_positions = self._joint_states._positions.copy()

                  # 计算速度和加速度（使用实际时间间隔）
                  if last_positions is not None:
                      self._joint_states._velocities = (current_positions - last_positions) / actual_dt

                      if last_velocities is not None:
                          self._joint_states._accelerations = (self._joint_states._velocities - last_velocities) / actual_dt
                      else:
                          self._joint_states._accelerations = np.zeros_like(self._joint_states._velocities)
                  else:
                      # 首次运行时初始化
                      self._joint_states._velocities = np.zeros_like(current_positions)
                      self._joint_states._accelerations = np.zeros_like(current_positions)

                  # 更新历史数据
                  last_positions = current_positions
                  last_velocities = self._joint_states._velocities.copy()

              # 频率控制：计算下次执行时间
              next_update_time += read_period
              current_time = time.perf_counter()
              sleep_time = next_update_time - current_time

              if sleep_time > 0:
                  time.sleep(sleep_time)
                  # 计算实际时间间隔（用于下次速度计算）
                  actual_dt = time.perf_counter() - loop_start_time
              else:
                  # 处理时间超过目标周期
                  actual_dt = current_time - loop_start_time
                  next_update_time = current_time
                  if actual_dt > read_period * 1.5:  # 超过50%阈值时警告
                      log.warning(f'State update frequency slow: expected {1.0/read_period:.1f}Hz, '
                                 f'actual {1.0/actual_dt:.1f}Hz')

          except Exception as e:
              log.error(f"Error in state update task: {e}")
              time.sleep(0.01)
              next_update_time = time.perf_counter()
              actual_dt = read_period  # 重置为目标周期

      log.info(f'Monte01 robot {self._ip} stopped update thread!')
    
    def _get_current_joint_positions(self) -> Tuple[bool, np.ndarray]:
        """
        Private method: Get current joint positions from both arms
        
        Returns:
            Tuple[bool, np.ndarray]: (success, joint_positions) 
                                   joint_positions is 14D array or zeros on failure
        """
        try:
            # Get left arm joint positions
            success_left, left_positions = self._robot.get_arm_servo_angle(COM_TYPE_LEFT)
            if not success_left:
                raise RuntimeError("Failed to get left arm positions")
            
            # Get right arm joint positions  
            success_right, right_positions = self._robot.get_arm_servo_angle(COM_TYPE_RIGHT)
            if not success_right:
                raise RuntimeError("Failed to get right arm positions")
            
            # Merge into 14D array: [left_7dof, right_7dof]
            all_positions = np.concatenate([left_positions, right_positions])
            return True, all_positions
            
        except Exception as e:
            log.info(f"Warning: Failed to get joint positions: {e}")
            return False, np.zeros(self._total_dof)
    
    def update_arm_states(self) -> None:
        """
        Update dual-arm joint states
        
        Get left and right arm joint positions from RobotLib, merge into 14D state vector
        """
        success, all_positions = self._get_current_joint_positions()
        
        if success:
            # Update safety checker with current state
            self.update_safety_state(all_positions)
            
            # For now, only support dual arms (14 DOF)
            # Future: extend for head/waist control based on flags
            self._joint_states._positions = all_positions
        else:
            # Use zero vector as fallback
            self._joint_states._positions = np.zeros(self._total_dof)
    
    def set_joint_command(self, mode: List[str], command: np.ndarray) -> None:
        """
        Set joint command
        
        Args:
            mode: Control mode list, currently only supports 'position'
            command: Joint command array, length should be total_dof
        """
        log.debug(f'[DEBUG] Monte01 received joint command: mode={mode}, command_shape={command.shape}')
        log.debug(f'[DEBUG] Command values: {command[:3]}... (showing first 3 joints)')
        
        # Check control mode
        for cur_mode in mode:
            if cur_mode != 'position':
                raise ValueError(f'Monte01 only supports position control, got: {cur_mode}')
        
        # Check command length
        if len(command) != self._total_dof:
            raise ValueError(f'Monte01 joint command length should be {self._total_dof}, '
                             f'but got {len(command)}')
        
        # Safety check: validate joint command
        is_safe, reason = self.check_joint_command_safety(command)
        if not is_safe:
            log.warn(f"[DEBUG] Safety check failed: {reason}")
            log.warn(f"[DEBUG] Command rejected for safety")
            return
        else:
            log.debug(f'[DEBUG] Safety check passed')
        
        # Decompose dual-arm command: first 7D to left arm, next 7D to right arm
        left_command = command[:self._left_dof]   # command[0:7]
        right_command = command[self._left_dof:self._left_dof + self._right_dof]  # command[7:14]
        
        try:
            # Send left arm command
            success_left = self._robot.set_arm_servo_angle_j(
                COM_TYPE_LEFT, 
                left_command.tolist(), 
                1.0,  # velocity
                0,    # acceleration 
                1     # sync
            )
            
            # Send right arm command
            success_right = self._robot.set_arm_servo_angle_j(
                COM_TYPE_RIGHT, 
                right_command.tolist(), 
                1.0,  # velocity
                0,    # acceleration
                1     # sync
            )
            
            if not (success_left and success_right):
                log.info(f"Warning: Joint command partially failed - Left: {success_left}, Right: {success_right}")
            else:
                # Command succeeded, commit as valid state
                self.update_safety_state(command)
                self.commit_safe_state()
            
            # Handle head control (if enabled)
            # if self._control_head and len(command) > self._total_dof:
            #     head_command = command[self._total_dof:self._total_dof + 2]
            #     log.info(f"Head control not implemented, ignoring command: {head_command}")
            
            # # Handle waist control (if enabled)
            # if self._control_waist and len(command) > self._total_dof + (2 if self._control_head else 0):
            #     waist_start = self._total_dof + (2 if self._control_head else 0)
            #     waist_command = command[waist_start:waist_start + 2]
            #     log.info(f"Waist control not implemented, ignoring command: {waist_command}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to execute joint command: {e}")
    
    def close(self) -> None:
        """
        Close robot connection and stop all threads
        """
        log.info("Closing Monte01 robot...")
        
        # Stop update thread
        self._thread_running = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        
        # Close robot connection
        if self._robot and ROBOTLIB_AVAILABLE:
            try:
                # Stop dual arm motion
                self._robot.set_arm_state(COM_TYPE_LEFT, ARM_STATE_STOP)
                self._robot.set_arm_state(COM_TYPE_RIGHT, ARM_STATE_STOP)
                
                # Disable dual arms
                self._robot.set_arm_enable(COM_TYPE_LEFT, ARM_DENABLE)
                self._robot.set_arm_enable(COM_TYPE_RIGHT, ARM_DENABLE)
                
                log.info("Monte01 robot closed successfully")
            except Exception as e:
                log.info(f"Warning: Error during robot shutdown: {e}")
        else:
            log.info("Monte01 mock robot closed")
    
    def get_dual_arm_positions(self) -> np.ndarray:
        """
        Get dual-arm joint positions (14D)
        
        Returns:
            np.ndarray: Dual-arm joint positions [left_7dof, right_7dof]
        """
        success, positions = self._get_current_joint_positions()
        return positions  # Already returns zeros on failure
    
    def get_body_positions(self) -> np.ndarray:
        """
        Get body joint positions (5D: body_joints + head_joints)
        
        Monte01 currently does not support body control, returns zero vector
        Based on URDF: body_joint_1, body_joint_2, body_joint_3, head_joint_1, head_joint_2
        
        Returns:
            np.ndarray: Body joint positions [body_3dof, head_2dof] = [0, 0, 0, 0, 0]
        """
        # if self._control_body:
        if True:  
            success, positions, _, _ = self._robot.get_joint_state(BODY_JOINT_IDS)
            if success and len(positions) == 0:
                log.debug("Received empty positions from robot, using zero positions")
                return np.zeros(len(BODY_JOINT_IDS))
            log.debug(f"[DEBUG] get_body_joint_positions from robot: success={success}, positions={positions}")
            log.debug(f"[DEBUG] BODY_JOINT_IDS: {BODY_JOINT_IDS}")
            log.debug(f"[DEBUG] Expected positions length: {len(BODY_JOINT_IDS)}")
            if positions is not None:
                log.debug(f"[DEBUG] Actual positions length: {len(positions)}")
            
            if not success or positions is None or len(positions) == 0:
                log.warning(f"Failed to get body joint positions or empty positions: success={success}, positions={positions}")
                log.warning(f"Using zero positions for body joints: {BODY_JOINT_IDS}")
                return np.zeros(len(BODY_JOINT_IDS))
            
            if len(positions) != len(BODY_JOINT_IDS):
                log.error(f"Position length mismatch! Expected {len(BODY_JOINT_IDS)}, got {len(positions)}")
                return np.zeros(len(BODY_JOINT_IDS))
                
            return np.array(positions)
        return np.zeros(5)
    
    def set_chassis_command(self, chassis_speed: List[float]) -> None:
        """
        Set chassis motion command
        
        Monte01 currently does not support wheeled chassis control
        
        Args:
            chassis_speed: Chassis speed command [linear, angular]
        """
        if len(chassis_speed) != 2:
            raise ValueError(f'Monte01 chassis command should be 2D, but got {len(chassis_speed)}D')
        
        if self._control_chassis:
            log.info(f"Warning: Chassis control not implemented, ignoring command: {chassis_speed}")
        else:
            log.info(f"Warning: Chassis control disabled in config, ignoring command: {chassis_speed}")
    