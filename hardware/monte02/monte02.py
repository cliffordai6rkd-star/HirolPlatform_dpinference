"""
Monte02 robot control class

Provides unified control interface for Monte02 dual-arm robot,
following AgibotG1's interface design.
Based on hardware/Monte02/config/Monte02_cfg.yaml configuration.
"""

import importlib.util
import inspect
import os
import threading
import time
from typing import List, Dict, Any, Tuple, Callable

import numpy as np

from hardware.base.arm import ArmBase
from hardware.base.utils import RobotJointState
from hardware.monte02.defs import *
import glog as log

# Conditional import of RobotLib SDK
try:
    import sys, glob
    sys.path.insert(0, os.path.abspath('dependencies/monte02_sdk/build'))
    from RobotLib import Robot

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
        
        def get_body_joint_state(self):
            """Mock body state: return 5-DoF zeros with a sane signature.
            Matches usage: success, q, dq, tau, t = get_body_joint_state()
            """
            zeros = np.zeros(5)
            return True, zeros, zeros.copy(), zeros.copy(), time.time()
        
        def clean_arm_err_warn_code(self, component_type: int) -> bool:
            arm_name = "Left" if component_type == COM_TYPE_LEFT else "Right"
            log.info(f"[Mock] Clean {arm_name} arm error/warning codes")
            return True
        
        def set_trunk_joint_enable(self, enable: int) -> bool:
            log.info(f"[Mock] Trunk joint enable: {enable}")
            return True

        def set_trunk_joint_mode(self, mode: int) -> bool:
            log.info(f"[Mock] Trunk joint mode: {mode}")
            return True

        def set_head_joint_enable(self, enable: int) -> bool:
            log.info(f"[Mock] Head joint enable: {enable}")
            return True

        def set_arm_servo_angle(self, component_type: int, angles: List[float], 
                                 velocity: float = 1.0, acceleration: float = 0, 
                                 sync: int = 1) -> bool:
            # Mirror to set_arm_servo_angle_j
            return self.set_arm_servo_angle_j(component_type, angles, velocity, acceleration, sync)
    
    Robot = MockRobotLib


class Monte02(ArmBase): #TODO: rename ArmBase to RobotBase, adjust related methods
    """
    Monte02 dual-arm robot unified control interface
    
    Provides AgibotG1-compatible interface for 14DOF dual-arm control.
    Based on RobotLib SDK with Mock mode fallback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Monte02 robot controller
        
        Args:
            config: Configuration dictionary containing 'Monte02' section
        """
        self._ip = config['ip']  # "192.168.11.3:50051"
        
        self._comm_freq = 200

        # Thread control
        self._thread_running = True
        self._update_thread = None
        
        # Initialize RobotLib
        self._robot = None

        super().__init__(config)
        self._total_dof = sum(self._dof)  # Sum of all DOF elements, e.g., [7, 7] -> 14
        # TODO: check
        self._joint_states._positions = np.zeros(self._total_dof)
        self._joint_states._velocities = np.zeros(self._total_dof)
        self._joint_states._accelerations = np.zeros(self._total_dof)
        self._joint_states._torques = np.zeros(self._total_dof)

    
    def initialize(self) -> bool:
        """
        Initialize robot hardware connection and control parameters
        
        Returns:
            bool: Whether initialization was successful
        """
        if self._is_initialized:
            return True
        
        try:
            # Initialize RobotLib instance via singleton
            from hardware.monte02.robotlib_manager import RobotAPI
            self._robot = RobotAPI.get_robot(self._ip, "", "")
            # Some SDK versions require trunk/head to be enabled before arm commands
            try:
                log.info(f"set_trunk_joint_enable")
                self._robot.set_trunk_joint_enable(ENABLE)
                log.info(f"set_head_joint_enable")
                self._robot.set_head_joint_enable(ENABLE)
                
                time.sleep(0.05)
            except Exception as _e:
                log.info(f"Non-fatal: pre-arm enable trunk/head setup failed: {_e}")
            # Clear error and warning codes
            CHECK(self._robot.clean_arm_err_warn_code(COM_TYPE_LEFT))
            CHECK(self._robot.clean_arm_err_warn_code(COM_TYPE_RIGHT), "Right arm clean_arm_err_warn_code")
            
            # Enable dual arms
            CHECK(self._robot.set_arm_enable(COM_TYPE_LEFT, ARM_ENABLE))
            CHECK(self._robot.set_arm_enable(COM_TYPE_RIGHT, ARM_ENABLE))
            
            # Set control mode to servo motion mode
            CHECK(self._robot.set_arm_mode(COM_TYPE_LEFT, ARM_MODE_SERVO_MOTION))
            CHECK(self._robot.set_arm_mode(COM_TYPE_RIGHT, ARM_MODE_SERVO_MOTION))
            
            # Set motion state
            CHECK(self._robot.set_arm_state(COM_TYPE_LEFT, ARM_STATE_SPORT))
            CHECK(self._robot.set_arm_state(COM_TYPE_RIGHT, ARM_STATE_SPORT))
            
            
            # Start state update thread
            self._update_thread = threading.Thread(target=self.update_state_task)
            self._update_thread.start()
            
            log.info(f"Monte02 robot initialized successfully (IP: {self._ip}) _robot: {self._robot}")
            return True
            
        except Exception as e:
            log.info(f"Warning: Monte02 initialization failed: {e}")
            log.info("Continuing with mock implementation")
            if self._robot is None:
                from hardware.monte02.robotlib_manager import RobotAPI
                self._robot = RobotAPI.get_robot(self._ip, "", "")  # singleton (mock if needed)
            
            
            return True
    
    def update_state_task(self) -> None:
      """
      State update thread task
      
      Periodically updates joint states, calculates velocities and accelerations
      """
      log.info(f'Monte02 robot {self._ip} started update thread!')

      read_period = 1.0 / self._comm_freq
      next_update_time = time.perf_counter()
      last_positions = None
      last_velocities = None
      actual_dt = read_period  # 初始化为目标周期

      while self._thread_running:
          try:
              loop_start_time = time.perf_counter()
              # 更新关节状态
              self.update_arm_states()

              # State update
              with self._lock:
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
                  self._joint_states._time_stamp = time.perf_counter()

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

      log.info(f'Monte02 robot {self._ip} stopped update thread!')
    
    def _get_current_joint_positions(self) -> Tuple[bool, np.ndarray]:
        """
        Private method: Get current joint positions from both arms
        
        Returns:
            Tuple[bool, np.ndarray]: (success, joint_positions) 
                                   joint_positions is 14D array or zeros on failure
        """
        # Start optimistic, AND with each subsystem read result
        ret = True
        
        success_body, body_positions, _, _, _ = self._robot.get_body_joint_state()
        if not success_body:
            log.warn("Read body joint positions failed!")
        ret &= success_body

        # Get left arm joint positions
        success_left, left_positions = self._robot.get_arm_servo_angle(COM_TYPE_LEFT)
        if not success_left:
            log.warn("Read left arm joint positions failed!")
        ret &= success_left
        
        # Get right arm joint positions  
        success_right, right_positions = self._robot.get_arm_servo_angle(COM_TYPE_RIGHT)
        if not success_right:
            log.warn("Read right arm joint positions failed!")
        ret &= success_right

        # Merge into 19D array: [body_5dof, left_7dof, right_7dof]
        if ret:
            all_positions = np.concatenate([body_positions, left_positions, right_positions])
        return ret, all_positions
    
    def update_arm_states(self) -> None:
        """
        Update dual-arm joint states
        
        Get left and right arm joint positions from RobotLib, merge into 14D state vector
        """
        success, all_positions = self._get_current_joint_positions()
        
        if success:
            self._joint_states._positions = all_positions
    
    def set_joint_command(self, mode: List[str], command: np.ndarray) -> None:
        """
        Set joint command
        
        Args:
            mode: Control mode list, currently only supports 'position'
            command: Joint command array, length should be total_dof
        """
        
        # Check control mode
        for cur_mode in mode:
            if cur_mode != 'position':
                raise ValueError(f'Monte02 only supports position control, got: {cur_mode}')
        
        # Check command length
        if len(command) != self._total_dof:
            raise ValueError(f'Monte02 joint command length should be {self._total_dof}, '
                             f'but got {len(command)}')
        
        
        # Decompose dual-arm command: first 7D to left arm, next 7D to right arm
        body_dof = self._dof[0]
        left_dof = self._dof[1]  # Left arm DOF (7)
        # right_dof = self._dof[2]  # Right arm DOF (7)
        body_command = command[:body_dof]
        left_command = command[body_dof:body_dof + left_dof]   # command[0:7]
        right_command = command[body_dof + left_dof:]  # command[7:14]
        
        # self._robot.set_trunk_joint_mode(TRUNK_JOINT_MODE_SERVOJ) # Dangerous ...
        self._robot.set_trunk_joint_mode(TRUNK_JOINT_MODE_PROFILE)
        self._robot.set_trunk_joint_position([1,2,3], body_command[:3].tolist())
        self._robot.set_head_joint_position([1,2], body_command[3:5].tolist())
        # Send left arm command
        success_left = self._robot.set_arm_servo_angle_j(
            COM_TYPE_LEFT, 
            left_command.tolist(),
            1.0,  # velocity
            0,    # acceleration 
            0     # sync
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
            log.warn(f"Joint command partially failed - Left: {success_left}, Right: {success_right}")
    
    def close(self) -> None:
        """
        Close robot connection and stop all threads
        """
        log.info("Closing Monte02 robot...")
        
        # Stop update thread
        self._thread_running = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        
        # Close robot connection
        if self._robot and ROBOTLIB_AVAILABLE:
            # Stop dual arm motion
            self._robot.set_arm_state(COM_TYPE_LEFT, ARM_STATE_STOP)
            self._robot.set_arm_state(COM_TYPE_RIGHT, ARM_STATE_STOP)
            
            # Disable dual arms
            self._robot.set_arm_enable(COM_TYPE_LEFT, DENABLE)
            self._robot.set_arm_enable(COM_TYPE_RIGHT, DENABLE)

            self._robot.set_trunk_joint_enable(DENABLE)
            self._robot.set_head_joint_enable(DENABLE)
            
            log.info("Monte02 robot closed successfully")
        else:
            log.info("Monte02 mock robot closed")
    
    def move_to_start(self):
        self._robot.set_trunk_joint_mode(TRUNK_JOINT_MODE_PROFILE)

        success, pos, vel, acc, tor = self._robot.get_body_joint_state()
        log.info(f"Success: {success}")
        log.info(f"pos: {pos}")

        log.info(f"set_trunk_joint_position")
        CHECK(self._robot.set_trunk_joint_position([1,2,3], [0] * 3))
        # time.sleep(0.5)

        log.info(f"set_head_joint_position")
        CHECK(self._robot.set_head_joint_position([1,2], [0.2] * 2))

        success, pos, vel, acc, tor = self._robot.get_body_joint_state()
        log.info(f"Success: {success}")
        log.info(f"pos: {pos}")
        # time.sleep(0.5)

        positions_left_start = [0.01174976211041212, 0.17985449731349945, 0.17040130496025085, 1.5592831373214722, 0.12087556719779968, 0.06636597961187363, -0.16235961019992828]
        positions_right_start = [-0.009328235872089863, 0.37675273418426514, -0.24538300931453705, 1.6782840490341187, 0.21086378395557404, 0.11790347844362259, 0.16573384404182434]
        speed = 0.3
        acc = 0
        self._robot.set_arm_servo_angle(1, positions_left_start, speed, acc, 0)
        self._robot.set_arm_servo_angle(2, positions_right_start, speed, acc, 1)

    def move_to_zero(self):
        positions_left_start = [0]*7
        positions_right_start = [0]*7
        speed = 0.3
        acc = 0
        self._robot.set_arm_servo_angle(1, positions_left_start, speed, acc, 0)
        self._robot.set_arm_servo_angle(2, positions_right_start, speed, acc, 1)

    
def __main__():
    """Simple CLI for quick bring-up and state inspection.

    Example:
        python hardware/monte02/monte02.py --config hardware/monte02/config/monte02_cfg.yaml \
        --duration 5 --rate 5
    """
    import argparse
    import yaml

    parser = argparse.ArgumentParser("Monte02 quick test")
    default_cfg = os.path.join(os.path.dirname(__file__), "config", "monte02_cfg.yaml")
    parser.add_argument("-c", "--config", default=default_cfg, help="path to monte02 yaml config")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to print joint states")
    parser.add_argument("--rate", type=float, default=5.0, help="print rate (Hz)")
    parser.add_argument("--demo-move", action="store_true",
                        help="after sampling states, resend current q as a position command (no motion)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = raw_cfg.get("monte02", raw_cfg)

    robot = Monte02(cfg)

    try:
        robot.move_to_start()
        # robot.move_to_zero()

        # # Allow background read thread to populate first sample
        # time.sleep(0.2)
        # log.info(f"DOF groups: {robot.get_dof()} (total={sum(robot.get_dof())})")

        # period = 1.0 / max(args.rate, 1e-3)
        # t_end = time.time() + max(args.duration, 0.0)
        # while time.time() < t_end:
        #     js = robot.get_joint_states()
        #     # Print compact
        #     log.info(f"q={np.round(js._positions, 3).tolist()}")
        #     time.sleep(period)

        # if args.demo_move:
        #     q = robot.get_joint_states()._positions
        #     # Use a single 'position' mode token; API accepts list[str]
        #     robot.set_joint_command(['position'], q)
        #     log.info("Re-sent current joint positions as a no-op command")

    finally:
        robot.close()

if __name__ == "__main__":
    __main__()
