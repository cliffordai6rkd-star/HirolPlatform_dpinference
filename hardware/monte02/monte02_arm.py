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


class Monte02_Arm(ArmBase):
    def __init__(self, config: Dict[str, Any]):
        self._ip = config['ip']  # "192.168.11.3:50051"
        self.component_type = 1 if config['side'] == 'left' else 2
        self._comm_freq = 200

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
        if self._is_initialized:
            return True
        
        # Initialize RobotLib instance via singleton
        from hardware.monte02.robotlib_manager import RobotAPI
        self._robot = RobotAPI.get_robot(self._ip, "", "")
        
        # Clear error and warning codes
        CHECK(self._robot.clean_arm_err_warn_code(self.component_type))
        
        # Enable dual arms
        CHECK(self._robot.set_arm_enable(self.component_type, ARM_ENABLE))
        
        # Set control mode to servo motion mode
        CHECK(self._robot.set_arm_mode(self.component_type, ARM_MODE_SERVO_MOTION))
        
        # Set motion state
        CHECK(self._robot.set_arm_state(self.component_type, ARM_STATE_SPORT))
        
        # Start state update thread
        self._update_thread = threading.Thread(target=self.update_state_task)
        self._update_thread.start()
        
        log.info(f"Monte02 robot initialized successfully (IP: {self._ip}) _robot: {self._robot}")
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
        
        # Get left arm joint positions
        ret, jps = self._robot.get_arm_servo_angle(self.component_type)
        if not ret:
            log.warn("Read left arm joint positions failed!")
        
        return ret, jps
    
    def update_arm_states(self) -> None:
        """
        Update dual-arm joint states
        """
        success, jps = self._get_current_joint_positions()

        if success:
            # Ensure numpy array for downstream arithmetic
            try:
                jps_arr = np.asarray(jps, dtype=float)
            except Exception:
                jps_arr = np.array(list(jps), dtype=float)

            # Defensive: enforce 7-DoF vector for a single arm
            if jps_arr.ndim != 1:
                jps_arr = jps_arr.flatten()
            if jps_arr.size > 7:
                jps_arr = jps_arr[:7]
            elif jps_arr.size < 7:
                raise Exception(f"jps size{jps_arr.size} < 7, this is NOT allowed!")
            
            self._joint_states._positions = jps_arr
    
    def set_joint_command(self, mode: Any, command: np.ndarray) -> None:
        """
        Set joint command
        
        Args:
            mode: Control mode list, currently only supports 'position'
            command: Joint command array, length should be total_dof
        """
        
        # Normalize control mode to list[str]
        if isinstance(mode, str):
            mode_list = [mode]
        else:
            mode_list = list(mode)

        # Check control mode
        for cur_mode in mode_list:
            if cur_mode != 'position':
                raise ValueError(f'Monte02 only supports position control, got: {cur_mode}')
        
        # Decompose dual-arm command: first 7D to left arm, next 7D to right arm
        sync = True if self.component_type == 1 else False
        success = self._robot.set_arm_servo_angle_j(
            self.component_type, 
            command.tolist(),
            1.0,  # velocity
            0,    # acceleration 
            sync     # sync
        )
        
        if not success:
            log.warn(f"Joint command partially failed")
    
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
            self._robot.set_arm_state(self.component_type, ARM_STATE_STOP)
            # Disable dual arms
            self._robot.set_arm_enable(self.component_type, DENABLE)
            log.info("Monte02 robot closed successfully")
        else:
            log.info("Monte02 mock robot closed")

    def move_to_start(self):

        positions_left_start = [0.01174976211041212, 0.17985449731349945, 0.17040130496025085, 1.5592831373214722, 0.12087556719779968, 0.06636597961187363, -0.16235961019992828]
        positions_right_start = [-0.009328235872089863, 0.37675273418426514, -0.24538300931453705, 1.6782840490341187, 0.21086378395557404, 0.11790347844362259, 0.16573384404182434]
        jp = positions_left_start if self.component_type == 1 else positions_right_start
        sync = 0 if self.component_type == 1 else 1
        speed = 0.3
        acc = 0
        self._robot.set_arm_servo_angle(self.component_type, jp, speed, acc, sync)

    def move_to_zero(self):
        sync = 0 if self.component_type == 1 else 1
        self._robot.set_arm_servo_angle(self.component_type, [0]*7, 0.3, 0, sync)

def __main__():
    """
    Simple smoke test for Monte02_Arm. Uses MockRobotLib if SDK is unavailable.
    Steps:
      - construct arm (left/right)
      - read initial joint state
      - move_to_start -> small position nudge -> move_to_zero
    """
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description="Monte02_Arm smoke test")
    parser.add_argument("--ip", default=os.getenv("MONTE02_IP", "192.168.11.3:50051"))
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--hold", type=float, default=1.0, help="hold time after each motion (s)")
    args = parser.parse_args()

    # For this single-arm wrapper, pass dof as a list to match sum(self._dof)
    config = {
        "ip": args.ip,
        "side": args.side,
        "dof": [7],
    }

    arm = None

    def _cleanup(*_):
        try:
            if arm is not None:
                arm.close()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        log.info(f"Creating Monte02_Arm for side={args.side}, ip={args.ip}")
        arm = Monte02_Arm(config)

        # Let state thread populate first reading
        time.sleep(0.2)
        js = arm.get_joint_states()
        try:
            q0 = np.round(np.array(js._positions, dtype=float), 3).tolist()
        except Exception:
            q0 = list(js._positions)
        log.info(f"Initial q: {q0}")

        log.info("Move to start...")
        arm.move_to_start()
        time.sleep(args.hold)
        js = arm.get_joint_states()
        try:
            q1 = np.round(np.array(js._positions, dtype=float), 3).tolist()
        except Exception:
            q1 = list(js._positions)
        log.info(f"After start q: {q1}")

        # Small position nudge
        try:
            cmd = np.array(js._positions, dtype=float).copy()
        except Exception:
            cmd = np.array(list(js._positions), dtype=float)
        if cmd.shape[0] == 7:
            cmd[0] += 0.05 if args.side == "left" else -0.05
        log.info("Send small position offset command...")
        arm.set_joint_command(["position"], cmd)
        time.sleep(args.hold)

        log.info("Move to zero...")
        arm.move_to_zero()
        time.sleep(args.hold)

        log.info("Test finished OK.")
    finally:
        if arm is not None:
            arm.close()

if __name__ == "__main__":
    __main__()
