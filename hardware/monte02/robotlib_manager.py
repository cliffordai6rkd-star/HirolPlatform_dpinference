"""
Centralized, singleton management for RobotLib.Robot.

Ensures the RobotLib extension module is imported once and the Robot
client is constructed exactly once per process, even if multiple hardware
wrappers (e.g., Monte02, CoreneticGripper2) are initialized.

Usage:
  from hardware.monte02.robotlib_manager import RobotAPI
  robot = RobotAPI.get_robot(ip)
"""
from __future__ import annotations

import os
import sys
import threading
import importlib
from typing import Optional, Tuple, Any
import glog as log

# Default SDK relative directory
MONTE02_SDK_DIR = os.path.join('dependencies', 'monte02_sdk', 'build')


class _MockRobot:
    """Lightweight mock that implements methods used by Monte02 and gripper."""
    def __init__(self, *args, **kwargs) -> None:
        self._left_arm_positions = [0.0] * 7
        self._right_arm_positions = [0.0] * 7
        log.info("[Mock] RobotLib Robot created")

    # Common arm API
    def set_arm_enable(self, component_type: int, enable: int) -> bool:
        log.info(f"[Mock] set_arm_enable({component_type}, {enable})")
        return True

    def set_arm_mode(self, component_type: int, mode: int) -> bool:
        log.info(f"[Mock] set_arm_mode({component_type}, {mode})")
        return True

    def set_arm_state(self, component_type: int, state: int) -> bool:
        log.info(f"[Mock] set_arm_state({component_type}, {state})")
        return True

    def get_arm_servo_angle(self, component_type: int) -> Tuple[bool, list[float]]:
        return (True, self._left_arm_positions[:] if component_type == 1 else self._right_arm_positions[:])

    def set_arm_servo_angle_j(self, component_type: int, angles: list[float], velocity: float = 1.0,
                               acceleration: float = 0.0, sync: int = 1) -> bool:
        if component_type == 1:
            self._left_arm_positions = list(angles)
        else:
            self._right_arm_positions = list(angles)
        return True

    # Body/head
    def get_body_joint_state(self):
        import numpy as np, time
        z = np.zeros(5)
        return True, z, z.copy(), z.copy(), time.time()

    def set_trunk_joint_enable(self, enable: int) -> bool: return True
    def set_trunk_joint_mode(self, mode: int) -> bool: return True
    def set_head_joint_enable(self, enable: int) -> bool: return True

    # Gripper subset
    def clean_gripper_err_code(self, component_type: int) -> bool: return True
    def get_gripper_err_code(self, component_type: int): return True, 0
    def set_gripper_enable(self, component_type: int, enable: int) -> bool: return True
    def set_gripper_mode(self, component_type: int, mode: int) -> bool: return True
    def get_gripper_position(self, component_type: int): return True, 0.0
    def set_gripper_effort(self, component_type: int, effort: float) -> bool: return True


class RobotAPI:
    _robot: Optional[Any] = None
    _lock = threading.Lock()
    _module_path: Optional[str] = None
    _ip: Optional[str] = None

    @classmethod
    def _ensure_module(cls) -> Any:
        """Import RobotLib once, preferring Monte02 SDK build dir."""
        # if 'RobotLib' in sys.modules:
        #     return sys.modules['RobotLib']

        import sys, glob
        sys.path.insert(0, os.path.abspath('dependencies/monte02_sdk/build'))
        from RobotLib import Robot

        try:
            mod = importlib.import_module('RobotLib')
            cls._module_path = getattr(mod, '__file__', None)
            log.info(f"RobotLib using (singleton): {cls._module_path}")
            return mod
        except Exception as e:
            log.info(f"RobotLib not available ({e}), falling back to Mock robot")
            return None

    @classmethod
    def get_robot(cls, ip: str = "", a: str = "", b: str = "") -> Any:
        """Get the singleton Robot client. First call creates it."""
        with cls._lock:
            if cls._robot is not None:
                # Warn on IP mismatch
                if ip and cls._ip and ip != cls._ip:
                    log.warning(f"RobotAPI: requested ip {ip} differs from existing {cls._ip}; reusing singleton")
                return cls._robot

            mod = cls._ensure_module()
            if mod is None:
                cls._robot = _MockRobot()
                cls._ip = ip
                return cls._robot

            try:
                # Construct Robot once
                RobotCls = getattr(mod, 'Robot')
                cls._robot = RobotCls(ip, a, b)
                cls._ip = ip
                return cls._robot
            except Exception as e:
                log.info(f"RobotAPI: failed to create Robot client ({e}); using Mock")
                cls._robot = _MockRobot()
                cls._ip = ip
                return cls._robot

