import abc
import numpy as np
from hardware.base.utils import RobotJointState
from hardware.base.safety_checker import SafetyChecker, SafetyLevel, SafetyLimits
import threading
import copy
from typing import Dict, Any, Optional, Tuple, Union, List
import glog as log
class ArmBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._dof = config["dof"]
        self._init_joint_positions = config.get("init_joint_positions", None)
        self._joint_states = RobotJointState()
        # self._tcp_pose = np.zeros(7) # [x, y, z, qx, qy, qz, qw]
        self._lock = threading.Lock()
        self._is_initialized = False
        
        self._is_initialized = self.initialize()
    
    def print_state(self):
        if not self._is_initialized:
            log.warn(f'Unitree g1 is still not initialized for printing joint state')
        
        print(f"Arm joint states[positions, velocity, torques]: "
              f"{self._joint_states._positions}, {self._joint_states._velocities}, {self._joint_states._torques}")
        # print(f'Arm TCP pose: {self._tcp_pose}')
    
    def get_dof(self):
        if not isinstance(self._dof, list):
            dof = [self._dof]
        else:
            dof = self._dof
        return dof
    
    def get_joint_states(self)-> RobotJointState: 
        if self._is_initialized:
            with self._lock:
                joint_state = copy.deepcopy(self._joint_states)
            # @TODO: hack
            # joint_state._accelerations = np.zeros(len(joint_state._accelerations))
            return joint_state
        else:
            raise RuntimeError("Arm is not initialized, cannot get joint states.")
        
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_arm_states(self):
        """
            This func should not be called from external
            Because this is called in the class thread
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_joint_command(self, mode: Union[str, List[str]], command: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def move_to_start(self):
        raise NotImplementedError
    
    