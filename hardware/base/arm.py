import abc
import numpy as np
import numpy as np
from hardware.base.utils import RobotJointState
from hardware.base.safety_checker import SafetyChecker, SafetyLevel, SafetyLimits
import threading
import copy
from typing import Dict, Any, Optional, Tuple
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
        
        # Initialize safety checker
        self._init_safety_checker(config)
        
        self._is_initialized = self.initialize()
    
    def print_state(self):
        print(f"Arm joint states[positions, velocity, torques]: "
              f"{self._joint_states._positions}, {self._joint_states._velocities}, {self._joint_states._torques}")
        # print(f'Arm TCP pose: {self._tcp_pose}')
    
    def get_dof(self):
        if not isinstance(self._dof, list):
            dof = [self._dof]
        else:
            dof = self._dof
        return dof
    
    # def get_tcp_pose(self):
    #     """
    #         return the tcp pose in the format [x,y,z,qx,qy,qz,qw]
    #     """
    #     if self.is_initialized:
    #         self._lock.acquire()
    #         tcp_pose = copy.deepcopy(self._tcp_pose)
    #         self._lock.release()
    #         return tcp_pose
    #     else:
    #         raise RuntimeError("Arm is not initialized, cannot get TCP pose.")

    def get_joint_states(self)-> RobotJointState: 
        if self._is_initialized:
            self._lock.acquire()
            joint_state = copy.deepcopy(self._joint_states)
            self._lock.release()
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
    def set_joint_command(self, mode: str | list[str], command: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def move_to_start(self):
        raise NotImplementedError
    
    def _init_safety_checker(self, config: Dict[str, Any]) -> None:
        """
        Initialize safety checker from configuration
        
        Args:
            config: Configuration dictionary
        """
        # Get safety level from config
        safety_level = SafetyLevel.NORMAL
        if 'safety_level' in config:
            safety_level = SafetyLevel(config['safety_level'])
        
        # Custom safety limits from config
        custom_limits = None
        if 'safety_limits' in config:
            limits_config = config['safety_limits']
            custom_limits = SafetyLimits(
                max_joint_change=limits_config.get('max_joint_change', 0.5),
                max_position_change=limits_config.get('max_position_change', 0.01),
                max_rotation_change=limits_config.get('max_rotation_change', 0.2),
                max_joint_velocity=limits_config.get('max_joint_velocity', 2.0),
                min_command_interval=limits_config.get('min_command_interval', 0.001)
            )
        
        # Get robot name from config or use default
        robot_name = config.get('robot_name', self.__class__.__name__)
        
        self._safety_checker = SafetyChecker(
            limits=custom_limits,
            safety_level=safety_level,
            robot_name=robot_name
        )
    
    def init_safety_state(self, joint_positions: Optional[np.ndarray] = None) -> None:
        """
        Initialize safety checker state
        
        Args:
            joint_positions: Initial joint positions, if None uses zero positions
        """
        if joint_positions is None:
            # Calculate total DOF
            total_dof = sum(self.get_dof()) if isinstance(self._dof, list) else self._dof
            joint_positions = np.zeros(total_dof)
        
        self._safety_checker.update_state(joint_positions=joint_positions)
        self._safety_checker.commit_valid_state()
        log.info(f"Safety checker initialized with state: {joint_positions}")
    
    def check_joint_command_safety(self, command: np.ndarray) -> Tuple[bool, str]:
        """
        Check if joint command is safe
        
        Args:
            command: Joint command array
            
        Returns:
            Tuple[bool, str]: (is_safe, reason)
        """
        return self._safety_checker.check_joint_command(command)
    
    def update_safety_state(self, joint_positions: np.ndarray) -> None:
        """
        Update safety checker with current robot state
        
        Args:
            joint_positions: Current joint positions
        """
        self._safety_checker.update_state(joint_positions=joint_positions)
    
    def commit_safe_state(self) -> None:
        """Commit current state as safe/valid"""
        self._safety_checker.commit_valid_state()
    
    # ============= Safety Management =============
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety check statistics"""
        return self._safety_checker.get_statistics()
    
    def print_safety_statistics(self) -> None:
        """Print safety check statistics"""
        self._safety_checker.print_statistics()
    
    def reset_safety_tracking(self) -> None:
        """Reset safety state tracking"""
        self._safety_checker.reset_tracking()
        log.info("Safety tracking reset")
    
    def get_valid_joint_positions(self) -> Optional[np.ndarray]:
        """Get last known safe joint positions for emergency rollback"""
        return self._safety_checker.get_valid_joint_positions()
    
    def emergency_rollback(self) -> bool:
        """
        Emergency rollback to last safe joint positions
        
        Returns:
            bool: Whether rollback was successful
        """
        valid_positions = self.get_valid_joint_positions()
        if valid_positions is not None:
            try:
                log.info("Executing emergency rollback to safe position...")
                # Temporarily disable safety check for rollback
                temp_limits = self._safety_checker.limits
                self._safety_checker.limits.max_joint_change = 10.0  # Allow large change
                
                self.set_joint_command(['position'], valid_positions)
                
                # Restore original limits
                self._safety_checker.limits = temp_limits
                log.info("Emergency rollback completed")
                return True
            except Exception as e:
                log.info(f"Emergency rollback failed: {e}")
                return False
        else:
            log.info("No valid rollback position available")
            return False
    
    