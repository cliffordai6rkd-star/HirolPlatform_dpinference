import abc
import numpy as np
import numpy as np
from hardware.base.utils import RobotJointState
import threading
import copy
class ArmBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._dof = config["dof"]
        self._joint_states = RobotJointState()
        # self._tcp_pose = np.zeros(7) # [x, y, z, qx, qy, qz, qw]
        self._lock = threading.Lock()
        self._is_initialized = False
        self._is_initialized = self.initialize()
    
    def print_state(self):
        print(f"Arm joint states[positions, velocity, torques]: "
              f"{self._joint_states._positions}, {self._joint_states._velocities}, {self._joint_states._torques}")
        # print(f'Arm TCP pose: {self._tcp_pose}')
    
    def get_dof(self):
        if len(self._dof) == 1:
            dof = [self._dof]
        return self._dof
    
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
    