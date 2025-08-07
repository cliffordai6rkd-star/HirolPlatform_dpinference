import abc
from hardware.base.utils import RobotJointState
import numpy as np
from collections import deque

class SimBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._traj_max_len = config["max_traj_len"]
        self._visulize_traj_data = deque(maxlen=self._traj_max_len)
        self._cur_traj_index = 0
        self.base_body_name = config.get("base_body", [])
        self._joint_names = config['joint_names']
        self._dof = config.get('dof', [len(self._joint_names)])
        # states
        self._joint_states = RobotJointState()
        
    @abc.abstractmethod
    def sim_thread(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_joint_states(self) -> RobotJointState:
        """Get the current joint states from the simulation."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """Get the current tcp poses from the simulation."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_joint_command(self, mode: list[str],  actuator_action:np.ndarray) -> np.ndarray:
        """set the joint commands for the simulation."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_camera_img(self, camera_name: str) -> np.ndarray | None:
        """get the image from camera"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_all_camera_images(self) -> list[dict] | None:
        """
            @brief: get all images form sim
            @return: None for no cameras
                a list contain all image info,
                dict: key: name value: camera name
                    key: resolution value: [height, width]
                    key: img value: image data 
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def parse_config(self) -> bool:
        """parse config to get the Mujoco model and data."""
        raise NotImplementedError

    @abc.abstractmethod
    def render(self):
        """Render the current state of the simulation."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def close(self):
        """close the simulation"""
        raise NotImplementedError
    
    def update_trajectory_data(self, data):
        self._visulize_traj_data.append(data)
    
    def get_dof(self):
        pass
    
