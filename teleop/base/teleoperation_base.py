import numpy as np
import abc

class TeleoperationDeviceBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._is_initialized = False
        self._is_initialized = self.initialize()
        
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def print_data(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def read_data(self):
        """
            This function is a inside function, could 
            not be called from external
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def parse_data_2_robot_target(self, mode: str) -> tuple[bool, dict, dict]:
        """Parse the data read from the teleoperation device to a robot target command.
            @params: mode: str, the mode of the robot command, ['absolute', 'relative']
            @return: 
                whether successfully get the data from the device: bool
                The dict with the key: ['single', 'left', 'right'] indicates which part of 
                robot's target; values: the 7D end effector pose target
                The second dict with same key; and the value indicates the devices other output
                which could be used for gripper/hand control
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    