import numpy
import abc
import warnings
from hardware.base.utils import GripperState

class GripperBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._state = GripperState()
        self._is_initialized = self.initialize()
        
    def print_state(self) -> None:
        if self._is_initialized() and not self._state is None:
            print(f'Gripper state: {self._state}')
        else:
            warnings.warn('Gripper is still not initialized or '
                          'the state is not updated')
    
    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_gripper_state(self) -> GripperState:
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_gripper_command(self, target) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def stop_gripper(self):
        raise NotImplementedError
    
    def close(self) -> bool:
        self.set_gripper_command(0)
   