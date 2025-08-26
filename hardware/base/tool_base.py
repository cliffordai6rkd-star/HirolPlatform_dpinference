import numpy as np
import abc
import warnings
import glog as log
import threading
import time
from typing import Union, Dict, List
from hardware.base.utils import ToolState, ToolControlMode, ToolType

class ToolBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._state = ToolState()
        self._tool_idle = True
        
        # Initialize control mode from config
        control_mode_str = config.get("control_mode", "binary")
        self._control_mode = ToolControlMode(control_mode_str)
        
        # Binary mode configuration
        if self._control_mode == ToolControlMode.BINARY:
            self._binary_threshold = config.get("binary_threshold", 0.5)
        
        # Incremental control state (for incremental mode only)
        if self._control_mode == ToolControlMode.INCREMENTAL:
            self._current_position_scaled = config.get("initial_position", 1.0)  # Default fully open position
            self._step_size = config.get("step_size", 0.02)  # Maximum position change per step
            self._last_move_time = 0.0
        self._is_initialized = False
        self._is_initialized = self.initialize()
        
    def print_state(self) -> None:
        if self._is_initialized and not self._state is None:
            print(f'Tool state: {self._state}')
            if self._control_mode == ToolControlMode.INCREMENTAL:
                print(f'Current width: {self._state._position:.3f}')
        else:
            warnings.warn('Tool is still not initialized or '
                          'the state is not updated')
    
    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_tool_state(self) -> ToolState:
        raise NotImplementedError
    
    def recover(self) -> ToolState:
        pass
    
    @abc.abstractmethod
    def set_hardware_command(self, command: np.array):
        """
            @brief: set the command to execute the hardware
        """
        raise NotImplementedError
    
    def set_tool_command(self, target: Union[float, List, np.ndarray, Dict]) -> bool:
        """
        Unified tool control interface supporting multiple control modes 
        and different tool categories.
        
        Args:
            target: Control target, dict is for dou_gripper, only contains keys ["left", "right"]
                - BINARY mode: numeric value (0.0-1.0), uses threshold to determine open/close
                  Examples: 0.3 -> close(0.0), 0.7 -> open(1.0) with threshold=0.5
                - INCREMENTAL mode: precise position value (0.0-1.0)
                  Examples: 0.3 -> 30% opening, 0.7 -> 70% opening
                
        Returns:
            bool: Control command execution success flag
            
        Raises:
            ValueError: Invalid control target format
        """
        if not self._tool_idle: 
            log.warn("Tool is currently working on other command, please wait for some time to set new command") 
            return False
        
        if self._state._tool_type != ToolType.GRIPPER and self._state._tool_type != ToolType.SUCTION:
            # @TODO: handle other tools
            return self.set_hardware_command(target)
        else:
            target = np.clip(target, 0, 1)
            if self._control_mode == ToolControlMode.BINARY:
                # Binary mode: extract value -> threshold judgment -> 0.0 or 1.0
                target = self._apply_binary_threshold(target)
                self._tool_idle = False
                success = self.set_hardware_command(target)
                # print(f'incremental tool target: {target}')
                self._tool_idle = True
            elif self._control_mode == ToolControlMode.INCREMENTAL:
                success = self._handle_gripper_incremental_command(target)
            else:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")
            return success
        
    
    @abc.abstractmethod
    def stop_tool(self):
        """
            @brief: stop the usage of the tool
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_tool_type_dict(self):
        raise NotImplementedError
   
    def close(self) -> bool:
        """
            @brief: set the tool state to be closed
        """
        return self.set_tool_command(0.0)
    
    def _apply_binary_threshold(self, value: float) -> float:
        return 1.0 if value >= self._binary_threshold else 0.0
    
    def _handle_gripper_incremental_command(self, target: float, is_wait: bool = False):
        """
        Handle gripper incremental command with smooth position transitions in a separate thread.
        
        Args:
            target: Target position (0.0-1.0)
            is_wait: Whether to wait for the thread to complete
        """
        self._tool_idle = False
        
        def incremental_execution():
            """
            Execute incremental movement in a while loop until target is reached.
            """
            while True:
                position_diff = target - self._current_position_scaled
                if abs(position_diff) < 1e-3:
                    break
                
                # Apply step size limit for smooth movement
                if abs(position_diff) > self._step_size:
                    # Move by step_size in the direction of target
                    step = self._step_size if position_diff > 0 else -self._step_size
                    new_position = self._current_position_scaled + step
                else:
                    # Close enough, move directly to target
                    new_position = target
                
                # Update internal position state
                self._current_position_scaled = new_position
                
                # Execute hardware command
                self.set_hardware_command(new_position)
                
                # Wait for next move based on frequency
                time.sleep(0.001)
                
            self._tool_idle = True
        
        # Start thread for incremental execution
        thread = threading.Thread(target=incremental_execution)
        thread.start()
        
        # Wait for thread completion if requested
        if is_wait:
            thread.join()
        return True