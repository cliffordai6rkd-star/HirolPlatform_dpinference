import numpy as np
import abc
import warnings
import time
from typing import Union, Dict, List
from hardware.base.utils import ToolState, GripperControlMode


class EdgeDetector:
    """独立的按钮上升沿检测器"""
    
    def __init__(self, initial_command: bool = True):
        """
        初始化上升沿检测器
        
        Args:
            initial_command: 初始命令状态 (True=开启)
        """
        self._last_button_state = False
        self._current_command = initial_command
    
    def detect_rising_edge(self, current_button: bool) -> bool:
        """
        检测按钮上升沿并切换命令状态
        
        Args:
            current_button: 当前按钮状态
            
        Returns:
            bool: 当前命令状态 (经过上升沿检测处理)
        """
        # 检测上升沿：从False到True的转换
        if not self._last_button_state and current_button:
            self._current_command = not self._current_command
        
        self._last_button_state = current_button
        return self._current_command

class ToolBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._state = ToolState()
        
        # Initialize control mode from config
        control_mode_str = config.get("control_mode", "binary")
        self._control_mode = GripperControlMode(control_mode_str)
        
        # Incremental control state
        self._current_position = 0.5  # Default middle position
        self._step_size = config.get("step_size", 0.02)
        self._move_frequency = config.get("move_frequency", 0.03)
        self._last_move_time = 0.0
        
        # Edge detector for binary mode button input
        if self._control_mode == GripperControlMode.BINARY:
            self._edge_detector = EdgeDetector(initial_command=True)
        
        self._is_initialized = self.initialize()
        
    def print_state(self) -> None:
        if self._is_initialized() and not self._state is None:
            print(f'Tool state: {self._state}')
        else:
            warnings.warn('Tool is still not initialized or '
                          'the state is not updated')
    
    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_tool_state(self) -> ToolState:
        raise NotImplementedError
    
    def set_tool_command(self, target: Union[float, Dict]) -> bool:
        """
        Unified tool control interface supporting multiple control modes.
        
        Args:
            target: Control target
                - BINARY mode: float (0-1 range position value)
                - INCREMENTAL mode: dict {'side': {'single': [bool, bool]}} button states
                
        Returns:
            bool: Control command execution success flag
            
        Raises:
            ValueError: Invalid control target format
        """
        if self._control_mode == GripperControlMode.BINARY:
            # Extract value from different input formats
            value = self._extract_binary_value(target)
            
            # Apply edge detection for button inputs (0.0 or 1.0)
            if isinstance(value, (int, float)) and value in [0.0, 1.0]:
                processed_target = self._process_button_input(bool(value))
                return self._set_binary_command(processed_target)
            else:
                return self._set_binary_command(float(value))
        elif self._control_mode == GripperControlMode.INCREMENTAL:
            if isinstance(target, dict):
                return self._set_incremental_command(target)
            elif isinstance(target, (list, np.ndarray)) and len(target) >= 2:
                # Direct button list format [open_button, close_button]
                return self._set_incremental_command(target)
            else:
                raise ValueError(f"Incremental mode requires dict or button list target, got {type(target)}")   
        else:
            raise ValueError(f"Unsupported control mode: {self._control_mode}")
    
    
    @abc.abstractmethod
    def stop_tool(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_tool_type_dict(self):
       raise NotImplementedError
    
    def _handle_incremental_control(self, buttons: List[bool]) -> float:
        """
        Incremental control core logic.
        
        Args:
            buttons: [open_button, close_button] button states list
            
        Returns:
            float: New target position value (0-1 range)
        """
        current_time = time.perf_counter()
        
        # Check frequency limit
        if (current_time - self._last_move_time) < self._move_frequency:
            return self._current_position
            
        # Quick exit if no buttons pressed
        if not (buttons[0] or buttons[1]):
            return self._current_position
            
        new_position = self._current_position
        
        # Button 0 - open gripper
        if buttons[0]:
            new_position = min(self._current_position + self._step_size, 1.0)
        # Button 1 - close gripper
        elif buttons[1]:
            new_position = max(self._current_position - self._step_size, 0.0)
            
        # Update state if position changed
        if new_position != self._current_position:
            self._current_position = new_position
            self._last_move_time = current_time
            
        return new_position
    
    def _extract_binary_value(self, target) -> float:
        """
        从不同格式的输入中提取二进制模式的值
        
        Args:
            target: 输入目标值
            
        Returns:
            float: 提取的值
            
        Raises:
            ValueError: 无效的输入格式
        """
        if isinstance(target, (int, float, np.number)):
            return float(target)
        elif isinstance(target, (list, np.ndarray)) and len(target) > 0:
            return float(target[0])
        elif isinstance(target, dict) and 'single' in target:
            buttons = target['single']
            if isinstance(buttons, (list, np.ndarray)) and len(buttons) > 0:
                return float(buttons[0])
            else:
                raise ValueError("Invalid 'single' format in dict target")
        else:
            raise ValueError(f"Binary mode requires numeric target or array, got {type(target)}")
    
    def _process_button_input(self, button_state: bool) -> float:
        """
        处理按钮输入的上升沿检测
        
        Args:
            button_state: 原始按钮状态
            
        Returns:
            float: 转换后的位置命令 (0.0 或 1.0)
        """
        if not hasattr(self, '_edge_detector'):
            return float(button_state)  # 简化fallback逻辑
            
        command_state = self._edge_detector.detect_rising_edge(button_state)
        return float(command_state)
    
    @abc.abstractmethod
    def _set_binary_command(self, target: float) -> bool:
        """
        Execute binary control command.
        
        Args:
            target: Position target (0-1 range)
            
        Returns:
            bool: Execution success
        """
        raise NotImplementedError
        
    def _set_incremental_command(self, target) -> bool:
        """
        Execute incremental control command.
        
        Args:
            target: Button states (list or dict)
            
        Returns:
            bool: Execution success
        """
        # Extract button states - handle different input formats
        buttons = None
        
        # Handle direct button list
        if isinstance(target, (list, np.ndarray)) and len(target) >= 2:
            buttons = target[:2]
        # Handle nested dict format from teleoperation
        elif isinstance(target, dict):
            if 'single' in target and isinstance(target['single'], list):
                buttons = target['single'][:2] if len(target['single']) >= 2 else [False, False]
            else:
                # Try to find buttons in nested structure
                for value in target.values():
                    if isinstance(value, dict) and 'single' in value:
                        buttons = value['single'][:2] if len(value['single']) >= 2 else [False, False]
                        break
                        
        if buttons is None:
            buttons = [False, False]
            
        # Calculate new position
        new_position = self._handle_incremental_control(buttons)
        
        # Execute binary command with new position
        return self._set_binary_command(new_position)

    def close(self) -> bool:
        if self._control_mode == GripperControlMode.BINARY:
            self.set_tool_command(0.0)
        else:
            self._set_binary_command(0.0)
   