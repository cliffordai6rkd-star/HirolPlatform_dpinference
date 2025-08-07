from hardware.base.tool_base import ToolBase
from hardware.base.utils import ToolType, ToolState
from hardware.fr3.franka_hand import FrankaHand
from hardware.monte01.xarm_gripper import XArmGripper
import numpy as np

class DuoTool(ToolBase):
    _tool_type: dict[str, ToolType] = None
    _tool: dict[str, ToolBase]
    def __init__(self, config):
        self._single_tool_class = {
            'franka_hand': FrankaHand,
            'xarm_gripper': XArmGripper
        }
        
        self._tool = {}
        
        # Initialize left tool if config exists
        left_tool_cfg = config.get("left", None)
        if left_tool_cfg is not None:
            left_calss_type = left_tool_cfg["type"]
            self._tool['left'] = self._single_tool_class[left_calss_type](left_tool_cfg)
        else:
            self._tool['left'] = None
        
        # Initialize right tool if config exists
        right_tool_cfg = config.get("right", None)
        if right_tool_cfg is not None:
            right_calss_type = right_tool_cfg["type"]
            self._tool['right'] = self._single_tool_class[right_calss_type](right_tool_cfg)
        else:
            self._tool['right'] = None
        
        # Check initialization status for existing tools only
        left_init = self._tool['left']._is_initialized if self._tool['left'] is not None else True
        right_init = self._tool['right']._is_initialized if self._tool['right'] is not None else True
        self._is_initialized = left_init and right_init
        
        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        # Initialize existing tools only
        left_success = self._tool['left'].initialize() if self._tool['left'] is not None else True
        right_success = self._tool['right'].initialize() if self._tool['right'] is not None else True
        return left_success and right_success
        
    def get_tool_state(self):
        # Get states from existing tools
        left_state = self._tool["left"].get_tool_state() if self._tool["left"] is not None else None
        right_state = self._tool["right"].get_tool_state() if self._tool["right"] is not None else None
        
        # Handle position concatenation
        positions = []
        if left_state is not None:
            positions.append(left_state._position)
        if right_state is not None:
            positions.append(right_state._position)
        self._state._position = np.hstack(positions) if positions else np.array([])
        
        # Handle force concatenation
        forces = []
        if left_state is not None:
            forces.append(left_state._force)
        if right_state is not None:
            forces.append(right_state._force)
        self._state._force = np.hstack(forces) if forces else np.array([])
        
        # Handle grasp state (both tools must be grasped, or single tool if only one exists)
        left_grasped = left_state._is_grasped if left_state is not None else True
        right_grasped = right_state._is_grasped if right_state is not None else True
        self._state._is_grasped = left_grasped and right_grasped
        
        # Handle tool types
        self._state._tool_type = {}
        if left_state is not None:
            self._state._tool_type['left'] = left_state._tool_type
        if right_state is not None:
            self._state._tool_type['right'] = right_state._tool_type
        
        return self._state
        
    def _set_binary_command(self, target: float) -> bool:
        """
        Execute binary command for all available tools.
        
        Args:
            target: Position value (0-1 range)
            
        Returns:
            bool: Success if all available tools execute successfully
        """
        success = True
        if self._tool["left"] is not None:
            success &= self._tool["left"]._set_binary_command(target)
        if self._tool["right"] is not None:
            success &= self._tool["right"]._set_binary_command(target)
        return success
        
    def set_tool_command(self, target):
        """
        Override to handle both unified and per-side control.
        
        Args:
            target: Either unified command or dict with side-specific commands
        """
        # Handle unified control mode (single value for all tools)
        if isinstance(target, (int, float, np.number)):
            return super().set_tool_command(target)
            
        # Handle side-specific control (dict format)
        if isinstance(target, dict):
            success = True
            # Send command to left tool if it exists and command is provided
            if "left" in target and self._tool["left"] is not None:
                target_left = target["left"]
                success &= self._tool["left"].set_tool_command(target_left)
            
            # Send command to right tool if it exists and command is provided  
            if "right" in target and self._tool["right"] is not None:
                target_right = target["right"]
                success &= self._tool["right"].set_tool_command(target_right)
            return success
            
        return False
        
    def stop_tool(self):
        if self._tool["left"] is not None:
            self._tool["left"].stop_tool()
        if self._tool["right"] is not None:
            self._tool["right"].stop_tool()
    
    def close(self):
        if self._tool["left"] is not None:
            self._tool["left"].close()
        if self._tool["right"] is not None:
            self._tool["right"].close()
    
    def get_tool_type_dict(self):
        if self._tool_type is None:
            self._tool_type = {}
            
            # Get tool types from existing tools
            if self._tool["left"] is not None:
                left_state = self._tool["left"].get_tool_state()
                self._tool_type['left'] = left_state._tool_type
                
            if self._tool["right"] is not None:
                right_state = self._tool["right"].get_tool_state()
                self._tool_type['right'] = right_state._tool_type
                
        return self._tool_type
        