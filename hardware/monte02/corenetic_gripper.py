from hardware.monte02.defs import *

from hardware.base.tool_base import ToolBase
from hardware.base.utils import ToolType
import glog as log
import time
import threading
import numpy as np
# Corenetic gripper constants

class CoreneticGripper2(ToolBase):
    _tool_type: ToolType = ToolType.GRIPPER
    
    def __init__(self, config: dict):
        self._ip = config["ip"]
        self._component_type = config["component_type"]  # COM_TYPE_LEFT or COM_TYPE_RIGHT
        super().__init__(config)
        self._state._tool_type = self._tool_type
        self._last_command = CORENETIC_GRIPPER_MAX_POSITION
        self._thread_running = True
        self._update_thread = threading.Thread(target=self.update_state)
        self._lock = threading.Lock()
        
        # Async command queue for low-latency mode
        self._command_queue = []
        self._queue_lock = threading.Lock()
        self._command_thread = threading.Thread(target=self._async_command_worker)
        self._command_thread.daemon = True
        self._command_thread.start()
        
    def initialize(self) -> bool:
        # Initialize RobotLib instance for Corenetic hardware
        from hardware.monte02.robotlib_manager import RobotAPI
        # Share the same Robot client (singleton)
        self.hardware = RobotAPI.get_robot("192.168.11.3:50051", "", "")
        log.info(f"Corenetic gripper connected to hardware at 192.168.11.3 (singleton) robot:{self.hardware}")

        CHECK(self.hardware.clean_gripper_err_code(self._component_type))
        CHECK(self.hardware.set_gripper_enable(self._component_type, GRIPPER_ENABLE))
        CHECK(self.hardware.set_gripper_mode(self._component_type, GRIPPER_MODE_POSITION_CTRL))

        CHECK(self.hardware.set_gripper_position(self._component_type, CORENETIC_GRIPPER_MAX_POSITION))
        time.sleep(1)
        log.info("Corenetic gripper initialized successfully")
            
        return True

    def set_hardware_command(self, command: np.array):
        log.debug(f'Setting Corenetic gripper command: {command}')
        target = np.clip(command, 0, 1)
        
        if hasattr(self, '_command_thread') and self._command_thread:
            # Async queue mode - ultra-low latency
            with self._queue_lock:
                # Only queue if different from last command
                if not self._command_queue or abs(self._command_queue[-1] - target) > 0.001:
                    self._command_queue.append(target)
                    # Limit queue size to prevent buildup
                    if len(self._command_queue) > 3:
                        self._command_queue = self._command_queue[-2:]  # Keep only last 2
            
            log.debug(f"Queued gripper command: {target}")
            return True

    def _set_binary_command(self, target: float) -> bool:
        """Move gripper to specified position. Returns True if successful, False otherwise."""
        
        target = np.clip(target, 0, 1)
        
        if hasattr(self, '_command_thread') and self._command_thread:
            # Async queue mode - ultra-low latency
            with self._queue_lock:
                # Only queue if different from last command
                if not self._command_queue or abs(self._command_queue[-1] - target) > 0.001:
                    self._command_queue.append(target)
                    # Limit queue size to prevent buildup
                    if len(self._command_queue) > 3:
                        self._command_queue = self._command_queue[-2:]  # Keep only last 2
            
            log.debug(f"Queued gripper command: {target}")
            return True

    def get_tool_state(self):
        try:
            success, pos, torque = self.hardware.get_gripper_position(self._component_type)
            if success:
                # log.info(f"Hardware gripper position: raw={pos:.6f}m")
                self._state._position = pos
                self._state._force = torque
            else:
                log.error(f'gripper communication test failed, success: {success}')
        except Exception as e:
            log.error(f'Error getting gripper position: {e}')
            
        return self._state
    
    def update_state(self):
        log.info(f'Starting state updating thread for Corenetic gripper {self._component_type}')
        
        read_frequency = 1000  # 10Hz update rate for gripper
        target_period = 1.0 / read_frequency
        next_update_time = time.perf_counter()
        
        while self._thread_running:
            loop_start_time = time.perf_counter()
            
            try:
                # Get gripper position
                success, position = self.hardware.get_gripper_position(self._component_type)
                
                with self._lock:
                    # Update position if successful
                    if success:
                        # Convert from 0-0.074m range to 0-1 normalized range
                        normalized_position = position / CORENETIC_GRIPPER_MAX_POSITION
                        normalized_position = np.clip(normalized_position, 0.0, 1.0)
                        self._state._position = normalized_position
                    else:
                        log.debug(f'Failed to get gripper position: success={success}')
                    
                    # Update grasp status using position-based heuristic
                    if hasattr(self._state, '_position'):
                        # Consider gripper closed/grasped when position is very small
                        self._state._is_grasped = (self._state._position < 0.1)
                        
            except Exception as e:
                log.error(f'Error updating gripper state: {e}')
            
            # Timing control - calculate next update time
            next_update_time += target_period
            current_time = time.perf_counter()
            sleep_time = next_update_time - current_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Handle timing overrun
                actual_period = current_time - loop_start_time
                if actual_period > target_period * 1.3:
                    log.warn(f'The Corenetic gripper {self._component_type} could not reach the update thread frequency!! '
                           f'Target: {read_frequency:.1f}Hz, Actual: {1.0/actual_period:.1f}Hz')
                # Reset timing to prevent drift
                next_update_time = current_time
                
        log.info(f'Corenetic gripper {self._component_type} stopped its update thread!!!!')
    
    def _async_command_worker(self):
        """Background worker for processing gripper commands asynchronously"""
        log.info(f'Starting async command worker for Corenetic gripper {self._component_type}')
        
        while self._thread_running:
            with self._queue_lock:
                if self._command_queue:
                    target = self._command_queue.pop(0)  # Get oldest command
                    # Clear duplicate commands in queue
                    self._command_queue = [cmd for cmd in self._command_queue if abs(cmd - target) > 0.001]
                else:
                    target = None
            
            if target is not None:
                try:
                    # Convert normalized position (0-1) to real position (0-0.074m)
                    position_real = target * CORENETIC_GRIPPER_MAX_POSITION
                    api_start = time.perf_counter()
                    
                    # Set gripper position using Corenetic API
                    success = self.hardware.set_gripper_position(self._component_type, position_real)
                    api_time = time.perf_counter() - api_start
                    
                    if not success:
                        log.debug(f"gripper async move failed: success={success}, position: {position_real}")
                    else:
                        log.debug(f"Hardware gripper async moved to position {position_real:.6f}m in {api_time*1000:.1f}ms")
                        
                    self._last_command = target
                except Exception as e:
                    log.error(f"Error in async gripper command: {e}")
            else:
                # No commands in queue, sleep briefly
                time.sleep(0.001)
                
        log.info(f'Corenetic gripper {self._component_type} stopped its async command worker!')

    def stop_tool(self):
        self._thread_running = False
        if hasattr(self, '_update_thread') and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        if hasattr(self, '_command_thread') and self._command_thread.is_alive():
            self._command_thread.join(timeout=2.0)
        log.info(f'Corenetic gripper component for {self._component_type} stopped')
        
    def get_tool_type_dict(self):
        tool_type_dict = {'single': self._tool_type}
        return tool_type_dict

    def get_force_sensor_data(self) -> list[float]:
        """Get gripper force sensor data [fx, fy, fz, mx, my, mz]"""
        if self.hardware:
            try:
                success, force = self.hardware.get_force_sensor_data(self._component_type)
                if success:
                    return force
                else:
                    log.error(f"Failed to get gripper force, success: {success}")
                    return [0.0] * 6
            except Exception as e:
                log.error(f"Error getting gripper force: {e}")
                return [0.0] * 6
        return [0.0] * 6

    def gripper_grasp(self, torque: float = 1.0):
        """Set gripper to grasp mode with specified torque"""
        if self.hardware:
            try:
                # Switch to torque control mode
                success = self.hardware.set_gripper_mode(self._component_type, GRIPPER_MODE_TORQUE_CTRL)
                log.info(f"set gripper torque mode, success={success}")
                
                if success:
                    success = self.hardware.set_gripper_effort(self._component_type, torque)
                    log.info(f"set gripper effort, success={success}")
                    return success
                else:
                    return False
                    
            except Exception as e:
                log.error(f"gripper grasp exception: {e}")
                return False
        return False
