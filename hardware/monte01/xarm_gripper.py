from hardware.base.tool_base import ToolBase
from hardware.monte01.xarm_api_manager import XArmAPIManager
import glog as log
from hardware.base.utils import ToolType
import time, threading
import numpy as np
from hardware.monte01.xarm_defs import *

class XArmGripper(ToolBase):
    _tool_type: ToolType = ToolType.GRIPPER
    def __init__(self, config: dict):
        self._ip = config["ip"]
        self._grasp_speed = config.get("grasp_speed", 5000)
        super().__init__(config)
        self._state._tool_type = self._tool_type
        self._last_command = 0.08
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
        # Get shared XArmAPI instance
        self.hardware = XArmAPIManager.get_instance(self._ip)
        
        if self.hardware is None:
            log.error(f"Failed to get shared XArmAPI instance for gripper at {self._ip}")
            self.is_valid = False
            self._is_initialized = False
            return False
            
        log.info(f"XArm gripper using shared API instance for {self._ip}")
        
        # Initialize gripper-specific settings
        try:
            code, version = self.hardware.get_gripper_version()
            if code == XARM_SUCCESS:
                log.info(f"Gripper version: {version}")
            else:
                log.warning(f"Failed to get gripper version, code: {code}")
        except Exception as e:
            log.warning(f"Could not get gripper version: {e}")
        
        try:
            self.hardware.clean_gripper_error()
            log.info("Gripper errors cleaned")
        except Exception as e:
            log.warning(f"Could not clean gripper errors: {e}")
            
        # Start the update thread
        # self._update_thread.start()
        # log.info("XArm gripper update thread started")

        # Try to initialize, but don't fail completely on individual errors
        init_errors = []
        
        code = self.hardware.set_control_modbus_baudrate(921600)
        log.info('set gripper baudrate: 921600, code={}'.format(code))
        if 0 != code:
            init_errors.append(f"baudrate: {code}")
        
        code = self.hardware.set_gripper_mode(XARM_GRIPPER_MODE_LOCATION)
        log.info('set gripper mode: location mode, code={}'.format(code))
        if 0 != code:
            init_errors.append(f"mode: {code}")

        code = self.hardware.set_gripper_enable(True)
        log.info('set gripper enable, code={}'.format(code))
        if 0 != code:
            init_errors.append(f"enable: {code}")
            
        code = self.hardware.set_gripper_speed(5000)
        log.info('set gripper speed, code={}'.format(code))
        if 0 != code:
            init_errors.append(f"speed: {code}")
        
        if init_errors:
            log.warning(f"Gripper initialization warnings: {init_errors}")
            # Still keep gripper valid for basic operations
            self.is_valid = True
        else:
            log.info("Gripper initialized successfully")
            
        # Set initialization flag and return success
        self._is_initialized = True
        return True

    def set_hardware_command(self, command):
        self._set_binary_command(command)

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
#         code, status = self.hardware.get_gripper_status()
#         if code != XARM_SUCCESS:
#             log.error(f"Failed to get gripper status, code: {code}")
# #         status & 0x03 == 0: stop state
# #         status & 0x03 == 1: move state
# #         status & 0x03 == 2: grasp state
#         log.info(f"Gripper status: {status & 0x03}")

        code, pos = self.hardware.get_gripper_position()
        if code != XARM_SUCCESS:
            log.error(f'gripper communication test failed, code: {code}')
        else:
            # 真机模式：将0-800范围的原始值转换为0-1范围的归一化值
            pos = pos / XARM_GRIPPER_MAX_POSITION
            # 确保值在0-1范围内
            pos = np.clip(pos, 0.0, 1.0)
            log.info(f"Hardware gripper position: raw={pos*XARM_GRIPPER_MAX_POSITION:.1f}, normalized={pos:.4f}")
            self._state._position = pos
        return self._state
    
    def update_state(self):
        log.info(f'Starting state updating thread for xArm gripper {self._ip}')
        
        read_frequency = 10  # 10Hz update rate for gripper
        target_period = 1.0 / read_frequency
        next_update_time = time.perf_counter()
        
        while self._thread_running:
            loop_start_time = time.perf_counter()
            
            try:
                # Get both gripper position and status
                code_pos, position = self.hardware.get_gripper_position()
                # code_status, status = self.hardware.get_gripper_status()
                
                with self._lock:
                    # Update position if successful
                    if code_pos == XARM_SUCCESS:
                        # Convert from 0-800 range to 0-1 normalized range
                        normalized_position = position / XARM_GRIPPER_MAX_POSITION
                        normalized_position = np.clip(normalized_position, 0.0, 1.0)
                        self._state._position = normalized_position
                    else:
                        log.debug(f'Failed to get gripper position: code={code_pos}')
                    
                    # Update grasp status using gripper status
                    # if code_status == XARM_SUCCESS:
                    #     # status & 0x03: 0=stop, 1=move, 2=grasp
                    #     gripper_state = status & 0x03
                    #     self._state._is_grasped = (gripper_state == 2)  # 2 means grasp state
                    #     log.debug(f'Gripper state: {gripper_state} (0=stop, 1=move, 2=grasp)')
                    # else:
                        # log.debug(f'Failed to get gripper status: code={code_status}')
                        # Fallback: use position-based heuristic
                    if hasattr(self._state, '_position'):
                        self._state._is_grasped = (self._state._position < 0.1) #TODO impl this
                        
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
                    log.warn(f'The xArm gripper {self._ip} could not reach the update thread frequency!! '
                           f'Target: {read_frequency:.1f}Hz, Actual: {1.0/actual_period:.1f}Hz')
                # Reset timing to prevent drift
                next_update_time = current_time
                
        log.info(f'xArm gripper {self._ip} stopped its update thread!!!!')
    
    def _async_command_worker(self):
        """Background worker for processing gripper commands asynchronously"""
        log.info(f'Starting async command worker for xArm gripper {self._ip}')
        
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
                    position_real = target * XARM_GRIPPER_MAX_POSITION
                    api_start = time.perf_counter()
                    code = self.hardware.set_gripper_position(position_real, wait=False)
                    api_time = time.perf_counter() - api_start
                    
                    if code != XARM_SUCCESS:
                        log.warn(f"gripper async move got code: {code}, position: {position_real}")
                    else:
                        log.debug(f"Hardware gripper async moved to position {position_real} in {api_time*1000:.1f}ms")
                        
                    self._last_command = target
                except Exception as e:
                    log.error(f"Error in async gripper command: {e}")
            else:
                # No commands in queue, sleep briefly
                time.sleep(0.001)
                
        log.info(f'xArm gripper {self._ip} stopped its async command worker!')

    def stop_tool(self):
        self._thread_running = False
        if hasattr(self, '_update_thread') and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        log.info(f'XArm gripper component for {self._ip} stopped')
        
    def get_tool_type_dict(self):
        tool_type_dict = {'single': self._tool_type}
        return tool_type_dict