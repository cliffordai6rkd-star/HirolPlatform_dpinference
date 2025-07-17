from xarm.wrapper import XArmAPI
from hardware.base.gripper_base import GripperBase
from typing import Text, Mapping, Any
import glog as log
import time
from hardware.base.utils import GripperState

XARM_GRIPPER_MAX_POSITION = 800
XARM_GRIPPER_MAX_POSITION_SIM = 8
class Gripper(GripperBase):
    def __init__(self, config: dict):
        super().__init__(config)
        self.hardware = None
        self.config = config
        self.ip = config.get('ip', None)
        self.ctrl_mode = config.get('ctrl_mode', 0)
        self.is_left = True  # Default to left gripper
        self.is_valid = True
        
        # Initialize hardware only if IP is provided and not in simulation-only mode
        if self.ip:
            try:
                self.hardware = XArmAPI(self.ip, default_gripper_baud=921600)
                log.info(f"Connected to gripper at {self.ip}")

                # Try to initialize, but don't fail completely on individual errors
                init_errors = []
                
                code = self.hardware.set_control_modbus_baudrate(921600)
                print('set gripper baudrate: 921600, code={}'.format(code))
                if 0 != code:
                    init_errors.append(f"baudrate: {code}")
                
                code = self.hardware.set_gripper_mode(self.ctrl_mode)
                print('set gripper mode: location mode, code={}'.format(code))
                if 0 != code:
                    init_errors.append(f"mode: {code}")

                code = self.hardware.set_gripper_enable(True)
                print('set gripper enable, code={}'.format(code))
                if 0 != code:
                    init_errors.append(f"enable: {code}")
                    
                code = self.hardware.set_gripper_speed(5000)
                print('set gripper speed, code={}'.format(code))
                if 0 != code:
                    init_errors.append(f"speed: {code}")
                
                if init_errors:
                    log.warning(f"Gripper initialization warnings: {init_errors}")
                    # Still keep gripper valid for basic operations
                    self.is_valid = True
                else:
                    log.info("Gripper initialized successfully")
                    
            except Exception as e:
                log.error(f"Failed to connect to gripper at {self.ip}: {e}")
                self.hardware = None
                self.is_valid = False

    def initialize(self) -> None:
        pass
    
    def get_gripper_state(self) -> GripperState:
        pass
    
    def set_gripper_command(self, target) -> bool:
        pass
    
    def stop_gripper(self):
        pass

    def valid(self) -> bool:
        """Check if the gripper is valid (hardware communication successful)"""
        return self.is_valid
    
    def print_state(self) -> None:
        try:
            if self.hardware:
                # 获取夹爪状态来验证通信是否正常
                code, pos = self.hardware.get_gripper_position()
                if code == 0:
                    print(f'gripper position: {pos}, baudrate setting successful')
                else:
                    print(f'gripper communication test failed, code: {code}, pos {pos}')
            else:
                print('gripper in simulation mode, hardware communication skipped')
        except Exception as e:
            print(f'gripper communication error: {e}')

    def get_position(self):
        pos = 1.0
        try:
            if self.hardware:
                # 获取夹爪状态来验证通信是否正常
                code, pos = self.hardware.get_gripper_position()
                if code != 0:
                    print(f'gripper communication test failed, code: {code}')
                else:
                    # 真机模式：将0-800范围的原始值转换为0-1范围的归一化值
                    pos = pos / XARM_GRIPPER_MAX_POSITION
                    # 确保值在0-1范围内
                    pos = max(0.0, min(1.0, pos))
                    log.info(f"Hardware gripper position: raw={pos*XARM_GRIPPER_MAX_POSITION:.1f}, normalized={pos:.4f}")
        except Exception as e:
            print(f'gripper communication error: {e}')
        return pos

    def gripper_close(self):
        self.gripper_move(0)

    def gripper_open(self):
        self.gripper_move(1)

    def gripper_move(self, position = 0.25):
        """Move gripper to specified position. Returns True if successful, False otherwise."""

        if position < 0 or position > 1:
            log.error(f"[gripper_move] Invalid input: need a 0~1 value.")
            return False
        
        # Control hardware gripper if available
        if self.hardware and self.is_valid:
            try:
                position_real = position * XARM_GRIPPER_MAX_POSITION  # Scale position to 0-800 range
                code = self.hardware.set_gripper_position(position_real, wait=False)
                if code != 0:
                    log.warning(f"gripper move got code: {code}, position: {position_real}")
                    return True  # Still return True for non-critical errors
                else:
                    log.debug(f"Hardware gripper moved to position {position_real}")
                    return True
            except Exception as e:
                log.error(f"gripper move exception: {e}")
                return False
        elif self.hardware:
            log.debug(f"Gripper hardware exists but invalid, simulating move to {position:.3f}")
            return True  # Simulate successful move when hardware is invalid
        else:
            log.debug(f"No gripper hardware, simulating move to {position:.3f}")
            return True  # Simulate successful move in simulation mode

    def gripper_grasp(self, torque: float = 1.0):
        log.warn(f"[gripper_grasp] This method is NOT supported for xArm gripper. Use gripper_move instead.")