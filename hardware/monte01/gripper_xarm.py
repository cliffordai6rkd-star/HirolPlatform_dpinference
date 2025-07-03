from xarm.wrapper import XArmAPI
from hardware.base.gripper import GripperBase
from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
from typing import Text, Mapping, Any
import glog as log
import time
XARM_GRIPPER_MAX_POSITION = 800
XARM_GRIPPER_MAX_POSITION_SIM = 8
class Gripper(GripperBase):
    def __init__(self, config: Mapping[Text, Any], ip:str, simulator: Monte01Mujoco, driver_joint_id,driver_joint_name, is_left:bool=True):
        super().__init__()
        self.simulator = simulator
        self.hardware = None
        self.is_left = is_left  # True for left gripper, False for right gripper
        self.driver_joint_id = driver_joint_id
        self.driver_joint_name = driver_joint_name
        self.is_valid = True
        # Initialize hardware only if IP is provided and not in simulation-only mode
        if ip:
            self.hardware = XArmAPI(ip, default_gripper_baud=921600)

            code = self.hardware.set_control_modbus_baudrate(921600)
            print('set gripper baudrate: 921600, code={}'.format(code))
            if 0!=code:
                self.is_valid = False
            # 夹爪初始化
            code = self.hardware.set_gripper_mode(config['ctrl_mode'])
            print('set gripper mode: location mode, code={}'.format(code))
            if 0!=code:
                self.is_valid = False
            # 设置夹爪波特率为 921600
            # code = self.hardware.set_control_modbus_baudrate(921600)
            # print('set gripper baudrate: 921600, code={}'.format(code))

            code = self.hardware.set_gripper_enable(True)
            print('set gripper enable, code={}'.format(code))
            if 0!=code:
                self.is_valid = False
            code = self.hardware.set_gripper_speed(5000)
            print('set gripper speed, code={}'.format(code))
            if 0!=code:
                self.is_valid = False

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
                pos = self.simulator.get_joint_positions([self.driver_joint_name])[0]
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
        if self.hardware:
            try:
                position_real = position * XARM_GRIPPER_MAX_POSITION  # Scale position to 0-800 range
                code = self.hardware.set_gripper_position(position_real, wait=False)
                if code != 0:
                    log.error(f"gripper move FAILED with code: {code}")
                    return False
                else:
                    log.info(f"Hardware gripper moved to position {position_real}")
            except Exception as e:
                log.error(f"gripper move exception: {e}")
                return False
        
        # Control simulation gripper
        if self.simulator:
            # Based on URDF analysis: drive gear joint controls gripper open/close
            # Scale to drive gear joint range (0 to 5.8469 radians)
            drive_joint_value = (1 - position) * XARM_GRIPPER_MAX_POSITION_SIM
            
            # Get driver joint ID 
            driver_joint_id = self.driver_joint_id
            # Direct position setting for less jittering
            try:
                joint_positions = {driver_joint_id: drive_joint_value}
                self.simulator.set_joint_positions(joint_positions)
                
                log.info(f"Simulation gripper moved to position {position} (drive joint: {drive_joint_value}, gripper: {'left' if self.is_left else 'right'})")
                return True
                
            except Exception as e:
                log.error(f"Error controlling gripper in simulation: {e}")
                # Fallback: direct position setting - only control driver joint
                joint_positions = {driver_joint_id: drive_joint_value}
                self.simulator.set_joint_positions(joint_positions)
                return True
        
        # If we reach here, no hardware or simulator was available
        return False

    def gripper_grasp(self, torque: float = 1.0):
        log.warn(f"[gripper_grasp] This method is NOT supported for xArm gripper. Use gripper_move instead.")