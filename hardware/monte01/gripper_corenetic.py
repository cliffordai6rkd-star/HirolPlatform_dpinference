import importlib.util
import os
from .defs import ROBOTLIB_SO_PATH
spec = importlib.util.spec_from_file_location(
    "RobotLib", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), ROBOTLIB_SO_PATH))
)
RobotLib_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RobotLib_module)
RobotLib = RobotLib_module.Robot

from hardware.base.gripper import GripperBase
from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
from typing import Text, Mapping, Any
import glog as log
import time

from .defs import *

CORENETIC_GRIPPER_MAX_POSITION = 0.074  # Maximum position for Corenetic gripper (0 to 0.074 meters)

# Gripper joint names in simulation
LEFT_GRIPPER_JOINT = "left_drive_gear_joint"
RIGHT_GRIPPER_JOINT = "right_drive_gear_joint"

# All gripper joint IDs (driver + follower joints)
LEFT_GRIPPER_JOINT_IDS = [13, 14, 15, 16, 17, 18, 19]  # left_drive_gear + all follower joints
RIGHT_GRIPPER_JOINT_IDS = [27, 28, 29, 30, 31, 32, 33]  # right_drive_gear + all follower joints

# Driver joint IDs (main control joints)
LEFT_DRIVER_JOINT_ID = 13   # left_drive_gear_joint
RIGHT_DRIVER_JOINT_ID = 27  # right_drive_gear_joint
class Gripper(GripperBase):
    def __init__(self, config: Mapping[Text, Any], ip:str, simulator: Monte01Mujoco, is_left:bool=True):
        super().__init__()
        self.simulator = simulator
        self.hardware = None
        self.is_left = is_left  # True for left gripper, False for right gripper
        self.component_type = is_left and COM_TYPE_LEFT or COM_TYPE_RIGHT
        # Initialize hardware only if IP is provided and not in simulation-only mode
        if ip:
            self.hardware = RobotLib("192.168.11.3:50051", "", "")

            success = self.hardware.set_gripper_enable(self.component_type, GRIPPER_ENABLE)
            log.info(f"set gripper enable, code={success}")

            success = self.hardware.set_gripper_mode(self.component_type, GRIPPER_MODE_POSITION_CTRL)
            log.info(f"set gripper mode, code={success}")

        self.print_state()

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
                position_real = position * CORENETIC_GRIPPER_MAX_POSITION  # Scale position to 0-0.074 meters
                success = self.hardware.set_gripper_position(self.component_type, position_real)

                if not success:
                    log.error(f"gripper move FAILED!")
                    return False
                else:
                    log.debug(f"Hardware gripper moved to position {position_real}")
            except Exception as e:
                log.error(f"gripper move exception: {e}")
                return False
        
        # Control simulation gripper
        if self.simulator:
            # Based on URDF analysis: drive gear joint controls gripper open/close
            # Scale to drive gear joint range (0 to 5.8469 radians)
            drive_joint_value = position * 5.8469  # 0 (closed) to 5.8469 (open)
            
            # Get driver joint ID and name
            driver_joint_id = LEFT_DRIVER_JOINT_ID if self.is_left else RIGHT_DRIVER_JOINT_ID
            joint_name = LEFT_GRIPPER_JOINT if self.is_left else RIGHT_GRIPPER_JOINT
            
            # Get current position for smooth movement
            try:
                current_pos = self.simulator.get_joint_positions([joint_name])[0]
                
                # Use smooth movement with smaller steps
                n = 5  # Reduce number of steps for much faster movement
                for i in range(n + 1):
                    # Smoothly interpolate from current to target position
                    drive_val = current_pos + (drive_joint_value - current_pos) * (i / n)
                    
                    # Only control the driver joint - follower joints should follow via equality constraints
                    joint_positions = {driver_joint_id: drive_val}
                    self.simulator.set_joint_positions(joint_positions)
                    time.sleep(0.005)  # 5ms per step, much faster
                
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

        