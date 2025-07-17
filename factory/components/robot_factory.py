from hardware.fr3.fr3_arm import Fr3Arm
from hardware.agibot_g1.agibot_g1 import AgibotG1
from hardware.monte01.monte01 import Monte01
from hardware.fr3.franka_hand import FrankaHand
from hardware.monte01.gripper_xarm import Gripper as Monte01Gripper
from simulation.mujoco.mujoco_sim import MujocoSim
# from hardware.sensors.cameras.realsense_camera import RealsenseCamera
# from hardware.sensors.cameras.agibot_cameras import AgibotCameras
import warnings
import threading
from hardware.base.utils import object_class_check
import numpy as np
from typing import Optional
import glog as log

# Used for loading one complete robot hw system from the factory
class RobotFactory:
    def __init__(self, config):
        self._config = config
        self._use_hardware = config['use_hardware']
        self._use_simulation = config['use_simulation']
        self._robot_type = config['robot']
        self._gripper_type = config['gripper']
        # self.sensor_type = config['sensors']
        self._simulation_type = config["simulation"]
               
        # object classes
        self._robot_classes = {
            'fr3': Fr3Arm,
            'agibot_g1': AgibotG1,
            'monte01': Monte01
        }
       
        self._gripper_classes = {
            'franka_hand': FrankaHand,
            'monte01_gripper': Monte01Gripper
        }
        self._sensor_classes = {}
        self._simulation_classes = {
            'mujoco': MujocoSim
        }
    
    def create_robot_system(self):
        # platforms
        if self._use_hardware:
            if not object_class_check(self._robot_classes, self._robot_type):
                raise ValueError
            self._robot = self._robot_classes[self._robot_type](self._config["robot_config"][self._robot_type])
            
            # Initialize gripper system
            self._grippers = {}
            if self._gripper_type and object_class_check(self._gripper_classes, self._gripper_type):
                gripper_config = self._config["gripper_config"][self._gripper_type]
                
                # Initialize left gripper
                if 'left_gripper' in gripper_config and gripper_config['left_gripper'].get('ip'):
                    self._grippers['left'] = self._gripper_classes[self._gripper_type](gripper_config['left_gripper'])
                    # log.info(f"Initialized left gripper: {self._gripper_type}")
                
                # Initialize right gripper (currently vacuum gripper - not implemented)
                if 'right_gripper' in gripper_config:
                    right_config = gripper_config['right_gripper']
                    if right_config.get('type') == 'vacuum_gripper':
                        log.info(f"Right gripper (vacuum) configured but not implemented yet")
                        # TODO: 实现吸盘夹爪控制
                        # self._grippers['right'] = VacuumGripper(right_config)
                    elif right_config.get('ip'):  # xarm gripper with IP
                        self._grippers['right'] = self._gripper_classes[self._gripper_type](right_config)
                        log.info(f"Initialized right gripper: {self._gripper_type}")
            # sensors
        if self._use_simulation:
            if not object_class_check(self._simulation_classes, self._simulation_type):
                raise ValueError
            self._simulation = self._simulation_classes[self._simulation_type](self._config["simulation_config"][self._simulation_type])
        
        # initialize all objects
        self._initialize()
        
    def _initialize(self):
        if self._use_hardware:
            if not self._robot.initialize():
                raise ValueError(f"robot hardware {self._robot_type} failed intialization")
            
            # Initialize grippers
            if hasattr(self, '_grippers'):
                for side, gripper in self._grippers.items():
                    if gripper.initialize():
                        log.info(f"{side} gripper initialized successfully")
                    else:
                        log.error(f"{side} gripper initialization failed")
                        # Note: Not raising error to allow partial system operation
            
    def get_joint_states(self):
        joint_states = None
        if self._use_simulation:
            joint_states = self._simulation.get_joint_states()
        if self._use_hardware:
            joint_states = self._robot.get_joint_states()
            
        return joint_states
    
    def get_robot_dofs(self):
        if self._use_simulation:
            sim_dofs = self._simulation.get_dof()
            dofs = sim_dofs
        if self._use_hardware:
            hw_dofs = self._robot.get_dof()
            if not isinstance(hw_dofs, list):
                dofs = [hw_dofs]

        if self._use_hardware and self._use_simulation:
            if sim_dofs != hw_dofs:
                raise ValueError("the simulation dofs did not match with hw dofs: "
                                 f"sim: {sim_dofs}, hw: {hw_dofs}")
        return dofs
    
    def get_gripper(self, side: str = 'left'):
        """Get gripper object for specified side (left/right)"""
        if hasattr(self, '_grippers') and side in self._grippers:
            return self._grippers[side]
        return None
    
    def get_all_grippers(self):
        """Get all available grippers"""
        if hasattr(self, '_grippers'):
            return self._grippers
        return {}
    
    def control_gripper(self, side: str, position: float) -> bool:
        """Control gripper position (0.0=close, 1.0=open)"""
        gripper = self.get_gripper(side)
        if gripper and gripper.valid():
            return gripper.gripper_move(position)
        else:
            log.warning(f"Gripper {side} not available or invalid")
            return False
        
    def set_joint_commands(self, joint_command, mode, execute_hardware: bool):
        # print('set_joint_commands ...')
        dofs = self.get_robot_dofs()
        # print(f'mode: {mode}, command: {joint_command}, dofs: {dofs}')
        if self._use_simulation:
            # mode assignment - handle dual arm case
            if len(dofs) > 1 and len(mode) > 1:
                # Dual arm with separate modes
                sim_mode = [mode[0]] * dofs[0]
                sim_mode_r = [mode[1]] * dofs[1]
                sim_mode = np.hstack((sim_mode, sim_mode_r))
            else:
                # Single arm or dual arm with same mode
                total_dofs = sum(dofs)
                sim_mode = [mode[0]] * total_dofs
            self._simulation.set_joint_command(sim_mode, joint_command)
        # log.info(f"[DEBUG] set_joint_commands called: _use_hardware={self._use_hardware}, execute_hardware={execute_hardware}")
        if self._use_hardware and execute_hardware:
            # log.info(f"Set joint command for hardware {self._robot_type} with mode: {mode}, command: {joint_command}")
            #TODO: check
            # if len(mode) == 1: 
            #     mode = mode[0]
            self._robot.set_joint_command(mode, joint_command)
            # log.warn(f"Hardware joint command execution not implemented yet for {self._robot_type}")
            pass
        else:
            log.info(f"[DEBUG] Hardware command skipped: use_hardware={self._use_hardware}, execute_hardware={execute_hardware}")
    
    def get_body_positions(self) -> Optional[np.ndarray]:
        """
        Get body joint positions from hardware
        
        Returns:
            Optional[np.ndarray]: 5D body positions [body_joint_1, body_joint_2, body_joint_3, head_joint_1, head_joint_2]
                                 None if hardware not available or not supported
        """
        if not self._use_hardware:
            return None
            
        if hasattr(self._robot, 'get_body_positions'):
            try:
                body_positions = self._robot.get_body_positions()
                log.debug(f"[BodySync] Got hardware body positions: {body_positions}")
                return body_positions
            except Exception as e:
                log.warning(f"[BodySync] Failed to get hardware body positions: {e}")
                return None
        else:
            log.debug(f"[BodySync] Robot {self._robot_type} does not support body position reading")
            return None
    
    def sync_body_positions(self) -> bool:
        """
        Synchronize hardware body positions to simulation
        
        Returns:
            bool: True if sync successful, False otherwise
        """
        if not (self._use_hardware and self._use_simulation):
            return False
            
        # Get body positions from hardware
        body_positions = self.get_body_positions()
        if body_positions is None:
            return False
            
        # Check if simulation supports body joint control
        if not hasattr(self._simulation, 'set_body_joint_command'):
            log.warning("[BodySync] Simulation does not support body joint control")
            return False
            
        # Sync to simulation
        try:
            success = self._simulation.set_body_joint_command(body_positions)
            if success:
                log.debug(f"[BodySync] Successfully synced body positions to simulation")
            return success
        except Exception as e:
            log.warning(f"[BodySync] Failed to sync body positions: {e}")
            return False
            