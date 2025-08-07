from hardware.base.arm import ArmBase
from hardware.base.tool_base import ToolBase
from hardware.base.utils import ToolType
from hardware.base.camera import CameraBase
from simulation.base.sim_base import SimBase
from hardware.fr3.fr3_arm import Fr3Arm
from hardware.duo_arm import DuoArm
from hardware.agibot_g1.agibot_g1 import AgibotG1
from hardware.monte01.monte01 import Monte01
from hardware.fr3.franka_hand import FrankaHand
from hardware.duo_tool import DuoTool
from simulation.mujoco.mujoco_sim import MujocoSim
from hardware.base.camera import CameraBase
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from hardware.sensors.cameras.opencv_camera import OpencvCamera
from hardware.sensors.cameras.agibot_cameras import AgibotCamera
from hardware.sensors.cameras.ros2_camera import Ros2Camera
import warnings
import threading
from hardware.base.utils import object_class_check
from controller.utils.weighted_moving_filter import WeightedMovingFilter
import numpy as np
from typing import Optional
import glog as log

# Used for loading one complete robot hw system from the factory
class RobotFactory:
    _robot: ArmBase
    _tool: ToolBase
    _simulation: SimBase
    def __init__(self, config):
        self._config = config
        self._use_hardware = config['use_hardware']
        self._use_simulation = config['use_simulation']
        self._robot_type = config['robot']
        self._gripper_type = config['gripper']
        self._simulation_type = config["simulation"]
        self._sensor_dicts = config.get("sensor_dicts", None)
        self._sensors = {}
        self._tool = None
            
        # object classes
        self._robot_classes = {
            'fr3': Fr3Arm,
            'agibot_g1': AgibotG1,
            'monte01': Monte01,
            'duo_arm': DuoArm
        }
       
        self._gripper_classes = {
            'franka_hand': FrankaHand,
            'duo_tool': DuoTool
        }
        
        self._camera_classes = {
            'realsense_camera': RealsenseCamera,
            'opencv_camera': OpencvCamera,
            'agibot_camera': AgibotCamera,
            'ros2_camera': Ros2Camera
        }
        
        self._simulation_classes = {
            'mujoco': MujocoSim
        }
    
    def create_robot_system(self):
        # platforms
        if self._use_hardware:
            if not object_class_check(self._robot_classes, self._robot_type):
                raise ValueError
            self._robot = self._robot_classes[self._robot_type](self._config["robot_config"][self._robot_type])
            
            # Initialize tool system - support both legacy _tool and modern _grippers
            # @TODO: get tool
            if object_class_check(self._gripper_classes, self._gripper_type):
                gripper_config = self._config["gripper_config"][self._gripper_type]
                
                # Check if this is a single tool (legacy) or dual gripper system
                if 'left_gripper' in gripper_config or 'right_gripper' in gripper_config:
                    # Modern dual gripper system
                    self._grippers = {}
                    
                    # Initialize left gripper
                    if 'left_gripper' in gripper_config and gripper_config['left_gripper'].get('ip'):
                        self._grippers['left'] = self._gripper_classes[self._gripper_type](gripper_config['left_gripper'])
                        log.info(f"Initialized left gripper: {self._gripper_type}")
                    
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
            # @TODO: get tool
            self._tool = self._gripper_classes[self._gripper_type](self._config["gripper_config"][self._gripper_type])
            
            # sensors
            if self._sensor_dicts is not None:
                if 'cameras' in self._sensor_dicts:
                    # cameras
                    cameras_info = self._sensor_dicts["cameras"]
                    cameras_objects = []
                    num_camera = 0
                    for cam_info in cameras_info:
                        if not object_class_check(self._camera_classes, cam_info['type']):
                            raise ValueError
                        cam_type = cam_info['type']
                        cam = self._camera_classes[cam_type](cam_info['cfg'][cam_type])
                        cameras_objects.append({'name': cam_info['name'], 'object': cam})
                        log.info(f"Add one hw camera {cam_info['name']}")
                        num_camera += 1
                    if num_camera:
                        self._sensors['camera'] = cameras_objects  
                # tactile 
                
                # FT
                
        if self._use_simulation:
            if not object_class_check(self._simulation_classes, self._simulation_type):
                raise ValueError
            self._simulation = self._simulation_classes[self._simulation_type](self._config["simulation_config"][self._simulation_type])
        
        # total dof
        total_dof = self.get_total_dofs()
        self.filter = WeightedMovingFilter([1.0/total_dof] * total_dof, total_dof)
        
        # initialize all objects
        self._initialize()
        
    def _initialize(self):
        if self._use_hardware:
            if not self._robot.initialize():
                raise ValueError(f"robot hardware {self._robot_type} failed intialization")
                    
            if hasattr(self, '_grippers'):
                for side, gripper in self._grippers.items():
                    if gripper.initialize():
                        log.info(f"{side} gripper initialized successfully")
                    else:
                        log.error(f"{side} gripper initialization failed")
                        # Note: Not raising error to allow partial system operation
                        
            # Initialize sensors
            if not self._tool.initialize():
                raise ValueError(f"tool hardware {self._gripper_type} failed intialization")
            if len(self._sensors) != 0:
                possible_sensor_types = ['camera', 'FT_sensor', 'tactile', 'imu']
                for sensor_type in possible_sensor_types:
                    if sensor_type in self._sensors:
                        for sensor in self._sensors[sensor_type]:
                            success = sensor["object"].initialize()
                            sensor_name = sensor["name"]
                            if not success:
                                raise ValueError(f"{sensor_type} {sensor_name} failed to initialize!!!")
                
    def get_joint_states(self):
        joint_states = None
        if self._use_simulation:
            joint_states = self._simulation.get_joint_states()
        if self._use_hardware:
            joint_states = self._robot.get_joint_states()
            
        return joint_states
    
    def get_tool_dict_state(self):
        if self._tool is None:
            return None
        
        cur_tool_state = self._tool.get_tool_state()
        if not isinstance(cur_tool_state, dict):
            tool_state = {"single": cur_tool_state}
        else: tool_state = cur_tool_state
        return tool_state
        
    def get_robot_dofs(self):
        if self._use_simulation:
            sim_dofs = self._simulation.get_dof()
            dofs = sim_dofs
        if self._use_hardware:
            hw_dofs = self._robot.get_dof()

        if self._use_hardware and self._use_simulation:
            if sim_dofs != hw_dofs:
                raise ValueError("the simulation dofs did not match with hw dofs: "
                                 f"sim: {sim_dofs}, hw: {hw_dofs}")
        return dofs
    
    def get_total_dofs(self):
        dofs = self.get_robot_dofs()
        total_dof = 0
        for dof in dofs:
            total_dof += dof
        return total_dof
    
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
        
    def set_joint_commands(self, joint_command, mode, execute_hardware: bool = False):
        # log.info(f'Set joint commands: {joint_command}, dim: {len(joint_command)},mode: {mode}')
        # filter command
        self.filter.add_data(joint_command)
        joint_command = self.filter.filtered_data
        
        dofs = self.get_robot_dofs()
        # log.info(f'mode: {mode}, command: {joint_command}')
        if self._use_simulation:
            # mode assignment
            sim_mode = [mode[0]] * dofs[0]
            if len(dofs) > 1:
                # Use mode[1] if available, otherwise use mode[0] for both arms
                # @TOOD: bug
                right_mode = mode[1] if len(mode) > 1 else mode[0]
                sim_mode_r = [right_mode] * dofs[1]
                sim_mode = np.hstack((sim_mode, sim_mode_r))
            total_dof = self.get_total_dofs()
            sim_mode = [mode[0]] * total_dof
            self._simulation.set_joint_command(sim_mode, joint_command)
        if self._use_hardware and execute_hardware: 
            if len(mode) == 1:
                mode = mode[0]
            self._robot.set_joint_command(mode, joint_command)
            
    def set_tool_command(self, tool_command):
        """
        Send tool command to the tool system.
        
        Args:
            tool_command: Tool control data from teleoperation interface
        """
        if not self._use_hardware:
            return 
        
        # Direct pass-through to tool layer which handles control mode logic
        self._tool.set_tool_command(tool_command)

    def close(self):
        if self._use_simulation:
            self._simulation.close()
        if self._use_hardware:
            self._robot.close()
            # @TODO: tools gradually add
            self._tool.stop_tool()
            log.info(f'All tools are successfully closed!!!')
            if len(self._sensors):
                # @TODO: gradually add the sensors
                possible_sensors = ['camera', 'FT_sensor', 'tactile', 'imu']
                for sensor_type in possible_sensors:
                    if sensor_type in self._sensors:
                        for sensor in self._sensors[sensor_type]:
                            sensor['object'].close()
                log.info(f'All sensors are closed successfully!!!!')
        log.info(f'Robot systyem is closed successfully!!!')
        
    def get_cameras_infos(self):
        cameras_data = None
        if self._use_simulation:
            cameras_data = self._simulation.get_all_camera_images()
        if'camera' in self._sensors and self._use_hardware:
            hw_camera_data = []
            cameras = self._sensors['camera']
            for cam in cameras:
                camera_name = cam['name']
                camera_object:CameraBase = cam['object']
                img = camera_object.capture_all_data()
                resolution = camera_object.get_resolution()
                if not img['image'] is None:
                    hw_camera_data.append({'name': camera_name+'_color', 'resolution': resolution,
                                        'img': img['image']})
                if not img['depth_map'] is None:
                    hw_camera_data.append({'name': camera_name+'_depth', 'resolution': resolution,
                                        'img': img['depth_map']})
                if not img['imu'] is None:
                    hw_camera_data.append({'name': camera_name+'_imu', 'resolution': resolution,
                                        'imu': img['imu']})
            if len(hw_camera_data):
                cameras_data = hw_camera_data
        return cameras_data
            
    def move_to_start(self):
        self._robot.move_to_start()
