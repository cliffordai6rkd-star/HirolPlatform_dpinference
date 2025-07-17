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

from typing import Text, Mapping, Any, Optional
import glog as log
import numpy as np
from motion.kinematics import PinocchioKinematicsModel as KinematicsModel

# Body joint IDs
BODY_JOINT_IDS = [1, 2, 3]

# position keys
WAIST_YAW = 1
WAIST_PITCH = 2
KNEE_PITCH = 3

BODY_KEYS_STRIDE = KNEE_PITCH

HEAD_PITCH = 1
HEAD_YAW = 2

class Trunk():
    def __init__(self, hardware_interface: Optional[RobotLib]):
        super().__init__()
        self.robot = hardware_interface
        
        # Initialize body joint names for kinematics
        self.body_joint_names = ['body_joint_1', 'body_joint_2', 'body_joint_3']
        
        # Cache for chest_to_world transformation
        self._cached_chest_to_world = None
        self._last_body_joints_for_cache = None
        
        # Initialize body kinematics if URDF path is provided
        self.body_kinematics = None
        try:
            urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets/monte_01/urdf/robot_description.urdf'))
            self.body_kinematics = KinematicsModel(urdf_path=urdf_path, base_link='base_link', end_effector_link='chest_link')
            log.info("Body kinematics initialized for coordinate transformations")
        except Exception as e:
            log.error(f"Failed to initialize body kinematics: {e}")
            self.body_kinematics = None
        
        if hardware_interface is not None:
            success = hardware_interface.set_trunk_joint_enable(True)
            if not success:
                log.error(f"set_trunk_joint_enable FAILED.")

            success = hardware_interface.set_head_joint_enable(True)
            if not success:
                log.error(f"set_head_joint_enable FAILED.")

            self.print_state()

    def set_trunk_joint_positions(self, positions_map: dict):
        """
            {
                id(int): position(radian), 
                ...
            }
        """
        if self.robot is not None:
            success = self.robot.set_trunk_joint_positions(list(positions_map.keys()), list(positions_map.values()))
            if not success:
                log.error(f"set_trunk_joint_positions FAILED!")

    def set_head_joint_positions(self, positions_map: dict):
        """
            {
                id(int): position(radian), 
                ...
            }
        """
        if self.robot is not None:
            success = self.robot.set_head_joint_positions(list(positions_map.keys()), list(positions_map.values()))
            if not success:
                log.error(f"set_head_joint_positions FAILED!")

    def get_body_joint_positions(self) -> np.ndarray:
        """Get body joint positions from robot or simulator"""
        if self.robot is not None:
            
            success, positions, _, _ = self.robot.get_joint_state(BODY_JOINT_IDS)
            if success and len(positions) == 0:
                log.debug("Received empty positions from robot, using zero positions")
                return np.zeros(len(BODY_JOINT_IDS))
            log.debug(f"[DEBUG] get_body_joint_positions from robot: success={success}, positions={positions}")
            log.debug(f"[DEBUG] BODY_JOINT_IDS: {BODY_JOINT_IDS}")
            log.debug(f"[DEBUG] Expected positions length: {len(BODY_JOINT_IDS)}")
            if positions is not None:
                log.debug(f"[DEBUG] Actual positions length: {len(positions)}")
            
            if not success or positions is None or len(positions) == 0:
                log.warning(f"Failed to get body joint positions or empty positions: success={success}, positions={positions}")
                log.warning(f"Using zero positions for body joints: {BODY_JOINT_IDS}")
                return np.zeros(len(BODY_JOINT_IDS))
            
            if len(positions) != len(BODY_JOINT_IDS):
                log.error(f"Position length mismatch! Expected {len(BODY_JOINT_IDS)}, got {len(positions)}")
                return np.zeros(len(BODY_JOINT_IDS))
                
            return np.array(positions)
        else:
            if hasattr(self, 'body_joint_names') and self.body_joint_names is not None:
                positions = self.simulator.get_joint_positions(self.body_joint_names)
                log.debug(f"[DEBUG] get_body_joint_positions from simulator: positions={positions}")
                return positions
            else:
                log.warning("No body_joint_names available in simulator mode")
                return np.zeros(len(BODY_JOINT_IDS))
    
    def get_chest_to_world_transform(self) -> np.ndarray:
        """Get transformation matrix from chest frame to world frame with caching"""
        try:
            if self.body_kinematics is not None:
                current_body_joints = self.get_body_joint_positions()
                
                # Check if we can use cached transformation
                if (self._cached_chest_to_world is not None and 
                    self._last_body_joints_for_cache is not None and
                    np.allclose(current_body_joints, self._last_body_joints_for_cache, atol=1e-3)):
                    return self._cached_chest_to_world
                
                # Compute new transformation and update cache
                if len(current_body_joints) == 0:
                    log.error("Empty body positions array, cannot compute FK")
                    return np.eye(4)
                    
                transform = self.body_kinematics.fk(current_body_joints)
                self._cached_chest_to_world = transform
                self._last_body_joints_for_cache = current_body_joints.copy()
                
                return transform
            else:
                log.warning("Body kinematics not available, returning identity matrix")
                return np.eye(4)
        except Exception as e:
            log.error(f"Failed to compute world to chest transform: {e}")
            return np.eye(4)
    
    def set_body_joint_positions(self, positions: np.ndarray):
        """Set body joint positions"""
        if self.robot is not None:
            # Use trunk joint position setting for real robot
            positions_map = {joint_id: positions[i] for i, joint_id in enumerate(BODY_JOINT_IDS)}
            self.set_trunk_joint_positions(positions_map)
        
        if self.simulator is not None:
            # Set body joint positions in simulator
            target_positions = {BODY_JOINT_IDS[i]: positions[i] for i in range(len(positions))}
            self.simulator.set_joint_positions(target_positions)

    def print_state(self):
        if self.robot is not None:
            success, pos, vel, effort = self.robot.get_joint_state([WAIST_YAW, WAIST_PITCH, KNEE_PITCH])
            log.info(f"Trunk state: \nsuccess: {success}\npos: {pos}\nvel: {vel}\neffort: {effort}\n")

            success, pos, vel, effort = self.robot.get_joint_state([HEAD_PITCH + BODY_KEYS_STRIDE, HEAD_YAW + BODY_KEYS_STRIDE])
            log.info(f"Head state: \nsuccess: {success}\npos: {pos}\nvel: {vel}\neffort: {effort}\n")