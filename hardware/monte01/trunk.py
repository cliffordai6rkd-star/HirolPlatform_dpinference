from hardware.base.trunk import TrunkBase
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
from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco

from typing import Text, Mapping, Any, Callable, Sequence, Union
import glog as log

# position keys
WAIST_YAW = 1
WAIST_PITCH = 2
KNEE_PITCH = 3

BODY_KEYS_STRIDE = KNEE_PITCH

HEAD_PITCH = 1
HEAD_YAW = 2

class Trunk(TrunkBase):
    def __init__(self, config: Mapping[Text, Any], hardware_interface: RobotLib, simulator: Monte01Mujoco):
        super().__init__()
        self.robot = hardware_interface
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
        success = self.robot.set_head_joint_positions(list(positions_map.keys()), list(positions_map.values()))
        if not success:
            log.error(f"set_head_joint_positions FAILED!")

    def print_state(self):
        success, pos, vel, effort = self.robot.get_joint_state([WAIST_YAW, WAIST_PITCH, KNEE_PITCH])
        log.info(f"Trunk state: \nsuccess: {success}\npos: {pos}\nvel: {vel}\neffort: {effort}\n")

        success, pos, vel, effort = self.robot.get_joint_state([HEAD_PITCH + BODY_KEYS_STRIDE, HEAD_YAW + BODY_KEYS_STRIDE])
        log.info(f"Head state: \nsuccess: {success}\npos: {pos}\nvel: {vel}\neffort: {effort}\n")