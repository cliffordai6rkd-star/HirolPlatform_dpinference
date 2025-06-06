from typing import Text, Mapping, Any
from threading import Thread

from hardware.monte01.arm import Arm
from hardware.base.robot import Robot

from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
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

from .camera import Camera


class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any], use_real_robot=False):
        
        self.robot = None
        
        if use_real_robot:
            self.robot = RobotLib("192.168.11.3:50051", "", "")
            # You can add any post-connection logic here, e.g., logging
            print("Robot connection established.")

        sim = Monte01Mujoco()
    
        # Start the simulation in a separate thread
        sim_thread = Thread(target=sim.start)
        sim_thread.start()

        self._arm_left = Arm(config=config['arm'], hardware_interface=self.robot, simulator=sim)
        self._arm_right = Arm(config=config['arm'], hardware_interface=self.robot, simulator=sim, isLeft=False)
        self.sim_thread = sim_thread
        self.sim = sim

        self.camera = Camera()

    def arm_left(self) -> Arm:
        return self._arm_left
    
    def arm_right(self) -> Arm:
        return self._arm_right
    
    def head_front_camera(self) -> Camera:
        return self.camera
