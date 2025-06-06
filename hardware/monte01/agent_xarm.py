from typing import Text, Mapping, Any
from threading import Thread

from hardware.monte01.arm_xarm import Arm
from hardware.base.robot import Robot

from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
import os

from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
from xarm.wrapper import XArmAPI

LEFT_ARM_IP = '192.168.11.11'
RIGHT_ARM_IP = '192.168.11.12'
class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any], use_real_robot=False):
        
        self.robot = None
        
        self.hil = None
        self.hir = None
        if use_real_robot:
            self.hil = XArmAPI(LEFT_ARM_IP)
            self.hir = XArmAPI(RIGHT_ARM_IP)
            # You can add any post-connection logic here, e.g., logging
            print("Robot connection established.")

        sim = Monte01Mujoco()
    
        # Start the simulation in a separate thread
        sim_thread = Thread(target=sim.start)
        sim_thread.start()

        self._arm_left = Arm(config=config['arm'], hardware_interface=self.hil, simulator=sim)
        self._arm_right = Arm(config=config['arm'], hardware_interface=self.hir, simulator=sim, isLeft=False)
        self.sim_thread = sim_thread
        self.sim = sim

    def arm_left(self) -> Arm:
        return self._arm_left
    
    def arm_right(self) -> Arm:
        return self._arm_right
