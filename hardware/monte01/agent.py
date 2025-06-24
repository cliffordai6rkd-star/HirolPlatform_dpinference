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

        # --- 改进后的加载策略：先预加载URDF，再创建手臂实例 ---
        print("正在预加载URDF模型...")
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['arm']['urdf_path']))
        Arm.preload_urdf(urdf_path)
        
        print("正在创建左臂实例...")
        self._arm_left_instance = Arm(config=config['arm'], hardware_interface=self.robot, simulator=self.sim, isLeft=True)
        
        print("正在创建右臂实例...")
        self._arm_right_instance = Arm(config=config['arm'], hardware_interface=self.robot, simulator=self.sim, isLeft=False)
        # --- 优化后的加载部分结束 ---

        self.camera = Camera()

    def arm_left(self) -> Arm:
        return self._arm_left
    
    def arm_right(self) -> Arm:
        return self._arm_right
    
    def head_front_camera(self) -> Camera:
        return self.camera
