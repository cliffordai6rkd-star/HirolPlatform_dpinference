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
import threading
import glog as log

class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any], use_real_robot=False):
        
        self.robot = None
        self._arms_ready = threading.Event() # 一個事件旗標，用來表示手臂是否已載入完成
        self._arm_left_instance = None
        self._arm_right_instance = None
        self.sim = None
        self._load_thread = threading.Thread(
            target=self._load_all_in_background, 
            args=(config, use_real_robot),
            daemon=True
        )
        self._load_thread.start()
        log.info("Agent 初始化已發起，正在背景載入模型...")

        self.camera = Camera()

    def _load_all_in_background(self, config: Mapping[Text, Any], use_real_robot: bool):
        """
        這個函式在背景執行緒中運行，包含了所有耗時的初始化操作。
        """
        print("背景載入執行緒已啟動...")
        
        if use_real_robot:
            self.robot = RobotLib("192.168.11.3:50051", "", "")
            print("Robot connection established.")

        # 模擬器的初始化通常很快，但也可以放在這裡
        self.sim = Monte01Mujoco()
        self.sim_thread = threading.Thread(target=self.sim.start, daemon=True)
        self.sim_thread.start()

        print("正在预加载URDF模型...")
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['arm']['urdf_path']))
        Arm.preload_urdf(urdf_path)
        #TODO: load urdf once, and construct reduced model separately
        # --- 這裡是最耗時的部分 ---
        print("正在載入左臂...")
        self._arm_left_instance = Arm(config=config['arm'], hardware_interface=self.robot, simulator=self.sim, isLeft=True)
        
        print("正在載入右臂...")
        self._arm_right_instance = Arm(config=config['arm'], hardware_interface=self.robot, simulator=self.sim, isLeft=False)
        # --- 耗時部分結束 ---

        print("所有手臂模型已載入完成！")
        # 設定事件，通知其他執行緒，手臂已經準備好了
        self._arms_ready.set()

    def arm_left(self) -> Arm:
        self._arms_ready.wait() # 等待事件被設定
        return self._arm_left_instance
    
    def arm_right(self) -> Arm:
        self._arms_ready.wait() # 等待事件被設定
        return self._arm_right_instance
    
    def wait_for_ready(self, timeout: float = None):
        """提供一個方法來等待所有資源載入完成。"""
        print("主程式正在等待所有資源載入完成...")
        ready = self._arms_ready.wait(timeout=timeout)
        if ready:
            print("資源已就緒！")
        else:
            print(f"等待超時 ({timeout}秒)！")
        return ready
    
    def head_front_camera(self) -> Camera:
        return self.camera
