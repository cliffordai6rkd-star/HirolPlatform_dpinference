from hardware.base.tool_base import ToolBase
from panda_py.libfranka import Gripper
from hardware.base.utils import ToolState, ToolType
import os, yaml, time, threading
import numpy as np
import copy
import warnings

class FrankaHand(ToolBase):
    _tool_type: ToolType = ToolType.GRIPPER
    def __init__(self, config):
        self._ip = config["ip"]
        self._grasp_force = config["grasp_force"]
        self._grasp_speed = config["grasp_speed"]
        super().__init__(config)
        self._state._tool_type = self._tool_type
        self._last_command = 0.08
        self._gripper_idle = True
        self._thread_running = True
        self._update_thread = threading.Thread(target=self.update_state)
        self._lock = threading.Lock()
        # self._update_thread.start()
        
        
    def initialize(self):
        self._gripper = Gripper(self._ip)
        # move to home
        success = self._gripper.move(0.08, 0.02)
        state = self._gripper.read_once()
        self._max_width = state.max_width
        return success
        
    def _set_binary_command(self, target: float) -> bool:
        """
            target command shoould be a float number with range [0,1]
            1 for fully open; 0 for fully closed
        """
        if not self._gripper_idle:
            return False 
        
        target = np.clip(target, 0 ,1)
        target = self._max_width * target
        # avoid continous calling 
        if np.isclose(target, self._last_command, rtol=0.0001):
            return True
        
        def grasp_task():
            self._gripper_idle = False
            if np.isclose(target, self._max_width):
                self._gripper.move(0.08, self._grasp_speed)
            else:
                self._gripper.grasp(target, self._grasp_speed, self._grasp_force,
                                    0.02, 0.06)
            # self._gripper.move(target, self._grasp_speed)
            self._state._position = target
            self._last_command = target
            self._gripper_idle = True
            
        gripper_thread = threading.Thread(target=grasp_task)
        gripper_thread.start()
        # gripper_thread.join()
        return True
    
    def get_tool_state(self):
        self._lock.acquire()
        state = copy.deepcopy(self._state)
        self._lock.release()
        return state
    
    def update_state(self):
        print(f'Starting state updating thread for franka hand {self._ip}')
        
        last_read_time = time.time()
        read_frequency = 1
        while self._thread_running:
            state = self._gripper.read_once()
            self._lock.acquire()
            self._state._position = state.width
            self._state._is_grasped = state.is_grasped
            self._lock.release()
            
            dt = time.time() - last_read_time
            if dt < 1.0 / read_frequency:
                sleep_time = 1.0 / read_frequency - dt
                time.sleep(sleep_time)
            elif dt > 1.3 / read_frequency:
                warnings.warn(f'The franka hand {self._ip} could not reach the update thread frequency!!')
            last_read_time = time.time()
        print(f'franka hand {self._ip} stopped its update thread!!!!')
            
    def stop_tool(self):
        self._thread_running = False
        # self._update_thread.join()
        self._gripper.stop()
        print(f'Franka hand {self._ip} successfully stopped!!!!')
        
    def get_tool_type_dict(self):
        tool_type_dict = {'single': self._tool_type}
        return tool_type_dict
    
if __name__ == '__main__':
    fr3_cfg = "hardware/fr3/config/fr3_cfg.yaml"
    config = "hardware/fr3/config/franka_hand_cfg.yaml"
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, "../..", config)
    fr3_cfg = os.path.join(cur_path, "../..", fr3_cfg)
    print(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'yaml data: {config}')
    with open(fr3_cfg, 'r') as stream:
        fr3_cfg = yaml.safe_load(stream)["fr3"]
    
    fr3_hand = FrankaHand(config=config["franka_hand"])
    from hardware.fr3.fr3_arm import Fr3Arm
    fr3 = Fr3Arm(fr3_cfg)
    fr3._fr3_robot.teaching_mode(True)
    
    while True:
        gripper_state = fr3_hand.get_gripper_state()
        print(f'gri: {gripper_state}')
        input_data = input('Please enter c for close, o to open: ')
        if input_data == 'c':
            fr3_hand.close()
        elif input_data == 'o':
            fr3_hand.set_gripper_command(0.075)
        time.sleep(0.01)
    