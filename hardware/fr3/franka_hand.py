from hardware.base.gripper_base import GripperBase
from panda_py.libfranka import Gripper
from hardware.base.utils import GripperState
import os, yaml, time, threading

class FrankaHand(GripperBase):
    def __init__(self, config):
        self._ip = config["ip"]
        self._grasp_force = config["grasp_force"]
        self._grasp_speed = config["grasp_speed"]
        super().__init__(config)
        
    def initialize(self):
        self._gripper = Gripper(self._ip)
        # move to home
        success = self._gripper.move(0.08, 0.02)
        state = self._gripper.read_once()
        self._max_width = state.max_width
        return success
        
    def set_gripper_command(self, target):
        target = self._max_width if target > self._max_width else target
        def grasp_task():
            # self._gripper.grasp(target, self._grasp_speed, self._grasp_force)
            self._gripper.move(target, self._grasp_speed)
        threading.Thread(target=grasp_task).start()
    
    def get_gripper_state(self):
        state = self._gripper.read_once()
        self._state._position = state.width
        self._state._is_grasped = state.is_grasped
        return self._state
    
    def stop_gripper(self):
        self._gripper.stop()
    
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
    