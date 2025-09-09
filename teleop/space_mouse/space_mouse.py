import numpy as np
import pyspacemouse
from teleop.base.teleoperation_base import TeleoperationDeviceBase
import time
import threading
import glog as log
from hardware.base.utils import ToolControlMode
from teleop.base.utils import RisingEdgeDetector

class SpaceMouse(TeleoperationDeviceBase):
    def __init__(self, config):
        self._frequency = config["frequency"]
        self._device_id = config["device_id"]
        self._scale = config["scale"]
        self._lower_threshold = config["low_threshold"]
        self._enable_rotation = config["enable_rotation"]
        self._tool_control_mode = config.get("tool_mode", "binary")
        self._tool_control_mode = ToolControlMode(self._tool_control_mode)
        print(f'space tool control mode: {self._tool_control_mode}')
        self._tool_incremental_step = config.get("incremental_step", 0.05)
        self._move_time = config.get("move_time", 0.005)
        if self._tool_control_mode == ToolControlMode.BINARY:
            self._rising_edge_detector = RisingEdgeDetector()
        else: self._last_change_time = time.perf_counter()
        self._tool_control_mode = ToolControlMode(self._tool_control_mode)
        self._last_command = 1.0 # True for open, False for close
        self._data = None
        self._device = None
        self.target_updated = False
        
        # super init
        super().__init__(config)
        # data update thread
        self._thread_running = True
        self.lock = threading.Lock()
        self._thread = threading.Thread(target=self.update_data)
        self._thread.start()
        time.sleep(0.5)
        # wait for thread
        print(f"The space mouse is initialized with state {self._is_initialized}")

    def initialize(self):
        if self._is_initialized:
            return True
        
        success = False
        devices = pyspacemouse.list_devices()
        if len(devices) <= 0:
            log.warn("No space mouse device found!!!")
            return False
        
        log.info(f'devices:', devices)
        self._device = pyspacemouse.open(device=devices[0],
                                        DeviceNumber=self._device_id)
        if not self._device is None:
            success = True
        return success
    
    def print_data(self):
        if not self._is_initialized or self._data is None:
            log.warn(f'The mouse object is not ready for printing data, '
                          f'is_initialized: {self._is_initialized},' 
                          f'has_data: {not self._data is None}')
            return 
        
        self.lock.acquire()
        cur_data = [self._data.x, self._data.y, self._data.z,
                self._data.roll , self._data.pitch, self._data.yaw]
        self.lock.release()
        log.info(f"Space mouse with device id {self._device_id}'s current data: {cur_data}")
    
    def close(self):
        self._thread_running = False
        self._thread.join()
        self._device.close()
        log.info(f"Close the space mouse with device id {self._device_id}")
        
    def read_data(self):
        if not self._is_initialized:
            log.warn(f"The space mouse with id {self._device_id} "
                          "is not initialized yet!!")
            return 
        
        self._data = self._device.read()
        
    def update_data(self):
        log.info(f"Space mouse with id {self._device_id} has started the data reading thread"
              f" with initialzed state: {self._is_initialized}!")
        
        start_time = time.time()
        # min_freq = 1e6
        while self._is_initialized and self._thread_running:
            self.lock.acquire()
            self.read_data()
            self.lock.release()
            
            self.target_updated = True
            dt = time.time() - start_time
            # cur_freq = (1.0/dt)
            # if cur_freq < min_freq:
            #     min_freq = cur_freq
            # log.info(f'read dt: {cur_freq}, min freq: {min_freq}')
            if  dt < (1.0 / self._frequency):
                sleep_time = (1.0 / self._frequency) - dt 
                time.sleep(sleep_time)
            elif dt > 1.3 / self._frequency:
                log.warn("The frequency for reading the space mouse data is slower than the" 
                              f"use specified frequency, expected: {self._frequency}, actual: {1.0 /dt}!")
            
            start_time = time.time()
                
        log.info(f'Space mouse with id {self._device_id} closes the data update thread!!!!')

    def parse_data_2_robot_target(self, mode: str) -> np.ndarray:
        if not self._is_initialized:
            log.warn(f"The space mouse with id {self._device_id} "
                          "is not initialized yet!!")
            return False, None, None

        if not self.target_updated:
            return False, None, None

        self.lock.acquire()
        # self._data.pitch, self._data.roll
        data = [-self._data.y, self._data.x, self._data.z,
                self._data.roll, self._data.pitch, self._data.yaw]
        # data = [0,0,0,
        #         0, self._data.roll,0]
        buttons = [self._data.buttons[0], self._data.buttons[1]]
        self.lock.release()
        
        if mode == 'absolute':
            raise ValueError("Unsupport mode for the absolute pose from 3d space mouse!")
        elif mode == 'relative':
            low_thresh_flag = np.all(np.array(np.abs(data)) < np.array(self._lower_threshold))
            low_thresh_flag = not low_thresh_flag
            target = np.zeros(6)
            if low_thresh_flag:
                target = np.array(self._scale) * np.array(data)
                if not self._enable_rotation:
                    target[3:] = 0
            pose_target = {'single': target}
            if self._tool_control_mode == ToolControlMode.BINARY:
                # 上升沿检查
                rising_edge = self._rising_edge_detector.update(buttons[0])
                if rising_edge:
                    self._last_command = not bool(self._last_command)
                buttons[0] = float(self._last_command)
            else:
                # continous control
                cur_time = time.perf_counter()
                if (cur_time - self._last_change_time) > self._move_time:
                    if buttons[0]:
                        self._last_command += self._tool_incremental_step
                    if buttons[1]:
                        self._last_command -= self._tool_incremental_step
                    self._last_command = np.clip(self._last_command, 0, 1)
                    self._last_change_time = time.perf_counter()
                buttons[0] = self._last_command
            tool_target = {'single': buttons}
            self.target_updated = False
            return True, pose_target, tool_target
        else:
            raise ValueError("Unsupported mode: {}".format(mode))
        
class DuoSpaceMouse(TeleoperationDeviceBase):
    def __init__(self, config):
        self.devices = {}
        self.devices['left'] = SpaceMouse(config['left'])
        self.devices['right'] = SpaceMouse(config['right'])
        self._is_initialized = self.devices['left']._is_initialized \
                            and self.devices['right']._is_initialized
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        self.devices['left'].initialize()
        self.devices['right'].initialize()
        return self.devices['left']._is_initialized \
                            and self.devices['right']._is_initialized
        
    def close(self):
        self.devices['left'].close()
        self.devices['right'].close()
        
    def read_data(self):
        pass
        
    def print_data(self):
        self.devices['left'].print_data()
        self.devices['right'].print_data()
        
    def parse_data_2_robot_target(self, mode):
        if not self._is_initialized:
            log.warn(f'One of devices is not initialized well, '
                            f'left: {self.devices["left"]._is_initialized}, '
                            f'right: {self.devices["right"]._is_initialized}')
        
        success, left_data, left_other = self.devices['left'].parse_data_2_robot_target(mode)
        if not success:
            log.warn('The left device did not successfully parse the data')
            return False, None, None
        success, right_data, right_other = self.devices['right'].parse_data_2_robot_target(mode)
        if not success:
            log.warn('The right device did not successfully parse the data')
            return False, None, None
            
        pose_target = {'left': left_data['single'], 'right': right_data['single']}
        tool_target = {'left': left_other['single'], 'right': right_other['single']}
        return True, pose_target, tool_target
                
if __name__ == '__main__':
    import yaml
    import os
    config = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    # single_space_mouse_cfg, duo_space_mouse_cfg
    cfg_file = os.path.join(cur_path, 'duo_space_mouse_cfg.yaml')
    log.info(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    log.info(f'yaml data: {config}')
    log.info(config)
    # space_mouse = SpaceMouse(config['space_mouse'])
    last_time = time.time()
    # while True:
    #     if time.time() - last_time > 1.0:
    #         space_mouse.print_data()
    #         last_time = time.time()
    
    duo_mouse = DuoSpaceMouse(config['duo_space_mouse'])
    while True:
        # if time.time() - last_time > 1.0:
        duo_mouse.print_data()
        res = duo_mouse.parse_data_2_robot_target('relative')
        log.info(f'duo data: {res[1]}')
        last_time = time.time()
            
        time.sleep(0.01)
   