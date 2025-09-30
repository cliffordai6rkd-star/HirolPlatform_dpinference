from teleop.base.teleoperation_base import TeleoperationDeviceBase
from pika.sense import Sense
from pika.tracker.vive_tracker import PoseData
from hardware.base.utils import convert_homo_2_7D_pose, negate_pose, transform_pose, transform_quat, pose_diff
import glog as log
import numpy as np
import time, threading, math
# from sshkeyboard import listen_keyboard, stop_listening
from pynput import keyboard
from typing import Dict, Tuple, Optional
import copy

class PikaTracker(TeleoperationDeviceBase):
    _tracker: dict[str, Sense]
    
    T_ROBOT_TRACKER = np.eye(4)
    T_TRACKER_ROBOT = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [ 0, 0, 0, 1]])
    # T_TRACKER_ROBOT = np.eye(4)
    
    # init pose for absolute delta pose calculation
    INIT_TARGET_POSE = [[0,0,0,0,0,0,1], [0,0,0,0,0,0,1]]
    
    def __init__(self, config):
        self._serial_port: dict = config.get("serial_ports")
        self._tracker = {}
        for key, port in self._serial_port.items():
            log.info(f'{key} port: {port}')
            if port is not None:
                tracker = Sense(port=port)
                self._tracker[key] = tracker
            else: self._tracker[key] = None
        self._rotation_enabled = config["rotation_enabled"]
        for key, rot_enable in self._rotation_enabled.items():
            if not rot_enable: continue
            assert len(rot_enable) == 3, f"{key} rotation enabled need to have three elemets indicating rpy but get {len(rot_selec) == 3}"

        # Vive Tracker configuration
        self._vive_config_path = config.get("vive_config_path", None)
        self._vive_lh_config = config.get("vive_lh_config", None)
        self._vive_args = config.get("vive_args", None)
        self._tracker_detection_attempts = config.get('num_attempts', 10)
        
        # Device configuration
        self._device_id: dict[str, str] = config["target_device"]  # Default tracker ID
        self._readed_device_id = None
        self._output_left = config.get("output_left", True)
        self._output_right = config.get("output_right", True)
        if self._output_left and self._output_right:
            self._index = {"left": 0, "right": 1}
        elif not self._output_left: 
            self._index = {"single": 1}
        elif not self._output_right:
            self._index = {"single": 0}
        
        # Initialize offset pose for relative positioning
        self._init_pose = config.get('init_pose', None)
        self._last_quat = [np.array([0,0,0,1]), np.array([0,0,0,1])]
        if not self._init_pose is None:
            self._init_pose_rot = [self._init_pose["initial_pose_left"][3:], 
                                    self._init_pose["initial_pose_right"][3:]]
            self._last_quat = self._init_pose_rot
            self._init_pose_trans = [self._init_pose["initial_pose_left"][:3], 
                                    self._init_pose["initial_pose_right"][:3]]
        self._device_enabled = False
        self._reset_pose_counter = 0
        
        super().__init__(config)
        if not self._is_initialized:
            raise ValueError
        
    def initialize(self) -> bool:
        """Initialize the Pika Sense device and Vive Tracker."""
        if self._is_initialized:
            return True
        
        try:
            get_right_device = {'left': False, 'right': False}
            # Initialize Vive Tracker
            for i, (key, tracker) in enumerate(self._tracker.items()):
                if not tracker:
                    get_right_device[key] = True
                    continue
                
                # Connect to Pika Sense device
                if not tracker.connect():
                    log.error("Failed to connect to Pika Sense device")
                    return False
                
                # Set up Vive Tracker configuration if provided
                if self._vive_config_path or self._vive_lh_config or self._vive_args:
                    tracker.set_vive_tracker_config(
                        config_path=self._vive_config_path,
                        lh_config=self._vive_lh_config,
                        args=self._vive_args,
                        pose_offset=True, use_uid=True
                    )
                
                for j in range(self._tracker_detection_attempts):
                    time.sleep(2.0)
                    vive_tracker_device_names = tracker.get_tracker_devices()
                    if len(vive_tracker_device_names) == 0:
                        log.error("Failed to initialize Vive Tracker")
                        continue
                    else:
                        log.info(f'{j}th {key} tracker device name: {self._readed_device_id}')
                    
                    if not self._readed_device_id or len(self._readed_device_id) < len(vive_tracker_device_names):
                        self._readed_device_id = vive_tracker_device_names
                    
                    for device_name in self._readed_device_id:
                        if device_name == self._device_id[key]:
                            get_right_device[key] = True
                            
                    if get_right_device["left"] and get_right_device["right"]:
                        break
                    
                if not get_right_device[key]:
                    log.error(f'Failed to get the correct tracker device for {key} with {self._device_id}, but get {vive_tracker_device_names}')
                    return False
                
            # keyboard listening for update init pose
            self._key_pressed = False
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press
            )
            self._keyboard_listener.start()
            log.info("PikaTracker keyboard listener started")
            
            # Wait for tracker data to stabilize
            time.sleep(1.0)
            
            log.info(f"PikaTracker initialized successfully with device: {self._serial_port} and device name: {self._device_id}")
            return True
            
        except Exception as e:
            log.error(f"Failed to initialize PikaTracker: {e}")
            return False
    
    def read_data(self):
        tracker_pose = None
        for key, tracker in self._tracker.items():
            if not tracker: continue
            
            pose = tracker.get_pose(None)
            if pose is None: return None, None
            else:
                if not tracker_pose or len(tracker_pose) < len(pose):
                    tracker_pose = pose
                
        pose_quat = {}
        all_pose_flag = {"left": False, "right": False}
        if not self._output_left:
            all_pose_flag["left"] = True
        if not self._output_right:
            all_pose_flag["right"] = True
        for device_name, cur_pose in tracker_pose.items():
            if all_pose_flag["left"] and all_pose_flag["right"]:
                break
            
            if self._output_left and self._device_id["left"] == device_name:
                pose_quat["left"] = np.zeros(7)
                pose_quat["left"][:3] = cur_pose.position
                pose_quat["left"][3:] = cur_pose.rotation  # [qx, qy, qz, qw]
                all_pose_flag["left"] = True
            if self._output_right and self._device_id["right"] == device_name:
                pose_quat["right"] = np.zeros(7)
                pose_quat["right"][:3] = cur_pose.position
                pose_quat["right"][3:] = cur_pose.rotation  # [qx, qy, qz, qw]
                all_pose_flag["right"] = True
        
        tool_data = {}
        for key, tracker in self._tracker.items():
            if not tracker:
                continue
            
            encoder = tracker.get_encoder_data()["rad"]
            distance = self._change_motor_rad_to_gripper_distance(encoder)
            normalized_value = distance / 90.0
            normalized_value = np.clip(normalized_value, 0.0, 1.0)
            command = tracker.get_command_state()
            tool_data[key] = np.array([normalized_value, command])
            
        if not self._output_left:
            pose_quat = dict(single=pose_quat["right"])
            tool_data = dict(single=tool_data["right"])
        if not self._output_right:
            pose_quat = dict(single=pose_quat["left"])
            tool_data = dict(single=tool_data["left"])
        return pose_quat, tool_data

    def _press_i(self):
        pose, _ = self.read_data()
        if pose is None:
            log.warning("Failed to read pose data for update init pose!!!")
            return
        for i, (key, cur_pose) in enumerate(pose.items()):
            cur_pose = self._process_raw_pose(cur_pose, key)
            self.INIT_TARGET_POSE[self._index[key]] = cur_pose
        self._device_enabled = True
        self._key_pressed = True 
        log.info("init pose is updated!!!")
        
    def _press_u(self):
        self._device_enabled = False
        log.info(f'Pika disabled!!!')

    def _on_key_press(self, key):
        try:
            if key.char == 'i' and not self._device_enabled:
                self._press_i()
            elif key.char == 'u' and self._device_enabled:
                self._press_u()
        except AttributeError:
              # ignore some hotkeys
              pass
          
    def _process_raw_pose(self, pose):
        res_pose = np.zeros(7)
        tracker_robot_quat = convert_homo_2_7D_pose(self.T_TRACKER_ROBOT)
        robot_tracker_quat = negate_pose(tracker_robot_quat)
        res_pose = transform_pose(transform_pose(robot_tracker_quat, pose), tracker_robot_quat)
        return res_pose

    def parse_data_2_robot_target(self, mode: str) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """Parse Vive Tracker pose data to robot target format."""
        if 'absolute' not in mode:
            log.warn('The pika tracker only supports absolute pose related teleoperation')
            return False, None, None
        
        if not self._is_initialized:
            log.warning("Device not initialized")
            return False, None, None
        
        # Get pose data from Vive Tracker, all dict
        pose_quat, tool_data = self.read_data()
        if pose_quat is None:
            log.warn(f"No pose data available for device: {self._serial_port}")
            return False, None, None
                
        # Prepare robot target dict
        pose_target = {} 
        tool_target = {}
        
        for key, value in pose_quat.items():
            # basis change
            cur_pose = self._process_raw_pose(value, key)
            
            # @TODO: zyx, select rotation
            # for i in range(3):
            #     if self._rotation_enabled and not self._rotation_enabled[i]:
            #         if mode == "absolute":
            #             cur_pose[3+i] = self._init_pose_rot[self._index[key]][i]
            #         else:
            #             cur_pose[3+i] = self
            
            if mode == "absolute_delta" and self._device_enabled:
                cur_pose = self._get_diff_trans(cur_pose, self._index[key])
            elif mode == "absolute":
                # rotation transform
                cur_pose = self.apply_init_offset(cur_pose, self._index[key])
            else: raise ValueError(f'{mode} mode is not supported for pika tracker')
            pose_target[key] = cur_pose
            tool_target[key] = np.hstack((tool_data[key], copy.deepcopy(self._key_pressed)))
        
        if self._key_pressed :
            if self._reset_pose_counter > 8:
                self._key_pressed = False
                self._reset_pose_counter = 0
            else: self._reset_pose_counter += 1
        if mode == "absolute" or (mode == "absolute_delta" and self._device_enabled):
            return True, pose_target, tool_target
        else: return False, None, None
            
    def print_data(self):
        data = self.parse_data_2_robot_target("absolute")
        log.info(f'pose: {data[1]}')
    
    def close(self):
        """Clean up resources and disconnect devices."""
        if not self._is_initialized:
            return 
        
        self._device_enabled = False
        for key, tracker in self._tracker.items():
            if tracker:
                tracker.disconnect()
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        log.info("PikaTracker disconnected successfully")
        return True
    
    def apply_init_offset(self, pose, id):
        new_pose = copy.deepcopy(pose)
        # basis convertion (mainly for rot)
        if self._init_pose is not None:
            init_rot = self._init_pose_rot[id]
            new_pose[3:] = transform_quat(pose[3:], init_rot)
        return new_pose
    
    def _get_diff_trans(self, cur_pose, index):
        diff_pose = pose_diff(cur_pose, self.INIT_TARGET_POSE[index])
        return diff_pose
    
    def _change_motor_rad_to_gripper_distance(self, angle_rad):
        distance = (self._get_distance(angle_rad) - self._get_distance(0)) * 2.0
        return distance
            
    def _get_distance(self, angle):
        angle = (180.0 - 43.99) / 180.0 * math.pi - angle
        height = 0.0325 * math.sin(angle)
        width_d = 0.0325 * math.cos(angle)
        width = math.sqrt(0.058**2 - (height - 0.01456)**2) + width_d
        return width*1000
    