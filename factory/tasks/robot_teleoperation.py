from teleop.space_mouse.space_mouse import SpaceMouse, DuoSpaceMouse
from factory.components.motion_factory import MotionFactory
from simulation.base.sim_base import SimBase
from hardware.base.utils import convert_homo_2_7D_pose, convert_7D_2_homo, Buffer, negate_pose, transform_pose
import warnings, os
import numpy as np
import threading, time, copy
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from sshkeyboard import listen_keyboard, stop_listening
import glog as log

def convert_rotation_to_matrix(mode, rotation):
    """
        Convert the target rotation to the format of rotation matrix,
        absolute mode outputs quat, relative mode output euler angle
        @return: 3x3 rotation matrix in np array
    """
    mat = None
    if mode == 'relative':
        mat = R.from_euler('xyz', rotation).as_matrix()
    elif mode == 'absolute':
        mat = R.from_quat(rotation).as_matrix()
    else:
        raise ValueError(f'The mode {mode} of rotation format is not supported!!!')
    return mat


# Used for different robot component into one robot system
class TeleoperationFactory:
    def __init__(self, config, robot_motion: MotionFactory):
        self._config = config
        self._robot_motion_system = robot_motion
        self._teleop_interface_type = config["teleop_interface"]
        self._interface_output_mode = config["inteface_output_mode"]
        self._use_simulation_target = config["use_simulation_target"]
        self._teleoperation_loop_time = config["teleoperation_loop_time"]
        self._teleop_thread_running = True
        self._enable_hardware = False
        self._enable_recording = False
        self._save_path_dir = config.get("save_path_prefix", None)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if not self._save_path_dir is None:
            self._save_path_dir = cur_path
        else:
            self._save_path_dir = os.path.join(cur_path, "../../dataset/data")
        self._save_path = None
        
        # object classes
        self._interface_classes = {
            'space_mouse': SpaceMouse,
            'duo_space_mouse': DuoSpaceMouse
        }
        self._robot_motion_system.update_execute_hardware(True) #TODO: remove this
        
        # Gripper control state tracking - simplified based on test_dual_mouse.py
        self._gripper_current_position = {'left': 0.5, 'right': 0.5}  # Track current gripper position (0.0=close, 1.0=open)
        self._gripper_min = 0.0
        self._gripper_max = 1.0
        self._gripper_step = 0.02  # Fixed step size (2% per press)
        self._gripper_move_frequency = 0.03  # Time interval between moves (33Hz)
        self._last_gripper_move_time = {'left': 0.0, 'right': 0.0}  # Last move timestamp

    def _keyboard_on_press(self, key):
        if key == 'h':
            self._enable_hardware = not self._enable_hardware
            log.info(f"Hardware execution status {self._enable_hardware}!!!.")
            self._robot_motion_system.update_execute_hardware(
                                        self._enable_hardware)               
        elif key == 'q':
            stop_listening()
            self._teleop_thread_running = False
            self.traj_visual_thread.join()
            self._robot_motion_system.close()
        elif key == 'r':
            self._enable_recording = not self._enable_recording
            if self._enable_recording:
                os.makedirs(self._save_path_dir, exist_ok=True)
    
    def create_robot_teleoperation_system(self) -> bool:
        # common objects
        self._interface = self._interface_classes[self._teleop_interface_type](self._config["interface_config"][self._teleop_interface_type])
        
        # initialize all objects
        self._initialize()
        
    def _initialize(self) -> bool:
        if not self._interface.initialize():
            # raise ValueError(f"Teleoperation interface {self._teleop_interface_type} failed intialization")
            log.error(f"Teleoperation interface {self._teleop_interface_type} failed intialization")
        self._robot_motion_system.create_motion_components()
        self._robot_system = self._robot_motion_system._robot_system
        self.world2base_pose = self._robot_system._simulation.get_body_pose(
                self._robot_system._simulation.base_body_name)
        log.info(f'world2base: {self.world2base_pose}')
        self.base2world_pose = negate_pose(self.world2base_pose)
        log.info(f'base2world: {self.base2world_pose}')
        ee_link = self._robot_motion_system.get_model_end_effector_name()
        
        
        # visualization of trajetcory thread task
        if self._robot_system._use_simulation and self._robot_motion_system._use_traj_planner:
            self.traj_visual_thread = threading.Thread(target = self.traj_visual_task, 
                                                args = (self._robot_motion_system._buffer,
                                                        self._robot_motion_system._buffer_lock,
                                                        self._robot_system._simulation))
            self.traj_visual_thread.start()
            
        # keyboard listener
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                        kwargs={"on_press": self._keyboard_on_press, 
                                                "until": None, "sequential": False,}, 
                                        daemon=True)
        listen_keyboard_thread.start()
        
        # Initialize gripper positions from current hardware state
        self._initialize_gripper_positions()
        
        mocap_target_site = self._config.get('targe_site_name', None)
        TCP_site = self._config.get("tcp_visualization_site", None)
        init_pose = {}
        robot_index = ['left', 'right']
        # teleoperation loop
        log.info(f'teleoperation loop started!!')
        target_period = self._teleoperation_loop_time
        next_run_time = time.perf_counter()
        slow_loop_count = 0
        while self._teleop_thread_running:
            loop_start_time = time.perf_counter()
            
            # get interface target
            inteface_output_mode = self._interface_output_mode
            success_get_target, ee_target, other_target  = \
                self._interface.parse_data_2_robot_target(inteface_output_mode)
            # log.debug(f'[DEBUG] Interface: success={success_get_target}, use_sim_target={self._use_simulation_target}, ee_target={ee_target}')
            
            # only for mujoco 
            if self._use_simulation_target and not mocap_target_site is None:
                ee_target = {}
                for i, target_site in enumerate(mocap_target_site):
                    key = robot_index[i]
                    if i == 0 and len(mocap_target_site) == 1:
                        key = 'single'
                    cur_sim_target = self._robot_system._simulation.get_site_pose(target_site, 'xyzw')
                    cur_sim_target = transform_pose(self.base2world_pose, cur_sim_target)
                    ee_target[key] = cur_sim_target
                inteface_output_mode = 'absolute'
            
            cur_tcp_pose = {}
            for i, (key, value) in enumerate(ee_link.items()):
                cur_tcp_pose[key] =  self._robot_motion_system.get_frame_pose(value, key)
                # visualize the curr tcp
                if not TCP_site is None:
                    cur_tcp = cur_tcp_pose[key]
                    # Transform TCP from pin model base_link to simulation world coordinates
                    tcp = transform_pose(self.world2base_pose, cur_tcp)
                    
                    cur_tcp_mocap = TCP_site[i]
                    tcp_mocap = cur_tcp_mocap.split('_')[0]
                    self._robot_system._simulation.set_target_mocap_pose(tcp_mocap, tcp)
                
            if success_get_target or self._use_simulation_target:
                high_level_command = np.array([])
                for i, (key, cur_ee_target) in enumerate(ee_target.items()):
                    # Extract the actual target array from nested dict structure
                    if isinstance(cur_ee_target, dict):
                        # Handle nested structure: {'single': array(...)}
                        if 'single' in cur_ee_target:
                            actual_target = cur_ee_target['single']
                        else:
                            # Find the first array value in the dict
                            actual_target = next(iter(cur_ee_target.values()))
                    else:
                        actual_target = cur_ee_target
                    
                    # Incremental target on the ee pose
                    if inteface_output_mode == 'relative':
                        # hack!!!
                        if len(init_pose) != len(ee_target):
                            init_pose[key] = cur_tcp_pose[key]
                        
                        # Transform 3D mouse delta from base frame to chest frame
                        delta_pos_base = actual_target[:3]
                        delta_rot_base = actual_target[3:]
                        
                        # Apply coordinate transformation from base frame to chest frame
                        delta_pos_chest, delta_rot_chest = self._transform_delta_from_base_to_chest(
                            delta_pos_base, delta_rot_base)
                        
                        # Apply transformed deltas to TCP pose in chest frame
                        init_pose[key][:3] += delta_pos_chest
                        cur_mat = R.from_quat(init_pose[key][3:]).as_matrix()
                        # ee_mat = R.from_euler('xyz', delta_rot_chest).as_matrix()
                        _,_, yaw = delta_rot_chest

                        # 忽略 XY 旋转（roll 和 pitch 强制设为 0）
                        modified_euler = [0, 0, yaw]  # 仅保留 Z 轴旋转

                        # 生成旋转矩阵
                        ee_mat = R.from_euler('xyz', modified_euler).as_matrix()
                        # log.info(f'delta rot: {ee_mat}')
                        ee_mat = cur_mat @ ee_mat
                        init_pose[key][3:] = R.from_matrix(ee_mat).as_quat()
                        # Update the target to the computed pose
                        final_target = init_pose[key]
                    else:
                        # For absolute mode, use the actual target directly
                        final_target = actual_target
                    
                    # log.info(f'ee site target {key}: {final_target}')
                    
                    # visualization of the target pose for ee 
                    # (Not using simulation for target tracking)
                    if not self._use_simulation_target and not mocap_target_site is None:
                        cur_mocap_target_site = mocap_target_site[i]
                        mocap_name = cur_mocap_target_site.split('_')[0]
                        # target_tcp = copy.deepcopy(ee_target_7D)
                        target_tcp = final_target
                        target_tcp = transform_pose(self.world2base_pose, target_tcp)

                        # Apply relative transform for target_tcp (Method 3 approach)
                        # chest_to_world_actual = self._get_chest_to_world_transform()
                        
                        # # Assume target_tcp is relative to neutral chest, transform to actual chest
                        # chest_neutral = np.eye(4)  # Neutral chest pose (identity)
                        # world_to_chest_actual = np.linalg.inv(chest_to_world_actual)
                        
                        # # Transform: neutral_chest -> world -> actual_chest
                        # relative_transform = world_to_chest_actual @ chest_neutral
                        # relative_transform_7d = convert_homo_2_7D_pose(relative_transform)
                        # target_tcp_transformed = transform_pose(relative_transform_7d, target_tcp)
                        # target_tcp = transform_pose(self.world2base_pose, target_tcp_transformed)
                        self._robot_system._simulation.set_target_mocap_pose(mocap_name, target_tcp)
                
                    high_level_command = np.hstack((high_level_command, final_target))
                # log.debug(f"[DEBUG] Calling update_high_level_command with: {high_level_command}")
                self._robot_motion_system.update_high_level_command(high_level_command)
                
                # Handle gripper control from button input (only if there's input)
            if other_target:
                self._handle_gripper_control(other_target)
            
            # Sync hardware body positions to simulation (always sync when hardware is available)
            if self._robot_system._use_hardware:
                try:
                    sync_success = self._robot_system.sync_body_positions()
                    if sync_success:
                        log.debug(f"[BodySync] Body positions synchronized successfully")
                    else:
                        log.debug(f"[BodySync] Body position sync failed or no data available")
                except Exception as e:
                    log.debug(f"[BodySync] Error during body sync: {e}")
            
            next_run_time += target_period
            current_time = time.perf_counter()
            sleep_time = next_run_time - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 处理时间超过目标周期
                actual_time = current_time - loop_start_time
                slow_loop_count += 1

                # 每100次慢循环警告一次，避免日志刷屏
                if slow_loop_count % 100 == 1:
                    expected_freq = 1.0 / target_period
                    actual_freq = 1.0 / actual_time
                    log.warning(f"Teleoperation frequency slow: expected {expected_freq:.1f}Hz, "
                                f"actual {actual_freq:.1f}Hz (warning #{slow_loop_count})")

                # 重置时间基准，避免更大的延迟
                next_run_time = current_time
        log.info(f'Teleoperation thread for reading is stopped!')
    
    def _get_chest_to_world_transform(self) -> np.ndarray:
        """Get chest_to_world transformation"""
        try:
            if (hasattr(self._robot_system, '_robot') and 
                hasattr(self._robot_system._robot, '_trunk') and 
                self._robot_system._robot._trunk is not None):
                return self._robot_system._robot._trunk.get_chest_to_world_transform()
            else:
                return np.eye(4)
        except Exception as e:
            log.error(f"Error getting chest_to_world transform: {e}")
            return np.eye(4)
    
    def _transform_delta_from_base_to_chest(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> tuple:
        """Transform position and rotation deltas from base frame to chest frame"""
        chest_to_world = self._get_chest_to_world_transform()
        world_to_chest = np.linalg.inv(chest_to_world)
        
        # Transform position delta
        R_world_to_chest = world_to_chest[:3, :3]
        delta_pos_chest = R_world_to_chest @ delta_pos
        
        # Transform rotation delta
        delta_rot_matrix = R.from_euler('xyz', delta_rot).as_matrix()
        delta_rot_chest_matrix = R_world_to_chest @ delta_rot_matrix @ R_world_to_chest.T
        delta_rot_chest = R.from_matrix(delta_rot_chest_matrix).as_euler('xyz')
        
        return delta_pos_chest, delta_rot_chest
    
    
    def _initialize_gripper_positions(self):
        """Initialize gripper positions from hardware - simplified approach"""
        grippers = self._robot_system.get_all_grippers()
        for side in ['left', 'right']:
            if side in grippers:
                gripper = grippers[side]
                if gripper and gripper.valid():
                    try:
                        current_pos = gripper.get_position()
                        self._gripper_current_position[side] = current_pos
                        log.info(f"Initialized {side} gripper position: {current_pos:.3f}")
                    except Exception as e:
                        log.warning(f"Failed to get {side} gripper position: {e}")
                        # Keep default position of 0.5
                else:
                    log.info(f"{side} gripper not available, using default position 0.5")
    
    def _handle_gripper_control(self, other_target):
        """Handle gripper control with buttons - optimized for minimal latency"""
        current_time = time.perf_counter()
        
        for side, buttons in other_target.items():
            if side not in ['left', 'right'] or not isinstance(buttons, dict) or 'single' not in buttons:
                continue
                
            current_buttons = buttons['single']
            if not current_buttons or len(current_buttons) < 2:
                continue
            
            # Quick exit if no buttons pressed
            if not (current_buttons[0] or current_buttons[1]):
                continue
                
            # Check frequency limit
            last_move_time = self._last_gripper_move_time[side]
            if (current_time - last_move_time) < self._gripper_move_frequency:
                continue
                
            current_position = self._gripper_current_position[side]
            
            # Button 0 - open gripper
            if current_buttons[0]:
                new_position = min(current_position + self._gripper_step, self._gripper_max)
                if new_position != current_position:
                    if self._robot_system.control_gripper(side, new_position):
                        self._gripper_current_position[side] = new_position
                        self._last_gripper_move_time[side] = current_time
            
            # Button 1 - close gripper  
            elif current_buttons[1]:
                new_position = max(current_position - self._gripper_step, self._gripper_min)
                if new_position != current_position:
                    if self._robot_system.control_gripper(side, new_position):
                        self._gripper_current_position[side] = new_position
                        self._last_gripper_move_time[side] = current_time
            
    def traj_visual_task(self, buffer: Buffer, lock: threading.Lock, sim: SimBase):
        while self._teleop_thread_running:
            start_time = time.perf_counter()
            lock.acquire()  
            buffer_size = buffer.size()
            # log.info(f'buffer size: {buffer_size}')
            if buffer_size !=0: 
                # log.info(f'get new traj point, buffer size: {buffer_size}')
                traj_data = buffer._data[buffer_size - 1]
                lock.release()  
                for type in self._robot_motion_system._robot_model.keys():
                    if 'single' in type:
                        cur_data = transform_pose(self.world2base_pose, traj_data[:7])
                        sim.update_trajectory_data(cur_data)
                    elif type == 'dual':
                        # For dual arm, process both left and right arm data
                        left_data = transform_pose(self.world2base_pose, traj_data[:7])
                        right_data = transform_pose(self.world2base_pose, traj_data[7:14])
                        combined_data = np.hstack((left_data, right_data))
                        sim.update_trajectory_data(combined_data)
                    else:
                        # Legacy left/right type handling
                        if 'left' in type:
                            cur_data = transform_pose(self.world2base_pose, traj_data[:7])
                        else:  # right
                            cur_data = transform_pose(self.world2base_pose, traj_data[7:14])
                        sim.update_trajectory_data(cur_data)
            else:
                lock.release()
                
            used_time = time.perf_counter() - start_time
            if used_time < 0.003:
                time.sleep(0.003 - used_time)
        log.info(f'teleoperation traj visual task stopped!!!!')
                
                