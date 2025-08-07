from teleop.base.teleoperation_base import TeleoperationDeviceBase
from teleop.space_mouse.space_mouse import SpaceMouse, DuoSpaceMouse
from teleop.XR.quest3.meta_quest3 import MetaQuest3, init_image_shared_mem
from factory.components.motion_factory import MotionFactory
from factory.components.robot_factory import RobotFactory
from simulation.base.sim_base import SimBase
from dataset.lerobot.data_process import EpisodeWriter
from hardware.base.utils import convert_homo_2_7D_pose, Buffer, negate_pose, transform_pose, object_class_check
from hardware.base.utils import ToolState
from hardware.base.img_utils import combine_image
from motion.duo_model import DuoRobotModel
import warnings, os
import numpy as np
import threading, time, copy
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from sshkeyboard import listen_keyboard, stop_listening
import glog as log
import cv2

# Used for different robot component into one robot system
class TeleoperationFactory:
    _robot_motion_system: MotionFactory
    _robot_system: RobotFactory
    _interface: TeleoperationDeviceBase
    def __init__(self, config, robot_motion: MotionFactory):
        self._config = config
        self._robot_motion_system = robot_motion
        self._teleop_interface_type = config["teleop_interface"]
        self._interface_output_mode = config["inteface_output_mode"]
        self._use_simulation_target = config["use_simulation_target"]
        self._teleoperation_loop_time = config["teleoperation_loop_time"]
        self._visual_data = config.get("visualize_data", False)
        self._task_description = config.get("task_description", None)
        self._task_description_goal = config.get("task_description_goal", None)
        self._task_description_step = config.get("task_description_step", None)
        self._data_record_frequency = config.get("data_record_frequency", 30)
        self._img_visualization = config.get("image_visualization", True)
        self._img_shm = None
        self._teleop_thread_running = True
        self._update_high_level_state = True
        self._is_initialized = False
        self._enable_hardware = False
        self._enable_recording = False
        self._save_path_dir = config.get("save_path_prefix", None)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if not self._save_path_dir is None:
            self._save_path_dir = os.path.join(cur_path, "../../dataset/data", 
                                               self._save_path_dir)
        else:
            self._save_path_dir = os.path.join(cur_path, "../../dataset/data")
        self.data_recorder = None
        
        # object classes
        self._interface_classes = {
            'space_mouse': SpaceMouse,
            'duo_space_mouse': DuoSpaceMouse,
            'meta_quest3': MetaQuest3
        }
        


    def create_robot_teleoperation_system(self) -> bool:
        if not object_class_check(self._interface_classes, self._teleop_interface_type):
            raise ValueError(f'Teleoperation interface {self._teleop_interface_type} is not supported')
        teleoperation_cfg = self._config["interface_config"][self._teleop_interface_type]
        if self._teleop_interface_type == 'meta_quest3':
            self._img_shm = init_image_shared_mem(teleoperation_cfg)
            self._xr_img_shape = teleoperation_cfg['image_shape']
            self._image_array = np.ndarray(self._xr_img_shape, 
                                           dtype=np.uint8, buffer=self._img_shm.buf)
        self._interface = self._interface_classes[self._teleop_interface_type](teleoperation_cfg)

        # initialize all objects
        self._initialize()
        
    def _initialize(self) -> bool:
        if not self._interface.initialize():
            log.error(f"Teleoperation interface {self._teleop_interface_type} failed intialization")
        self._robot_motion_system.create_motion_components()
        self._robot_system = self._robot_motion_system._robot_system
        
        # base frames
        if self._robot_system._use_simulation:
            if len(self._robot_system._simulation.base_body_name) == 0:
                self.world2base_pose = [np.array([0, 0, 0, 0, 0, 0, 1])]
                self.base2world_pose = [negate_pose(self.world2base_pose[0])]
            else:
                self.world2base_pose = []
                self.base2world_pose = []
                for cur_base_body in self._robot_system._simulation.base_body_name:
                    cur_world2base = self._robot_system._simulation.get_body_pose(cur_base_body)
                    self.world2base_pose.append(cur_world2base)
                    self.base2world_pose.append(negate_pose(cur_world2base))
            log.info(f'world2base: {self.world2base_pose}')
            log.info(f'base2world: {self.base2world_pose}')
       
        self.ee_link = self._robot_motion_system.get_model_end_effector_link_list()
        log.info(f'ee links: {self.ee_link}')
        
        
        # visualization of trajetcory thread task
        if self._robot_system._use_simulation and self._robot_motion_system._use_traj_planner:
            self._traj_visual_thread = threading.Thread(target = self.traj_visual_task, 
                                                args = (self._robot_motion_system._buffer,
                                                        self._robot_motion_system._buffer_lock,
                                                        self._robot_system._simulation))
            self._traj_visual_thread.start()
            
        # data recording thread
        self._data_recording_thread = threading.Thread(target=self.add_teleoperation_data)
        self._data_recording_thread.start()
            
        # keyboard listener
        self._is_initialized = True
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                        kwargs={"on_press": self._keyboard_on_press, 
                                                "until": None, "sequential": False,}, 
                                        daemon=True)
        listen_keyboard_thread.start()
        
        
        mocap_target_site = self._config.get('target_site_name', None)
        TCP_site = self._config.get("tcp_visualization_site", None)
        # @TODO: check how to integrate with hand
        self._init_pose = {}
        # @TODO: BUG HERE FOR DIMENSION MATCHING
        self._robot_index = ['left', 'right'] if len(self.ee_link) == 2 else ["single"]
        
        # teleoperation loop
        log.info(f'teleoperation loop started!!')
        target_period = self._teleoperation_loop_time
        next_run_time = time.perf_counter()
        slow_loop_count = 0
        while self._teleop_thread_running:
            loop_start_time = time.perf_counter()
            
            # get interface target
            inteface_output_mode = self._interface_output_mode
            success_get_target, ee_target, tool_target  = \
                self._interface.parse_data_2_robot_target(inteface_output_mode)
            # log.info(f'[DEBUG] Interface: success={success_get_target}, ee_target_keys={list(ee_target.keys()) if ee_target else None}')
            
            # only for mujoco 
            if self._use_simulation_target and not mocap_target_site is None:
                ee_target = {}
                for i, target_site in enumerate(mocap_target_site):
                    # @TODO: BUG HERE FOR DIMENSION MATCHING
                    key = self._robot_index[i]
                    # if i == 0 and len(mocap_target_site) == 1:
                    #     key = 'single'
                    cur_sim_target = self._robot_system._simulation.get_site_pose(target_site, 'xyzw')
                    cur_base_pose = self.base2world_pose[0] if len(self.base2world_pose) == 1 else self.base2world_pose[i]
                    cur_sim_target = transform_pose(cur_base_pose, cur_sim_target)
                    ee_target[key] = cur_sim_target
                inteface_output_mode = 'absolute'
            
            cur_tcp_pose = {}
            for i, cur_ee_link in enumerate(self.ee_link):
                key = self._robot_index[i]
                cur_tcp_pose[key] =  self._robot_motion_system.get_frame_pose(cur_ee_link, key)
                # visualize the curr tcp
                if not TCP_site is None:
                    cur_tcp = cur_tcp_pose[key]
                    cur_world2base = self.world2base_pose[0] if len(self.world2base_pose) == 1 else self.world2base_pose[i]
                    tcp = transform_pose(cur_world2base, cur_tcp)
                    
                    cur_tcp_mocap = TCP_site[i]
                    tcp_mocap = cur_tcp_mocap.split('_')[0]
                    self._robot_system._simulation.set_target_mocap_pose(tcp_mocap, tcp)
                
            if success_get_target or self._use_simulation_target:
                high_level_command = np.array([])
                for i, (key, cur_ee_target) in enumerate(ee_target.items()):
                    # Incremental target on the ee pose
                    if inteface_output_mode == 'relative':
                        # hack!!!
                        if len(self._init_pose) != len(ee_target):
                            self._init_pose[key] = cur_tcp_pose[key]
                        
                        self._init_pose[key][:3] += cur_ee_target[:3]
                        cur_mat = R.from_quat(self._init_pose[key][3:]).as_matrix()
                        ee_mat = R.from_euler('xyz', cur_ee_target[3:]).as_matrix()
                        # print(f'delta rot: {ee_mat}')
                        ee_mat = cur_mat @ ee_mat
                        self._init_pose[key][3:] = R.from_matrix(ee_mat).as_quat()
                        ee_target[key] = self._init_pose[key]
                    
                    # log.info(f'ee site target {key}: {ee_target[key]}')
                    
                    # visualization of the target pose for ee 
                    # (Not using simulation for target tracking)
                    if not self._use_simulation_target and not mocap_target_site is None:
                        cur_mocap_target_site = mocap_target_site[i]
                        mocap_name = cur_mocap_target_site.split('_')[0]
                        # target_tcp = copy.deepcopy(ee_target_7D)
                        target_tcp = ee_target[key]
                        cur_world2base = self.world2base_pose[0] if len(self.world2base_pose) == 1 else self.world2base_pose[i]
                        target_tcp = transform_pose(cur_world2base, target_tcp)
                        self._robot_system._simulation.set_target_mocap_pose(mocap_name, target_tcp)
                
                    high_level_command = np.hstack((high_level_command, ee_target[key]))
                
                if self._update_high_level_state:
                    self._robot_motion_system.update_high_level_command(high_level_command)
                if success_get_target:
                    self._robot_system.set_tool_command(tool_target)
                
            # Direct gripper control - tool layer handles incremental vs binary mode
            if success_get_target and tool_target:
                self._robot_system.set_tool_command(tool_target)

            
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
    
            
    def traj_visual_task(self, buffer: Buffer, lock: threading.Lock, sim: SimBase):
        while self._teleop_thread_running:
            start_time = time.perf_counter()
            lock.acquire()  
            buffer_size = buffer.size()
            # log.info(f'buffer size: {buffer_size}')
            if buffer_size !=0: 
                traj_data = buffer._data[buffer_size - 1]
                lock.release()  
                for i, _ in enumerate(self.ee_link):
                    cur_world2base = self.world2base_pose[0] if len(self.world2base_pose) == 1 else self.world2base_pose[i]
                    if i == 0:
                        cur_data = transform_pose(cur_world2base, traj_data[:7])
                    else:
                        cur_data = transform_pose(cur_world2base, traj_data[7:14])
                    sim.update_trajectory_data(cur_data)
            else:
                lock.release()
                
            used_time = time.perf_counter() - start_time
            if used_time < 0.003:
                time.sleep(0.003 - used_time)
        log.info(f'teleoperation traj visual task stopped!!!!')

    def add_teleoperation_data(self):
        log.info(f'Add teleoperation data thread started!!!')
        
        start_time = time.time()
        while self._teleop_thread_running:
            # parse camera data
            cameras_data = self._robot_system.get_cameras_infos()
            image_list = []
            if cameras_data is not None:
                cur_colors = {}
                cur_depths = {}
                cur_imus = {}
                for cam_data in cameras_data:
                    name = cam_data['name']
                    if 'color' in name:
                        cur_colors[name] = cam_data['img']
                        image_list.append(cam_data['img'])
                    if 'depth' in name:
                        cur_depths[name] = cam_data['img']
                    if 'imu' in name:
                        cur_imus[name] = cam_data['imu']
                        
            # image visualization 
            if len(image_list) and (self._img_visualization or self._img_shm is not None):
                combined_imgs = image_list[0]
                for i in range(1, len(image_list)):
                    combined_imgs = combine_image(combined_imgs, image_list[i])
                if self._img_visualization:
                    cv2.imshow('combined image', combined_imgs)
                    cv2.waitKey(1)
                # xr visualization 
                if self._img_shm is not None:
                    combined_imgs = cv2.resize(combined_imgs, self._xr_img_shape[1::-1])
                    np.copyto(self._image_array, np.array(combined_imgs))
                    
            # recording episode data
            if self._enable_recording and self.data_recorder is not None:
                # get robot propioceptive info 
                joint_states = {}; ee_states = {}; gripper_state = {}
                all_joint_states = self._robot_system.get_joint_states()
                
                # Get end effector links properly
                ee_links = self.ee_link
                if isinstance(ee_links, list):
                    robot_index = ['left', 'right'] if len(ee_links) == 2 else ["single"]
                    for i, cur_ee_link in enumerate(ee_links):
                        key = robot_index[i]
                        sliced_joint_states = self._robot_motion_system.get_type_joint_state(
                                                        all_joint_states, key)
                        joint_states[key] = {}
                        joint_states[key]["position"] = sliced_joint_states._positions.tolist()
                        joint_states[key]["velocitie"] = sliced_joint_states._velocities.tolist()
                        joint_states[key]["acceleration"] = sliced_joint_states._accelerations.tolist()
                        joint_states[key]["torque"] = sliced_joint_states._torques.tolist()
                        cur_ee_pose = self._robot_motion_system.get_frame_pose(cur_ee_link, key)
                        ee_states[key] = cur_ee_pose.tolist()
                        
                        cur_tool_state = self._robot_system._tool.get_tool_state()
                        if not isinstance(cur_tool_state, dict):
                            tool_state = {"single": cur_tool_state}
                        else: 
                            tool_state = cur_tool_state
                        gripper_state[key] = {}
                        # @TODO: get tools info, all elements
                        gripper_state[key]['position'] = tool_state[key]._position
                elif isinstance(ee_links, dict):
                    for key, cur_ee_link in ee_links.items():
                        sliced_joint_states = self._robot_motion_system.get_type_joint_state(
                                                        all_joint_states, key)
                        joint_states[key] = {}
                        joint_states[key]["position"] = sliced_joint_states._positions.tolist()
                        joint_states[key]["velocitie"] = sliced_joint_states._velocities.tolist()
                        joint_states[key]["acceleration"] = sliced_joint_states._accelerations.tolist()
                        joint_states[key]["torque"] = sliced_joint_states._torques.tolist()
                        cur_ee_pose = self._robot_motion_system.get_frame_pose(cur_ee_link, key)
                        ee_states[key] = cur_ee_pose.tolist()
                        
                        cur_tool_state = self._robot_system._tool.get_tool_state()
                        if not isinstance(cur_tool_state, dict):
                            tool_state = {"single": cur_tool_state}
                        else: 
                            tool_state = cur_tool_state
                        gripper_state[key] = {}
                        gripper_state[key]['position'] = tool_state[key]._position
                
                # get sensor readings
                colors = None
                depths = None
                imus = None
                if len(cur_colors):
                    colors = cur_colors
                if len(cur_depths):
                    depths = cur_depths
                if len(cur_imus):
                    imus = cur_imus
                
                # @TODO: get tactile data
                tactiles = None
                
                self.data_recorder.add_item(colors=colors, depths=depths, tools=gripper_state,
                                            joint_states=joint_states, ee_states=ee_states,
                                            imus=imus, tactiles=tactiles)
                
            used_time = time.time() - start_time
            if used_time < (1.0 / self._data_record_frequency):
                time.sleep((1.0 / self._data_record_frequency) - used_time)
            start_time = time.time()
            
        print(f'Add teleoperation data thread stopped!!!')  
                
    def _keyboard_on_press(self, key):
        if key == 'h':
            self._enable_hardware = not self._enable_hardware
            print(f"{'='*15}Hardware execution status {self._enable_hardware}!!!.{'='*15}")
            self._robot_motion_system.update_execute_hardware(
                                        self._enable_hardware)               
        elif key == 'q' and self._is_initialized:
            print(f"{'='*15}Closing the teleoperation thread!!!{'='*15}")
            stop_listening()
            self._teleop_thread_running = False
            if self._robot_system._use_simulation and self._robot_motion_system._use_traj_planner:
                self._traj_visual_thread.join()
            self._data_recording_thread.join()
            self._robot_motion_system.close()
            self._interface.close()
            if self.data_recorder is not None:
                self.data_recorder.close()
            if self._img_visualization:
                cv2.destroyAllWindows()
        elif key == 'r':
            self._enable_recording = not self._enable_recording
            if self._enable_recording and self.data_recorder is None:
                os.makedirs(self._save_path_dir, exist_ok=True)
                print(f"{'='*15}Build data recoreder at {self._save_path_dir}{'='*15}")
                self.data_recorder = EpisodeWriter(task_dir=self._save_path_dir, 
                                                rerun_log=self._visual_data,
                                                task_description=self._task_description,
                                                task_description_goal=self._task_description_goal,
                                                task_description_steps=self._task_description_step)
            if self._enable_recording: # start record a new episode
                # Record the episode data
                if not self.data_recorder.create_episode():
                    warnings.warn(f'Episode write failed to create a episode for recording data!!!!')
                else:
                    print(f"{'='*15}Data recorder started to write the episode data!!!!{'='*15}")
            else: # finish the episode write
                self.data_recorder.save_episode()
                time.sleep(0.5)
                print(f"{'='*15}Data recorder stoped recording the episode data!!!!{'='*15}")
        elif key == 'o':
            # move to start
            self._update_high_level_state = False
            self._robot_motion_system.update_execute_hardware(False)
            print(f"{'='*20}, Blocking the Motion process to reset the robot to init state{'='*20}")
            self._robot_motion_system.move_to_start_blocking()
            self._init_pose = {}
            time.sleep(1.5)
            self._update_high_level_state = True
            self._robot_motion_system.update_execute_hardware(True)
            self._robot_motion_system.clear_traj_buffer()
            time.sleep(0.5)
            print(f"{'='*20}, Motion resumes normal!!!{'='*20}")
            