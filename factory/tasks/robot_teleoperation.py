from teleop.base.teleoperation_base import TeleoperationDeviceBase
from teleop.space_mouse.space_mouse import SpaceMouse, DuoSpaceMouse

# Try to import MetaQuest3, fall back to mock if not available
try:
    from teleop.XR.quest3.meta_quest3 import MetaQuest3, init_image_shared_mem
except (ImportError, ModuleNotFoundError):
    import glog as log
    log.warning("teleop.XR.quest3.meta_quest3 not available, MetaQuest3 interface will not work")
    # Create mock classes to allow import
    class MetaQuest3(TeleoperationDeviceBase):
        def __init__(self, config):
            raise NotImplementedError("MetaQuest3 is not available. Install XR dependencies to use this interface.")
    def init_image_shared_mem(config):
        raise NotImplementedError("MetaQuest3 is not available. Install XR dependencies to use this interface.")
    
from teleop.pika_tracker.pika_tracker import PikaTracker
from factory.components.motion_factory import MotionFactory, Robot_Space
from factory.components.robot_factory import RobotFactory
from simulation.base.sim_base import SimBase
from dataset.lerobot.data_process import EpisodeWriter
from hardware.base.utils import convert_homo_2_7D_pose, Buffer, negate_pose, transform_pose, object_class_check
from hardware.base.utils import ToolState, ToolType
from hardware.base.img_utils import combine_image, combine_images_2x2_grid
from teleop.base.utils import RisingEdgeDetector
import warnings, os
import numpy as np

# 确保teleop模式环境变量被设置
os.environ['TELEOP_MODE'] = 'true'
import threading, time, copy
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from sshkeyboard import listen_keyboard, stop_listening
import glog as log
import cv2
from tools.performance_profiler import PerformanceProfiler, timer

# Used for different robot component into one robot system
class TeleoperationFactory:
    _robot_motion_system: MotionFactory
    _robot_system: RobotFactory
    _interface: TeleoperationDeviceBase
    def __init__(self, config, robot_motion: MotionFactory):
        self._config = config
        self._robot_motion_system = robot_motion
        self._teleop_interface_type = config["teleop_interface"]
        self._teleop_target = None
        self._interface_output_mode = config["inteface_output_mode"]
        self._use_simulation_target = config["use_simulation_target"]
        self._teleoperation_loop_time = config["teleoperation_loop_time"]
        self._reset_arm_command = config.get("reset_arm_command", None)
        self._reset_space = config.get("reset_space", None)
        self._reset_space = Robot_Space(self._reset_space) if not self._reset_space is None else None
        log.info(f'reset space: {self._reset_space}')
        self._reset_tool_command = config.get("reset_tool_command", None)
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
        self._tool_action = {}
        self._tool_action_lock = threading.Lock()
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
            'meta_quest3': MetaQuest3,
            'pika_tracker': PikaTracker,
        }

    def create_robot_teleoperation_system(self) -> bool:
        if not object_class_check(self._interface_classes, self._teleop_interface_type):
            raise ValueError(f'Teleoperation interface {self._teleop_interface_type} is not supported')
        teleoperation_cfg = self._config["interface_config"][self._teleop_interface_type]
        self._interface = None
        if not self._use_simulation_target:
            if self._teleop_interface_type == 'meta_quest3':
                self._img_shm = init_image_shared_mem(teleoperation_cfg)
                self._xr_img_shape = teleoperation_cfg['image_shape']
                self._image_array = np.ndarray(self._xr_img_shape, 
                                            dtype=np.uint8, buffer=self._img_shm.buf)
                self._reset_rising_edges = dict(reset_hardware=RisingEdgeDetector(), 
                                                reset_record=RisingEdgeDetector(),
                                                quit=RisingEdgeDetector())
            self._interface = self._interface_classes[self._teleop_interface_type](teleoperation_cfg)
        else:
            if not 'target_site_name' in self._config:
                raise ValueError(f'Teleoperation with simulation target without defined target site!')

        # initialize all objects
        self._initialize()
        
    def _initialize(self) -> bool:
        if self._interface:
            if not self._interface.initialize():
                log.error(f"Teleoperation interface {self._teleop_interface_type} failed intialization")
        self._robot_motion_system.create_motion_components()
        self._robot_system = self._robot_motion_system._robot_system
        if not self._interface and not self._robot_system._use_simulation:
            raise ValueError(f'Teleoperation with simulation target but not enable simulation in config!!!')
        
        # base frames
        self.world2base_pose, self.base2world_pose = self._robot_motion_system.get_sim_base_world_transform()
        log.info(f'world2base (real robot): {self.world2base_pose}')
        log.info(f'base2world (real robot): {self.base2world_pose}')
       
        self.ee_link = self._robot_motion_system.get_model_end_effector_link_list()
        log.info(f'ee links: {self.ee_link}')
        
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
        # @TODO: check how to integrate with hand
        self._init_pose = {}
        self._robot_index = self._robot_motion_system.get_model_types()
        self._ee_index = ['left', 'right'] if len(self.ee_link) > 1 else ['single']
        log.info(f'ee index in robot teleoperation: {self._ee_index}')
        
        # performance profiler
        PerformanceProfiler.clear_stats()
        
        # teleoperation loop
        log.info(f'teleoperation loop started!!')
        target_period = self._teleoperation_loop_time
        next_run_time = time.perf_counter()
        slow_loop_count = 0
        while self._teleop_thread_running:
            loop_start_time = time.perf_counter()
            
            # get interface target
            interface_output_mode = self._interface_output_mode
            with timer("get_interface_target", "robot_teleoperation_"):
                if self._interface:
                    success_get_target, ee_target, tool_target  = \
                        self._interface.parse_data_2_robot_target(interface_output_mode)
                    # log.info(f'tool target: {tool_target}')
                else: success_get_target=False; ee_target=None; tool_target=None
                            
            # only for mujoco 
            if self._use_simulation_target and not mocap_target_site is None and self._robot_system._use_simulation:
                ee_target = {}
                for i, target_site in enumerate(mocap_target_site):
                    key = self._ee_index[i]
                    # log.info(f'target site in sim: {target_site} for {key}')
                    cur_sim_target = self._robot_system._simulation.get_site_pose(target_site, 'xyzw')
                    cur_base_pose = self.base2world_pose[0] if len(self.base2world_pose) == 1 else self.base2world_pose[i]
                    cur_sim_target = transform_pose(cur_base_pose, cur_sim_target)
                    ee_target[key] = cur_sim_target
                interface_output_mode = 'absolute'
            
            # get curr tcp
            cur_tcp_pose = {}
            for i, cur_ee_link in enumerate(self.ee_link):
                key = self._robot_index[i] if len(self._robot_index) > 1 else self._robot_index[0]
                cur_tcp_pose[self._ee_index[i]] = self._robot_motion_system.get_frame_pose(cur_ee_link, key)
                # log.info(f'tcp pose {cur_tcp_pose[key]} for {key}')
            self._robot_motion_system.sim_visualize_tcp(cur_tcp_pose)
            
            # parse the teleoperation target to robot
            if (success_get_target or (self._robot_system._use_simulation and self._use_simulation_target)) and self._update_high_level_state:
                high_level_command = np.array([])
                with timer("parse_target", "robot_teleoperation_"):
                    for i, (key, cur_ee_target) in enumerate(ee_target.items()):
                        # Incremental target on the ee pose
                        if interface_output_mode == 'relative':
                            if len(self._init_pose) != len(ee_target):
                                self._init_pose[key] = cur_tcp_pose[key]
                            
                            self._init_pose[key][:3] += cur_ee_target[:3]
                            cur_mat = R.from_quat(self._init_pose[key][3:]).as_matrix()
                            ee_mat = R.from_euler('xyz', cur_ee_target[3:]).as_matrix()
                            ee_mat = cur_mat @ ee_mat
                            self._init_pose[key][3:] = R.from_matrix(ee_mat).as_quat()
                            ee_target[key] = self._init_pose[key]
                        elif interface_output_mode == 'absolute_delta':
                            # @TODO: check: after reset, the init pose could only be reset!!!!
                            if tool_target[key][-1] and len(self._init_pose) != len(self.ee_link):
                                self._init_pose[key] = cur_tcp_pose[key]
                                log.info(f"{'='*10} updated the robot neutral pose for {key}: {self._init_pose[key]} {'='*10} ")
                            if len(self._init_pose) == 0: break
                            ee_target[key] = transform_pose(self._init_pose[key], cur_ee_target, True)
                        elif interface_output_mode != "absolute":
                            raise ValueError(f"Teleoperation interface {interface_output_mode} is not supported")
                        # @TODO: if mode is absolute your reset target will be same as before inside simulation
                    
                        high_level_command = np.hstack((high_level_command, ee_target[key]))
                    
                    # skip the current, bug here to check @TODO: zyx
                    if len(high_level_command) == 0: 
                        log.info(f'Len of high level command is 0!')
                        success_get_target = False
                    else:
                        self._robot_motion_system.update_high_level_command(high_level_command)
                        self._teleop_target = high_level_command
                        # visualize targets in sim (Not using simulation for target tracking)
                        if not self._use_simulation_target:
                            self._robot_motion_system.sim_visualize_targets(ee_target)
                
                if success_get_target:
                    # coupling with vr
                    if self._teleop_interface_type == "meta_quest3":
                        whether_to_reset_hardware = self._reset_rising_edges["reset_hardware"].update(float(tool_target['reset'][0]))
                        if whether_to_reset_hardware:
                            self._reset_hardware()
                        whether_to_reset_record = self._reset_rising_edges["reset_record"].update(float(tool_target['reset'][1]))
                        if whether_to_reset_record:
                            self._reset_recording()
                        if tool_target["reset"][2]:
                            self._reset_robot()
                        whether_to_quit = self._reset_rising_edges["quit"].update(float(tool_target['reset'][3]))
                        if whether_to_quit:
                            self.close()
                    
                    
                    tool_type_dict = self._robot_system.get_tool_type_dict()
                    if self._enable_recording and tool_type_dict is not None:
                        with self._tool_action_lock:
                            for key, tool_command in tool_target.items():
                                tool_command = np.array(tool_command[:-1])
                                if tool_type_dict[key] == ToolType.GRIPPER or \
                                   tool_type_dict[key] == ToolType.SUCTION:
                                    if isinstance(tool_command, np.ndarray) and tool_command.ndim != 0:
                                        tool_command = tool_command[0].astype(np.float32)
                                tool_target[key] = tool_command
                                # make sure the action is list/float for writing json
                                if not isinstance(tool_command, np.ndarray):
                                    tool_command = float(tool_command)
                                else: tool_command = tool_command.tolist()
                                self._tool_action[key] = dict(tool=dict(
                                    position=tool_command, time_stamp=time.perf_counter()))
                    # log.info(f'tool target: {tool_target}')
                    # tool_target = dict(single=np.array([0,0]))
                    self._robot_motion_system.set_tool_command(tool_target)
            # for torque control, handling pausing period of teleoperation device
            elif self._teleop_target is not None and self._update_high_level_state:
                self._robot_motion_system.update_high_level_command(self._teleop_target)

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
                    # log.warning(f"Teleoperation frequency slow: expected {expected_freq:.1f}Hz, "
                                # f"actual {actual_freq:.1f}Hz (warning #{slow_loop_count})")

                # 重置时间基准，避免更大的延迟
                next_run_time = current_time
        log.info(f'Teleoperation thread for reading is stopped!')
    
    def add_teleoperation_data(self):
        log.info(f'Add teleoperation data thread started!!!')
        
        start_time = time.perf_counter()
        while self._teleop_thread_running:
            cameras_data = self._robot_system.get_cameras_infos()
            image_list = []
            cur_colors = {}; cur_depths = {}; cur_imus = {}
            if cameras_data is not None:
                
                for cam_data in cameras_data:
                    name = cam_data['name']
                    if 'color' in name:
                        cur_colors[name] = {"data": cam_data['img'], "time_stamp": cam_data['time_stamp']}
                        image_list.append(cam_data['img'])
                    if 'depth' in name:
                        cur_depths[name] = {"data": cam_data['img'], "time_stamp": cam_data['time_stamp']}
                    if 'imu' in name:
                        cur_imus[name] = {"data": cam_data['imu'], "time_stamp": cam_data['time_stamp']}

            # image visualization
            if len(image_list) and (self._img_visualization or self._img_shm is not None):
                combined_imgs = combine_images_2x2_grid(image_list)
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
                tool_state_dict = self._robot_system.get_tool_dict_state()
                ft_data = self._robot_system.get_ft_data()
                
                # Get end effector links properly
                for i, cur_ee_link in enumerate(self.ee_link):
                    key = self._ee_index[i]
                    robot_key = self._robot_index[i] if len(self._robot_index) > 1 else self._robot_index[0]
                    sliced_joint_states = self._robot_motion_system.get_type_joint_state(
                                                    all_joint_states, robot_key)
                    joint_states[key] = {}
                    joint_states[key]["position"] = sliced_joint_states._positions.tolist()
                    joint_states[key]["velocity"] = sliced_joint_states._velocities.tolist()
                    joint_states[key]["acceleration"] = sliced_joint_states._accelerations.tolist()
                    joint_states[key]["torque"] = sliced_joint_states._torques.tolist()
                    joint_states[key]["time_stamp"] = sliced_joint_states._time_stamp
                    cur_ee_pose = self._robot_motion_system.get_frame_pose_with_joint_state(
                                all_joint_states, cur_ee_link, robot_key, need_vel=True)
                    ee_states[key] = {}
                    ee_states[key]["pose"] = cur_ee_pose[:7].tolist()
                    ee_states[key]["twist"] = cur_ee_pose[7:13].tolist()
                    ee_states[key]["time_stamp"] = sliced_joint_states._time_stamp
                    # @TODO: parsed FT sensor data, wait to be checked!!!
                    if ft_data:
                        if len(self._ee_index) == 1:
                            ee_states[key]["ft"] = ft_data [0]["data"]
                            ee_states[key]["ft_time_stamp"] = ft_data[0]["time_stamp"]
                        else:
                            for cur_ft_data in ft_data:
                                if key in cur_ft_data["name"]:
                                    ee_states[key]["ft"] = cur_ft_data["data"]
                                    ee_states[key]["ft_time_stamp"] = cur_ft_data["time_stamp"]
                                    break
                            
                        
                    # get tool state
                    if tool_state_dict is not None:
                        gripper_state[key] = {}
                        if isinstance(tool_state_dict[key]._position, np.ndarray):
                            tool_state_dict[key]._position = tool_state_dict[key]._position.tolist()
                        gripper_state[key]['position'] = tool_state_dict[key]._position
                        gripper_state[key]["time_stamp"] = tool_state_dict[key]._time_stamp

                # get sensor readings
                colors = None; depths = None; imus = None
                if len(cur_colors):
                    colors = cur_colors
                if len(cur_depths):
                    depths = cur_depths
                if len(cur_imus):
                    imus = cur_imus
                
                # get tactile data
                tactiles = self._robot_system.get_tactile_data()
                
                # get actions
                motion_action = self._robot_motion_system.get_latest_action()
                if motion_action is not None and len(self._tool_action) != 0:
                    with self._tool_action_lock:
                        actions = copy.deepcopy(self._tool_action)
                    for key in list(actions.keys()):
                        actions[key]["joint"] = motion_action[key]["joint"]
                        actions[key]["ee"] = motion_action[key]["ee"]
                    # log.info(f'action: {actions}, type:{type(actions["single"]["tool"]["position"])}')
                    # log.info(f'update data, ee states: {ee_states["single"]["ft"]}, {type(ee_states["single"]["ft"])}')
                    self.data_recorder.add_item(colors=colors, depths=depths, tools=gripper_state,
                                                joint_states=joint_states, ee_states=ee_states,
                                                imus=imus, tactiles=tactiles, actions=actions)
                
            used_time = time.perf_counter() - start_time
            if used_time < (1.0 / self._data_record_frequency):
                time.sleep((1.0 / self._data_record_frequency) - used_time)
            start_time = time.perf_counter()
            
        log.info(f'Add teleoperation data thread stopped!!!')  
                
    def _keyboard_on_press(self, key):
        if key == 'h':
            self._reset_hardware()     
        elif key == 'q' and self._is_initialized:
            self.close()
        elif key == 'r':
            self._reset_recording()
        elif key == 'o':
            self._reset_robot()
    
    def _reset_recording(self):
        self._enable_recording = not self._enable_recording
        if self._enable_recording and self.data_recorder is None:
            os.makedirs(self._save_path_dir, exist_ok=True)
            log.info(f"{'='*15}Build data recoreder at {self._save_path_dir}{'='*15}")
            self.data_recorder = EpisodeWriter(task_dir=self._save_path_dir, 
                                            rerun_log=self._visual_data,
                                            task_description=self._task_description,
                                            task_description_goal=self._task_description_goal,
                                            task_description_steps=self._task_description_step)
        if self._enable_recording: # start record a new episode
            # Record the episode data
            self._robot_motion_system.change_update_action_status(True)
            if not self.data_recorder.create_episode():
                warnings.warn(f'Episode write failed to create a episode for recording data!!!!')
            else:
                log.info(f"{'='*15}Data recorder started to write the episode data!!!!{'='*15}")
        else: # finish the episode write
            self.data_recorder.save_episode()
            self._robot_motion_system.change_update_action_status(False)
            time.sleep(0.5)
            log.info(f"{'='*15}Data recorder stoped recording the episode data!!!!{'='*15}")
    
    def _reset_robot(self):
        # move to start
        self._update_high_level_state = False
        log.info(f"{'='*20}, Blocking the Motion process to reset the robot to init state{'='*20}")
        log.info(f'reset space: {self._reset_space}, command: {self._reset_arm_command}')
        self._robot_motion_system.reset_robot_system(self._reset_arm_command, self._reset_space,
                                                        self._reset_tool_command)
        if not self._robot_motion_system._use_traj_planner:
            self._robot_motion_system.clear_traj_buffer()
            self._robot_motion_system.wait_buffer_empty()
            self._robot_motion_system.clear_high_level_command()
        # @TODO: Reset target!!!!
        self._init_pose = {}
        self._teleop_target = None
        time.sleep(0.01)
        self._update_high_level_state = True
        log.info(f"{'='*20}, Motion resumes normal!!!{'='*20}")
    
    def _reset_hardware(self):
        if self._enable_hardware:
            return 
        
        self._enable_hardware = not self._enable_hardware
        log.info(f"{'='*15}Hardware execution status {self._enable_hardware}!!!.{'='*15}")
        self._robot_motion_system.update_execute_hardware(
                                    self._enable_hardware)            
    
    def close(self):
        log.info(f"{'='*15}Closing the teleoperation thread!!!{'='*15}")
        self._update_high_level_state = False
        self._robot_motion_system.update_execute_hardware(False)
        stop_listening()
        self._teleop_thread_running = False
        self._data_recording_thread.join()
        self._robot_motion_system.close()
        self._interface.close()
        if self.data_recorder is not None:
            self.data_recorder.close()
        if self._img_visualization:
            cv2.destroyAllWindows()
            
