from hardware.base.arm import ArmBase
from hardware.base.tool_base import ToolBase
from hardware.base.utils import ToolType
from hardware.base.camera import CameraBase
from simulation.base.sim_base import SimBase
from hardware.fr3.fr3_arm import Fr3Arm
from hardware.duo_arm import DuoArm
from hardware.agibot_g1.agibot_g1 import AgibotG1
from hardware.monte01.monte01 import Monte01
from hardware.fr3.franka_hand import FrankaHand
from hardware.duo_tool import DuoTool
from simulation.mujoco.mujoco_sim import MujocoSim
from hardware.base.camera import CameraBase
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from hardware.sensors.cameras.opencv_camera import OpencvCamera
from hardware.sensors.cameras.agibot_cameras import AgibotCamera
from hardware.sensors.cameras.ros2_camera import Ros2Camera
from hardware.sensors.ft_sensor.ati_ft import AtiFt
from hardware.sensors.cameras.network_camera import NetworkCamera
from hardware.base.tactile_base import TactileBase
from hardware.sensors.paxini_tactile.paxini_serial_sensor import PaxiniSerialSensor
from hardware.sensors.paxini_tactile.paxini_network_sensor import PaxiniNetworkSensor
import threading
import time, copy
from hardware.base.utils import object_class_check
from controller.utils.weighted_moving_filter import WeightedMovingFilter
import numpy as np
from typing import Optional, Union, List, Dict, Any
import glog as log
from factory.components.learning_inference_factory import LearningInferenceFactory

# Import smoother modules
from smoother.smoother_base import SmootherBase
from smoother.critical_damped_smoother import CriticalDampedSmoother
from smoother.adaptive_critical_damped_smoother import AdaptiveCriticalDampedSmoother
from smoother.ruckig_smoother import RuckigSmoother

from tools.performance_profiler import timer

# Used for loading one complete robot hw system from the factory
class RobotFactory:
    _robot: ArmBase
    _tool: ToolBase
    _simulation: SimBase
    _smoother: Optional[SmootherBase]
    
    _learning_inference_engine: Optional[Any]
    _learning_data_adapter: Optional[Any]
    def __init__(self, config):
        self._config = config
        self._use_hardware = config['use_hardware']
        self._use_simulation = config['use_simulation']
        self._robot_type = config['robot']
        self._gripper_type = config.get('gripper', None)
        self._simulation_type = config["simulation"]
        self._sensor_dicts = config.get("sensor_dicts", None)
        self._sensors = {}
        self._tool = None
        self._enable_hardware = False
        
        # Smoother configuration
        self._use_smoother = config.get('use_smoother', False)
        self._smoother = None
        
        # Async control mode
        self._async_mode = False
        self._async_thread = None
        self._async_running = False
        self._async_frequency = config.get('async_control_frequency', 800.0)
        
        # robot action
        self._last_robot_action = {}
        self._last_robot_action_lock = threading.Lock()
        self._update_action = False
        
        # @TODO: delete Learning inference components
        self._learning_config = config.get("learning", None)
        self._learning_inference_engine = None
        self._learning_data_adapter = None
            
        # object classes
        self._robot_classes = {
            'fr3': Fr3Arm,
            'agibot_g1': AgibotG1,
            'monte01': Monte01,
            'duo_arm': DuoArm
        }
       
        self._gripper_classes = {
            'franka_hand': FrankaHand,
            'duo_tool': DuoTool
        }
        
        self._camera_classes = {
            'realsense_camera': RealsenseCamera,
            'opencv_camera': OpencvCamera,
            'agibot_camera': AgibotCamera,
            'ros2_camera': Ros2Camera,
            'network_camera': NetworkCamera,
        }
        
        self._tactile_classes = {
            'paxini_serial_sensor': PaxiniSerialSensor,
            'paxini_network_sensor': PaxiniNetworkSensor,
        }
        
        self._simulation_classes = {
            'mujoco': MujocoSim
        }
        
        self._smoother_classes = {
            'critical_damped': CriticalDampedSmoother,
            'adaptive_critical_damped': AdaptiveCriticalDampedSmoother,
            'ruckig': RuckigSmoother
        }
    
    def create_robot_system(self):
        # platforms
        if self._use_hardware:
            if not object_class_check(self._robot_classes, self._robot_type):
                raise ValueError
            self._robot = self._robot_classes[self._robot_type](self._config["robot_config"][self._robot_type])
            
            # Initialize tool system - support both legacy _tool and modern _grippers
            # @TODO: get tool
            if self._gripper_type is not None:
                if object_class_check(self._gripper_classes, self._gripper_type):
                    self._tool = self._gripper_classes[self._gripper_type](self._config["gripper_config"][self._gripper_type])
            
            # sensors
            if self._sensor_dicts is not None:
                if 'cameras' in self._sensor_dicts:
                    # cameras
                    cameras_info = self._sensor_dicts["cameras"]
                    log.info(f"{cameras_info}")
                    cameras_objects = []
                    num_camera = 0
                    for cam_info in cameras_info:
                        if not object_class_check(self._camera_classes, cam_info['type']):
                            log.error(f"ValueError")
                            raise ValueError
                        cam_type = cam_info['type']
                        log.info(f"{cam_type}")
                        log.info(f" : {cam_info['cfg'][cam_type]}")
                        log.info(f" : {self._camera_classes[cam_type]}")
                        cam = self._camera_classes[cam_type](cam_info['cfg'][cam_type])

                        cameras_objects.append({'name': cam_info['name'], 'object': cam})
                        log.info(f"Add one hw camera {cam_info['name']}")
                        num_camera += 1
                    if num_camera:
                        self._sensors['camera'] = cameras_objects  
                        
                # tactile sensors
                if 'tactile' in self._sensor_dicts:
                    tactile_info = self._sensor_dicts["tactile"]
                    log.info(f"Tactile sensors config: {tactile_info}")
                    tactile_objects = []
                    num_tactile = 0
                    for tactile_info_item in tactile_info:
                        if not object_class_check(self._tactile_classes, tactile_info_item['type']):
                            log.error(f"Unknown tactile sensor type: {tactile_info_item['type']}")
                            raise ValueError(f"Unknown tactile sensor type: {tactile_info_item['type']}")
                        tactile_type = tactile_info_item['type']
                        log.info(f"Tactile sensor type: {tactile_type}")
                        log.info(f"Tactile config: {tactile_info_item['cfg']}")
                        
                        # Extract configuration based on sensor type
                        config_key = 'paxini_serial_sensor' if 'serial' in tactile_type else 'paxini_network_sensor'
                        tactile_config = tactile_info_item['cfg'][config_key]
                        
                        tactile_sensor = self._tactile_classes[tactile_type](tactile_config)
                        tactile_objects.append({'name': tactile_info_item['name'], 'object': tactile_sensor})
                        log.info(f"Added tactile sensor: {tactile_info_item['name']}")
                        num_tactile += 1
                    if num_tactile:
                        self._sensors['tactile'] = tactile_objects
                
                # FT
            
        if self._use_simulation:
            if not object_class_check(self._simulation_classes, self._simulation_type):
                raise ValueError
            self._simulation = self._simulation_classes[self._simulation_type](self._config["simulation_config"][self._simulation_type])
        
        # total dof
        total_dof = self.get_total_dofs()
        # Comment out filter as we'll use smoother instead when enabled
        # self.filter = WeightedMovingFilter([1.0/total_dof] * total_dof, total_dof)
        
        # Create smoother if enabled
        if self._use_smoother:
            log.info('Creating smoother!!!!')
            self._create_smoother(total_dof)
            print("use smoother",self._use_smoother)
        
        # initialize all objects
        self._initialize()
        
    def _create_smoother(self, dof: int) -> None:
        """Create smoother instance based on configuration"""
        try:
            smoother_type = self._config.get('smoother_type', 'critical_damped')
            self._smoother_config = self._config.get('smoother_config', {})[smoother_type]
            
            if not object_class_check(self._smoother_classes, smoother_type):
                log.warning(f"Unknown smoother type: {smoother_type}, smoother disabled")
                self._use_smoother = False
                return
            
            # Create smoother instance
            self._smoother = self._smoother_classes[smoother_type](self._smoother_config, dof)
            
            # Get initial joint positions for smoother initialization
            # This will be properly initialized after robot initialization
            log.info(f"Smoother created: {smoother_type} with omega_n={self._smoother_config.get('omega_n', 25.0)}")
            
        except Exception as e:
            log.error(f"Failed to create smoother: {e}")
            self._use_smoother = False
            self._smoother = None
    
        # Initialize learning inference components if configured
        # 注意：ACT推理运行器有独立的学习组件初始化逻辑，此处可跳过
        # self._initialize_learning_components()
        
    def _initialize(self):
        if self._use_hardware:
            if not self._robot.initialize():
                raise ValueError(f"robot hardware {self._robot_type} failed intialization")
                    
            if hasattr(self, '_grippers'):
                for side, gripper in self._grippers.items():
                    if gripper.initialize():
                        log.info(f"{side} gripper initialized successfully")
                    else:
                        log.error(f"{side} gripper initialization failed")
                        # Note: Not raising error to allow partial system operation
                        
            # Initialize sensors
            if self._tool is not None:
                if not self._tool.initialize():
                    raise ValueError(f"tool hardware {self._gripper_type} failed intialization")
            if len(self._sensors) != 0:
                possible_sensor_types = ['camera', 'FT_sensor', 'tactile', 'imu']
                for sensor_type in possible_sensor_types:
                    if sensor_type in self._sensors:
                        for sensor in self._sensors[sensor_type]:
                            success = sensor["object"].initialize()
                            sensor_name = sensor["name"]
                            if not success:
                                raise ValueError(f"{sensor_type} {sensor_name} failed to initialize!!!")
        
        # Initialize smoother with current joint positions
        if self._use_smoother and self._smoother is not None:
            try:
                initial_joints = self.get_joint_states()
                if initial_joints is not None:
                    initial_positions = initial_joints._positions
                    self._smoother.start(initial_positions)
                    log.info("Smoother initialized and started")
                else:
                    log.warning("Could not get initial joint positions for smoother")
                    self._use_smoother = False
            except Exception as e:
                log.error(f"Failed to initialize smoother: {e}")
                self._use_smoother = False
        
        # Auto-enable async control if configured
        self._auto_enable_async = self._config.get('auto_enable_async_control', False)
        if self._auto_enable_async and self._use_smoother:
            if self.enable_async_control():
                log.info("Async control auto-enabled based on configuration")
            else:
                log.warning("Failed to auto-enable async control")
                
    def get_joint_states(self):
        joint_states = None
        if self._use_simulation:
            joint_states = self._simulation.get_joint_states()
        if self._use_hardware:
            joint_states = self._robot.get_joint_states()
            
        return joint_states
    
    def get_tool_dict_state(self):
        if self._tool is None:
            return None
        
        cur_tool_state = self._tool.get_tool_state()
        if not isinstance(cur_tool_state, dict):
            tool_state = {"single": cur_tool_state}
        else: tool_state = cur_tool_state
        return tool_state
        
    def get_robot_dofs(self):
        if self._use_simulation:
            sim_dofs = self._simulation.get_dof()
            dofs = sim_dofs
        if self._use_hardware:
            hw_dofs = self._robot.get_dof()
            dofs = hw_dofs

        if self._use_hardware and self._use_simulation:
            if sim_dofs != hw_dofs:
                raise ValueError("the simulation dofs did not match with hw dofs: "
                                 f"sim: {sim_dofs}, hw: {hw_dofs}")
        return dofs
    
    def get_total_dofs(self):
        dofs = self.get_robot_dofs()
        total_dof = 0
        for dof in dofs:
            total_dof += dof
        return total_dof
        
    def set_joint_commands(self, joint_command, mode, execute_hardware: bool = False,
                           update_action = False, change_action_status = False):
        self._enable_hardware = execute_hardware
        if change_action_status:
            self._update_action = update_action
        # Check if we should use smoother
        should_use_smoother = self._should_use_smoother(mode)
        # log.info(f"Should use smoother: {should_use_smoother}, mode: {mode}")
        
        if should_use_smoother:
            # Update smoother target
            self._smoother.update_target(np.array(joint_command))
            
            # In async mode, just update target and return
            # The async thread will handle sending commands
            if self._async_mode:
                return  # Early return in async mode
            
            # Synchronous mode: get smoothed command and send it
            smoothed_command, is_active = self._smoother.get_command()
            if is_active:
                joint_command = smoothed_command
        
        # Only execute direct commands in synchronous mode or when smoother not used
        if not self._async_mode or not should_use_smoother:
            # log.info(f"Set joint command: mode: {mode}")
            self.set_robot_joint_command(joint_command, mode, execute_hardware, update_action)
                
    def set_robot_joint_command(self, joint_command, mode, execute_hardware:bool = True,
                                update_action = False):
        dofs = self.get_robot_dofs()
        if self._use_simulation:
            # mode assignment
            sim_mode = [mode[0]] * dofs[0]
            # log.info(f'mode: {mode}')
            if len(dofs) > 1:
                sim_mode_r = [mode[1]] * dofs[1]
                sim_mode = np.hstack((sim_mode, sim_mode_r))
                # @TODO: handle dof other than arms @zyx
                # total_dof = self.get_total_dofs()
                # sim_mode = [mode[0]] * total_dof
            self._simulation.set_joint_command(sim_mode, joint_command)
        if self._use_hardware and execute_hardware: 
            if len(mode) == 1:
                mode = mode[0]
            self._robot.set_joint_command(mode, joint_command)
        
        # update joint action data
        if update_action:
            with self._last_robot_action_lock:
                dof_list = self.get_robot_dofs()
                index = ["single"] if len(dof_list) == 1 else ["left", "right"]
                dof_list = [0] + dof_list
                for i, key in enumerate(index):
                    self._last_robot_action[key] = dict(
                        joint=dict(position=joint_command[dof_list[i]:dof_list[i+1]].tolist(), 
                                time_stamp=time.perf_counter())
                    )
        
    def _update_robot_action(self, action_dict: dict):
        if len(self._last_robot_action) == 0:
            return False
        
        self._last_robot_action_lock.acquire()
        for i, key in enumerate(list(self._last_robot_action.keys())):
            action_dict[key] = {}
            action_dict[key] = copy.deepcopy(self._last_robot_action[key])
        self._last_robot_action_lock.release()
        return True
        
    def _should_use_smoother(self, mode: Union[str, List[str]]) -> bool:
        """Check if smoother should be used for current mode"""
        # log.info(f'mode: {mode}, use: {self._use_smoother}, smoother: {self._smoother}')
        if not self._use_smoother or self._smoother is None:
            return False
        
        # Check if mode is position or velocity
        if isinstance(mode, list):
            # Check all modes
            return all(m in ['position', 'velocity'] for m in mode)
        else:
            return mode in ['position', 'velocity']
            
    def set_tool_command(self, tool_command: dict[str, np.ndarray]):
        if self._tool is None:
            log.debug(f'Tool is not connected')
            return False
        
        tool_type_dict = self._tool.get_tool_type_dict()
        for key, tool_type in tool_type_dict.items():
            if tool_type == ToolType.GRIPPER or tool_type == ToolType.SUCTION:
                tool_command[key] = np.array(tool_command[key])
                if tool_command[key].ndim != 0: 
                    tool_command[key] = tool_command[key][0]
            else:
                tool_command[key] = np.array(tool_command[key][::-1])
                
        if 'single' in tool_command:
            tool_command = tool_command["single"]
        return self._tool.set_tool_command(tool_command)
    
    def get_tool_type_dict(self):
        if self._tool is None:
            return None
        
        tool_type_dict = self._tool.get_tool_type_dict()
        return tool_type_dict

    def close(self):
        # Stop async control first if enabled
        if self._async_mode:
            self.disable_async_control()
        
        # Stop smoother if enabled
        if self._use_smoother and self._smoother is not None:
            try:
                self._smoother.stop()
                log.info('Smoother stopped successfully')
            except Exception as e:
                log.warning(f'Error stopping smoother: {e}')
        
        if self._use_simulation:
            self._simulation.close()
        if self._use_hardware:
            self._robot.close()
            # @TODO: tools gradually add
            if self._tool is not None:
                self._tool.stop_tool()
                log.info(f'All tools are successfully closed!!!')
            if len(self._sensors):
                # @TODO: gradually add the sensors
                possible_sensors = ['camera', 'FT_sensor', 'tactile', 'imu']
                for sensor_type in possible_sensors:
                    if sensor_type in self._sensors:
                        for sensor in self._sensors[sensor_type]:
                            sensor['object'].close()
                log.info(f'All sensors are closed successfully!!!!')
        log.info(f'Robot systyem is closed successfully!!!')
        
    def get_cameras_infos(self):
        cameras_data = None
        if self._use_simulation:
            cameras_data = self._simulation.get_all_camera_images()
        if'camera' in self._sensors and self._use_hardware:
            hw_camera_data = []
            cameras = self._sensors['camera']
            for cam in cameras:
                camera_name = cam['name']
                camera_object:CameraBase = cam['object']
                img = camera_object.capture_all_data()
                resolution = camera_object.get_resolution()
                if not img['image'] is None:
                    hw_camera_data.append({'name': camera_name+'_color', 'resolution': resolution,
                                        'img': img['image'],'time_stamp': img["time_stamp"]})
                if not img['depth_map'] is None:
                    hw_camera_data.append({'name': camera_name+'_depth', 'resolution': resolution,
                                        'img': img['depth_map'],'time_stamp': img["time_stamp"]})
                if not img['imu'] is None:
                    hw_camera_data.append({'name': camera_name+'_imu', 'resolution': resolution,
                                        'imu': img['imu'],'time_stamp': img["time_stamp"]})
            if len(hw_camera_data):
                cameras_data = hw_camera_data
        return cameras_data
    
    def get_tactile_data(self):
        """Get tactile sensor data from all tactile sensors"""
        tactile_data = {}
        if 'tactile' in self._sensors and self._use_hardware:
            tactile_sensors = self._sensors['tactile']
            for tactile in tactile_sensors:
                sensor_name = tactile['name']
                sensor_object: TactileBase = tactile['object']
                success, data, timestamp = sensor_object.read_tactile_data()
                if success and data is not None:
                    tactile_data[sensor_name] = {
                        'data': data,
                        'timestamp': timestamp,
                        'shape': data.shape
                    }
                else:
                    log.warning(f"Failed to read data from tactile sensor: {sensor_name}")
        return tactile_data
            
    def move_to_start(self, joint_commands = None, mode = None):
        """
        Move robot to start position
        Args:
            joint_commands: If provided, use smoother to move smoothly to target
                           If None, use robot's default move_to_start (immediate)
        """
        if joint_commands is not None and self._use_smoother and self._smoother is not None:
            # Use smoother to move smoothly to specified position
            log.info(f"Moving to target position with smoother...")
            
            # Update smoother target
            self._smoother.update_target(joint_commands, immediate=False)
            
            # Create temporary control loop to actually move the robot
            if mode is None:
                raise ValueError("Mode must be specified when using smoother for move_to_start")
            log.info(f"Move to start mode: {mode}, command: {joint_commands}")
            self.set_joint_commands(joint_commands, mode, execute_hardware=self._enable_hardware)
            time.sleep(2.0)
            
        else:
            # joint_commands is None: use robot's default move_to_start (immediate reset)
            # Pause smoother before immediate movement
            self.pause_smoother()
            
            # Execute immediate move to default start position
            if self._use_simulation:
                self._simulation.move_to_start(None)
            if self._use_hardware:
                self._robot.move_to_start()
            
            # Resume smoother and sync to current position
            time.sleep(0.1)
            self.resume_smoother()
    
    def pause_smoother(self) -> None:
        """Pause smoother (for reset/special operations)"""
        if self._use_smoother and self._smoother is not None:
            self._smoother.pause()
            log.info("Smoother paused")
    
    def resume_smoother(self, sync_to_current: bool = True) -> None:
        """Resume smoother after pause"""
        if self._use_smoother and self._smoother is not None:
            if sync_to_current:
                # Sync to current joint positions
                current_joints = self.get_joint_states()
                if current_joints is not None:
                    self._smoother.update_target(current_joints._positions, immediate=True)
            self._smoother.resume(sync_to_current)
            log.info("Smoother resumed")
    
    def set_smoother_omega_n(self, omega_n: float) -> None:
        """Update smoother natural frequency (if adaptive smoother)"""
        if self._use_smoother and self._smoother is not None:
            if hasattr(self._smoother, '_omega_n'):
                self._smoother._omega_n = np.clip(omega_n, 10.0, 50.0)
                log.info(f"Smoother omega_n set to {omega_n:.1f} rad/s")
    
    def get_smoother_state(self) -> Optional[dict]:
        """Get current smoother state for debugging"""
        if self._use_smoother and self._smoother is not None:
            return self._smoother.get_motion_state()
        return None
    
    def enable_async_control(self) -> bool:
        """
        Enable async control mode - smoother runs in background thread and directly sends commands
        This creates true decoupling between low-freq planning and high-freq control
        
        Returns:
            bool: True if successfully enabled, False otherwise
        """
        if not hasattr(self, '_config'):
            auto_enable_async = self._config.get('auto_enable_async_control', False)
            if not auto_enable_async or not hasattr(self, 'enable_async_control'):
                log.error("Auto enable async is failed by user config")
                return False
        
        if not self._use_smoother or self._smoother is None:
            log.error("Cannot enable async control without smoother")
            return False
        
        if self._async_mode:
            log.warning("Async control already enabled")
            return True
        
        # Start async command thread
        self._async_running = True
        self._async_thread = threading.Thread(
            target=self._async_command_loop,
            daemon=True,
            name="AsyncControlLoop"
        )
        self._async_thread.start()
        self._async_mode = True
        
        log.info(f"Async control enabled at {self._async_frequency}Hz")
        return True
    
    def disable_async_control(self) -> None:
        """Disable async control mode and return to synchronous mode"""
        if not self._async_mode:
            return
        
        # Stop async thread
        self._async_running = False
        if self._async_thread and self._async_thread.is_alive():
            self._async_thread.join(timeout=1.0)
            if self._async_thread.is_alive():
                log.warning("Async control thread did not stop cleanly")
        
        self._async_mode = False
        self._async_thread = None
        log.info("Async control disabled, returned to synchronous mode")
    
    def _async_command_loop(self) -> None:
        """
        Async command loop - continuously sends smoothed commands to robot
        This runs in a separate thread at high frequency (e.g., 800Hz)
        """
        dt = 1.0 / self._async_frequency
        next_time = time.perf_counter()
        slow_loop_count = 0
        
        log.info(f"Starting async command loop at {self._async_frequency}Hz")
        
        dofs = self.get_robot_dofs()
        while self._async_running:
            loop_start = time.perf_counter()
            
            # Get smoothed command from smoother
            with timer("async_smoother", "robot_factory"):
                if self._smoother is not None:
                    smoothed_command, is_active = self._smoother.get_command()
                    
                    if is_active:
                        mode = ["position"] * len(dofs)
                        self.set_robot_joint_command(smoothed_command, mode,
                                            execute_hardware=self._enable_hardware,
                                            update_action=self._update_action)

            # Timing management
            next_time += dt
            sleep_time = next_time - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Performance warning
                slow_loop_count += 1
                if slow_loop_count % 1000 == 1:
                    actual_dt = time.perf_counter() - loop_start
                    log.warning(f"Async control loop running slow: {actual_dt*1000:.1f}ms "
                              f"(target: {dt*1000:.1f}ms)")
                next_time = time.perf_counter()
        
        log.info("Async command loop stopped")
    
    # ========== 学习推理相关方法 ==========
    
    def _initialize_learning_components(self) -> None:
        """初始化学习推理组件."""
        if self._learning_config is None:
            log.info("📚 未配置学习推理组件，跳过初始化")
            return
        
        try:
            algorithm = self._learning_config.get("algorithm", "ACT")
            ckpt_dir = self._learning_config.get("checkpoint_dir")
            
            if not ckpt_dir:
                log.warning("⚠️ 学习配置中未指定checkpoint_dir，跳过学习组件初始化")
                return
            
            # 验证检查点目录
            if not LearningInferenceFactory.validate_checkpoint_directory(ckpt_dir, algorithm):
                log.error(f"❌ 检查点目录验证失败: {ckpt_dir}")
                return
            
            # 创建学习推理流水线
            inference_config = self._learning_config.get("inference_config", {})
            inference_engine, data_adapter = LearningInferenceFactory.create_learning_pipeline(
                algorithm=algorithm,
                robot_type=self._robot_type,
                ckpt_dir=ckpt_dir,
                config=inference_config
            )
            
            # 保存引用
            self._learning_inference_engine = inference_engine
            self._learning_data_adapter = data_adapter
            
            # 设置到机器人对象
            if self._use_hardware and self._robot:
                self._robot.set_learning_inference(inference_engine, data_adapter)
            
            log.info(f"✅ 学习推理组件初始化成功: {algorithm}")
            
        except Exception as e:
            log.error(f"❌ 学习推理组件初始化失败: {str(e)}")
            log.warning("⚠️ 继续运行，但学习功能不可用")
    
    def setup_learning_inference(
        self, 
        algorithm: str,
        ckpt_dir: str,
        inference_config: Dict[str, Any]
    ) -> bool:
        """动态设置学习推理组件.
        
        Args:
            algorithm: 学习算法名称
            ckpt_dir: 检查点目录
            inference_config: 推理配置
            
        Returns:
            bool: 设置是否成功
        """
        try:
            # 创建学习推理流水线
            inference_engine, data_adapter = LearningInferenceFactory.create_learning_pipeline(
                algorithm=algorithm,
                robot_type=self._robot_type,
                ckpt_dir=ckpt_dir,
                config=inference_config
            )
            
            # 更新引用
            self._learning_inference_engine = inference_engine
            self._learning_data_adapter = data_adapter
            
            # 设置到机器人对象
            if self._use_hardware and self._robot:
                self._robot.set_learning_inference(inference_engine, data_adapter)
            
            log.info(f"✅ 学习推理组件动态设置成功: {algorithm}")
            return True
            
        except Exception as e:
            log.error(f"❌ 学习推理组件设置失败: {str(e)}")
            return False
    
    def get_learning_prediction(self, camera_data: Dict[str, np.ndarray]) -> Optional[list]:
        """获取学习策略预测.
        
        Args:
            camera_data: 相机数据字典
            
        Returns:
            Optional[list]: 预测的动作序列
        """
        if not self._use_hardware or not self._robot:
            log.error("❌ 硬件模式下才支持学习推理")
            return None
        
        if not self._robot.is_learning_enabled():
            log.error("❌ 学习推理组件未启用")
            return None
        
        return self._robot.get_learning_prediction(camera_data)
    
    def execute_learned_actions(
        self, 
        actions: list,
        execution_mode: str = "position"
    ) -> bool:
        """执行学习策略输出的动作.
        
        Args:
            actions: 动作序列
            execution_mode: 执行模式
            
        Returns:
            bool: 执行是否成功
        """
        if not self._use_hardware or not self._robot:
            log.error("❌ 硬件模式下才支持动作执行")
            return False
        
        return self._robot.execute_learned_action_sequence(actions, execution_mode=execution_mode)
    
    def run_learning_control_loop(
        self, 
        camera_data: Dict[str, np.ndarray],
        execution_mode: str = "position"
    ) -> bool:
        """运行学习控制循环.
        
        Args:
            camera_data: 相机数据
            execution_mode: 执行模式
            
        Returns:
            bool: 控制循环是否成功
        """
        if not self._use_hardware or not self._robot:
            log.error("❌ 硬件模式下才支持学习控制循环")
            return False
        
        return self._robot.run_learning_control_loop(camera_data, execution_mode)
    
    def is_learning_enabled(self) -> bool:
        """检查学习推理是否已启用."""
        if not self._use_hardware or not self._robot:
            return False
        return self._robot.is_learning_enabled()
    
    def disable_learning_inference(self) -> None:
        """禁用学习推理功能."""
        if self._use_hardware and self._robot:
            self._robot.disable_learning_inference()
        
        self._learning_inference_engine = None
        self._learning_data_adapter = None
        
        log.info("🛑 RobotFactory学习推理功能已禁用")
