from motion.pin_model import RobotModel
from controller.controller_base import IKController
from controller.impedance_controller import ImpedanceController
from controller.whole_body_ik import WholeBodyIk
from trajectory.cartesian_trajectory import CartessianTrajectory
from hardware.base.utils import Buffer
import threading
from hardware.base.utils import object_class_check, TrajectoryState
import time, warnings
from factory.components.robot_factory import RobotFactory
from hardware.base.utils import convert_homo_2_7D_pose, get_joint_slice_value, RobotJointState
import copy
import numpy as np
import glog as log

# Import performance profiler
from tools.performance_profiler import PerformanceProfiler, timer

# Used for different robot component into one robot system
class MotionFactory:
    def __init__(self, config, robot: RobotFactory):
        self._config = config
        self._use_traj_planner = config["use_trajectory_planner"]
        if self._use_traj_planner:
            # indicate the plan type is cartesian or joint space
            self._plan_type = config["plan_type"]
            # indicate the planner interpolation type (cartesian polynomial)
            self._trajectory_planner_type = config["trajectory_planner_type"]
            self._traj_frequency = config["traj_frequency"]
        self._high_level_command = None
        self._high_level_updated = False
        self._control_frequency = config["control_frequency"]
        self._robot_system = robot
        self._execute_hardware = False
            
        # object classes
        self._controller_classes = {
            'ik': IKController,
            'impedance': ImpedanceController,
            'whole_body_ik': WholeBodyIk,
            'dual_whole_body_ik': WholeBodyIk  # Same class, dual config
        }
        self._trajectory_classes = {
            'cart_polynomial': CartessianTrajectory
        }
        
    def create_motion_components(self):
        log.info("Starting MotionFactory component creation...")
        
        # Clear previous performance statistics
        PerformanceProfiler.clear_stats()
        
        # traj planner objects
        if self._use_traj_planner:
            if not object_class_check(self._trajectory_classes, self._trajectory_planner_type):
                raise ValueError
            # Dynamically choose buffer config based on robot type
            buffer_config_key = "buffer"
            # Check if we have dual-arm configuration
            for model_cfg in self._config["model_config"]:
                if model_cfg.get("type") == "dual":
                    buffer_config_key = "duo_buffer"
                    break
            
            buffer_config = self._config["trajectory_config"][buffer_config_key]
            self._buffer = Buffer(buffer_config["size"], buffer_config["dim"])
            self._buffer_lock = threading.Lock()
            self._trajectory = self._trajectory_classes[
                                self._trajectory_planner_type](
                                    self._config["trajectory_config"][self._trajectory_planner_type],
                                    self._buffer, self._buffer_lock)
        model_config = self._config["model_config"]
        self._robot_model = {}
        self._controller = {}
        for i, cur_model_cfg in enumerate(model_config):
            model_type = cur_model_cfg['type']
            model_name = cur_model_cfg['name']
            self._robot_model[model_type] = RobotModel(cur_model_cfg['cfg'][model_name])
            cur_controller_cfg = self._config["controller_config"][i]
            cur_controller_type = cur_controller_cfg['name']
            if not object_class_check(self._controller_classes, cur_controller_type):
                raise ValueError
            if model_type != cur_controller_cfg['type']:
                raise ValueError("model & controller are not matched, "
                                 f"model type: {model_type}, controller_type: {cur_controller_cfg['type']}")
            self._controller[model_type] = self._controller_classes[
                                cur_controller_type](
                                    cur_controller_cfg['cfg'][cur_controller_type], 
                                    self._robot_model[model_type])
        
        # initialize all objects
        self._initialize()
        log.info("MotionFactory component creation completed")
        
    def _initialize(self):
        # robot init
        log.info("[DEBUG] MotionFactory._initialize() starting")
        self._robot_system.create_robot_system()
        
        # thread starting
        controller_thread = threading.Thread(target=self._controller_task)
        controller_thread.start()
        
        if self._use_traj_planner:
            traj_thread = threading.Thread(target=self._traj_task)
            traj_thread.start()
        
    def _controller_task(self):
        target = dict()
        iteration_count = 0

        ctrl_period = 1.0 / self._control_frequency
        next_run_time = time.perf_counter()
        slow_loop_count = 0
        while True:
            loop_start_time = time.perf_counter()
            iteration_count += 1
            
            with timer("controller_total", "motion_factory_"):
                # Target preparation
                if self._high_level_command is not None:
                    if not self._use_traj_planner and self._high_level_updated:
                        target = self._get_controller_target(self._high_level_command)
                        self._high_level_updated = False
                    elif self._use_traj_planner:
                        self._buffer_lock.acquire()
                        get_data, data = self._buffer.pop_data()
                        self._buffer_lock.release()
                        if get_data:
                            target = self._get_controller_target(data)
            
            # Controller execution
            if len(target) != 0:
                success = True; joint_target = np.array([]); joint_mode = []
                
                with timer("get_joint_states", "motion_factory_"):
                    curr_joint_state = self._robot_system.get_joint_states()
                
                with timer("controller_computation", "motion_factory_"):
                    for key, controller in self._controller.items():
                        sliced_joint_state = self._get_type_joint_state(curr_joint_state, key)
                        
                        # Add body joints to joint state for whole body IK
                        if sliced_joint_state._positions is not None:
                            complete_positions = self._add_body_joints_to_positions(sliced_joint_state._positions, key)
                            sliced_joint_state._positions = complete_positions
                            
                            # Also need to extend velocities and accelerations for body joints
                            expected_nq = self._robot_model[key].model.nq
                            current_size = len(sliced_joint_state._velocities) if sliced_joint_state._velocities is not None else 0
                            if current_size < expected_nq:
                                body_joints_needed = expected_nq - current_size
                                # Add zero velocities for body joints
                                if sliced_joint_state._velocities is not None:
                                    complete_velocities = np.concatenate([
                                        np.zeros(body_joints_needed),  # Zero body velocities
                                        sliced_joint_state._velocities
                                    ])
                                    sliced_joint_state._velocities = complete_velocities
                                
                                # Add zero accelerations for body joints  
                                if sliced_joint_state._accelerations is not None:
                                    complete_accelerations = np.concatenate([
                                        np.zeros(body_joints_needed),  # Zero body accelerations
                                        sliced_joint_state._accelerations
                                    ])
                                    sliced_joint_state._accelerations = complete_accelerations
                        
                        cur_target = target
                        if 'left' in key or 'right' in key:
                            cur_target = cur_target[key]
                        cur_success, cur_joint_target, cur_joint_mode = controller.compute_controller(
                                cur_target, robot_state=sliced_joint_state)
        
                        success = success and cur_success
                        if cur_success:
                            # Remove body joints from controller output (keep only arm joints for hardware/sim)
                            if len(cur_joint_target) > 14:  # If controller returned body+arm joints
                                body_joints_count = len(cur_joint_target) - 14
                                arm_joint_target = cur_joint_target[body_joints_count:]  # Skip body joints, keep arm joints
                                joint_target = np.hstack((joint_target, arm_joint_target))
                            else:
                                joint_target = np.hstack((joint_target, cur_joint_target))
                            joint_mode.append(cur_joint_mode)
                        else: 
                            break
                
                with timer("hardware_execution", "motion_factory_"):
                    if success:
                        self._robot_system.set_joint_commands(joint_target, joint_mode,
                                                              self._execute_hardware)
                    else:
                        log.warning(f"Controller failed to compute valid joint commands for target: {target}")
            
            next_run_time += ctrl_period
            current_time = time.perf_counter()
            sleep_time = next_run_time - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 处理超时情况
                actual_time = current_time - loop_start_time
                slow_loop_count += 1

                # 控制警告频率：每1000次慢循环警告一次
                if slow_loop_count % 1000 == 1:
                    expected_freq = self._control_frequency
                    actual_freq = 1.0 / actual_time
                    log.warning(f"Controller frequency slow: expected {expected_freq:.1f}Hz, "
                                f"actual {actual_freq:.1f}Hz (warning #{slow_loop_count})")

                # 重置时间基准
                next_run_time = current_time

            # 性能统计（降低频率，减少日志量）
            if iteration_count % 5000 == 0:
                log.info(f"=== Motion Factory Performance (iteration {iteration_count}) ===")
                PerformanceProfiler.print_stats(sort_by='avg_ms', top_n=5)
    
    def _traj_task(self):
        # print('Trajectory thread started!!!')
        
        while True:
            start_time = time.perf_counter()
            
            if self._high_level_updated:
                tcp = np.array([])
                nums_target = 0
                for key, model in self._robot_model.items():
                    ee_link_name = model.ee_link
                    if isinstance(ee_link_name, list):
                        for cur_ee_link in ee_link_name:
                            cur_tcp = self.get_frame_pose(cur_ee_link, key)
                            tcp = np.hstack((tcp, cur_tcp))
                            nums_target += 1
                    else:
                        cur_tcp = self.get_frame_pose(ee_link_name, key)
                        tcp = np.hstack((tcp, cur_tcp))
                        nums_target += 1
                        
                traj_target = TrajectoryState()
                traj_target._zero_order_values = np.vstack((tcp,
                                                            self._high_level_command))
                traj_target._first_order_values = np.zeros((2,7*nums_target))
                traj_target._second_order_values = np.zeros((2,7*nums_target))
                self._high_level_updated = False
                # traj planning
                self._trajectory.plan_trajectory(traj_target)
            
            use_time = time.perf_counter() - start_time
            if use_time < (1.0 / self._traj_frequency):
                sleep_time = (1.0 / self._traj_frequency) - use_time
                time.sleep(sleep_time)
            # else:
            #     warnings.warn(f"The trajectory frequency is slow, expected: {self._traj_frequency} "
            #                   f"actual: {1.0 / use_time}")
    
    def update_high_level_command(self, command):
        self._high_level_command = copy.deepcopy(command)
        self._high_level_updated = True
        
    def get_frame_pose(self, frame_name, model_type):
        """
            @brief: get frame pose in 7D format [x, y, z, qx, qy, qz, qw]
            @params:
                frame_name: the frame link name
                model_type: ['single', 'left', 'right', 'dual']
        """
        origin_joint_states = self._robot_system.get_joint_states()
        
        # For dual arm model, determine the actual model type based on frame name
        actual_model_type = model_type
        if model_type in ['left', 'right'] and 'dual' in self._robot_model:
            actual_model_type = 'dual'
            
        sliced_joint_states = self._get_type_joint_state(origin_joint_states, actual_model_type)
        joint_position = sliced_joint_states._positions
        
        # Add body joints to the joint position vector if model includes body joints
        joint_position = self._add_body_joints_to_positions(joint_position, actual_model_type)
        
        # Safety check: if joint_position is empty, use neutral configuration
        if len(joint_position) == 0:
            # print(f"WARNING: {actual_model_type} joint_position is empty, using neutral configuration")
            import pinocchio as pin
            neutral_q = pin.neutral(self._robot_model[actual_model_type].model)
            joint_position = neutral_q
            
        pose = self._robot_model[actual_model_type].get_frame_pose(frame_name, joint_position, 
                                                      need_update = True)
        pose = convert_homo_2_7D_pose(pose)
        return pose
    
    def get_model_end_effector_name(self):
        ee_name = {}
        for (key, value) in self._robot_model.items():
            if key == 'dual':
                # For dual arm model, create separate entries for left and right
                frame_names = value.ee_link  # ['left_link_tcp', 'right_link_tcp']
                ee_name['left'] = frame_names[0]   # 'left_link_tcp'
                ee_name['right'] = frame_names[1]  # 'right_link_tcp'
            else:
                ee_name[key] = value.ee_link
        return ee_name
    
    def update_execute_hardware(self, enable_hardware):
        self._execute_hardware = enable_hardware
    
    def print_performance_stats(self):
        """Print current performance statistics on demand"""
        log.info("=== MotionFactory Performance Statistics ===")
        PerformanceProfiler.print_stats(sort_by='avg_ms', top_n=8)
        
    def close(self):
        log.info("=== MotionFactory Final Performance Statistics ===")
        PerformanceProfiler.print_summary()
        PerformanceProfiler.print_stats(sort_by='total_ms', top_n=10)
    
    def _get_controller_target(self, data):
        target = {}
        for type, model in self._robot_model.items():
            if type == 'dual':
                # For dual arm model, create targets for both end effectors
                frame_names = model.ee_link  # ['left_link_tcp', 'right_link_tcp']
                left_frame = frame_names[0]  # 'left_link_tcp'  
                right_frame = frame_names[1] # 'right_link_tcp'
                
                # Fixed index mapping: left=[0:7], right=[7:14]
                target[left_frame] = data[0:7]   # Left arm target
                target[right_frame] = data[7:14] # Right arm target
            else:
                # Legacy single arm logic
                frame_name = model.ee_link
                start, end = 0, 7
                if 'right' in type:
                    start, end = 7, 14
                    
                if 'single' in type:
                    target[frame_name] = data[start:end]
                else:
                    sub_target = {frame_name: data[start:end]}
                    target = {type: sub_target}
        return target
    
    def _get_type_joint_state(self, joint_states: RobotJointState, model_type: str):
        if model_type == 'dual':
            # For dual arm model, return complete joint states (14 joints)
            # The controller will handle internal arm separation
            return joint_states
        else:
            # Legacy logic for separate left/right models
            dof = self._robot_model[model_type].nv
            if 'single' in model_type or 'left' in model_type:
                sliced_joint_states = get_joint_slice_value(0, dof, 
                                                            joint_states)
            else:  # right arm
                left_dof = self._robot_model['left'].nv
                sliced_joint_states = get_joint_slice_value(left_dof, left_dof+dof, 
                                                            joint_states)
            return sliced_joint_states
    
    def _add_body_joints_to_positions(self, arm_joint_positions, model_type):
        """
        Add body joint positions to arm joint positions for complete pin model
        
        Args:
            arm_joint_positions: Joint positions for arms only
            model_type: Model type string
            
        Returns:
            np.ndarray: Complete joint positions including body joints
        """
        import pinocchio as pin
        
        # Get the expected number of joints from the pin model
        expected_nq = self._robot_model[model_type].model.nq
        current_size = len(arm_joint_positions)
        
        # If sizes match, no need to add body joints
        if current_size == expected_nq:
            return arm_joint_positions
            
        # If pin model expects more joints, we need to add body joints
        if current_size < expected_nq:
            # Calculate how many body joints we need to add
            body_joints_needed = expected_nq - current_size
            
            # Get body joint positions from hardware
            body_positions = None
            if self._robot_system._use_hardware:
                try:
                    body_positions = self._robot_system.get_body_positions()
                    if body_positions is not None:
                        log.debug(f"Got body positions from hardware: {body_positions}")
                except Exception as e:
                    log.warning(f"Failed to get body positions: {e}")
            
            # Create complete joint position vector
            if body_positions is not None and len(body_positions) >= body_joints_needed:
                # Use actual body joint positions
                complete_positions = np.concatenate([
                    body_positions[:body_joints_needed],  # Body joints first
                    arm_joint_positions                    # Then arm joints
                ])
                # log.info(f"Added {body_joints_needed} body joints to arm positions")
            else:
                # Use neutral body joint positions
                neutral_q = pin.neutral(self._robot_model[model_type].model)
                complete_positions = neutral_q.copy()
                # Update arm joint positions in the neutral vector
                complete_positions[body_joints_needed:] = arm_joint_positions
                log.debug(f"Used neutral body joints, updated arm positions")
                
            return complete_positions
        else:
            # Current size is larger than expected, just return as is
            log.warning(f"Joint position size ({current_size}) > expected ({expected_nq})")
            return arm_joint_positions
    