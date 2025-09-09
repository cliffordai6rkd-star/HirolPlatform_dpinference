from __future__ import annotations

from trajectory.trajectory_base import TrajectoryBase
from motion.model_base import ModelBase
from motion.pin_model import RobotModel
from motion.duo_model import DuoRobotModel
from controller.controller_base import ControllerBase, IKController
from controller.whole_body_ik import WholeBodyIk
from controller.impedance_controller import ImpedanceController

from controller.cartesian_impedance_controller import CartesianImpedanceController
from controller.duo_controller import DuoController
from trajectory.cartesian_trajectory import CartessianTrajectory
from hardware.base.utils import Buffer
import threading
from hardware.base.utils import object_class_check, TrajectoryState
import time
from factory.components.robot_factory import RobotFactory
from hardware.base.utils import convert_homo_2_7D_pose, get_joint_slice_value, \
    RobotJointState, negate_pose, transform_pose
import copy
import numpy as np
import glog as log
from typing import Dict

# Import performance profiler
from tools.performance_profiler import PerformanceProfiler, timer
from enum import Enum

class Robot_Space(Enum):
    CARTESIAN_SPACE = "cartesian"
    JOINT_SPACE = "joint"

# Used for different robot component into one robot system
class MotionFactory:
    _robot_system: RobotFactory
    _trajectory: TrajectoryBase
    _robot_model: ModelBase
    _controller: ControllerBase
    def __init__(self, config, robot: RobotFactory):
        self._config = config
        self._model_type = config["model_type"]
        self._controller_type = config["controller_type"]
        self._use_traj_planner = config["use_trajectory_planner"]
        if self._use_traj_planner:
            # indicate the plan type is cartesian or joint space
            self._buffer_type = config["buffer_type"]
            self._plan_type = config["plan_type"]
            # indicate the planner interpolation type (cartesian polynomial)
            self._trajectory_planner_type = config["trajectory_planner_type"]
            self._traj_frequency = config["traj_frequency"]
        self._high_level_command = None
        self.enable_high_level_update = True
        self._high_level_updated = False
        self._control_frequency = config["control_frequency"]
        self._robot_system = robot
        self._execute_hardware = False
        self._blocking_motion = False
        self._latest_action = {}
        self._latest_action_lock = threading.Lock()
        self._update_action = False
            
        # object classes
        self._model_classes = {
            "model": RobotModel,
            "duo_model": DuoRobotModel
        }
        self._controller_classes = {
            'ik': IKController,
            'impedance': ImpedanceController,
            'whole_body_ik': WholeBodyIk,
            'cartesian_impedance': CartesianImpedanceController,
            'duo_controller': DuoController,
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
            buffer_config = self._config["trajectory_config"][self._buffer_type]
            self._buffer = Buffer(buffer_config["size"], buffer_config["dim"])
            self._buffer_lock = threading.Lock()
            self._trajectory = self._trajectory_classes[
                                self._trajectory_planner_type](
                                    self._config["trajectory_config"][self._trajectory_planner_type],
                                    self._buffer, self._buffer_lock)
            
        if not object_class_check(self._model_classes, self._model_type):
            raise ValueError
        model_config = self._config["model_config"]
        model_name = model_config["name"]
        model_config = model_config["cfg"]
        self._robot_model = self._model_classes[self._model_type](model_config[model_name])
        
        if not object_class_check(self._controller_classes, self._controller_type):
            raise ValueError
        controller_config = self._config["controller_config"]
        self._controller = self._controller_classes[self._controller_type](
                                controller_config[self._controller_type], self._robot_model)
        
        # initialize all objects
        self._initialize()
        log.info("MotionFactory component creation completed")
        
    def _initialize(self):
        # robot init
        log.info("[DEBUG] MotionFactory._initialize() starting")
        self._robot_system.create_robot_system()
        
        # Auto-enable async control if configured
        if self._robot_system.enable_async_control():
            log.info("Async control enabled in MotionFactory based on configuration")
        else:
            log.warning("Failed to enable async control in MotionFactory")
        
        # thread starting
        self._controller_thread_running = True
        self._controller_thread = threading.Thread(target=self._controller_task)
        self._controller_thread.start()
        
        if self._use_traj_planner:
            self._traj_thread_running = True
            self._traj_thread = threading.Thread(target=self._traj_task)
            self._traj_thread.start()
        
        self._ee_links = self.get_model_end_effector_link_list()
        self._ee_index = ["single"] if len(self._ee_links) == 1 else ["left", "right"]
        self._sim_world2base, _ = self.get_sim_base_world_transform()
        
    def _controller_task(self):
        log.info('Controller thread started!!!')
        iteration_count = 0

        ctrl_period = 1.0 / self._control_frequency
        next_run_time = time.perf_counter()
        slow_loop_count = 0
        while self._controller_thread_running:
            loop_start_time = time.perf_counter()
            iteration_count += 1
            
            with timer("controller_total", "motion_factory_"):
                # Target preparation
                target = []
                if self._high_level_command is not None:
                    if not self._use_traj_planner and self._high_level_updated:
                        target = self._get_controller_target(self._high_level_command)
                        self._high_level_updated = True
                    elif self._use_traj_planner:
                        self._buffer_lock.acquire()
                        get_data, data, time_stamp = self._buffer.pop_data()
                        self._buffer_lock.release()
                        # @TODO: test, online planning
                        current_time_stamp = time.perf_counter()
                        if get_data and current_time_stamp - time_stamp > 0.1:
                            log.debug(f'traj execution slow, clearing outdated data, time: {current_time_stamp - time_stamp}')
                            self._buffer.clear_outdated_data(current_time_stamp)
                            get_data = False
                        if get_data:
                            # traj visualization in sim
                            if self._robot_system._use_simulation:
                                for i, _ in enumerate(self._ee_links):
                                    cur_world2base = self._sim_world2base[0] if len(self._sim_world2base) == 1 else self._sim_world2base[i]
                                    if i == 0:
                                        cur_traj_pint_sim = transform_pose(cur_world2base, data[:7])
                                    else:
                                        cur_traj_pint_sim = transform_pose(cur_world2base, data[7:14])
                                    self._robot_system._simulation.update_trajectory_data(cur_traj_pint_sim)
                            target = self._get_controller_target(data)
                    # log.info(f'controller target: {target}')
            
            # Controller execution
            if len(target) != 0 and not self._blocking_motion:
                curr_joint_state = None
                with timer("get_joint_states", "motion_factory_"):
                    curr_joint_state = self._robot_system.get_joint_states()
                    # log.info(f'Current joint state: pos {curr_joint_state._positions}, vel {curr_joint_state._velocities}')
                
                with timer("controller_computation", "motion_factory_"):
                    success, joint_target, joint_mode = self._controller.compute_controller(
                                                        target, robot_state=curr_joint_state)
                
                with timer("hardware_execution", "motion_factory_"):
                    if success:
                        joint_mode = joint_mode if isinstance(joint_mode, list) else [joint_mode]
                        # log.info(f'controller mode: {joint_mode}')
                        self._robot_system.set_joint_commands(joint_target, joint_mode,
                                self._execute_hardware, update_action=self._update_action,
                                change_action_status=True)
                        # update action
                        if self._update_action:
                            self._latest_action_lock.acquire()
                            get_robot_action_status = self._robot_system._update_robot_action(self._latest_action)
                            if get_robot_action_status:
                                for i, ee_target_dict in enumerate(target):
                                    key = self._ee_index[i]
                                    self._latest_action[key]["ee"] = dict(
                                            pose=list(ee_target_dict.values())[0].tolist(), time_stamp=time.perf_counter())
                            self._latest_action_lock.release()
                        
                        # Check if robot recovered from error and reset controller
                        if self._robot_system.check_robot_recovery():
                            log.info("Robot recovery detected, resetting controller...")
                            ee_links = self.get_model_end_effector_link_list()
                            if ee_links:
                                curr_joint_state = self._robot_system.get_joint_states()
                                for frame_name in ee_links:
                                    log.info(f"Resetting controller for frame: {frame_name}")
                                    self._controller.reset(frame_name, curr_joint_state)
                                log.info("Controller reset after recovery completed")
                    else:
                        # if use hardware; directly map the hardware joints to sim
                        if self._robot_system._use_hardware \
                            and self._robot_system._use_simulation:
                            joint_mode = ["position"] * len(self._ee_links)
                            self._robot_system.set_joint_commands(curr_joint_state._positions, joint_mode, False)
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
        
        log.info(f'Motion factory controller thread stopped!!!')
    
    def _traj_task(self):
        log.info('Trajectory thread started!!!')
        
        model_types = self.get_model_types()
        while self._traj_thread_running:
            start_time = time.perf_counter()
            
            if self._high_level_updated and not self._blocking_motion:
                nums_target = 0
                tcp = np.array([]); twist = np.array([]); acc = np.array([])
                for i, cur_ee_link in enumerate(self._ee_links):
                    model_type = model_types[i] if len(model_types) > 1 else model_types[0]
                    frame_motion = self.get_frame_pose(cur_ee_link, model_type, need_vel=True, need_acc=True)
                    cur_tcp = frame_motion[:7]
                    tcp = np.hstack((tcp, cur_tcp))
                    twist = np.hstack((twist, frame_motion[7:13], 0))
                    frame_motion[13:19] = np.zeros(6)
                    # @TODO: figure out how to clear buffer
                    acc = np.hstack((acc, frame_motion[13:19], 0))
                    nums_target += 1
                if len(tcp) != len(self._high_level_command):
                    log.warn(f'dim not match for tcp and high level command: {len(tcp)}, {self._high_level_command}')
                    continue
                traj_target = TrajectoryState()
                traj_target._zero_order_values = np.vstack((tcp,
                                                            self._high_level_command))
                traj_target._first_order_values = np.vstack((twist,
                                                             np.zeros(7*nums_target)))
                traj_target._second_order_values = np.vstack((acc,
                                                             np.zeros(7*nums_target)))
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
        
        log.info(f'Motion factory trajectory thread stopped!!!')
    
    def set_next_pose_target(self, pose_target):
        target_dim = 7 if len(self._ee_links) == 1 else 14
        if len(pose_target) != target_dim:
            log.warn(f'The high level command has wrong len, expected: {target_dim}, but get: {len(pose_target)}')
            return
        self._high_level_command = copy.deepcopy(pose_target)
        self._high_level_updated = True
    
    def update_high_level_command(self, command):
        '7d pose or two 7d pose'
        if not self.enable_high_level_update:
            return 
       
        self.set_next_pose_target(command)
    
    def clear_high_level_command(self):
        self._high_level_command = None    
    
    def set_joint_positions(self, joint_commands, is_continous_joint_command = True):
        self._blocking_motion = True
        position_mode = ["position"]
        if len(self._ee_links) > 1: position_mode = position_mode * 2
        self._robot_system.set_joint_commands(joint_commands, position_mode, 
                                              self._execute_hardware)
        if not is_continous_joint_command:
            self._blocking_motion = False
            
    def only_set_simulation(self, command):
        if self._robot_system._use_simulation:
            mode = ['position'] * len(command)
            self._robot_system._simulation.set_joint_command(mode, command)
            
    def get_frame_pose_with_joint_state(self, joint_states: RobotJointState, frame_name,
                                        model_type, need_vel = False, need_acc = False):
        pose = self._robot_model.get_frame_pose(frame_name, joint_states._positions,
                                                need_update=True, model_type=model_type)
        pose = convert_homo_2_7D_pose(pose)
        if need_vel:
            twist = self._robot_model.get_frame_twist(frame_name, joint_states._positions,
                                                    joint_states._velocities, need_update=True,
                                                    model_type=model_type)
            pose = np.hstack((pose, twist))
        if need_acc:
            acc = self._robot_model.get_frame_acc(frame_name, joint_states._positions,
                                                joint_states._velocities, joint_states._accelerations,
                                                need_update=True, model_type=model_type)
            pose = np.hstack((pose, acc))
        return pose
        
    def get_frame_pose(self, frame_name, model_type, need_vel = False, need_acc = False):
        """
            @brief: get frame pose in 7D format [x, y, z, qx, qy, qz, qw]
            @params:
                frame_name: the frame link name
                model_type: ['single', 'left', 'right', 'dual']
        """
        joint_states = self._robot_system.get_joint_states()
        return self.get_frame_pose_with_joint_state(joint_states,
                frame_name, model_type, need_vel, need_acc)
    
    def set_tool_command(self, tool_command: list[np.ndarray] | Dict):
        tool_type_dict = self._robot_system.get_tool_dict_state()
        if tool_type_dict is None:
            log.warn(f'There is no tool for the robot config!!!')
            return False
        
        # log.info(f'motion tool: {tool_command}')
        parsed_tool_command = {}
        if len(tool_type_dict) == 1:
            if not isinstance(tool_command, dict):
                parsed_tool_command["single"] = tool_command
            else: parsed_tool_command = tool_command
        else: # for duo tool
            if len(tool_type_dict) != len(tool_command):
                log.error(f'tool command should have len of two for duo tool command but get {len(tool_command)}')
                return False
            if not isinstance(tool_command, dict):
                parsed_tool_command["left"] = tool_command[0]
                parsed_tool_command["right"] = tool_command[1]
            else: parsed_tool_command = tool_command
        # log.info(f'after tool: {parsed_tool_command}')
        return self._robot_system.set_tool_command(parsed_tool_command)

    def reset_robot_system(self, arm_command: list[float] | None = None, 
                           space: Robot_Space = Robot_Space.JOINT_SPACE,
                           tool_command: Dict[str, np.ndarray] = None):
        mode = ["position"] * len(self._ee_links)
        if space == Robot_Space.CARTESIAN_SPACE:
            if arm_command is not None:
                self.enable_high_level_update = False
                # wait for the trajectory done
                self.clear_traj_buffer()
                self.wait_buffer_empty()
                log.info('Trajectory buffer has all been consumed for cartesian space reset!!!')
                # @TODO: attach to current tcp
                self.set_next_pose_target(arm_command)
                time.sleep(2.0)
                self.enable_high_level_update = True
            else:
                self.move_to_start_blocking()
        else:
            self.move_to_start_blocking(arm_command, mode)
        
        # Reset controller
        robot_state = self._robot_system.get_joint_states()
        ee_links = self._ee_links
        if ee_links:
            for frame_name in ee_links:
                log.info(f'Resetting controller with frame: {frame_name}')
                self._controller.reset(frame_name, robot_state)
            log.info('All controllers reset completed')
        
        if tool_command is not None:
            self.set_tool_command(tool_command)
    
    def update_execute_hardware(self, enable_hardware):
        self._execute_hardware = enable_hardware
        
    def change_update_action_status(self, update_action):
        self._update_action = update_action
       
    def close(self):
        self._controller_thread_running = False
        self._controller_thread.join()
        if self._use_traj_planner:
            self._traj_thread_running = False
            self._traj_thread.join()
        log.info(f'Motion factory threads are successfully closed for all!!!')
        
        log.info("=== MotionFactory Final Performance Statistics ===")
        PerformanceProfiler.print_summary()
        PerformanceProfiler.print_stats(sort_by='total_ms', top_n=10)
        self._robot_system.close()

    def get_model_dof_list(self):
        """
            return [0, dof1, dof2] or [0, dof]
        """
        model_dof = self._robot_model.get_model_dof()
        if not isinstance(model_dof, list):
            model_dof = [model_dof]
        model_dof.insert(0, 0) # [0 dof_left dof_right]
        return model_dof
    
    def get_tool_dof_list(self):
        pass
    
    def get_model_end_effector_link_list(self):
        end_effector_links = self._robot_model.get_model_end_links()
        
        if not isinstance(end_effector_links, list):
            end_effector_links = [end_effector_links]
        return end_effector_links

    def _get_controller_target(self, data):
        target = []
        dimensions = [0, 7, 14]
        for i, cur_target_frame in enumerate(self._ee_links):
            cur_target = {cur_target_frame: data[dimensions[i]:dimensions[i+1]]}
            target.append(cur_target)
        return target
    
    def get_type_joint_state(self, joint_states: RobotJointState, model_type: str):
        dofs = self.get_model_dof_list()
        if 'single' in model_type or 'left' in model_type:
            sliced_joint_states = get_joint_slice_value(dofs[0], dofs[1], 
                                                        joint_states)
        else:
            sliced_joint_states = get_joint_slice_value(dofs[1], dofs[1]+dofs[2], joint_states)
        return sliced_joint_states
    
    def move_to_start_blocking(self, joint_commands = None, mode=None):
        self._blocking_motion = True
        if joint_commands is not None: joint_commands = np.array(joint_commands)
        self._robot_system.move_to_start(joint_commands, mode=mode)
        self._blocking_motion = False
        
    def clear_traj_buffer(self):
        if self._use_traj_planner:
            self._buffer.clear()

    def wait_buffer_empty(self):
        if self._use_traj_planner:
            while self._buffer.size():
                time.sleep(0.001)

    def get_model_types(self):
        return ['single'] if self._model_type == 'model' else ['left', 'right']
    
    def get_latest_action(self):
        if len(self._latest_action) == 0 or not self._update_action:
            return None
        
        with self._latest_action_lock:
            latest_action = copy.deepcopy(self._latest_action)
        return latest_action
    
    def get_sim_base_world_transform(self):
        if not self._robot_system._use_simulation:
            return None, None
        
        world2base_pose = [np.array([0, 0, 0, 0, 0, 0, 1])]
        base2world_pose = [negate_pose(world2base_pose[0])]
        if len(self._robot_system._simulation.base_body_name) != 0:
            world2base_pose = []
            base2world_pose = []
            for cur_base_body in self._robot_system._simulation.base_body_name:
                cur_world2base = self._robot_system._simulation.get_body_pose(cur_base_body)
                world2base_pose.append(cur_world2base)
                base2world_pose.append(negate_pose(cur_world2base))
        return world2base_pose, base2world_pose
