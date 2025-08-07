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
        self._high_level_updated = False
        self._control_frequency = config["control_frequency"]
        self._robot_system = robot
        self._execute_hardware = False
        self._blocking_motion = False
            
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
        
        # thread starting
        self._controller_thread_running = True
        self._controller_thread = threading.Thread(target=self._controller_task)
        self._controller_thread.start()
        
        if self._use_traj_planner:
            self._traj_thread_running = True
            self._traj_thread = threading.Thread(target=self._traj_task)
            self._traj_thread.start()
        
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
                        get_data, data = self._buffer.pop_data()
                        self._buffer_lock.release()
                        # log.info(f'data: {data}, get data: {get_data}, buffer size: {self._buffer.size()}')
                        if get_data:
                            target = self._get_controller_target(data)
                    # log.info(f'controller target: {target}')
                # log.info(f'target use time {target_time - start_time}s')
            
            # Controller execution
            if len(target) != 0 and not self._blocking_motion:
                curr_joint_state = None
                with timer("get_joint_states", "motion_factory_"):
                    curr_joint_state = self._robot_system.get_joint_states()
                    # log.info(f'Current joint state: pos {curr_joint_state._positions}, vel {curr_joint_state._velocities}')
                
                with timer("controller_computation", "motion_factory_"):
                    success, joint_target, joint_mode = self._controller.compute_controller(
                                                        target, robot_state=curr_joint_state)
                    # log.info(f'Controller output - success: {success}, joint_target: {joint_target}, joint_mode: {joint_mode}')
                
                with timer("hardware_execution", "motion_factory_"):
                    if success:
                        joint_mode = joint_mode if isinstance(joint_mode, list) else [joint_mode]
                        self._robot_system.set_joint_commands(joint_target, joint_mode,
                                                            self._execute_hardware)
                    else:
                        self._robot_system.set_joint_commands(curr_joint_state._positions, ["position"], False)
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
        
        ee_links = self.get_model_end_effector_link_list()
        model_types = ["left", "right"] if isinstance(self._robot_model, DuoRobotModel) else ["single"]
        while self._traj_thread_running:
            start_time = time.perf_counter()
            
            if self._high_level_updated and not self._blocking_motion:
                nums_target = 0
                tcp = np.array([])
                for i, cur_ee_link in enumerate(ee_links):
                    cur_tcp = self.get_frame_pose(cur_ee_link, model_types[i])
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
            else:
                warnings.warn(f"The trajectory frequency is slow, expected: {self._traj_frequency} "
                              f"actual: {1.0 / use_time}")
        
        log.info(f'Motion factory trajectory thread stopped!!!')
    
    def update_high_level_command(self, command):
        '7d pose or two 7d pose'
        self._high_level_command = copy.deepcopy(command)
        self._high_level_updated = True
        
    def get_frame_pose(self, frame_name, model_type):
        """
            @brief: get frame pose in 7D format [x, y, z, qx, qy, qz, qw]
            @params:
                frame_name: the frame link name
                model_type: ['single', 'left', 'right', 'dual']
        """
        joint_states = self._robot_system.get_joint_states()
        # print(f'posi: {joint_states._positions}')
        pose = self._robot_model.get_frame_pose(frame_name, joint_states._positions,
                                                need_update=True, model_type=model_type)
        pose = convert_homo_2_7D_pose(pose)
        return pose
    
    def update_execute_hardware(self, enable_hardware):
        self._execute_hardware = enable_hardware
       
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
    
    def get_model_end_effector_link_list(self):
        end_effector_links = self._robot_model.get_model_end_links()
        
        if not isinstance(end_effector_links, list):
            end_effector_links = [end_effector_links]
        return end_effector_links

    def _get_controller_target(self, data):
        target = []
        end_effector_links = self.get_model_end_effector_link_list()
        dimensions = [0, 7, 14]
        for i, cur_target_frame in enumerate(end_effector_links):
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
    
    def move_to_start_blocking(self):
        if not self._robot_system._use_hardware:
            return 
    
        self._blocking_motion = True
        self._robot_system.move_to_start()
        self._blocking_motion = False
        
    def clear_traj_buffer(self):
        if self._use_traj_planner:
            self._buffer.clear()
