import abc
from factory.components.gym_interface import GymApi
from hardware.base.utils import ToolControlMode
from factory.tasks.inferences_tasks.utils import display_images, AnimationPlotter
from dataset.utils import ActionType, Action_Type_Mapping_Dict, ObservationType
import threading, time, cv2, os, copy
from sshkeyboard import listen_keyboard, stop_listening
import glog as log
import torch as th
import numpy as np
from collections import deque
from factory.tasks.inferences_tasks.utils.action_aggreagator import ActionAggregator, WeightMode

class InferenceBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._gym_robot = GymApi(config)
        self._status_ok = True
        self._quit = False
        
        self._action_type = config["action_type"]
        self._action_type = Action_Type_Mapping_Dict[self._action_type]
        # Command action types share the same execution semantics as their state-based
        # counterparts; normalize for downstream slicing and GymApi execution.
        self._exec_action_type = self._normalize_action_type(self._action_type)
        if self._exec_action_type != self._action_type:
            log.info(f'Normalize action type from {self._action_type} to {self._exec_action_type} for execution')
        self._gym_robot.set_action_type(self._exec_action_type)
        self._action_ori_type = config.get("action_orientation_type", "euler")
        self._obs_type = config.get("observation_type", ObservationType.JOINT_POSITION_ONLY)
        
        # tool related config
        self._tool_position_dof = config.get("tool_position_dof", 1)
        self._tool_min: float | list = config.get(f'tool_min', 0)
        self._tool_max: float | list = config.get(f'tool_max', 90)
        self._tool_control_mode = config.get(f'tool_control_mode', ToolControlMode.BINARY)
        if self._tool_control_mode == ToolControlMode.BINARY:
            self._last_gripper_open = [True, True]
            self._tool_open_threshold = config.get(f'tool_open_thresh', 45)
            self._tool_close_threshold = config.get(f'tool_close_thresh', 45)
        self._num_episodes = config.get("num_episodes", 10)
        
        # used for latter aggregation 
        self._max_timestamps = config.get(f"max_timestamps", 1500)
        self._execution_action_chunk_size = config.get(f'execution_chunk_steps', 4)
        self._predicted_action_chunks = -1 # policy output action chunk size
        self._infer_frequency = config.get(f'infer_frequency', 10)
        self._weight_mode = WeightMode(config.get('action_weight_mode', "no"))
        self._weight_gain = config.get("action_weight_gain", 0.01)
        self._action_aggregation = None
        self._execution_interruption = False
        self._async_execution = config.get("async_execution", False)
        
        # display
        self._enable_display = config.get("enable_display", True)
        self._display_window_name = config["display_window_name"]
        
        # keyboard listening
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                kwargs={"on_press": self._keyboard_on_press, 
                        "until": None, "sequential": False,}, daemon=True)
        listen_keyboard_thread.start()
        
        # animation plotter, @TODO: pad state and action to same len
        self._joint_positions = None
        self._lock = threading.Lock()
        # Check if plotting should be enabled (can be disabled for performance)
        self._enable_plotting = config.get("enable_plotting", False)
        self._plotter = None

    @staticmethod
    def _normalize_action_type(action_type: ActionType) -> ActionType:
        command_to_state = {
            ActionType.COMMAND_JOINT_POSITION: ActionType.JOINT_POSITION,
            ActionType.COMMAND_JOINT_POSITION_DELTA: ActionType.JOINT_POSITION_DELTA,
            ActionType.COMMAND_END_EFFECTOR_POSE: ActionType.END_EFFECTOR_POSE,
            ActionType.COMMAND_END_EFFECTOR_POSE_DELTA: ActionType.END_EFFECTOR_POSE_DELTA,
        }
        return command_to_state.get(action_type, action_type)
        
    def update_plotter(self, state, action):
        if self._enable_plotting:
            if self._plotter is None:
                joint_state_names = [f'state_{i}' for i in range(len(state))]
                action_names = [f'action_{i}' for i in range(len(action))]
                if len(joint_state_names) < len(action_names):
                    joint_state_names += [f'state_mask_{i}' for i in range(len(action) - len(state))]
                if len(joint_state_names) > len(action_names):
                    action_names += [f'action_mask_{i}' for i in range(len(state) - len(action))]
                self._plotter = AnimationPlotter(joint_state_names, action_names)
                self._plotter.start_animation()
                self._plotter.start_main_thread_updater()
                log.info(f'Plotter is successfully settled and ready for draw plots')
            else:
                if len(state) < len(action):
                    state = np.hstack((state, [0] * (len(action) - len(state))))
                if len(state) > len(action):
                    action = np.hstack((action, [0] * (len(state) - len(action))))
                self._plotter.update_signal(state, action)
    
    @abc.abstractmethod
    def convert_from_gym_obs(self, gym_obs = None):
        if gym_obs is None:
            gym_obs = self._gym_robot.get_observation()
        return gym_obs
    
    def convert_to_gym_action_single_step(self, action, raw_action=None):
        dofs = self._gym_robot._robot_motion.get_model_dof_list()[1:]
        cur_action = action
        # log.info(f'Executing action for {i}th action: {cur_action}')
        with self._lock:
            joint_state = copy.deepcopy(self._joint_positions)
        self.update_plotter(joint_state, cur_action)
        
        # iterates with the dof list
        action = {'arm': np.array([]), 'tool': np.array([])}
        action_index = 0
        gripper_position_dof = self._tool_position_dof
        # log.info(f'len dof: {len(dofs)}')
        for j in range(len(dofs)):
            if self._exec_action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_POSITION_DELTA]:
                index_l = gripper_position_dof*j + action_index
                index_r = gripper_position_dof*j + dofs[j] + action_index
                action_index = index_r+gripper_position_dof
                # log.info(f'arm index for joint: {index_l}, {index_r}')
                cur_arm_action = cur_action[index_l:index_r]
            elif self._exec_action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA]:
                pose_dof = 6 if self._action_ori_type == "euler" else 7
                # log.info(f'arm index for pose: {action_index}')
                cur_arm_action = cur_action[action_index:action_index+pose_dof].copy()
                cur_arm_action[3:] = raw_action[action_index+3:action_index+pose_dof]
                index_r = action_index+pose_dof
                action_index = index_r+gripper_position_dof 
            else:
                raise ValueError(f"Unsupported action type: {self._action_type}")

            action["arm"] = np.hstack((action["arm"], cur_arm_action))
            cur_tool_action = cur_action[index_r:index_r+gripper_position_dof].copy()
            # log.info(f'cur tool action from model action for {j}: {cur_tool_action}, len {len(cur_tool_action)}')
            
            if self._tool_control_mode == ToolControlMode.BINARY:
                if self._last_gripper_open[j]:
                    log.info(f'open, action {cur_tool_action[0]}')
                    cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_open_threshold else 0.0
                else:
                    log.info(f'close, action {cur_tool_action[0]}')
                    cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_close_threshold else 0.0
                if cur_tool_action[0] > 0.0085:
                    self._last_gripper_open[j] = True
                else: self._last_gripper_open[j] = False
            else: cur_tool_action /= self._tool_max
            action["tool"] = np.hstack((action["tool"], cur_tool_action))
            log.info(f'tranformed tool action for {j}: {cur_tool_action}')
        return action
    
    def convert_to_gym_action(self, model_action: np.ndarray):
        """
            @brief: convert action type and execute it one by one
            model_action: dim (chunk_size, action_size)
            assuming all chunk size is need to be executed
        """
        dofs = self._gym_robot._robot_motion.get_model_dof_list()[1:]
        log.info(f'dofs: {dofs}')
        for i in range(model_action.shape[0]):
            if self._execution_interruption or not self._status_ok:
                log.info(f"Action execution interrupted with interruption {self._execution_interruption}, status {self._status_ok}")
                break
            
            start_time = time.perf_counter()
            cur_action = model_action[i]
            with self._lock:
                joint_state = copy.deepcopy(self._joint_positions)
            self.update_plotter(joint_state, cur_action)
            
            # iterates with the dof list
            action = {'arm': np.array([]), 'tool': np.array([])}
            action_index = 0
            gripper_position_dof = self._tool_position_dof
            # log.info(f'len dof: {len(dofs)}')
            convert_start = time.perf_counter()
            for j in range(len(dofs)):
                if self._exec_action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_POSITION_DELTA]:
                    index_l = gripper_position_dof*j + action_index
                    index_r = gripper_position_dof*j + dofs[j] + action_index
                    action_index = index_r+gripper_position_dof
                    # log.info(f'arm index for joint: {index_l}, {index_r}')
                    cur_arm_action = cur_action[index_l:index_r]
                elif self._exec_action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA]:
                    pose_dof = 6 if self._action_ori_type == "euler" else 7
                    # log.info(f'arm index for pose: {action_index}')
                    cur_arm_action = cur_action[action_index:action_index+pose_dof]
                    index_r = action_index+pose_dof
                    action_index = index_r+gripper_position_dof 
                else:
                    raise ValueError(f"Unsupported action type: {self._action_type}")

                action["arm"] = np.hstack((action["arm"], cur_arm_action))
                cur_tool_action = cur_action[index_r:index_r+gripper_position_dof].copy()
                log.info(f'cur tool action from model action for {j}: {cur_tool_action}, len {len(cur_tool_action)}')
                
                if self._tool_control_mode == ToolControlMode.BINARY:
                    if self._last_gripper_open[j]:
                        log.info(f'open, action {cur_tool_action[0]}')
                        cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_open_threshold else 0.0
                    else:
                        log.info(f'close, action {cur_tool_action[0]}')
                        cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_close_threshold else 0.0
                    if cur_tool_action[0] > 0.0085:
                        self._last_gripper_open[j] = True
                    else: self._last_gripper_open[j] = False
                else: cur_tool_action /= self._tool_max
                action["tool"] = np.hstack((action["tool"], cur_tool_action))
                # log.info(f'tranformed tool action for {j}: {cur_tool_action}')
            loop_time = time.perf_counter() - convert_start    
            
            # log.info(f'gym action: {action}')
            # time.sleep(0.01)
            step_start = time.perf_counter()
            res = self._gym_robot.step(action)
            step_time = time.perf_counter() - step_start
            dt = time.perf_counter() - start_time
            if dt < 1.0 / 50.0:
                sleep_time = (1.0 / 50) -  dt
                time.sleep(sleep_time)
            else: log.warn(f"{'=='*15} Execution is slow: {1.0 / dt:.3f}Hz {loop_time:.5f} {step_time:.5f}  {res[4]['obs_time']:.5f} {'=='*15}")
            if self._async_execution:
                time.sleep(0.001)

    @abc.abstractmethod
    def policy_reset(self):
        # raise NotImplementedError
        pass
    
    @abc.abstractmethod
    def policy_prediction(self, obs) -> th.Tensor | np.ndarray:
        """
            predict policy action from obs
            action should be numpy action
        """
        # raise NotImplementedError
        pass

    @abc.abstractmethod
    def start_inference(self):
        """Start the main inference loop.
        
        Continuously processes observations, runs inference, and executes actions
        until interrupted by keyboard input.
        """
        raise NotImplementedError
    
    def start_common_inference(self):
        # rollout episodes
        query_frequency = int(50 / self._infer_frequency)
        for episode_id in range(self._num_episodes):
            if self._quit: break
            
            if self._action_aggregation:
                self._action_aggregation.reset()
            self._gym_robot.reset()
            self._last_gripper_open = [True, True]
            self.policy_reset()
            self._status_ok = True
            obs = None
            log.info(f'Starting the {episode_id} th episodes')
            # inference timestamp
            pred_action_chunk = None
            for t in range(self._max_timestamps):
                step_loop_start = time.perf_counter()
                # if self._action_ori_type == "quaternion" and self._weight_mode != WeightMode.NO_WEIGHT:
                #     raise ValueError(f'Action aggregation does not support quaternion action orientation')
                    
                if not self._status_ok or self._quit: 
                    break
                
                if obs is None: 
                    obs = self.convert_from_gym_obs(obs)
                
                if t % query_frequency == 0:
                    # 仅在查询周期到达时预测并追加新的动作块
                    pred_action_chunk = self.policy_prediction(obs)
                    # log.info(f'predicted action shape: {pred_action_chunk.shape}')
                    if self._action_aggregation is None:
                        predicted_action_chunk_size = pred_action_chunk.shape[0]
                        self._action_aggregation = ActionAggregator(
                            query_frequency=query_frequency,
                            chunk_size=predicted_action_chunk_size,
                            max_timestamps=self._max_timestamps,
                            action_size=pred_action_chunk.shape[1],
                            k=self._weight_gain,
                        )
                    # 追加最新的动作块（以当前全局时间 t 作为块起点）
                    self._action_aggregation.add_action_chunk(t, pred_action_chunk)
                # calculate aggregated action
                aggregated_action = self._action_aggregation.aggregation_action(t, self._weight_mode)
                
                # log.info(f'aggregated action: {aggregated_action}')
                # 与聚合动作对应的原始行（用于姿态类动作的原始旋转覆盖等）
                row_idx = t % query_frequency
                raw_row = pred_action_chunk[row_idx] if pred_action_chunk is not None else aggregated_action
                gym_action = self.convert_to_gym_action_single_step(aggregated_action, raw_row)
                res = self._gym_robot.step(gym_action)
                gym_obs = res[0]
                obs = self.convert_from_gym_obs(gym_obs)
                # 上层循环节流到约50Hz
                elapsed = time.perf_counter() - step_loop_start
                target_dt = 1.0 / 50.0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                # for 循环自动递增 t，无需手动自增
                                
    @abc.abstractmethod
    def close(self):
        pass
    
    def image_display(self, gym_obs):
        if self._enable_display and gym_obs.get("colors"):
            display_images(gym_obs["colors"], self._display_window_name)    
    
    
    def _keyboard_on_press(self, key: str) -> None:
        # quit
        if key == 'q':
            log.info("Quit command received, shutting down...")
            print(f"{'='*15}Closing the inference thread!!!{'='*15}")
            self._gym_robot.close()
            stop_listening()
            self._quit = True
            self._status_ok = False
            # self._plotter.clear_data()
            # self._plotter.stop_animation()
            time.sleep(1.5)
            self.close()
            cv2.destroyAllWindows()
        # reset
        elif key == 'r':
            log.info("Reset command received")
            self._gym_robot.reset()
        elif key == 'd':
            self._status_ok = False
            log.info(f"Set done to True for current episode!!!")
    
