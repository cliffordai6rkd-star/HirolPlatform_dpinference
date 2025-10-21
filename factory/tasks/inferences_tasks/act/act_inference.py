#!/usr/bin/env python3
"""
Clean ACT Inference Task for HIROLRobotPlatform
Refactored version with no try-except and no fallback mechanisms
"""

import time
import os
import glog as log
import numpy as np
import torch
from typing import Dict, Any, Optional

# Base class import
from factory.tasks.inferences_tasks.inference_base import InferenceBase

# 从dependencies/act导入ACT推理组件
from dependencies.act.policy_inference import ACTInference

# 从project导入data_adapter
from dependencies.act.data_adapter import create_data_adapter

# Import refactored utility modules - direct imports, no try-except
from factory.tasks.inferences_tasks.utils.camera_handler import CameraHandler
from factory.tasks.inferences_tasks.utils.performance_monitor import PerformanceMonitor
from factory.tasks.inferences_tasks.utils.keyboard_handler import KeyboardHandler
from factory.tasks.inferences_tasks.utils.state_processor import StateProcessor
from factory.tasks.inferences_tasks.utils.gripper_controller import (
    create_gripper_controller, GripperStateLogger
)

# 导入任务类型系统 - direct import, no try-except
from factory.tasks.inferences_tasks.utils.task_types import TaskType, TaskTypeFactory

# Time statistics
from tools.performance_profiler import timer, PerformanceProfiler

GRIPPER_OPEN_WIDTH = 90

class ACT_Inferencer(InferenceBase):
    """
    Clean ACT inference task - no try-except, no fallback
    All dependencies are required and must be available
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # 任务类型检测和配置合并
        config = self._init_task_type_system(config)

        super().__init__(config)

        # ACT-specific configurations
        self._frequency = config.get("frequency", 20.0)
        self._dt = 1.0 / self._frequency

        # Episode step limit configuration
        self._max_step_nums = config.get("max_step_nums", config.get("max_episode_length", float('inf')))
        log.info(f"📊 Max steps per episode: {self._max_step_nums}")

        # Load ACT model and initialize inference parameters
        self._device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self._device}")

        self._ckpt_dir = config["checkpoint_path"]
        self._robot_type = config.get("robot_type", "fr3")

        # Initialize ACT inference engine
        self._init_act_engine(config)

        # Clear gripper plots directory before starting
        self._clear_gripper_plots_directory()

        # Initialize modular components - all required, no fallback
        self.camera_handler = CameraHandler(config, self._device)
        self.performance_monitor = PerformanceMonitor(config)
        self.keyboard_handler = KeyboardHandler()
        self.state_processor = StateProcessor(config)

        # ACT action sequence management
        self.action_sequence = None
        self.action_index = 0
        self.action_chunk_size = config.get("action_chunk_size", 100)
        self.action_sampling_interval = config.get("action_sampling_interval", 1)
        self.failed_actions_count = 0
        self.max_failed_actions = config.get("max_failed_actions", 5)

        # Sliding window inference parameters
        self.sliding_window_size = config.get("sliding_window_size", 10)
        self.steps_since_last_prediction = 0
        self.force_repredict = False

        # Action interpolation for smoothness
        self.enable_action_interpolation = config.get("enable_action_interpolation", True)
        self.last_executed_action = None
        self.interpolation_steps = config.get("interpolation_steps", 3)

        # Initialize temporal aggregation (replaces action aggregator)
        self._init_temporal_aggregation(config)

        # Initialize gripper controllers
        self._init_gripper_controllers(config)

        # Initialize joint visualizer
        self._init_joint_visualizer(config)

        # Camera latency compensation
        self.camera_latency_ms = config.get("camera_latency_ms", 80.0)
        self.action_delay_compensation_steps = 0

        # Image display configuration
        self.visualization_config = config.get("visualization", {})
        # 覆盖基类的display设置，使用visualization.enable_image_display
        if self.visualization_config.get("enable_image_display", False):
            self._enable_display = True
        self.display_window_size = self.visualization_config.get("display_window_size", [640, 480])
        self.display_windows_initialized = False

        self.gripper_action_previous = 1.0  # 用于记录上一个夹爪动作

        # 🆕 Episode对比记录器
        self.episode_first_inputs = {}  # 记录每个episode的第一个输入
        self.current_episode_num = 0
        self.step_count = 0

        log.info("ACT_Inferencer initialized successfully")

    def _init_task_type_system(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """初始化任务类型系统并合并任务特定配置"""
        # 检测任务类型
        self.task_type = TaskType.from_config(config)
        log.info(f"🎯 检测到任务类型: {self.task_type.display_name}")

        # 调试：记录合并前的配置
        log.info(f"🔍 合并前config有max_step_nums: {'max_step_nums' in config}")
        if 'max_step_nums' in config:
            log.info(f"🔍 合并前config['max_step_nums'] = {config['max_step_nums']}")

        # 加载并合并任务特定配置
        from factory.tasks.inferences_tasks.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        merged_config = config_loader.merge_with_task_config(config, self.task_type.value)
        log.info(f"✅ 成功合并任务特定配置")

        # 调试：记录合并后的配置
        log.info(f"🔍 合并后merged_config有max_step_nums: {'max_step_nums' in merged_config}")
        if 'max_step_nums' in merged_config:
            log.info(f"🔍 合并后merged_config['max_step_nums'] = {merged_config['max_step_nums']}")

        # 验证合并后的配置
        TaskTypeFactory.validate_task_config(self.task_type, merged_config)

        # 应用任务特定的max_step_nums（如果合并后的配置没有）
        if "max_step_nums" not in merged_config:
            default_steps = TaskTypeFactory.get_default_max_steps(self.task_type)
            merged_config["max_step_nums"] = default_steps
            log.info(f"📏 应用任务特定步数限制: {default_steps} 步")
        else:
            log.info(f"📏 使用合并后的步数限制: {merged_config['max_step_nums']} 步")

        # 获取任务特征
        task_characteristics = TaskTypeFactory.get_task_characteristics(self.task_type)
        log.info(f"📋 任务特征: {task_characteristics}")

        return merged_config

    def _init_act_engine(self, config: Dict[str, Any]) -> None:
        """Initialize ACT inference engine and data adapter"""
        learning_config = config.get("learning", {})

        # Validate checkpoint directory
        if not os.path.exists(self._ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self._ckpt_dir}")

        # Auto-detect robot configuration
        if self._robot_type == "monte01":
            state_dim = 16  # Monte01 dual-arm: 16 DOF
            camera_names = learning_config.get("camera_names", ["left_ee_cam", "right_ee_cam", "third_person_cam"])
            log.info(f"🤖 Detected Monte01 dual-arm mode, using {state_dim} DOF")
        elif self._robot_type == "fr3":
            state_dim = learning_config.get("state_dim", 8)  # FR3: 7 DOF arm + 1 DOF gripper
            camera_names = learning_config.get("camera_names", ["ee_cam", "third_person_cam", "side_cam"])
            log.info(f"🤖 Detected FR3 single-arm mode, using {state_dim} DOF")
        else:
            raise Exception(f"not Support")

        # Extract ACT parameters
        act_kwargs = {
            key: learning_config[key] for key in learning_config
            if key not in ["state_dim", "camera_names", "robot_type"]
        }

        # Create ACT inference engine
        self.act_engine = ACTInference(
            ckpt_dir=self._ckpt_dir,
            state_dim=state_dim,
            camera_names=camera_names,
            **act_kwargs
        )

        # Create data adapter
        self.data_adapter = create_data_adapter(self._robot_type)

        log.info(f"✅ ACT inference engine initialized:")
        log.info(f"   - Robot type: {self._robot_type}")
        log.info(f"   - State dimension: {state_dim}")
        log.info(f"   - Cameras: {camera_names}")
        log.info(f"   - Checkpoint: {self._ckpt_dir}")

    def _init_temporal_aggregation(self, config: Dict[str, Any]) -> None:
        """Initialize temporal aggregation system (replaces action aggregator)"""
        temporal_agg_config = config.get("action_aggregation", {})
        self.temporal_agg_enabled = temporal_agg_config.get("enabled", False)

        if self.temporal_agg_enabled:
            # Temporal aggregation parameters
            self.temporal_agg_k = temporal_agg_config.get("k", 0.01)  # decay factor
            self.max_timesteps = config.get("max_step_nums", config.get("max_episode_length", 1000))
            self.num_queries = self.action_chunk_size

            # Initialize all_time_actions tensor
            state_dim = config.get("learning", {}).get("state_dim", 8)  # FR3: 7 joints + 1 gripper

            # Store actions for all timesteps: [max_timesteps, max_timesteps+num_queries, state_dim]
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries, state_dim]).cuda()
            self.current_timestep = 0

            log.info("✅ Temporal aggregation enabled")
            log.info(f"   - Decay factor k: {self.temporal_agg_k}")
            log.info(f"   - Max timesteps: {self.max_timesteps}")
            log.info(f"   - State dimension: {state_dim}")
        else:
            self.all_time_actions = None
            self.current_timestep = 0
            log.info("ℹ️ Temporal aggregation disabled")

    def _init_gripper_controllers(self, config: Dict[str, Any]) -> None:
        """Initialize gripper controllers"""
        self.gripper_postprocess_config = config.get("gripper_postprocess", {})
        self.gripper_postprocess_enabled = self.gripper_postprocess_config.get("enabled", True)

        if not self.gripper_postprocess_enabled:
            log.info("🔧 Gripper post-processing disabled")
            self.gripper_controller = None
            self.gripper_state = None
            return

        # 使用统一的任务感知控制器工厂
        if self._robot_type == "monte01":
            # Dual-arm gripper controllers
            left_config = self.gripper_postprocess_config.get("left_gripper", self.gripper_postprocess_config)
            right_config = self.gripper_postprocess_config.get("right_gripper", self.gripper_postprocess_config)

            # Create separate configs for each arm
            left_full_config = config.copy()
            left_full_config['gripper_postprocess'] = left_config
            right_full_config = config.copy()
            right_full_config['gripper_postprocess'] = right_config

            self.left_gripper_controller = create_gripper_controller(left_config, left_full_config)
            self.right_gripper_controller = create_gripper_controller(right_config, right_full_config)
            self.gripper_controller = self.left_gripper_controller  # Backward compatibility

            log.info(f"✅ Monte01 dual-arm task-aware gripper controllers initialized")
        else:
            # Single-arm gripper controller
            self.gripper_controller = create_gripper_controller(
                self.gripper_postprocess_config,
                config  # 传递完整配置以支持任务类型检测
            )
            log.info(f"✅ {self._robot_type} task-aware gripper controller initialized")

        # Initialize gripper state
        if hasattr(self.gripper_controller, 'state'):
            self.gripper_state = self.gripper_controller.state
        else:
            self.gripper_state = None

    def _init_joint_visualizer(self, config: Dict[str, Any]) -> None:
        """Initialize joint position visualizer"""
        self.joint_vis_config = config.get("joint_position_visualization", {})

    def start_inference(self) -> None:
        """Main inference loop following the unified pattern"""
        for episode_num in range(self._num_episodes):
            if self._quit:
                break

            # Reset for new episode
            self._gym_robot.reset()

            # 🆕 重置夹爪控制器状态
            if self.gripper_controller:
                self.gripper_controller.reset()
                log.info("🔄 夹爪控制器状态已重置")

            self._reset_action_sequence()
            self._status_ok = True

            # Reset temporal aggregation for new episode
            if self.temporal_agg_enabled:
                self.all_time_actions.zero_()
                self.current_timestep = 0
                log.info("🔄 Temporal aggregation reset for new episode")

            # Store current episode number for comparison
            self.current_episode_num = episode_num

            log.info(f'Starting ACT inference episode {episode_num}')

            # Log initial gripper state
            self._log_gripper_state("episode_start")

            # Calculate delay compensation
            if self.camera_latency_ms > 0 and self._frequency > 0:
                control_period_ms = 1000.0 / self._frequency
                self.action_delay_compensation_steps = int(round(self.camera_latency_ms / control_period_ms))
                log.info(f"📷 Camera latency compensation: {self.camera_latency_ms}ms, skipping {self.action_delay_compensation_steps} actions")
            else:
                self.action_delay_compensation_steps = 0

            # Initialize display windows
            self._initialize_display_windows()

            step_count = 0
            self.step_count = 0  # 🆕 全局步数计数器

            while self._status_ok:
                step_start_time = time.perf_counter()

                # Check if maximum steps reached
                if step_count >= self._max_step_nums:
                    log.info(f"🔄 Reached maximum steps ({self._max_step_nums}), returning to start position...")
                    break

                # Get observations
                with timer("gym_obs", "act_inferencer"):
                    act_obs = self.convert_from_gym_obs()

                # Predict actions if needed
                if self._should_predict_new_actions():
                    with timer("act_inference_time", "act_inferencer"):
                        self._predict_new_action_sequence(act_obs)

                # Execute next action
                if self.action_sequence is not None:
                    with timer("gym_step", "act_inferencer"):
                        action = self._get_next_action()
                        self.convert_to_gym_action(action)

                    step_count += 1
                    self.step_count += 1  # 🆕 更新全局步数计数器
                    self.steps_since_last_prediction += 1

                    # Update temporal aggregation timestep
                    if self.temporal_agg_enabled:
                        self.current_timestep += 1

                    if step_count % 100 == 0:
                        log.info(f"📊 Executed {step_count} steps (max: {self._max_step_nums})")

                # Control frequency
                elapsed = time.perf_counter() - step_start_time
                sleep_time = max(0, self._dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            log.info(f"🎯 Episode {episode_num} completed with {step_count} steps")

            # Clean up keyboard listener
            self.keyboard_handler.stop_keyboard_listener()

            # Print episode performance statistics if enabled
            if self.performance_monitor.is_enabled():
                self.performance_monitor.print_episode_performance_stats(episode_num, step_count)

    def convert_from_gym_obs(self) -> Dict[str, torch.Tensor]:
        """Convert gym observations to ACT format"""
        gym_obs = super().convert_from_gym_obs()

        with timer("image_display", "act_inference"):
            self.image_display(gym_obs)

        # Process robot state using state processor
        with timer("robot_state_extraction", "act_inference"):
            state_components = self.state_processor.extract_robot_state(gym_obs)

        # Process camera observations using camera handler
        with timer("camera_obs_extraction", "act_inference"):
            camera_obs = self.camera_handler.extract_camera_observations(gym_obs)

        return {
            'state': state_components,
            'images': camera_obs
        }

    def convert_to_gym_action(self, act_action: np.ndarray) -> None:
        """Convert ACT action to gym format and execute"""
        with timer("action_conversion_total", "act_inference"):
            with timer("action_preprocessing", "act_inference"):
                if torch.is_tensor(act_action):
                    act_action = act_action.cpu().numpy()

                # 确保是数组格式，处理标量输入
                act_action = np.atleast_1d(np.asarray(act_action)).flatten()

                # Apply temporal aggregation if enabled
                if self.temporal_agg_enabled and self.all_time_actions is not None:
                    act_action = self._apply_temporal_aggregation(act_action)

                # Apply action interpolation for smoothness
                if self.enable_action_interpolation and self.last_executed_action is not None:
                    alpha = 0.3
                    act_action = (1 - alpha) * self.last_executed_action + alpha * act_action

            # Convert to gym action format
            with timer("action_format_conversion", "act_inference"):
                gym_action = self._convert_act_action_to_gym_format(act_action)

            # Execute action
            with timer("action_execution", "act_inference"):
                success = self._execute_gym_action(gym_action)

            if success:
                self.last_executed_action = act_action.copy()
                self.failed_actions_count = 0
            else:
                self.failed_actions_count += 1

    def _convert_act_action_to_gym_format(self, act_action: np.ndarray) -> Dict[str, Any]:
        """Convert ACT action to gym action format"""
        # 确保 act_action 是数组
        act_action = np.atleast_1d(act_action)

        if self._robot_type == "monte01":
            # Monte01 dual-arm: split 16D action
            left_arm_action = act_action[:8]
            right_arm_action = act_action[8:16]

            left_gripper_action = np.clip(left_arm_action[7] / 0.074, 0.0, 1.0)
            right_gripper_action = np.clip(right_arm_action[7] / 0.074, 0.0, 1.0)
            return {
                'arm': np.concatenate([left_arm_action[:7], right_arm_action[:7]]),  # 14D joint command
                'tool': np.array([left_gripper_action, right_gripper_action])  # 2D tool command
            }
        else:
            # Single-arm robot (FR3)
            joint_action = act_action[:7] if len(act_action) >= 7 else act_action
            gripper_action = act_action[7] if len(act_action) > 7 else 1.0

            gripper_action = np.clip(gripper_action / GRIPPER_OPEN_WIDTH, 0.0, 1.0)
            return {
                'arm': joint_action.astype(np.float32),
                'tool': np.array([gripper_action]).astype(np.float32)
            }

    def _execute_gym_action(self, gym_action: Dict[str, Any]) -> bool:
        """Execute gym action with gripper post-processing"""
        if self.gripper_controller is not None and 'tool' in gym_action:
            with timer("gripper_control_processing", "act_inference"):
                gripper_val = gym_action['tool'][0] if len(gym_action['tool']) > 0 else 1.0
                gym_action['tool'] = np.array([gripper_val])
                log.info(f"gym_action['tool'] => {gym_action['tool']}")
                # Log gripper action details (every 20 steps to avoid spam)
                # if hasattr(self, '_gripper_log_counter'):
                #     self._gripper_log_counter += 1
                # else:
                #     self._gripper_log_counter = 1

                # current_obs = self._gym_robot.get_observation()

                # # 检查当前夹爪宽度，如果小于5mm则强制打开
                # current_gripper_width = None
                # safety_failed = False
                # # 获取当前夹爪状态
                # tools_dict = current_obs.get("tools", {})
                # if tools_dict:
                #     # 获取第一个工具（通常是夹爪）
                #     tool_key = list(tools_dict.keys())[0]  # 'single', 'left', 'right'等
                #     tool_data = tools_dict[tool_key]
                #     if isinstance(tool_data, dict) and "position" in tool_data:
                #         current_gripper_width = float(tool_data["position"])

                #         # 安全检查：使用配置的最小抓取宽度进行检查
                #         min_safe_width = self.gripper_controller.min_grasp_width
                #         if current_gripper_width < min_safe_width and self.gripper_controller.grasp_check_enabled:
                #             safety_failed = True
                #             log.warning(f"⚠️ 检测到夹爪过紧！当前宽度={current_gripper_width:.4f}m < {min_safe_width:.4f}m，需要打开")

                #         # 定期记录夹爪宽度（每50步）
                #         if self._gripper_log_counter % 50 == 1:
                #             log.info(f"📏 当前夹爪宽度: {current_gripper_width:.4f}m")

                # execute_command, command_value, force_repredict = self.gripper_controller.process(
                #     gripper_val,
                #     safety_failed=safety_failed,
                #     end_effector_pose=None
                # )

                # if force_repredict:
                #     self.force_repredict = True

                # if execute_command and command_value is not None:
                #     # 使用gripper controller返回的命令值
                #     log.info(f"🔧 夹爪控制: 输出={command_value:.4f} (控制器修改)")
                #     gym_action['tool'] = np.array([command_value])
                #     self.gripper_action_previous = gym_action['tool'][0]
                # else:
                #     gym_action['tool'] = np.array([self.gripper_action_previous])

        with timer("robot_step_execution", "act_inference"):
            self._gym_robot.step(gym_action)

        # Save camera images when close/open actions are detected
        if self.gripper_controller is not None and 'tool' in gym_action:
            self.camera_handler.check_and_save_action_images(gym_action, self._gym_robot)

        return True

    def _should_predict_new_actions(self) -> bool:
        """Check if new prediction is needed"""
        # No action sequence cached
        if self.action_sequence is None:
            return True

        # Too many consecutive failures
        if self.failed_actions_count >= self.max_failed_actions:
            return True

        # Force repredict flag set (e.g., by gripper controller)
        if self.force_repredict:
            self.force_repredict = False
            return True

        # Action sequence exhausted
        real_index = self.action_index * self.action_sampling_interval
        return real_index >= len(self.action_sequence)

    def _predict_new_action_sequence(self, obs_dict: Dict[str, Any]) -> None:
        """Predict new action sequence using ACT"""
        log.info("🤖 Executing ACT inference for new action sequence...")

        # Prepare state and images for ACT
        with timer("state_preparation", "act_inference"):
            state = obs_dict['state']
            images = obs_dict['images']

            # 🆕 记录传给ACT模型的状态
            log.info(f"📊 ACT模型输入 (Episode {self.current_episode_num}, Step {self.step_count}):")
            log.info(f"   - State shape: {state.shape}")
            log.info(f"   - State values: {state}")

            # 特别关注夹爪状态（假设是state的最后一个元素）
            if len(state) >= 8:  # FR3是8维状态
                gripper_value = state[-1]
                log.info(f"   - 夹爪输入值: {gripper_value:.4f}")

        # Run ACT inference
        with timer("model_inference", "act_inference"):
            predicted_actions = self.act_engine.predict(state=state, images=images)

            # 🆕 记录模型输出的夹爪动作
            if predicted_actions.shape[-1] >= 8:
                gripper_action = predicted_actions[0, -1] if predicted_actions.ndim > 1 else predicted_actions[-1]
                log.info(f"   - ACT输出夹爪动作: {gripper_action:.4f}")

                # 检查并修正异常的夹爪动作值
                original_gripper_action = gripper_action
                if gripper_action > GRIPPER_OPEN_WIDTH:
                    log.warning(f"⚠️ ACT输出夹爪动作异常: {gripper_action:.4f} > GRIPPER_OPEN_WIDTHm，已clip到GRIPPER_OPEN_WIDTH")
                    gripper_action = GRIPPER_OPEN_WIDTH
                    predicted_actions[0, -1] = gripper_action  # 修正原始数组
                elif gripper_action < 0:
                    log.warning(f"⚠️ ACT输出夹爪动作异常: {gripper_action:.4f} < 0，已clip到0.0")
                    gripper_action = 0.0
                    predicted_actions[0, -1] = gripper_action  # 修正原始数组

                if original_gripper_action != gripper_action:
                    log.info(f"   - 修正后夹爪动作: {gripper_action:.4f}")

            # Store actions in temporal aggregation if enabled
            if self.temporal_agg_enabled and self.all_time_actions is not None:
                self._store_predicted_actions(predicted_actions)

        log.info(f"🔍 ACT output: shape={predicted_actions.shape}, type={type(predicted_actions)}")

        # Process action chunk output
        with timer("action_postprocessing", "act_inference"):
            if predicted_actions.ndim == 2 and predicted_actions.shape[0] > 1:
                # Get action sequence chunk
                all_actions = predicted_actions[:self.action_chunk_size]

                # Apply delay compensation
                if self.action_delay_compensation_steps > 0:
                    compensated_actions = all_actions[self.action_delay_compensation_steps:]
                    if len(compensated_actions) == 0:
                        compensated_actions = all_actions[-1:]
                    self.action_sequence = compensated_actions
                    log.info(f"📦 Cached compensated action sequence: {all_actions.shape} → {len(self.action_sequence)}")
                else:
                    self.action_sequence = all_actions
                    log.info(f"📦 Cached action sequence: {all_actions.shape}")
            else:
                # Single-step action output
                if predicted_actions.ndim > 1:
                    predicted_actions = predicted_actions.squeeze()
                self.action_sequence = predicted_actions.reshape(1, -1)
                log.info(f"📍 Single-step action cached: {self.action_sequence.shape}")

            # Reset indices and counters
            self.action_index = 0
            self.failed_actions_count = 0
            self.steps_since_last_prediction = 0

            log.info(f"✅ New action sequence cached: shape={self.action_sequence.shape}")

    def _get_next_action(self) -> np.ndarray:
        """Get next action from sequence"""
        real_index = self.action_index * self.action_sampling_interval
        action = self.action_sequence[real_index]

        # 确保 action 是数组
        if np.isscalar(action):
            action = np.array([action])
        elif action.ndim == 0:  # 0维数组（标量）
            action = action.reshape(1)

        self.action_index += 1
        return action

    def _reset_action_sequence(self) -> None:
        """Reset action sequence cache"""
        self.action_sequence = None
        self.action_index = 0
        self.failed_actions_count = 0

        if self.temporal_agg_enabled and self.all_time_actions is not None:
            self.all_time_actions.zero_()
            self.current_timestep = 0

        if self.gripper_controller is not None:
            self.gripper_controller.reset()

        self.gripper_action_previous = 1

        # Log current gripper state after reset
        self._log_gripper_state("after_reset")
        log.info("🔄 Action sequence reset")

    def _initialize_display_windows(self) -> None:
        """Initialize image display windows"""
        if not self._enable_display or self.display_windows_initialized:
            return

        # Simple display initialization - can be enhanced later
        self.display_windows_initialized = True
        log.info("🖥️ Image display windows initialized")

    def _log_gripper_state(self, context: str = "") -> None:
        """Log current gripper state for debugging"""
        GripperStateLogger.log_gripper_state(self._gym_robot, self.gripper_controller, context)

    def _clear_gripper_plots_directory(self) -> None:
        """Clear gripper plots directory before starting inference"""
        import shutil
        from pathlib import Path

        plots_dir = Path("./logs/gripper_plots")

        if plots_dir.exists():
            # Remove entire directory
            shutil.rmtree(plots_dir)
            log.info(f"🗑️ Removed gripper plots directory: {plots_dir}")

        # Create fresh directory
        plots_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"📁 Created fresh gripper plots directory: {plots_dir}")

    def close(self) -> None:
        """Clean up ACT-specific resources"""
        # Print final performance report
        if self.performance_monitor.is_enabled():
            self.performance_monitor.print_final_performance_report()

        # Stop joint plotter
        if hasattr(self, 'joint_plotter') and self.joint_plotter is not None:
            self.joint_plotter.stop_plotting()

        # Clean up temporal aggregation
        if hasattr(self, 'all_time_actions') and self.all_time_actions is not None:
            self.all_time_actions = None

        # Clean up modular components
        if hasattr(self, 'keyboard_handler') and self.keyboard_handler is not None:
            self.keyboard_handler.cleanup()

        # Clean up ACT engine
        if hasattr(self, 'act_engine'):
            del self.act_engine

        super().close()

    def _store_predicted_actions(self, predicted_actions: torch.Tensor) -> None:
        """Store predicted actions in temporal aggregation buffer"""
        if not self.temporal_agg_enabled or self.all_time_actions is None:
            return

        # Convert to tensor if needed
        if isinstance(predicted_actions, np.ndarray):
            predicted_actions = torch.from_numpy(predicted_actions).cuda()

        # Store actions: all_time_actions[current_t, current_t:current_t+num_queries] = predicted_actions
        t = self.current_timestep
        if t < self.max_timesteps and predicted_actions.shape[0] > 0:
            end_t = min(t + self.num_queries, self.max_timesteps + self.num_queries)
            actual_queries = end_t - t
            self.all_time_actions[t, t:end_t] = predicted_actions[:actual_queries]
            log.info(f"📦 Stored {actual_queries} actions for timestep {t}")

    def _apply_temporal_aggregation(self, current_action: np.ndarray) -> np.ndarray:
        """Apply temporal aggregation to get smoothed action"""
        if not self.temporal_agg_enabled or self.all_time_actions is None:
            return current_action

        t = self.current_timestep
        if t >= self.max_timesteps:
            return current_action

        # Get all actions for current timestep
        actions_for_curr_step = self.all_time_actions[:, t]

        # Find which timesteps have valid (non-zero) actions for current step
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
        valid_actions = actions_for_curr_step[actions_populated]

        if len(valid_actions) == 0:
            log.warning(f"⚠️ No valid actions found for timestep {t}, using current action")
            return current_action

        # Calculate exponential weights
        k = self.temporal_agg_k
        exp_weights = np.exp(-k * np.arange(len(valid_actions)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)

        # Compute weighted average
        aggregated_action = (valid_actions * exp_weights).sum(dim=0, keepdim=True)

        # Convert back to numpy
        result = aggregated_action.squeeze(0).cpu().numpy()

        log.info(f"🔄 Temporal aggregation: {len(valid_actions)} actions → timestep {t}")
        return result


def main():
    """Main function for testing clean ACT inference"""
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml

    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "factory/tasks/inferences_tasks/act/config/fr3_act_inference_cfg.yaml",
            "help": "Path to the config file"
        }
    }
    args = parse_args("ACT inference (clean)", arguments)

    # Load configuration
    config = dynamic_load_yaml(args.config)

    act_executor = ACT_Inferencer(config)
    act_executor.start_inference()


if __name__ == "__main__":
    main()