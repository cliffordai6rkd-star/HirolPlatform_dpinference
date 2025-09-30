#!/usr/bin/env python3
"""
State Processor for ACT Inference
Handles robot state extraction and processing
"""

import numpy as np
import glog as log
from typing import Dict, Any


class StateProcessor:
    """Handles robot state extraction and processing for ACT inference"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize state processor

        Args:
            config: Configuration dictionary
        """
        self._robot_type = config.get("robot_type", "fr3")
        self._obs_contain_ee = config.get("obs_contain_ee", False)

        # 智能日志控制
        self._last_gripper_value = None
        self._log_counter = 0
        self._gripper_change_threshold = 0.001  # 夹爪值变化阈值
        self._log_every_n_steps = 100  # 每N步强制打印一次

        log.info("✅ State processor initialized")

    def extract_robot_state(self, gym_obs: Dict[str, Any]) -> np.ndarray:
        """Extract robot state from gym observations"""
        state_components = np.array([])

        # Extract joint positions and tool states
        for key, joint_state in gym_obs.get('joint_states', {}).items():
            joint_positions = np.array(joint_state["position"], dtype=np.float32)
            log.debug(f"Processing key: {key}, joint_positions shape: {joint_positions.shape}")

            # Add end-effector pose if configured
            if self._obs_contain_ee:
                ee_pose = gym_obs.get('ee_states', {}).get(key, None)
                log.debug(f"ee_pose for {key}: {ee_pose}, type: {type(ee_pose)}")
                if ee_pose is not None:
                    # 处理字典格式的 ee_pose
                    if isinstance(ee_pose, dict):
                        log.debug(f"Processing dict ee_pose for {key}: {ee_pose}")
                        # 尝试按顺序提取 pose, position, 或 x, y, z
                        if 'pose' in ee_pose:
                            ee_pose = ee_pose['pose']
                        elif 'position' in ee_pose:
                            ee_pose = ee_pose['position']
                        elif all(k in ee_pose for k in ['x', 'y', 'z']):
                            ee_pose = [ee_pose['x'], ee_pose['y'], ee_pose['z']]
                        else:
                            # 如果字典格式未知，跳过
                            log.warning(f"Unknown ee_pose format for {key}: {ee_pose}")
                            ee_pose = None

                    if ee_pose is not None:
                        ee_pose = np.atleast_1d(ee_pose)
                        joint_positions = np.hstack((joint_positions, ee_pose))
                        log.debug(f"After adding ee_pose, joint_positions shape: {joint_positions.shape}")

            # Add tool state - 更健壮的处理
            tools_value = gym_obs.get("tools", {}).get(key, None)
            log.debug(f"tools_value for {key}: {tools_value}, type: {type(tools_value)}")

            # 智能日志控制 - 只在值变化显著时打印
            should_log = False
            if key == 'single':
                self._log_counter += 1

            if tools_value is not None:
                # 处理不同格式的 tools 数据
                if isinstance(tools_value, dict):
                    tools_data = tools_value.get("position", None)
                    log.debug(f"Extracted tools_data from dict: {tools_data}")

                    # 检查是否需要打印日志
                    if key == 'single' and tools_data is not None:
                        current_value = float(tools_data)
                        if (self._last_gripper_value is None or
                            abs(current_value - self._last_gripper_value) > self._gripper_change_threshold or
                            self._log_counter % self._log_every_n_steps == 1):
                            should_log = True
                        self._last_gripper_value = current_value
                else:
                    # 检查是否需要打印日志
                    if key == 'single' and tools_data is not None:
                        current_value = float(tools_data)
                        if (self._last_gripper_value is None or
                            abs(current_value - self._last_gripper_value) > self._gripper_change_threshold or
                            self._log_counter % self._log_every_n_steps == 1):
                            should_log = True
                            log.info(f"🔧 夹爪状态提取 (key={key}):")
                            log.info(f"   - 直接使用tools_value: {tools_data}")
                        self._last_gripper_value = current_value

                if tools_data is not None:
                    tools_data = np.atleast_1d(tools_data)
                    joint_positions = np.hstack((joint_positions, tools_data))
                    log.debug(f"After adding tools_data, joint_positions shape: {joint_positions.shape}")

                    # 只在需要时打印最终值
                    if should_log and key == 'single':
                        log.info(f"   - 最终夹爪值添加到state: {tools_data[0]:.4f}")

            state_components = np.hstack((state_components, joint_positions))
            log.debug(f"Final joint_positions for {key}: shape={joint_positions.shape}, state_components shape: {state_components.shape}")

        log.debug(f"Final state_components shape: {state_components.shape}, dtype: {state_components.dtype}")
        return state_components.astype(np.float32)

    def get_robot_type(self) -> str:
        """Get configured robot type"""
        return self._robot_type