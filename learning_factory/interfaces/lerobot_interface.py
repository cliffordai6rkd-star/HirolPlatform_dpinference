#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Composable interface for running LeRobot policies against RobotMotion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset.utils import ActionType
from factory.tasks.robot_motion import RobotMotion
from hardware.base.utils import transform_pose

from learning_factory.datasets.utils import load_info, load_stats
from learning_factory.utils.orientation import (
    _get_orientation_dim,
    _orientation_to_rotation,
    _pose_to_representation,
    _split_pose_and_gripper,
)

def _extract_first_dim(feature: dict[str, Any] | None) -> int | None:
    if not feature:
        return None
    shape = feature.get("shape")
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0])
    dims = feature.get("dims")
    if isinstance(dims, int):
        return int(dims)
    if isinstance(dims, (list, tuple)) and dims:
        return int(dims[0])
    return None


@dataclass
class DatasetLayout:
    repo_id: str
    root: Path
    info: dict[str, Any]
    features: dict[str, Any]
    stats: dict[str, Any] | None
    image_keys: list[str] = field(default_factory=list)
    state_key: str | None = None
    state_dim: int | None = None
    action_key: str | None = None
    action_dim: int | None = None
    fps: float | None = None

    @classmethod
    def from_dataset(cls, root: str | Path, repo_id: str) -> "DatasetLayout":
        root_path = Path(root)
        info = load_info(root_path)
        try:
            stats = load_stats(root_path)
        except FileNotFoundError:
            stats = info.get("stats")

        features = info.get("features", {})
        image_keys = sorted(key for key in features if key.startswith("observation.images."))

        state_key = "observation.state" if "observation.state" in features else None
        state_dim = _extract_first_dim(features.get(state_key))

        action_key = None
        for candidate in ("action", "actions", "policy.action", "policy.actions"):
            if candidate in features:
                action_key = candidate
                break
        action_dim = _extract_first_dim(features.get(action_key))

        fps = None
        raw_fps = info.get("fps")
        if isinstance(raw_fps, (int, float)):
            fps = float(raw_fps)

        return cls(
            repo_id=repo_id,
            root=root_path,
            info=info,
            features=features,
            stats=stats,
            image_keys=image_keys,
            state_key=state_key,
            state_dim=state_dim,
            action_key=action_key,
            action_dim=action_dim,
            fps=fps,
        )

    def as_policy_metadata(self) -> Any:
        class _PolicyMeta:
            def __init__(self, layout: DatasetLayout) -> None:
                self.repo_id = layout.repo_id
                self.root = layout.root
                self.info = layout.info
                self.features = layout.features
                self.stats = layout.stats

        return _PolicyMeta(self)


@dataclass
class ObservationHistory:
    joints: np.ndarray | None = None
    tool: np.ndarray | None = None
    ee_position: np.ndarray | None = None
    ee_quaternion: np.ndarray | None = None
    ee_orientation: np.ndarray | None = None

    def reset(self) -> None:
        self.joints = None
        self.tool = None
        self.ee_position = None
        self.ee_quaternion = None
        self.ee_orientation = None


@dataclass
class CommandTracker:
    joint_target: np.ndarray | None = None
    pose_target: np.ndarray | None = None
    gripper: float | None = None
    command_joint_target: np.ndarray | None = None
    command_pose_target: np.ndarray | None = None
    command_gripper: float | None = None

    def reset(self) -> None:
        self.joint_target = None
        self.pose_target = None
        self.gripper = None
        self.command_joint_target = None
        self.command_pose_target = None
        self.command_gripper = None

    def sync_with_robot(self, robot_motion: RobotMotion) -> None:
        state = robot_motion.get_state()
        joints = state.get("q")
        pose = state.get("pose")
        gripper_pos = state.get("gripper_pos")

        if joints is not None:
            joints_arr = np.asarray(joints, dtype=np.float64)
            self.joint_target = joints_arr.copy()
            if self.command_joint_target is None:
                self.command_joint_target = joints_arr.copy()
        if pose is not None:
            pose_arr = np.asarray(pose, dtype=np.float64)
            self.pose_target = pose_arr.copy()
            if self.command_pose_target is None:
                self.command_pose_target = pose_arr.copy()
        if gripper_pos is not None:
            if isinstance(gripper_pos, dict):
                first_key = next(iter(gripper_pos))
                raw_grip = gripper_pos[first_key]
            else:
                raw_grip = gripper_pos
            grip_array = np.asarray(raw_grip, dtype=np.float64).reshape(-1)
            if grip_array.size == 0:
                grip = 0.0
            else:
                grip = float(grip_array[0])
            self.gripper = grip
            if self.command_gripper is None:
                self.command_gripper = grip

    def update_joint(self, target: np.ndarray, *, is_command: bool) -> None:
        arr = np.asarray(target, dtype=np.float64)
        if is_command:
            self.command_joint_target = arr.copy()
        else:
            self.joint_target = arr.copy()
            if self.command_joint_target is None:
                self.command_joint_target = arr.copy()

    def update_pose(self, target: np.ndarray, *, is_command: bool) -> None:
        arr = np.asarray(target, dtype=np.float64)
        if is_command:
            self.command_pose_target = arr.copy()
        else:
            self.pose_target = arr.copy()
            if self.command_pose_target is None:
                self.command_pose_target = arr.copy()

    def update_gripper(self, value: float, *, is_command: bool) -> None:
        if is_command:
            self.command_gripper = float(value)
        else:
            self.gripper = float(value)
            if self.command_gripper is None:
                self.command_gripper = float(value)

    def snapshot(self) -> dict[str, Any]:
        return {
            "joint_target": None if self.joint_target is None else self.joint_target.copy(),
            "pose_target": None if self.pose_target is None else self.pose_target.copy(),
            "gripper": self.gripper,
            "command_joint_target": None
            if self.command_joint_target is None
            else self.command_joint_target.copy(),
            "command_pose_target": None
            if self.command_pose_target is None
            else self.command_pose_target.copy(),
            "command_gripper": self.command_gripper,
        }


class ObservationBuilder:
    def __init__(
        self,
        layout: DatasetLayout,
        observation_type: str,
        observation_orientation: str,
    ) -> None:
        self._layout = layout
        self._observation_type = observation_type.lower()
        self._observation_orientation = observation_orientation
        self._history = ObservationHistory()

    @property
    def history(self) -> ObservationHistory:
        return self._history

    def reset(self) -> None:
        self._history.reset()

    def build(self, robot_motion: RobotMotion) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
        observation: dict[str, np.ndarray] = {}

        robot_system = getattr(robot_motion, "_robot_system", None)
        imgs = {}
        if robot_system is not None and hasattr(robot_system, "get_cameras_infos"):
            info_list = robot_system.get_cameras_infos() or []
            for cam in info_list:
                name = cam.get("name")
                img = cam.get("img")
                if name is not None and img is not None:
                    imgs[name] = img

        for feature_key in self._layout.image_keys:
            camera_name = feature_key.split(".")[-1]
            img = imgs.get(camera_name, imgs.get(feature_key))
            if img is None:
                raise RuntimeError(f"Camera '{camera_name}' required by feature '{feature_key}' not found.")
            observation[feature_key] = img

        state_key = self._layout.state_key
        if state_key is None:
            return observation, None

        if self._observation_type == "mask":
            if self._layout.state_dim is None:
                raise ValueError("Dataset metadata missing state_dim required for mask observation.")
            observation[state_key] = np.zeros(self._layout.state_dim, dtype=np.float32)
            return observation, None

        state = robot_motion.get_state()
        state_vector = self._construct_state_vector(state).astype(np.float32, copy=False)

        if self._layout.state_dim is not None and state_vector.size != self._layout.state_dim:
            raise ValueError(
                f"Observation state dimension mismatch: expected {self._layout.state_dim}, got {state_vector.size}"
            )

        observation[state_key] = state_vector
        return observation, state

    def _construct_state_vector(self, state: dict[str, Any]) -> np.ndarray:
        obs_type = self._observation_type
        vector_parts: list[np.ndarray] = []

        if obs_type == "mask":
            if self._layout.state_dim is None:
                raise ValueError("Dataset metadata missing state_dim required for mask observation.")
            return np.zeros(self._layout.state_dim, dtype=np.float64)

        joints_raw = state.get("q")
        gripper_raw = state.get("gripper_pos")
        pose_raw = state.get("pose")

        if obs_type in {"joint_position", "joint_position_delta"}:
            if joints_raw is None:
                raise ValueError("Robot state missing 'q' for joint-based observation.")
            joints = np.asarray(joints_raw, dtype=np.float64)
            if obs_type.endswith("delta"):
                previous = self._history.joints
                if previous is None or previous.shape != joints.shape:
                    joint_component = np.zeros_like(joints)
                else:
                    joint_component = joints - previous
            else:
                joint_component = joints
            self._history.joints = joints

            vector_parts.append(joint_component)

            if gripper_raw is not None:
                grip_arr = np.array([float(gripper_raw)], dtype=np.float64)
                vector_parts.append(grip_arr)
                self._history.tool = grip_arr

        elif obs_type in {"end_effector_pose", "end_effector_pose_delta"}:
            if pose_raw is None:
                raise ValueError("Robot state missing 'pose' for end-effector observation.")
            pose_arr = np.asarray(pose_raw, dtype=np.float64)
            position, orientation, quat = _pose_to_representation(pose_arr, self._observation_orientation)

            if obs_type.endswith("delta"):
                prev_pos = self._history.ee_position
                prev_quat = self._history.ee_quaternion
                delta_rot = None
                if (
                    prev_pos is not None
                    and prev_quat is not None
                    and prev_pos.shape == position.shape
                    and prev_quat.shape == quat.shape
                ):
                    prev_rot = _orientation_to_rotation(prev_quat, "quaternion")
                    cur_rot = _orientation_to_rotation(quat, "quaternion")
                    delta_pos = prev_rot.inv().apply(position - prev_pos)
                    delta_rot = prev_rot.inv() * cur_rot
                else:
                    delta_pos = np.zeros_like(position)

                orient_type = self._observation_orientation.lower()
                if delta_rot is None:
                    orient_dim = _get_orientation_dim(self._observation_orientation)
                    if orient_type in {"quaternion", "quat"}:
                        delta_ori = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                    else:
                        delta_ori = np.zeros(orient_dim, dtype=np.float64)
                elif orient_type == "euler":
                    delta_ori = delta_rot.as_euler("xyz", degrees=False)
                elif orient_type in {"quaternion", "quat"}:
                    delta_ori = delta_rot.as_quat()
                elif orient_type in {"rotvec", "rotation_vector", "axis_angle"}:
                    delta_ori = delta_rot.as_rotvec()
                else:
                    raise NotImplementedError(
                        f"Unsupported observation orientation '{self._observation_orientation}'."
                    )
                vector_parts.append(np.concatenate([delta_pos, delta_ori], dtype=np.float64))
            else:
                vector_parts.append(np.concatenate([position, orientation], dtype=np.float64))

            self._history.ee_position = position
            self._history.ee_quaternion = quat
            self._history.ee_orientation = orientation
            if gripper_raw is not None:
                grip_arr = np.array([float(gripper_raw)], dtype=np.float64)
                vector_parts.append(grip_arr)
                self._history.tool = grip_arr

        else:
            raise ValueError(f"Unsupported observation type '{self._observation_type}'.")

        if not vector_parts:
            raise ValueError(f"Failed to construct observation state for type '{self._observation_type}'.")

        return np.concatenate(vector_parts, dtype=np.float64)


class ActionApplier:
    def __init__(self, action_type: ActionType, rotation_repr: str, gripper_max: float | None) -> None:
        self._action_type = action_type
        self._rotation_repr = rotation_repr
        self._gripper_max = gripper_max

    def apply(
        self,
        robot_motion: RobotMotion,
        action: torch.Tensor | np.ndarray,
        tracker: CommandTracker | None = None,
    ) -> dict[str, Any]:
        action_np = self._to_numpy(action)
        info: dict[str, Any] = {"action_type": self._action_type.name}

        if self._action_type == ActionType.JOINT_POSITION:
            joint_target, remainder = self._split_joint_action(robot_motion, action_np, tracker)
            robot_motion.send_joint_command(joint_target)
            if tracker is not None:
                tracker.update_joint(joint_target, is_command=False)
            info["joint_target"] = joint_target
            grip = self._send_gripper(robot_motion, remainder, tracker, is_command=False)
            if grip is not None:
                info["gripper"] = grip
            return info

        if self._action_type == ActionType.JOINT_POSITION_DELTA:
            base = self._resolve_joint_reference(robot_motion, tracker, prefer_command=False)
            target = base + action_np[: base.size]
            robot_motion.send_joint_command(target)
            if tracker is not None:
                tracker.update_joint(target, is_command=False)
            info["joint_target"] = target
            grip = self._send_gripper(robot_motion, action_np[base.size :], tracker, is_command=False)
            if grip is not None:
                info["gripper"] = grip
            return info

        if self._action_type == ActionType.END_EFFECTOR_POSE:
            pose_values, remainder = _split_pose_and_gripper(action_np, self._rotation_repr)
            target_pose = self._build_absolute_pose(pose_values)
            robot_motion.send_pose_command(target_pose)
            if tracker is not None:
                tracker.update_pose(target_pose, is_command=False)
            info["pose_target"] = target_pose
            grip = self._send_gripper(robot_motion, remainder, tracker, is_command=False)
            if grip is not None:
                info["gripper"] = grip
            return info

        if self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
            pose_values, remainder = _split_pose_and_gripper(action_np, self._rotation_repr)
            base_pose = self._resolve_pose_reference(robot_motion, tracker, prefer_command=False)
            target_pose = self._build_delta_pose(base_pose, pose_values)
            robot_motion.send_pose_command(target_pose)
            if tracker is not None:
                tracker.update_pose(target_pose, is_command=False)
            info["pose_target"] = target_pose
            grip = self._send_gripper(robot_motion, remainder, tracker, is_command=False)
            if grip is not None:
                info["gripper"] = grip
            return info

        if self._action_type == ActionType.COMMAND_JOINT_POSITION:
            joint_target, remainder = self._split_joint_action(robot_motion, action_np, tracker)
            robot_motion.send_joint_command(joint_target)
            if tracker is not None:
                tracker.update_joint(joint_target, is_command=True)
                tracker.update_joint(joint_target, is_command=False)
            info["command_joint_target"] = joint_target
            grip = self._send_gripper(robot_motion, remainder, tracker, is_command=True)
            if grip is not None:
                info["command_gripper"] = grip
                if tracker is not None:
                    tracker.update_gripper(grip, is_command=False)
            return info

        if self._action_type == ActionType.COMMAND_JOINT_POSITION_DELTA:
            base = self._resolve_joint_reference(robot_motion, tracker, prefer_command=True)
            target = base + action_np[: base.size]
            robot_motion.send_joint_command(target)
            if tracker is not None:
                tracker.update_joint(target, is_command=True)
                tracker.update_joint(target, is_command=False)
            info["command_joint_target"] = target
            grip = self._send_gripper(robot_motion, action_np[base.size :], tracker, is_command=True)
            if grip is not None:
                info["command_gripper"] = grip
                if tracker is not None:
                    tracker.update_gripper(grip, is_command=False)
            return info

        if self._action_type == ActionType.COMMAND_END_EFFECTOR_POSE:
            pose_values, remainder = _split_pose_and_gripper(action_np, self._rotation_repr)
            target_pose = self._build_absolute_pose(pose_values)
            robot_motion.send_pose_command(target_pose)
            if tracker is not None:
                tracker.update_pose(target_pose, is_command=True)
                tracker.update_pose(target_pose, is_command=False)
            info["command_pose_target"] = target_pose
            grip = self._send_gripper(robot_motion, remainder, tracker, is_command=True)
            if grip is not None:
                info["command_gripper"] = grip
                if tracker is not None:
                    tracker.update_gripper(grip, is_command=False)
            return info

        if self._action_type == ActionType.COMMAND_END_EFFECTOR_POSE_DELTA:
            pose_values, remainder = _split_pose_and_gripper(action_np, self._rotation_repr)
            base_pose = self._resolve_pose_reference(robot_motion, tracker, prefer_command=True)
            target_pose = self._build_delta_pose(base_pose, pose_values)
            robot_motion.send_pose_command(target_pose)
            if tracker is not None:
                tracker.update_pose(target_pose, is_command=True)
                tracker.update_pose(target_pose, is_command=False)
            info["command_pose_target"] = target_pose
            grip = self._send_gripper(robot_motion, remainder, tracker, is_command=True)
            if grip is not None:
                info["command_gripper"] = grip
                if tracker is not None:
                    tracker.update_gripper(grip, is_command=False)
            return info

        raise NotImplementedError(f"Action type '{self._action_type}' is not supported during inference.")

    def _to_numpy(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float64)
        action = action.astype(np.float64, copy=False)
        if action.ndim > 1:
            action = action.reshape(-1)
        return action

    def _resolve_joint_reference(
        self,
        robot_motion: RobotMotion,
        tracker: CommandTracker | None,
        *,
        prefer_command: bool,
    ) -> np.ndarray:
        if tracker is not None:
            candidate = tracker.command_joint_target if prefer_command else tracker.joint_target
            if candidate is None and prefer_command:
                candidate = tracker.joint_target
            if candidate is not None:
                return candidate.copy()
        state = robot_motion.get_state()
        joints = state.get("q")
        if joints is None:
            raise ValueError("Robot state missing 'q' while resolving joint reference.")
        return np.asarray(joints, dtype=np.float64)

    def _resolve_pose_reference(
        self,
        robot_motion: RobotMotion,
        tracker: CommandTracker | None,
        *,
        prefer_command: bool,
    ) -> np.ndarray:
        if tracker is not None:
            candidate = tracker.command_pose_target if prefer_command else tracker.pose_target
            if candidate is None and prefer_command:
                candidate = tracker.pose_target
            if candidate is not None:
                return candidate.copy()
        state = robot_motion.get_state()
        pose = state.get("pose")
        if pose is None:
            raise ValueError("Robot state missing 'pose' while resolving pose reference.")
        return np.asarray(pose, dtype=np.float64)

    def _split_joint_action(
        self,
        robot_motion: RobotMotion,
        action: np.ndarray,
        tracker: CommandTracker | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        reference = self._resolve_joint_reference(robot_motion, tracker, prefer_command=False)
        joint_dim = reference.size
        if action.size < joint_dim:
            raise ValueError(f"Received action of size {action.size} for {joint_dim}-D joint command.")
        return action[:joint_dim], action[joint_dim:]

    def _send_gripper(
        self,
        robot_motion: RobotMotion,
        gripper_slice: np.ndarray,
        tracker: CommandTracker | None,
        *,
        is_command: bool,
    ) -> float | None:
        if gripper_slice.size == 0:
            return None
        raw = float(gripper_slice[0])
        if self._gripper_max and self._gripper_max > 0:
            raw = raw / self._gripper_max
        command = float(np.clip(raw, 0.0, 1.0))
        keys = getattr(robot_motion, "_ee_index", None)
        if isinstance(keys, list) and keys:
            command_dict = {key: command for key in keys}
        else:
            command_dict = {"single": command}
        robot_motion.send_gripper_command(command_dict)
        if tracker is not None:
            tracker.update_gripper(command, is_command=is_command)
        return command

    def _build_absolute_pose(self, values: np.ndarray) -> np.ndarray:
        position = values[:3]
        orientation_values = values[3:]
        if self._rotation_repr.lower() in {"quaternion", "quat"}:
            quat = np.asarray(orientation_values, dtype=np.float64)
            norm = np.linalg.norm(quat)
            if norm > 1e-6:
                quat = quat / norm
            else:
                quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            quat = _orientation_to_rotation(np.asarray(orientation_values, dtype=np.float64), self._rotation_repr).as_quat()
        return np.concatenate([position, quat], dtype=np.float64)

    def _build_delta_pose(self, base_pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
        delta_pos = delta[:3]
        delta_rot_values = delta[3:]
        delta_rotation = _orientation_to_rotation(
            np.asarray(delta_rot_values, dtype=np.float64), self._rotation_repr
        ).as_quat()
        delta_pose = np.concatenate([delta_pos, delta_rotation], dtype=np.float64)
        return transform_pose(np.asarray(base_pose, dtype=np.float64), delta_pose)


class LeRobotInterface:
    def __init__(
        self,
        robot_motion_config: str,
        dataset_layout: DatasetLayout,
        action_type: ActionType,
        action_orientation: str,
        observation_type: str,
        observation_orientation: str,
        gripper_max: float | None,
        enable_hardware: bool,
    ) -> None:
        self.layout = dataset_layout
        self.robot_motion = RobotMotion(robot_motion_config)
        self._auto_enable_hardware = enable_hardware
        self._hardware_enabled = False
        if enable_hardware:
            self.enable_hardware()

        self._command_tracker = CommandTracker()
        self._observation_builder = ObservationBuilder(
            dataset_layout,
            observation_type=observation_type,
            observation_orientation=observation_orientation,
        )
        self._action_applier = ActionApplier(
            action_type=action_type,
            rotation_repr=action_orientation,
            gripper_max=gripper_max,
        )
        self._target_dt = 1.0 / dataset_layout.fps if dataset_layout.fps else None
        self._last_action: dict[str, Any] | None = None

    @property
    def target_dt(self) -> float | None:
        return self._target_dt

    @property
    def command_tracker(self) -> CommandTracker:
        return self._command_tracker

    def enable_hardware(self) -> None:
        if not self._hardware_enabled:
            self.robot_motion.enable_hardware()
            self._hardware_enabled = True

    def disable_hardware(self) -> None:
        if self._hardware_enabled:
            self.robot_motion.disable_hardware()
            self._hardware_enabled = False

    @staticmethod
    def _copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Create a shallow copy of the observation with duplicated numpy arrays.

        Copying ensures that downstream asynchronous processing can safely mutate the
        observation without interacting with buffers owned by the control loop.
        """
        copied: dict[str, np.ndarray] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    def reset(
        self, *, reset_robot: bool = True, copy_observation: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._auto_enable_hardware and not self._hardware_enabled:
            self.enable_hardware()

        if reset_robot:
            try:
                self.robot_motion.reset_to_home(space=self.robot_motion._reset_space)
            except Exception as exc:
                logging.warning("Failed to reset robot: %s", exc)

        self._command_tracker.reset()
        try:
            self._command_tracker.sync_with_robot(self.robot_motion)
        except Exception as exc:
            logging.warning("Unable to synchronize command tracker from robot state: %s", exc)

        self._observation_builder.reset()
        observation, state = self._observation_builder.build(self.robot_motion)
        if copy_observation:
            observation = self._copy_observation(observation)
        info = self._compose_info(state, action_info=None)
        return observation, info

    def step(
        self, action: torch.Tensor | np.ndarray, *, copy_observation: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        action_info = self._action_applier.apply(self.robot_motion, action, self._command_tracker)
        self._last_action = action_info

        observation, state = self._observation_builder.build(self.robot_motion)
        if copy_observation:
            observation = self._copy_observation(observation)
        info = self._compose_info(state, action_info=action_info)
        return observation, info

    def close(self) -> None:
        if self._hardware_enabled:
            try:
                self.robot_motion.disable_hardware()
            except Exception as exc:
                logging.warning("Failed to disable hardware cleanly: %s", exc)
            finally:
                self._hardware_enabled = False
        self.robot_motion.close()

    def _compose_info(self, state: dict[str, Any] | None, action_info: dict[str, Any] | None) -> dict[str, Any]:
        info: dict[str, Any] = {"command_tracker": self._command_tracker.snapshot()}
        if state is not None:
            info["robot_state"] = state
        if action_info is not None:
            info["last_action"] = action_info
        return info


__all__ = [
    "ActionApplier",
    "CommandTracker",
    "DatasetLayout",
    "LeRobotInterface",
    "ObservationBuilder",
    "ObservationHistory",
]
