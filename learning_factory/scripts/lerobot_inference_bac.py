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

"""
Run online inference on a real robot controlled through RobotMotion.

Usage:
    python HIROLRobotPlatform/learning_factory/scripts/lerobot_inference.py \
  --config HIROLRobotPlatform/learning_factory/configs/robot_motion_inference_act.yaml
  
  python learning_factory/scripts/lerobot_inference.py --config learning_factory/configs/robot_motion_inference_act.yaml


Example YAML:
    robot_motion_config: factory/tasks/config/robot_motion_fr3_cfg.yaml
    dataset:
      repo_id: fr3_pick_and_place_3dmouse
      root: /home/hanyu/code/HIROLRobotPlatform/dataset/assets/fr3_pick_and_place_3dmouse
    policy:
      path: outputs/train/fr3_act_local/checkpoints/080000/pretrained_model
    device: cuda
    enable_hardware: true
    task: collect rice and place in bowl
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from factory.tasks.robot_motion import RobotMotion

try:
    from dataset.utils import ActionType, Action_Type_Mapping_Dict
except ImportError:  # Backward compatibility with older naming
    from dataset.utils import ActionType, dict_str2action_type as Action_Type_Mapping_Dict

from learning_factory.configs.policies import PreTrainedConfig
from learning_factory.policies.factory import make_policy, make_pre_post_processors
from learning_factory.processor import make_default_processors
from learning_factory.utils.control_utils import init_keyboard_listener, predict_action
from learning_factory.utils.import_utils import register_third_party_devices
from learning_factory.utils.utils import get_safe_torch_device, init_logging
from learning_factory.datasets.utils import load_info, load_stats
from hardware.base.utils import transform_pose
import random


ObservationStateMode = Literal["live", "zeros", "omit"]

@dataclass
class RobotMotionInferenceConfig:
    robot_motion_config: str
    dataset_repo_id: str
    dataset_root: str
    policy_path: str
    device: str = "cuda"
    enable_hardware: bool = True
    task: str | None = None
    gripper_max: float | None = None
    action_type: ActionType = ActionType.JOINT_POSITION
    action_orientation: str = "euler"
    observation_type: str = "joint_position"
    observation_orientation: str = "euler"
    observation_state_mode: ObservationStateMode = "live"
    contain_ee_obs: bool = False
    max_episodes: int | None = None
    episode_timeout_s: float | None = 90


class InferenceDatasetMeta:
    """Minimal dataset metadata shim for policy factory."""

    def __init__(self, repo_id: str, root: Path):
        self.repo_id = repo_id
        self.root = root
        self.info = load_info(root)
        self.features = self.info["features"]
        self.stats = load_stats(root)


@dataclass
class CommandState:
    """Track the last commanded targets so delta policies integrate correctly."""

    joint_target: np.ndarray | None = None
    pose_target: np.ndarray | None = None
    gripper: float | None = None

    def initialize_from_robot(self, robot_motion: RobotMotion) -> None:
        """Latch the current robot state as the starting command reference."""
        state = robot_motion.get_state()
        joints = state.get("q")
        self.joint_target = np.array(joints, dtype=np.float64) if joints is not None else None
        pose = state.get("pose")
        self.pose_target = np.array(pose, dtype=np.float64) if pose is not None else None
        gripper_pos = state.get("gripper_pos")
        self.gripper = float(gripper_pos) if gripper_pos is not None else None

    def clear(self) -> None:
        """Drop cached targets (e.g. between episodes if reset fails)."""
        self.joint_target = None
        self.pose_target = None
        self.gripper = None


@dataclass
class ObservationHistory:
    """Cache previous robot observations so delta features can be reconstructed."""

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


def _normalize_bool_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        return None
    return bool(value)


def _resolve_observation_state_mode(
    cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> ObservationStateMode:
    raw_mode = (
        cfg.get("observation_state_mode")
        or cfg.get("state_mode")
        or dataset_cfg.get("observation_state_mode")
        or dataset_cfg.get("state_mode")
    )
    omit_flag = _normalize_bool_flag(cfg.get("omit_observation_state"))
    if omit_flag is None:
        omit_flag = _normalize_bool_flag(dataset_cfg.get("omit_observation_state"))
    zero_flag = _normalize_bool_flag(cfg.get("zero_observation_state"))
    if zero_flag is None:
        zero_flag = _normalize_bool_flag(dataset_cfg.get("zero_observation_state"))

    if omit_flag and zero_flag:
        raise ValueError("`omit_observation_state` and `zero_observation_state` cannot both be true.")
    if omit_flag:
        return "omit"
    if zero_flag:
        return "zeros"

    if raw_mode is None:
        return "live"

    text = str(raw_mode).strip().lower()
    mapping: dict[str, ObservationStateMode] = {
        "live": "live",
        "default": "live",
        "robot": "live",
        "zeros": "zeros",
        "zero": "zeros",
        "0": "zeros",
        "omit": "omit",
        "none": "omit",
        "empty": "omit",
        "skip": "omit",
    }
    if text not in mapping:
        raise ValueError(f"Unsupported observation_state_mode '{raw_mode}'.")
    return mapping[text]


def _build_observation(
    robot_motion: RobotMotion,
    image_feature_keys: list[str],
    state_key: str | None,
    state_mode: ObservationStateMode,
    observation_type: str,
    observation_orientation: str,
    observation_history: ObservationHistory,
    expected_state_dim: int | None,
    include_joint_absolute_ee: bool,
) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}

    # Camera images ---------------------------------------------------------------
    camera_infos = robot_motion._robot_system.get_cameras_infos()  
    camera_map: dict[str, np.ndarray] = {}
    if camera_infos:
        for cam in camera_infos:
            camera_map[cam["name"]] = cam["img"]

    for feature_key in image_feature_keys:
        camera_name = feature_key.split(".")[-1]
        img = camera_map.get(camera_name)
        if img is None:
            img = camera_map.get(feature_key)
        if img is None:
            raise RuntimeError(f"Camera '{camera_name}' required by feature '{feature_key}' not found.")
        observation[feature_key] = img


    # Robot state ----------------------------------------------------------------
    if state_key is not None:
        if state_mode == "zeros":
            if expected_state_dim is None:
                raise ValueError("Expected state dimension is required when observation_state_mode='zeros'.")
            observation[state_key] = np.zeros(expected_state_dim, dtype=np.float32)
            return observation
        if state_mode != "live":
            raise ValueError(f"Unsupported observation_state_mode '{state_mode}' encountered in inference.")
        state = robot_motion.get_state()
        state_vector = _construct_state_vector(
            state=state,
            observation_type=observation_type,
            observation_orientation=observation_orientation,
            history=observation_history,
            include_joint_absolute_ee=include_joint_absolute_ee,
        ).astype(np.float32, copy=False)

        if expected_state_dim is not None and state_vector.size != expected_state_dim:
            raise ValueError(
                f"Observation state dimension mismatch: expected {expected_state_dim}, got {state_vector.size}"
            )

        observation[state_key] = state_vector

    return observation


def _get_orientation_dim(rotation_repr: str) -> int:
    rotation_repr = rotation_repr.lower()
    if rotation_repr in {"euler"}:
        return 3
    if rotation_repr in {"quaternion", "quat"}:
        return 4
    if rotation_repr in {"rotvec", "rotation_vector", "axis_angle"}:
        return 3
    raise NotImplementedError(f"Orientation representation '{rotation_repr}' is not supported.")


def _orientation_to_rotation(values: np.ndarray, rotation_repr: str) -> R:
    rotation_repr = rotation_repr.lower()
    if rotation_repr == "euler":
        return R.from_euler("xyz", values, degrees=False)
    if rotation_repr in {"quaternion", "quat"}:
        return R.from_quat(values)
    if rotation_repr in {"rotvec", "rotation_vector", "axis_angle"}:
        return R.from_rotvec(values)
    raise NotImplementedError(f"Orientation representation '{rotation_repr}' is not supported.")


def _split_pose_and_gripper(action_vec: np.ndarray, rotation_repr: str) -> tuple[np.ndarray, np.ndarray]:
    orient_dim = _get_orientation_dim(rotation_repr)
    pose_dim = 3 + orient_dim
    if action_vec.size < pose_dim:
        raise ValueError(
            f"Expected at least {pose_dim} values for pose with '{rotation_repr}' orientation, got {action_vec.size}"
        )
    pose_values = action_vec[:pose_dim]
    remainder = action_vec[pose_dim:]
    return pose_values, remainder


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Return quaternion conjugate for [x, y, z, w] ordering."""
    q = np.asarray(quat, dtype=np.float64)
    if q.size != 4:
        raise ValueError(f"Quaternion expected to have 4 components, got {q.size}")
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication for [x, y, z, w] ordering."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float64)


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (np.asarray(values, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def _pose_to_representation(pose: np.ndarray, rotation_repr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split pose into position, chosen orientation representation, and quaternion."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.size < 7:
        raise ValueError(f"Pose must contain position + quaternion (expected >=7 values, got {pose.size})")
    position = pose[:3]
    quat = pose[3:7]
    repr_type = rotation_repr.lower()
    if repr_type == "euler":
        orientation = R.from_quat(quat).as_euler("xyz", degrees=False)
    elif repr_type in {"quaternion", "quat"}:
        orientation = quat
    elif repr_type in {"rotvec", "rotation_vector", "axis_angle"}:
        orientation = R.from_quat(quat).as_rotvec()
    else:
        raise NotImplementedError(f"Unsupported observation orientation representation '{rotation_repr}'.")
    return position, orientation, quat


def _construct_state_vector(
    state: dict[str, Any],
    observation_type: str,
    observation_orientation: str,
    history: ObservationHistory,
    include_joint_absolute_ee: bool,
) -> np.ndarray:
    """Build the robot state vector matching the dataset formatting."""
    obs_type = observation_type.lower()
    vector_parts: list[np.ndarray] = []

    joints_raw = state.get("q")
    gripper_raw = state.get("gripper_pos")
    pose_raw = state.get("pose")

    if obs_type in {"joint_position", "joint_position_delta"}:
        if joints_raw is None:
            raise ValueError("Robot state missing 'q' for joint-based observation.")
        joints = np.asarray(joints_raw, dtype=np.float64)
        if obs_type.endswith("delta"):
            prev_joints = history.joints
            if prev_joints is None or prev_joints.shape != joints.shape:
                joint_component = np.zeros_like(joints)
            else:
                joint_component = joints - prev_joints
        else:
            joint_component = joints
        history.joints = joints

        if obs_type == "joint_position" and include_joint_absolute_ee:
            if pose_raw is None:
                raise ValueError("Robot state missing 'pose' while `contain_ee_obs` is enabled.")
            pose_arr = np.asarray(pose_raw, dtype=np.float64)
            if pose_arr.size < 7:
                raise ValueError(f"Expected pose with 7 values (xyz + quat), got {pose_arr.size}")
            # Insert EE pose before joint positions to match dataset loader ordering.
            vector_parts.append(pose_arr[:7])
            history.ee_position = pose_arr[:3]
            history.ee_quaternion = pose_arr[3:7]
            history.ee_orientation = pose_arr[3:7]

        vector_parts.append(joint_component)

        if gripper_raw is not None:
            grip_arr = np.array([float(gripper_raw)], dtype=np.float64)
            tool_component = grip_arr
            history.tool = grip_arr
            vector_parts.append(tool_component)

    elif obs_type in {"end_effector_pose", "end_effector_pose_delta"}:
        if pose_raw is None:
            raise ValueError("Robot state missing 'pose' for end-effector observation.")
        pose_arr = np.asarray(pose_raw, dtype=np.float64)
        position, orientation, quat = _pose_to_representation(pose_arr, observation_orientation)

        if obs_type.endswith("delta"):
            prev_pos = history.ee_position
            prev_quat = history.ee_quaternion
            if prev_pos is None or prev_pos.shape != position.shape or prev_quat is None or prev_quat.shape != quat.shape:
                delta_pos = np.zeros_like(position)
                delta_rot = R.identity()
            else:
                prev_rot = R.from_quat(prev_quat)
                delta_pos = prev_rot.inv().apply(position - prev_pos)
                cur_rot = R.from_quat(quat)
                delta_rot = prev_rot.inv() * cur_rot

            orient_type = observation_orientation.lower()
            if orient_type == "euler":
                ori_component = delta_rot.as_euler("xyz", degrees=False)
            elif orient_type in {"quaternion", "quat"}:
                ori_component = delta_rot.as_quat()
            elif orient_type in {"rotvec", "rotation_vector", "axis_angle"}:
                ori_component = delta_rot.as_rotvec()
            else:
                raise NotImplementedError(f"Unsupported observation orientation '{observation_orientation}'.")

            vector_parts.append(np.concatenate([delta_pos, ori_component], dtype=np.float64))
        else:
            vector_parts.append(np.concatenate([position, orientation], dtype=np.float64))

        history.ee_position = position
        history.ee_quaternion = quat
        history.ee_orientation = orientation

    else:
        raise ValueError(f"Unsupported observation type '{observation_type}'.")

    if not vector_parts:
        raise ValueError(f"Failed to construct observation state for type '{observation_type}'.")

    return np.concatenate(vector_parts, dtype=np.float64)


def _build_absolute_pose(pose_values: np.ndarray, rotation_repr: str) -> np.ndarray:
    position = pose_values[:3]
    orientation_values = pose_values[3:]
    if rotation_repr.lower() in {"quaternion", "quat"}:
        quat = np.array(orientation_values, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            quat = quat / norm
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat = _orientation_to_rotation(np.asarray(orientation_values, dtype=np.float64), rotation_repr).as_quat()
    return np.concatenate([position, quat], dtype=np.float64)


def _build_delta_pose(base_pose: np.ndarray, pose_delta: np.ndarray, rotation_repr: str) -> np.ndarray:
    current_pose = np.asarray(base_pose, dtype=np.float64)
    delta_pos = pose_delta[:3]
    delta_rot_values = pose_delta[3:]
    delta_rotation = _orientation_to_rotation(np.asarray(delta_rot_values, dtype=np.float64), rotation_repr).as_quat()
    delta_pose = np.concatenate([delta_pos, delta_rotation], dtype=np.float64)

    target_pos = transform_pose(current_pose, delta_pose)
    # target_pos[3:] = np.array([1,0,0,0], dtype=np.float64)  # Keep fixed orientation for inference
    return target_pos


def _send_gripper(
    robot_motion: RobotMotion,
    gripper_slice: np.ndarray,
    gripper_max: float | None,
    command_state: CommandState | None,
) -> None:
    if gripper_slice.size == 0:
        return
    raw = float(gripper_slice[0])
    if gripper_max and gripper_max > 0:
        raw = raw / gripper_max
    gripper_cmd = float(np.clip(raw, 0.0, 1.0))
    robot_motion.send_gripper_command({"single": gripper_cmd})
    if command_state is not None:
        command_state.gripper = gripper_cmd


def _apply_action(
    robot_motion: RobotMotion,
    action: torch.Tensor,
    action_type: ActionType,
    rotation_repr: str = "euler",
    gripper_max: float | None = None,
    command_state: CommandState | None = None,
) -> None:
    action_np = action.detach().cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np[0]

    if action_type == ActionType.JOINT_POSITION:
        if command_state is not None and command_state.joint_target is not None:
            joint_dim = command_state.joint_target.size
        else:
            joint_dim = len(robot_motion.get_state()["q"])
        joint_command = action_np[:joint_dim]
        robot_motion.send_joint_command(joint_command)
        if command_state is not None:
            command_state.joint_target = np.array(joint_command, dtype=np.float64)
        _send_gripper(robot_motion, action_np[joint_dim:], gripper_max, command_state)
    elif action_type == ActionType.JOINT_POSITION_DELTA:
        if command_state is None or command_state.joint_target is None:
            current_q = np.asarray(robot_motion.get_state()["q"], dtype=np.float64)
            if command_state is not None:
                command_state.joint_target = current_q.copy()
        base_joint = (
            command_state.joint_target
            if command_state is not None and command_state.joint_target is not None
            else np.asarray(robot_motion.get_state()["q"], dtype=np.float64)
        )
        joint_dim = base_joint.size
        delta = action_np[:joint_dim]
        target = base_joint + delta
        robot_motion.send_joint_command(target)
        if command_state is not None:
            command_state.joint_target = target
        _send_gripper(robot_motion, action_np[joint_dim:], gripper_max, command_state)
    elif action_type == ActionType.END_EFFECTOR_POSE:
        pose_values, gripper_slice = _split_pose_and_gripper(action_np, rotation_repr)
        target_pose = _build_absolute_pose(pose_values, rotation_repr)
        robot_motion.send_pose_command(target_pose)
        if command_state is not None:
            command_state.pose_target = target_pose
        _send_gripper(robot_motion, gripper_slice, gripper_max, command_state)
    elif action_type == ActionType.END_EFFECTOR_POSE_DELTA:
        pose_values, gripper_slice = _split_pose_and_gripper(action_np, rotation_repr)
        if command_state is None or command_state.pose_target is None:
            pose_target = np.asarray(robot_motion.get_state()["pose"], dtype=np.float64)
            if command_state is not None:
                command_state.pose_target = pose_target.copy()
        base_pose = (
            command_state.pose_target
            if command_state is not None and command_state.pose_target is not None
            else np.asarray(robot_motion.get_state()["pose"], dtype=np.float64)
        )
        target_pose = _build_delta_pose(base_pose, pose_values, rotation_repr)
        robot_motion.send_pose_command(target_pose)
        if command_state is not None:
            command_state.pose_target = target_pose
        _send_gripper(robot_motion, gripper_slice, gripper_max, command_state)
    else:
        raise NotImplementedError(f"Action type '{action_type}' is not supported in inference.")


def _extract_feature_keys(features: dict[str, dict]) -> tuple[list[str], str | None]:
    image_keys = [key for key in features if key.startswith("observation.images.")]
    state_key = "observation.state" if "observation.state" in features else None
    return image_keys, state_key


def run_inference(cfg: RobotMotionInferenceConfig) -> None:
    seed = 42  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    
    init_logging()
    register_third_party_devices()

    device = get_safe_torch_device(cfg.device, log=True)

    dataset_root = Path(cfg.dataset_root)
    dataset_meta = InferenceDatasetMeta(cfg.dataset_repo_id, dataset_root)
    image_feature_keys, state_key = _extract_feature_keys(dataset_meta.features)
    expected_state_dim: int | None = None
    state_mode = cfg.observation_state_mode
    if state_mode == "omit":
        state_key = None
    else:
        if state_key is not None:
            state_feature = dataset_meta.features.get(state_key, {})
            state_shape = state_feature.get("shape")
            if isinstance(state_shape, (list, tuple)) and len(state_shape) > 0:
                expected_state_dim = int(state_shape[0])
            elif isinstance(state_shape, int):
                expected_state_dim = int(state_shape)
            if expected_state_dim is None:
                dims = state_feature.get("dims")
                if isinstance(dims, int):
                    expected_state_dim = int(dims)
                elif isinstance(dims, (list, tuple)) and len(dims) > 0:
                    expected_state_dim = int(dims[0])
        if state_mode == "zeros" and state_key is None:
            raise ValueError(
                "Dataset features do not include 'observation.state' but observation_state_mode='zeros' was requested."
            )

    include_joint_ee = bool(cfg.contain_ee_obs and cfg.observation_type.lower() == "joint_position")

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = cfg.policy_path
    policy_cfg.device = cfg.device

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset_meta)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    dataset_stats = dataset_meta.stats or dataset_meta.info.get("stats")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.policy_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    if hasattr(preprocessor, "eval"):
        preprocessor.eval()
    if hasattr(postprocessor, "eval"):
        postprocessor.eval()
    _, _, robot_observation_processor = make_default_processors()

    robot_motion = RobotMotion(cfg.robot_motion_config)
    if cfg.enable_hardware:
        robot_motion.enable_hardware()
        try:
            robot_motion.reset_to_home(space=robot_motion._reset_space)
        except Exception as exc:
            logging.warning("Failed to reset robot before inference: %s", exc)

    command_state = CommandState()
    observation_history = ObservationHistory()
    observation_history.reset()
    try:
        command_state.initialize_from_robot(robot_motion)
    except Exception as exc:
        logging.warning("Unable to initialize command references from robot state: %s", exc)
        command_state.clear()
        observation_history.reset()

    logging.info(
        "Using action type '%s' with orientation '%s'",
        cfg.action_type.name.lower(),
        cfg.action_orientation,
    )
    logging.info(
        "Observation type '%s' with orientation '%s' (contain_ee_obs=%s)",
        cfg.observation_type,
        cfg.observation_orientation,
        include_joint_ee,
    )
    logging.info("Observation state mode set to '%s'.", state_mode)

    target_dt = 1.0 / dataset_meta.info.get("fps", 15)

    listener, events = init_keyboard_listener()
    manual_control = listener is not None
    episode_timeout = cfg.episode_timeout_s
    if episode_timeout is not None and episode_timeout <= 0:
        episode_timeout = None
    episode_limit = cfg.max_episodes if cfg.max_episodes is not None else float("inf")
    episode_results: list[str] = []
    episode_idx = 0

    logging.info("Starting inference loop")
    if episode_timeout is not None:
        logging.info("Episode timeout set to %.1f seconds.", episode_timeout)
    else:
        logging.info("Episode timeout disabled.")
    try:
        while episode_idx < episode_limit and not events["stop_recording"]:
            logging.info("Starting episode %d", episode_idx + 1)
            events["episode_success"] = False
            events["episode_failure"] = False
            events["proceed_next_episode"] = False
            events["exit_early"] = False
            events["rerecord_episode"] = False
            observation_history.reset()

            episode_start = time.perf_counter()
            timed_out = False
            while not events["stop_recording"]:
                loop_start = time.perf_counter()
                try:
                    raw_obs = _build_observation(
                        robot_motion=robot_motion,
                        image_feature_keys=image_feature_keys,
                        state_key=state_key,
                        state_mode=state_mode,
                        observation_type=cfg.observation_type,
                        observation_orientation=cfg.observation_orientation,
                        observation_history=observation_history,
                        expected_state_dim=expected_state_dim,
                        include_joint_absolute_ee=include_joint_ee,
                    )
                except (RuntimeError, ValueError) as exc:
                    logging.warning("Skipping step: %s", exc)
                    time.sleep(target_dt)
                    continue

                robot_obs = robot_observation_processor(raw_obs)
                action = predict_action(
                    observation=robot_obs,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy_cfg.use_amp,
                    task=cfg.task,
                    robot_type=None,
                )
                
                print("Action:", action)
                
                _apply_action(
                    robot_motion,
                    action,
                    cfg.action_type,
                    cfg.action_orientation,
                    cfg.gripper_max,
                    command_state,
                )
                elapsed = time.perf_counter() - loop_start
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if events["episode_success"] or events["episode_failure"]:
                    break
                if events["exit_early"] and not (events["episode_success"] or events["episode_failure"]):
                    logging.info("Episode %d interrupted by user.", episode_idx + 1)
                    break
                if episode_timeout is not None and (time.perf_counter() - episode_start) >= episode_timeout:
                    timed_out = True
                    logging.info("Episode %d reached timeout (%.1f s).", episode_idx + 1, episode_timeout)
                    break

            if events["stop_recording"]:
                logging.info("Stop requested by user. Exiting inference loop.")
                break

            outcome = None
            if events["episode_success"]:
                outcome = "success"
            elif events["episode_failure"]:
                outcome = "failure"
            elif timed_out:
                outcome = "timeout"

            if outcome is None and manual_control:
                logging.info("Mark episode %d outcome: press 's' for success or 'f' for failure.", episode_idx + 1)
                while not events["stop_recording"]:
                    if events["episode_success"]:
                        outcome = "success"
                        break
                    if events["episode_failure"]:
                        outcome = "failure"
                        break
                    time.sleep(0.05)
            elif outcome is None:
                outcome = "failure"

            if events["stop_recording"]:
                logging.info("Stop requested while awaiting outcome. Exiting inference loop.")
                break

            episode_results.append(outcome)
            logging.info("Episode %d result: %s", episode_idx + 1, outcome)

            events["episode_success"] = False
            events["episode_failure"] = False
            events["exit_early"] = False
            events["rerecord_episode"] = False

            try:
                robot_motion.reset_to_home(space=robot_motion._reset_space)
            except Exception as exc:
                logging.warning("Failed to reset robot: %s", exc)
                command_state.clear()
                observation_history.reset()
            else:
                try:
                    command_state.initialize_from_robot(robot_motion)
                except Exception as exc:
                    logging.warning("Failed to refresh command references after reset: %s", exc)
                    command_state.clear()
                    observation_history.reset()
                else:
                    observation_history.reset()

            if manual_control:
                logging.info("Press Enter to start the next episode or ESC to exit.")
                events["proceed_next_episode"] = False
                while not events["stop_recording"]:
                    if events["proceed_next_episode"]:
                        events["proceed_next_episode"] = False
                        break
                    time.sleep(0.05)
                if events["stop_recording"]:
                    logging.info("Stop requested before the next episode. Exiting inference loop.")
                    break
            else:
                time.sleep(0.1)

            if hasattr(policy, "reset"):
                policy.reset()

            episode_idx += 1
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user.")
    finally:
        if listener is not None:
            listener.stop()
        robot_motion.disable_hardware()
        robot_motion.close()

    if episode_results:
        success_count = sum(res == "success" for res in episode_results)
        failure_count = sum(res == "failure" for res in episode_results)
        timeout_count = sum(res == "timeout" for res in episode_results)
        logging.info(
            "Episode summary: %d total | %d success | %d failure | %d timeout",
            len(episode_results),
            success_count,
            failure_count,
            timeout_count,
        )


def _load_config(path: Path) -> RobotMotionInferenceConfig:
    with path.open() as f:
        raw_cfg = yaml.safe_load(f)

    dataset_cfg = raw_cfg.get("dataset", {})
    policy_cfg = raw_cfg.get("policy", {})
    gripper_cfg = raw_cfg.get("gripper", {})

    required = {
        "robot_motion_config": raw_cfg.get("robot_motion_config"),
        "dataset_repo_id": dataset_cfg.get("repo_id"),
        "dataset_root": dataset_cfg.get("root"),
        "policy_path": policy_cfg.get("path"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required configuration entries: {missing}")

    max_episodes = raw_cfg.get("max_episodes")
    if max_episodes is not None:
        max_episodes = int(max_episodes)
        if max_episodes <= 0:
            raise ValueError("`max_episodes` must be a positive integer.")

    episode_timeout = raw_cfg.get("episode_timeout_s", raw_cfg.get("episode_timeout"))
    if episode_timeout is None:
        episode_timeout = 30.0
    else:
        episode_timeout = float(episode_timeout)
        if episode_timeout <= 0:
            episode_timeout = None

    raw_action_type = raw_cfg.get("action_type", dataset_cfg.get("action_type"))
    if raw_action_type is None:
        raw_action_type = ActionType.JOINT_POSITION
    if isinstance(raw_action_type, ActionType):
        action_type = raw_action_type
    else:
        str_action_type = str(raw_action_type).lower()
        mapping = {k.lower(): v for k, v in Action_Type_Mapping_Dict.items()}
        lookup_key = str_action_type
        if lookup_key not in mapping and "." in lookup_key:
            lookup_key = lookup_key.split(".")[-1]
        if lookup_key not in mapping:
            raise ValueError(f"Unsupported action type '{raw_action_type}'.")
        action_type = mapping[lookup_key]

    raw_orientation = (
        raw_cfg.get("action_orientation_type")
        or raw_cfg.get("action_orientation")
        or raw_cfg.get("action_ori_type")
        or dataset_cfg.get("action_orientation_type")
        or dataset_cfg.get("action_ori_type")
    )
    action_orientation = str(raw_orientation).lower() if raw_orientation is not None else "euler"

    obs_type = raw_cfg.get("observation_type") or dataset_cfg.get("observation_type")
    observation_type = str(obs_type).lower() if obs_type is not None else "joint_position"

    raw_obs_orientation = (
        raw_cfg.get("observation_orientation_type")
        or raw_cfg.get("observation_orientation")
        or raw_cfg.get("observation_ori_type")
        or dataset_cfg.get("observation_orientation_type")
        or dataset_cfg.get("observation_orientation")
        or dataset_cfg.get("observation_ori_type")
    )
    observation_orientation = (
        str(raw_obs_orientation).lower()
        if raw_obs_orientation is not None
        else action_orientation
    )
    state_mode = _resolve_observation_state_mode(raw_cfg, dataset_cfg)

    contain_ee_value = raw_cfg.get("contain_ee_obs", dataset_cfg.get("contain_ee_obs", False))
    if isinstance(contain_ee_value, str):
        contain_ee_obs = contain_ee_value.strip().lower() in {"1", "true", "yes", "y"}
    else:
        contain_ee_obs = bool(contain_ee_value)

    return RobotMotionInferenceConfig(
        robot_motion_config=str(required["robot_motion_config"]),
        dataset_repo_id=str(required["dataset_repo_id"]),
        dataset_root=str(required["dataset_root"]),
        policy_path=str(required["policy_path"]),
        device=str(raw_cfg.get("device", "cuda")),
        enable_hardware=bool(raw_cfg.get("enable_hardware", True)),
        task=raw_cfg.get("task"),
        gripper_max=gripper_cfg.get("max_position"),
        action_type=action_type,
        action_orientation=action_orientation,
        observation_type=observation_type,
        observation_orientation=observation_orientation,
        observation_state_mode=state_mode,
        contain_ee_obs=contain_ee_obs,
        max_episodes=max_episodes,
        episode_timeout_s=episode_timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run policy inference using RobotMotion.")
    parser.add_argument("--config", required=True, help="Path to inference YAML configuration.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    run_inference(cfg)


if __name__ == "__main__":
    main()
