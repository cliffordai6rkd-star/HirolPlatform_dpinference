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

This script instantiates the RobotMotion high-level interface and uses the same
policy/pre-post-processing pipeline as `lerobot_record.py`, but without creating
or updating a dataset. The policy drives the robot directly in closed loop.

Example:
    python lerobot_inference.py \\
        --robot_motion_config factory/tasks/config/robot_motion_fr3_cfg.yaml \\
        --dataset.repo_id fr3_pick_and_place_3dmouse \\
        --dataset.root /home/hanyu/code/HIROLRobotPlatform/dataset/assets/fr3_pick_and_place_3dmouse \\
        --policy.path outputs/train/fr3_act_local/checkpoints/080000/pretrained_model
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from factory.tasks.robot_motion import RobotMotion
from learning_factory.datasets.lerobot_dataset import LeRobotDatasetMetadata

from learning_factory.configs import parser
from learning_factory.configs.policies import PreTrainedConfig
from learning_factory.datasets.utils import load_info
from learning_factory.policies.factory import make_policy, make_pre_post_processors
from learning_factory.policies.pretrained import PreTrainedPolicy
from learning_factory.processor import PolicyProcessorPipeline
from learning_factory.processor import make_default_processors
from learning_factory.utils.control_utils import predict_action
from learning_factory.utils.import_utils import register_third_party_devices
from learning_factory.utils.utils import get_safe_torch_device, init_logging


@dataclass
class InferenceDatasetConfig:
    """Minimal dataset config to fetch metadata/statistics."""

    repo_id: str
    root: str


@dataclass
class RobotMotionInferenceConfig:
    """High-level configuration for online inference."""

    robot_motion_config: str
    dataset: InferenceDatasetConfig
    policy: PreTrainedConfig
    device: str | None = None
    control_dt: float = 0.05
    enable_hardware: bool = True
    task: str | None = None

    def __post_init__(self):
        # Allow CLI override for pretrained policy path, same as lerobot_record.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy.pretrained_path is None:
            raise ValueError("policy.pretrained_path must be provided for inference.")

        # Ensure device field is set for downstream helpers.
        if self.device is None:
            self.device = self.policy.device or "cuda"
        self.policy.device = self.device


def _build_observation(
    robot_motion: RobotMotion,
    image_feature_keys: list[str],
    state_key: str | None,
) -> dict[str, np.ndarray]:
    """
    Collect the latest observation from RobotMotion and format it using dataset feature names.
    """
    observation: dict[str, np.ndarray] = {}

    # Camera images ---------------------------------------------------------------------
    camera_infos = robot_motion._robot_system.get_cameras_infos()  # noqa: SLF001
    camera_map: dict[str, np.ndarray] = {}
    if camera_infos:
        for cam in camera_infos:
            # Some drivers prepend full feature path, some keep the short name.
            camera_map[cam["name"]] = cam["img"]

    for feature_key in image_feature_keys:
        # Feature format: observation.images.<camera_name>
        camera_name = feature_key.split(".")[-1]
        img = camera_map.get(camera_name) or camera_map.get(feature_key)
        if img is None:
            raise RuntimeError(
                f"Camera '{camera_name}' required by feature '{feature_key}' not found in RobotMotion stream."
            )
        observation[feature_key] = img

    # Robot state ----------------------------------------------------------------------
    if state_key is not None:
        state = robot_motion.get_state()
        joints = np.asarray(state["q"], dtype=np.float32)
        gripper = np.array([state.get("gripper_pos", 0.0)], dtype=np.float32)
        observation[state_key] = np.concatenate([joints, gripper], dtype=np.float32)

    return observation


def _apply_action(robot_motion: RobotMotion, action: torch.Tensor) -> None:
    """
    Dispatch the policy action to RobotMotion (joint + gripper).
    """
    action_np = action.detach().cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np[0]

    robot_state = robot_motion.get_state()
    joint_dim = len(robot_state["q"])

    joint_command = action_np[:joint_dim]
    robot_motion.send_joint_command(joint_command)

    if action_np.size > joint_dim:
        gripper_cmd = float(np.clip(action_np[joint_dim], 0.0, 1.0))
        robot_motion.send_gripper_command({"single": gripper_cmd})


def _build_feature_lists(meta: LeRobotDatasetMetadata) -> tuple[list[str], str | None]:
    """Extract observation feature keys for images and state."""
    image_keys = [key for key, ft in meta.features.items() if key.startswith("observation.images.")]
    state_key = "observation.state" if "observation.state" in meta.features else None
    return image_keys, state_key


@parser.wrap()
def run_inference(cfg: RobotMotionInferenceConfig) -> None:
    """
    Main entry point: load policy + processors, instantiate RobotMotion, run closed-loop control.
    """
    init_logging()
    register_third_party_devices()

    device = get_safe_torch_device(cfg.device, log=True)

    # Load dataset metadata & statistics ------------------------------------------------
    dataset_root = Path(cfg.dataset.root)
    dataset_meta = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=dataset_root)
    dataset_stats = dataset_meta.stats or load_info(dataset_root).get("stats")

    image_feature_keys, state_key = _build_feature_lists(dataset_meta)

    # Instantiate policy + processors ---------------------------------------------------
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
        },
    )
    preprocessor.eval()
    postprocessor.eval()

    _, _, robot_observation_processor = make_default_processors()

    # Instantiate RobotMotion -----------------------------------------------------------
    robot_motion = RobotMotion(cfg.robot_motion_config)
    if cfg.enable_hardware:
        robot_motion.enable_hardware()

    logging.info("Starting inference loop")
    try:
        while True:
            try:
                raw_observation = _build_observation(robot_motion, image_feature_keys, state_key)
            except RuntimeError as err:
                logging.warning("Skipping step: %s", err)
                time.sleep(cfg.control_dt)
                continue

            robot_obs = robot_observation_processor(raw_observation)
            action = predict_action(
                observation=robot_obs,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=cfg.policy.use_amp,
                task=cfg.task,
                robot_type=None,
            )

            _apply_action(robot_motion, action)
            time.sleep(cfg.control_dt)
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user.")
    finally:
        robot_motion.disable_hardware()
        robot_motion.close()


if __name__ == "__main__":
    run_inference()
