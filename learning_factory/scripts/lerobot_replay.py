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

"""Replay a stored LeRobot dataset episode on RobotMotion."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dataset.utils import ActionType, Action_Type_Mapping_Dict
from learning_factory.datasets.lerobot_dataset import LeRobotDataset
from learning_factory.interfaces.lerobot_interface import DatasetLayout, LeRobotInterface
from learning_factory.utils.import_utils import register_third_party_devices
from learning_factory.utils.robot_utils import busy_wait
from learning_factory.utils.utils import init_logging


@dataclass
class DatasetSection:
    repo_id: str
    root: str
    episode: int = 0


@dataclass
class ActionSection:
    type: ActionType = ActionType.COMMAND_END_EFFECTOR_POSE_DELTA
    orientation: str = "quaternion"


@dataclass
class ObservationSection:
    type: str = "end_effector_pose"
    orientation: str = "quaternion"


@dataclass
class ReplayConfig:
    robot_motion_config: str
    dataset: DatasetSection
    action: ActionSection = ActionSection()
    observation: ObservationSection = ObservationSection()
    gripper_max: float | None = None
    enable_hardware: bool = True
    reset_robot: bool = True
    strict_fps: bool = True


def _normalize_action_type(raw: Any) -> ActionType:
    if isinstance(raw, ActionType):
        return raw
    text = str(raw).lower()
    lookup = text.split(".")[-1]
    mapping = {k.lower(): v for k, v in Action_Type_Mapping_Dict.items()}
    if lookup not in mapping:
        raise ValueError(f"Unsupported action type '{raw}'.")
    return mapping[lookup]


def _parse_config(path: Path) -> ReplayConfig:
    with path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    dataset_cfg = raw_cfg.get("dataset", {})
    action_cfg = raw_cfg.get("action", {})
    observation_cfg = raw_cfg.get("observation", {})
    gripper_cfg = raw_cfg.get("gripper", {})

    for key in ("robot_motion_config",):
        if key not in raw_cfg or not raw_cfg[key]:
            raise ValueError(f"Missing required config key '{key}'.")
    for key in ("repo_id", "root"):
        if key not in dataset_cfg or not dataset_cfg[key]:
            raise ValueError(f"Missing required dataset config key '{key}'.")

    dataset = DatasetSection(
        repo_id=str(dataset_cfg["repo_id"]),
        root=str(dataset_cfg["root"]),
        episode=int(dataset_cfg.get("episode", 0)),
    )
    action = ActionSection(
        type=_normalize_action_type(action_cfg.get("type", ActionSection.type)),
        orientation=str(action_cfg.get("orientation", ActionSection.orientation)).lower(),
    )
    observation = ObservationSection(
        type=str(observation_cfg.get("type", ObservationSection.type)).lower(),
        orientation=str(observation_cfg.get("orientation", ObservationSection.orientation)).lower(),
    )

    return ReplayConfig(
        robot_motion_config=str(raw_cfg["robot_motion_config"]),
        dataset=dataset,
        action=action,
        observation=observation,
        gripper_max=gripper_cfg.get("max_position"),
        enable_hardware=bool(raw_cfg.get("enable_hardware", True)),
        reset_robot=bool(raw_cfg.get("reset_robot", True)),
        strict_fps=bool(raw_cfg.get("strict_fps", True)),
    )


def replay_episode(cfg: ReplayConfig) -> None:
    init_logging()
    logging.info("Replay configuration: %s", cfg)
    register_third_party_devices()

    layout = DatasetLayout.from_dataset(cfg.dataset.root, cfg.dataset.repo_id)
    interface = LeRobotInterface(
        robot_motion_config=cfg.robot_motion_config,
        dataset_layout=layout,
        action_type=cfg.action.type,
        action_orientation=cfg.action.orientation,
        observation_type=cfg.observation.type,
        observation_orientation=cfg.observation.orientation,
        gripper_max=cfg.gripper_max,
        enable_hardware=cfg.enable_hardware,
    )
    layout.image_keys = []
    layout.image_keys.clear()

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.episode],
        tolerance_s=1e-4,
    )
    frames = dataset.hf_dataset.filter(lambda item: item["episode_index"] == cfg.dataset.episode)
    if len(frames) == 0:
        raise ValueError(f"No frames found for episode {cfg.dataset.episode}.")

    observation, info = interface.reset(reset_robot=cfg.reset_robot)
    target_dt = interface.target_dt
    if target_dt is None:
        fps = layout.fps or dataset.fps or 15.0
        target_dt = 1.0 / float(fps)
    logging.info("Frames: %d | target dt: %.4fs | strict FPS: %s", len(frames), target_dt, cfg.strict_fps)

    try:
        for idx, frame in enumerate(frames):
            loop_start = time.perf_counter()
            action = np.asarray(frame["action"], dtype=np.float64)
            observation, info = interface.step(action)

            if cfg.strict_fps:
                elapsed = time.perf_counter() - loop_start
                remaining = target_dt - elapsed
                if remaining > 0:
                    busy_wait(remaining)

            if idx % 50 == 0:
                logging.info("Replayed %d / %d frames", idx + 1, len(frames))
    except KeyboardInterrupt:
        logging.info("Replay interrupted by user.")
    finally:
        interface.close()
        logging.info("Replay finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a LeRobot dataset episode on RobotMotion.")
    parser.add_argument("--config", type=Path, required=True, help="Replay configuration YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _parse_config(args.config)
    replay_episode(cfg)


if __name__ == "__main__":
    main()
