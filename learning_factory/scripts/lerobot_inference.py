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
"""Run online inference on a real robot controlled through RobotMotion."""

from __future__ import annotations

import argparse
import logging
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import threading
import torch
import yaml

from dataset.utils import ActionType, Action_Type_Mapping_Dict

from learning_factory.async_inference.helpers import FPSTracker
from learning_factory.configs.policies import PreTrainedConfig
from learning_factory.interfaces.lerobot_interface import DatasetLayout, LeRobotInterface
from learning_factory.policies.factory import make_policy, make_pre_post_processors
from learning_factory.processor import make_default_processors
from learning_factory.utils.control_utils import init_keyboard_listener, predict_action
from learning_factory.utils.import_utils import register_third_party_devices
from learning_factory.utils.utils import get_safe_torch_device, init_logging


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
    max_episodes: int | None = None
    episode_timeout_s: float | None = 90.0
    async_enabled: bool = False
    async_workers: int = 1
    async_log_stats: bool = False


def _load_config(path: Path) -> RobotMotionInferenceConfig:
    with path.open() as f:
        raw_cfg = yaml.safe_load(f)

    dataset_cfg = raw_cfg.get("dataset", {})
    policy_cfg = raw_cfg.get("policy", {})
    gripper_cfg = raw_cfg.get("gripper", {})
    async_cfg = raw_cfg.get("async_inference", {})

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
        lookup_key = str_action_type.split(".")[-1] if "." in str_action_type else str_action_type
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
        str(raw_obs_orientation).lower() if raw_obs_orientation is not None else action_orientation
    )

    async_enabled = bool(async_cfg.get("enabled", False))
    async_workers = int(async_cfg.get("workers", 1))
    if async_workers <= 0:
        raise ValueError("`async_inference.workers` must be a positive integer.")
    async_log_stats = bool(async_cfg.get("log_stats", False))

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
        max_episodes=max_episodes,
        episode_timeout_s=episode_timeout,
        async_enabled=async_enabled,
        async_workers=async_workers,
        async_log_stats=async_log_stats,
    )


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

    dataset_layout = DatasetLayout.from_dataset(cfg.dataset_root, cfg.dataset_repo_id)
    dataset_meta = dataset_layout.as_policy_metadata()

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = cfg.policy_path
    policy_cfg.device = cfg.device

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset_meta)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    dataset_stats = dataset_layout.stats or dataset_layout.info.get("stats")
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

    interface = LeRobotInterface(
        robot_motion_config=cfg.robot_motion_config,
        dataset_layout=dataset_layout,
        action_type=cfg.action_type,
        action_orientation=cfg.action_orientation,
        observation_type=cfg.observation_type,
        observation_orientation=cfg.observation_orientation,
        gripper_max=cfg.gripper_max,
        enable_hardware=cfg.enable_hardware,
    )

    target_dt = interface.target_dt
    if target_dt is None:
        fps = dataset_layout.fps or dataset_layout.info.get("fps") or 15
        target_dt = 1.0 / float(fps)
    async_executor: ThreadPoolExecutor | None = None
    async_policy_lock = threading.Lock()
    async_fps_tracker: FPSTracker | None = None

    if cfg.async_enabled:
        async_executor = ThreadPoolExecutor(
            max_workers=cfg.async_workers, thread_name_prefix="lerobot-async"
        )
        if cfg.async_log_stats:
            target_fps = (1.0 / target_dt) if target_dt and target_dt > 0 else 0.0
            async_fps_tracker = FPSTracker(target_fps=target_fps)

    def _predict_async_action(observation: dict[str, np.ndarray]) -> torch.Tensor:
        robot_obs = robot_observation_processor(observation)
        with async_policy_lock:
            return predict_action(
                observation=robot_obs,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy_cfg.use_amp,
                task=cfg.task,
                robot_type=None,
            )

    def _submit_async_action(observation: dict[str, np.ndarray]) -> Future[torch.Tensor]:
        if async_executor is None:
            raise RuntimeError("Async executor is not initialized.")
        return async_executor.submit(_predict_async_action, observation)

    listener, events = init_keyboard_listener()
    manual_control = listener is not None

    episode_timeout = cfg.episode_timeout_s
    if episode_timeout is not None and episode_timeout <= 0:
        episode_timeout = None
    episode_limit = cfg.max_episodes if cfg.max_episodes is not None else float("inf")
    episode_results: list[str] = []
    episode_idx = 0
    require_hardware_reset = True

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

            try:
                observation, obs_info = interface.reset(
                    reset_robot=require_hardware_reset, copy_observation=cfg.async_enabled
                )
            except (RuntimeError, ValueError) as exc:
                logging.error("Failed to reset interface before episode %d: %s", episode_idx + 1, exc)
                break
            require_hardware_reset = False
            action_future: Future[torch.Tensor] | None = None
            if cfg.async_enabled:
                action_future = _submit_async_action(observation)

            episode_start = time.perf_counter()
            timed_out = False

            while not events["stop_recording"]:
                loop_start = time.perf_counter()
                if cfg.async_enabled:
                    if action_future is None:
                        action_future = _submit_async_action(observation)
                    try:
                        action = action_future.result()
                    except Exception as exc:
                        logging.warning("Skipping action due to processing error: %s", exc)
                        action_future = _submit_async_action(observation)
                        if target_dt and target_dt > 0:
                            time.sleep(target_dt)
                        continue
                else:
                    try:
                        robot_obs = robot_observation_processor(observation)
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
                    except Exception as exc:
                        logging.warning("Skipping action due to processing error: %s", exc)
                        if target_dt and target_dt > 0:
                            time.sleep(target_dt)
                        continue

                try:
                    observation, obs_info = interface.step(action, copy_observation=cfg.async_enabled)
                except (RuntimeError, ValueError) as exc:
                    logging.warning("Failed to apply action: %s", exc)
                    if cfg.async_enabled:
                        action_future = _submit_async_action(observation)
                    if target_dt and target_dt > 0:
                        time.sleep(target_dt)
                    continue
                if cfg.async_enabled:
                    action_future = _submit_async_action(observation)
                    if cfg.async_log_stats and async_fps_tracker is not None:
                        metrics = async_fps_tracker.calculate_fps_metrics(time.perf_counter())
                        logging.debug(
                            "Async inference FPS: %.2f (target %.2f)",
                            metrics["avg_fps"],
                            metrics["target_fps"],
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

            if cfg.async_enabled and action_future is not None:
                if not action_future.done():
                    action_future.cancel()
                action_future = None

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

            if manual_control:
                try:
                    interface.reset(reset_robot=True)
                    logging.info("Robot reset completed. Press Enter when ready to begin the next episode.")
                    require_hardware_reset = False
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Failed to reset robot after episode %d: %s", episode_idx + 1, exc)
                    require_hardware_reset = True

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
                require_hardware_reset = True

            if hasattr(policy, "reset"):
                policy.reset()

            episode_idx += 1
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user.")
    finally:
        if listener is not None:
            listener.stop()
        if async_executor is not None:
            async_executor.shutdown(wait=True)
        interface.close()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LeRobot policy inference on RobotMotion.")
    parser.add_argument("--config", type=str, required=True, help="Path to inference YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(Path(args.config))
    run_inference(cfg)


if __name__ == "__main__":
    main()
