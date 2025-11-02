#!/usr/bin/env python3
"""
LeRobot Inference Task

Lightweight wrapper that plugs a LeRobot pretrained policy (e.g., ACT) into the
HIROL Gym interface. We rely on the common InferenceBase for action execution
and keyboard controls, and only implement the policy-specific glue:
- load LeRobot policy + processors from a checkpoint directory
- convert Gym observations to LeRobot observation dict
- run policy.predict_action_chunk and return a numpy chunk for aggregation
"""

from __future__ import annotations

from factory.tasks.inferences_tasks.inference_base import InferenceBase

from typing import Dict, Any, Tuple, Optional
import os
import sys
from pathlib import Path
import glog as log
import numpy as np
import torch
import time
import math
import threading

# Configuration constants - use environment variables for flexibility
BASE_DIR = Path(os.environ.get('LEROBOT_BASE_DIR', '/workspace/dependencies/lerobot'))
OUTPUT_DIR = Path(os.environ.get('LEROBOT_OUTPUT_DIR', str(BASE_DIR / 'outputs' / 'train')))

# Ensure vendored LeRobot package is on PYTHONPATH
_this_file = Path(__file__).resolve()
for _parent in _this_file.parents:
    _lerobot_src = _parent / "dependencies" / "lerobot" / "src"
    if _lerobot_src.is_dir():
        _lerobot_src_str = str(_lerobot_src)
        if _lerobot_src_str not in sys.path:
            sys.path.insert(0, _lerobot_src_str)
        break

# RTC helpers (Real-Time Chunking)
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker

# LeRobot imports (local vendored in dependencies/lerobot)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, LAST_CHECKPOINT_LINK


def _resolve_pretrained_dir(checkpoint_path: str) -> str:
    """Resolve a usable pretrained_model directory from a user-provided path.

    Accepts any of the following and returns the directory that contains
    the LeRobot policy config.json/model.safetensors:
    - <path>/pretrained_model
    - <path> (if it already contains config.json)
    - <path>/last/pretrained_model (symlink case)
    - <path>/<step>/pretrained_model
    """
    cp = os.path.expanduser(checkpoint_path)
    candidates: list[str] = []
    if os.path.isdir(cp):
        # direct pretrained_model dir
        pm = os.path.join(cp, PRETRAINED_MODEL_DIR)
        candidates.append(pm)
        # directly a pretrained dir (has config.json)
        candidates.append(cp)
        # last symlink
        candidates.append(os.path.join(cp, LAST_CHECKPOINT_LINK, PRETRAINED_MODEL_DIR))
        # scan one-level subdirs for step dirs
        if os.path.isdir(cp) and os.access(cp, os.R_OK):
            for name in os.listdir(cp):
                sub = os.path.join(cp, name, PRETRAINED_MODEL_DIR)
                candidates.append(sub)

    for c in candidates:
        cfg = os.path.join(c, "config.json")
        if os.path.isfile(cfg):
            log.info(f"Resolved pretrained model dir: {c}")
            return c

    # fall back to provided path (will let PreTrainedConfig raise if invalid)
    return cp


def _apply_modes_from_checkpoint_suffix(cfg: Dict[str, Any]) -> None:
    """Parse checkpoint_path suffix like '*_q2q', '*_dq2dq', '*_ee2ee', '*_dee2dee'
    and update cfg['observation_type'], cfg['action_type'], and orientation if needed.
    """
    import re

    ckpt = str(cfg.get("checkpoint_path", ""))
    pair = re.search(
        r"(dee|ee|dq|q|mask|ft)2(cdee|cee|cdq|cq|dee|ee|dq|q)(?![A-Za-z0-9])",
        ckpt.lower(),
    )
    # Default to q2q when suffix is absent
    if not pair:
        obs_key, act_key = "q", "q"
    else:
        obs_key, act_key = pair.group(1), pair.group(2)

    # Map to InferenceBase/GymApi expected strings
    obs_map = {
        "dee": "delta_ee_pose",
        "ee": "ee_pose",
        "dq": "delta_joint_position",
        "q": "joint_position",
        "mask": "mask",
        "ft": "ft",
    }
    act_map = {
        "dee": "end_effector_pose_delta",
        "ee": "end_effector_pose",
        "dq": "joint_position_delta",
        "q": "joint_position",
        "cdee": "command_end_effector_pose_delta",
        "cee": "command_end_effector_pose",
        "cdq": "command_joint_position_delta",
        "cq": "command_joint_position",
    }

    cfg["observation_type"] = obs_map[obs_key]
    cfg["action_type"] = act_map[act_key]
    # Use quaternion orientation for EE actions
    if act_key in ("ee", "dee", "cee", "cdee"):
        cfg["action_orientation_type"] = "quaternion"
    src = "suffix" if pair else "default"
    log.info(
        f"Applied {src} modes: obs='{obs_key}' -> '{cfg['observation_type']}', "
        f"act='{act_key}' -> '{cfg['action_type']}', ori='{cfg.get('action_orientation_type','euler')}'"
    )


class Lerobot_Inferencer(InferenceBase):
    def __init__(self, config: Dict[str, Any]) -> None:
        # Derive action/observation modes from checkpoint suffix before base init
        _apply_modes_from_checkpoint_suffix(config)
        log.info(config)
        # Super sets up GymApi, keyboard listener, aggregation knobs, etc.
        super().__init__(config)

        # Device selection
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self._device}")

        # GymApi is not thread-safe; protect it when using RTC background threads
        self._robot_lock = threading.Lock()

        # Load dataset metadata (used to infer features and provide stats)
        ds_cfg = config.get("dataset", {})
        if not ds_cfg:
            raise ValueError("Config missing 'dataset' section: {repo_id, root}")
        repo_id = ds_cfg.get("repo_id")
        root = ds_cfg.get("root")
        if not repo_id or not root:
            raise ValueError("'dataset.repo_id' and 'dataset.root' must be provided")
        # Validate root path exists
        root_path = Path(root).expanduser()
        if not root_path.exists():
            log.warning(f"Dataset root path {root_path} does not exist, will attempt to create it")
            root_path.mkdir(parents=True, exist_ok=True)
        self._ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        log.info(f"Loaded dataset meta: repo='{repo_id}', fps={self._ds_meta.fps}")

        # Load policy config and instantiate policy + processors
        pretrained_dir = _resolve_pretrained_dir(config.get("checkpoint_path", ""))
        # Load policy config from disk; override device if needed
        policy_cfg: PreTrainedConfig = PreTrainedConfig.from_pretrained(pretrained_dir)
        policy_cfg.pretrained_path = os.path.abspath(pretrained_dir)
        if self._device.type != str(policy_cfg.device):
            log.info(f"Override policy device: {policy_cfg.device} -> {self._device}")
            policy_cfg.device = self._device.type

        # ----- RTC configuration (for pi0/pi05/smolvla etc.) -----
        self._rtc_config: RTCConfig | None = getattr(policy_cfg, "rtc_config", None)
        self._rtc_enabled: bool = False
        self._action_queue: ActionQueue | None = None
        self._latency_tracker: LatencyTracker | None = None
        self._rtc_get_actions_threshold: int = int(
            config.get("rtc_action_queue_size_to_get_new_actions", 30)
        )

        rtc_cfg_from_yaml = config.get("rtc")
        if rtc_cfg_from_yaml is not None:
            # If checkpoint has no rtc_config, create one from YAML
            if self._rtc_config is None:
                self._rtc_config = RTCConfig(**rtc_cfg_from_yaml)
            else:
                # Override existing fields from YAML
                for k, v in rtc_cfg_from_yaml.items():
                    if hasattr(self._rtc_config, k):
                        setattr(self._rtc_config, k, v)
            if hasattr(policy_cfg, "rtc_config"):
                policy_cfg.rtc_config = self._rtc_config

        if self._rtc_config is not None and getattr(self._rtc_config, "enabled", False):
            self._rtc_enabled = True
            log.info(f"RTC enabled with config: {self._rtc_config}")
        else:
            # If RTC is off, we don't need a prefill threshold
            self._rtc_get_actions_threshold = 0

        # Optionally enable ACT 内置时间聚合 (temporal ensembling) from config
        ens_coeff = config.get("temporal_ensemble_coeff", None)
        if ens_coeff is not None and hasattr(policy_cfg, "temporal_ensemble_coeff"):
            if isinstance(ens_coeff, (int, float)):
                policy_cfg.temporal_ensemble_coeff = float(ens_coeff)
                log.info(
                    f"Using temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff} "
                    f"for ACT temporal ensembling."
                )
            else:
                log.warning(
                    f"Invalid temporal_ensemble_coeff={ens_coeff}; "
                    f"keeping model default {getattr(policy_cfg, 'temporal_ensemble_coeff', None)}"
                )

        # Instantiate policy using dataset meta (sets input/output feature shapes)
        self._policy = make_policy(cfg=policy_cfg, ds_meta=self._ds_meta)
        self._policy.eval()

        # If policy supports RTC, (re)initialize its RTC processor
        if self._rtc_enabled and hasattr(self._policy, "init_rtc_processor"):
            self._policy.init_rtc_processor()
            log.info("Initialized RTC processor on policy.")
            # Verify RTC initialization succeeded
            if not hasattr(self._policy, 'rtc_processor') or self._policy.rtc_processor is None:
                log.warning("Failed to init RTC processor, disabling RTC")
                self._rtc_enabled = False
                self._rtc_config = None
                self._action_queue = None

        # Build pre/post processors with dataset stats for proper normalization
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg, dataset_stats=self._ds_meta.stats,
            pretrained_path=pretrained_dir,
            preprocessor_overrides={"device_processor": {"device": self._device.type}},
        )

        # Cache action chunk size to align with aggregator
        if hasattr(self._policy.config, "chunk_size"):
            self._predicted_action_chunks = int(getattr(self._policy.config, "chunk_size", 1))
        if hasattr(self._policy.config, "n_action_steps"):
            self._execution_action_chunk_size = int(
                getattr(self._policy.config, "n_action_steps", self._execution_action_chunk_size)
            )

        log.info(
            f"Policy ready. chunk_size={self._predicted_action_chunks}, n_action_steps={self._execution_action_chunk_size}"
        )

    def _initialize_episode(self, episode_id: int) -> None:
        """Initialize robot and policy state for a new episode."""
        self._gym_robot.reset()
        self._last_gripper_open = [True, True]
        self.policy_reset()
        self._status_ok = True
        log.info(f"Starting episode {episode_id}")

    def _execute_single_step(self, action_vec: np.ndarray, target_dt: float, 
                            start_time: float) -> Tuple[Dict, np.ndarray]:
        """Execute a single action step and handle timing."""
        gym_action = self.convert_to_gym_action_single_step(action_vec, action_vec)
        res = self._gym_robot.step(gym_action)
        gym_obs = res[0]
        obs = self.convert_from_gym_obs(gym_obs)
        
        elapsed = time.perf_counter() - start_time
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)
        
        return gym_obs, obs

    def start_inference(self) -> None:
        """Run inference.

        - RTC 关闭时：保持原有 ACT/SmolVLA 单步推理行为；
        - RTC 打开时：使用 ActionQueue + LatencyTracker，后台线程请求动作块，
          主线程按数据集 FPS 从队列取 action 执行。
        """
        # 数据集 fps，缺省/异常时退回 15Hz
        # Allow config override to slow down execution if hardware is slower
        fps = float(self._config.get("target_fps", getattr(self._ds_meta, "fps", 15.0)))
        log.info(f"start_inference fps: {fps}")
        target_dt = 1.0 / fps

        # -------- 非 RTC：原有单步推理路径 --------
        if not self._rtc_enabled:
            for episode_id in range(self._num_episodes):
                if self._quit:
                    break

                self._initialize_episode(episode_id)
                obs = None

                t = 0
                while self._status_ok and not self._quit and t < self._max_timestamps:
                    loop_start = time.perf_counter()

                    if obs is None:
                        # 首帧直接从 GymApi 取观测
                        obs = self.convert_from_gym_obs(obs)

                    # 单步推理（例如 ACT 使用内部队列）
                    action_vec = self._predict_single_action(obs)

                    # 转成 GymApi action 并执行一步
                    gym_obs, obs = self._execute_single_step(action_vec, target_dt, loop_start)

                    t += 1

                log.info(
                    f"Episode {episode_id} finished "
                    f"(steps={t}, status_ok={self._status_ok}, quit={self._quit})"
                )
            return

        # -------- RTC：后台 chunk 线程 + 主线程执行 --------
        log.info("RTC enabled: using ActionQueue + LatencyTracker for chunked inference.")

        execution_horizon = getattr(self._rtc_config, "execution_horizon", None)
        if execution_horizon is None:
            execution_horizon = 0

        if self._latency_tracker is None:
            self._latency_tracker = LatencyTracker()

        def get_actions_loop():
            """Background thread: request new action chunks when the queue is low."""
            time_per_step = target_dt
            while not self._quit and self._status_ok:
                if self._action_queue.qsize() <= self._rtc_get_actions_threshold:
                    current_time = time.perf_counter()
                    action_index_before_inference = self._action_queue.get_action_index()
                    prev_actions = self._action_queue.get_left_over()

                    inference_latency = self._latency_tracker.p95()
                    inference_delay = 0
                    if inference_latency is not None and inference_latency > 0.0:
                        inference_delay = int(math.ceil(inference_latency / time_per_step))

                    # Get one observation for the next chunk
                    with self._robot_lock:
                        gym_obs = self._gym_robot.get_observation()
                    obs = self.convert_from_gym_obs(gym_obs)

                    original_actions, postprocessed_actions = self._predict_action_chunk_torch(
                        obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                        execution_horizon=execution_horizon,
                    )

                    new_latency = time.perf_counter() - current_time
                    new_delay = int(math.ceil(new_latency / time_per_step))
                    self._latency_tracker.add(new_latency)

                    if self._rtc_get_actions_threshold < execution_horizon + new_delay:
                        log.warning(
                            "[RTC] rtc_action_queue_size_to_get_new_actions too small; "
                            "it should be >= execution_horizon + inference_delay."
                        )

                    self._action_queue.merge(
                        original_actions,
                        postprocessed_actions,
                        real_delay=new_delay,
                        action_index_before_inference=action_index_before_inference,
                    )
                else:
                    time.sleep(0.001)

        for episode_id in range(self._num_episodes):
            if self._quit:
                break

            self._initialize_episode(episode_id)

            # Reset RTC queue and latency tracker per episode
            self._action_queue = ActionQueue(self._rtc_config)
            if self._latency_tracker is None:
                self._latency_tracker = LatencyTracker()
            else:
                self._latency_tracker.reset()

            log.info(f"Starting episode {episode_id} (RTC mode)")

            worker = threading.Thread(
                target=get_actions_loop,
                daemon=True,
                name=f"RTC-GetActions-{episode_id}",
            )
            worker.start()

            t = 0
            obs = None
            while self._status_ok and not self._quit and t < self._max_timestamps:
                loop_start = time.perf_counter()

                action_tensor = self._action_queue.get()
                if action_tensor is None:
                    time.sleep(0.001)
                    continue

                action_vec = action_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

                gym_action = self.convert_to_gym_action_single_step(action_vec, action_vec)
                with self._robot_lock:
                    res = self._gym_robot.step(gym_action)
                gym_obs = res[0]
                obs = self.convert_from_gym_obs(gym_obs)

                elapsed = time.perf_counter() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

                t += 1

            self._status_ok = False
            if worker.is_alive():
                worker.join(timeout=1.0)

            log.info(
                f"[RTC] Episode {episode_id} finished "
                f"(steps={t}, status_ok={self._status_ok}, quit={self._quit})"
            )

    def policy_reset(self):
        # Reset policy internal queues if supported (e.g., ACT)
        if hasattr(self._policy, "reset") and callable(getattr(self._policy, "reset")):
            self._policy.reset()

    def convert_from_gym_obs(self, gym_obs=None) -> dict[str, np.ndarray]:
        """Convert GymApi observation to LeRobot observation format.

        Returns a flat dict with keys expected by LeRobot processors:
        - 'observation.state': 1D float32 array
        - 'observation.images.<cam_name>': HxWxC uint8/float32 images
        """
        gym_obs = super().convert_from_gym_obs(gym_obs)

        # optional display
        self.image_display(gym_obs)

        # Build state vector (follow GymApi get_observation order)
        state_vec = np.array([], dtype=np.float32)
        temp_joint_pos = np.array([], dtype=np.float32)
        for _, cur_state in (gym_obs.get("state") or {}).items():
            cur_state = np.asarray(cur_state, dtype=np.float32).reshape(-1)
            # Track joints for plotting only (the very first joint block per key)
            temp_joint_pos = np.hstack((temp_joint_pos, cur_state))
            state_vec = np.hstack((state_vec, cur_state))
        # log.info(f'state vec: {state_vec}')
        # Debug: z-score 诊断，检查当前推理 state 与数据集 stats 对齐情况
        # try:
        #     ds = self._ds_meta.stats["observation.state"]
        #     import numpy as _np
        #     mu = _np.array(ds["mean"]).astype(_np.float32).reshape(-1)
        #     std = _np.array(ds["std"]).astype(_np.float32).reshape(-1)
        #     x = state_vec.astype(_np.float32).reshape(-1)
        #     if x.shape[0] == mu.shape[0]:
        #         z = _np.abs((x - mu) / (std + 1e-8))
        #         # 打印整体分布和前几维
        #         import glog as _log
        #         _log.info(f"state z-abs: mean={_np.mean(z):.3f}, max={_np.max(z):.3f}, first5={z[:5]}")
        #     else:
        #         _log.warning(f"Dim mismatch: inference state len={x.shape[0]} vs dataset {mu.shape[0]}")
        # except Exception as e:
        #     pass

        # Update joint cache for plotting/action overlays
        with self._lock:
            self._joint_positions = temp_joint_pos

        # Build image dict
        obs: dict[str, np.ndarray] = {"observation.state": state_vec}
        for cam_name, img in (gym_obs.get("colors") or {}).items():
            if img is None:
                continue
            obs[f"observation.images.{cam_name}"] = img

        return obs


    def _predict_action_chunk_torch(
        self,
        obs: dict[str, np.ndarray],
        inference_delay: int | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper: returns (original_actions, postprocessed_actions) as torch tensors with shape (T, A)."""
        obs_t = prepare_observation_for_inference(
            observation=obs,
            device=self._device,
            task=None,
            robot_type=self._config.get("robot_type", None),
        )
        proc_obs = self._preprocessor(obs_t)

        predict_kwargs: Dict[str, Any] = {}
        if self._rtc_enabled and (
            inference_delay is not None
            or prev_chunk_left_over is not None
            or execution_horizon is not None
        ):
            if inference_delay is not None:
                predict_kwargs["inference_delay"] = inference_delay
            if prev_chunk_left_over is not None:
                predict_kwargs["prev_chunk_left_over"] = prev_chunk_left_over
            if execution_horizon is not None:
                predict_kwargs["execution_horizon"] = execution_horizon

        # RTC diffusion models (e.g. SmolVLA) need gradients inside their own
        # predict_action_chunk / rtc_processor, so we must NOT wrap this call
        # in torch.inference_mode(). Keep inference_mode only for non-RTC usage.
        if self._rtc_enabled:
            actions = self._policy.predict_action_chunk(proc_obs, **predict_kwargs)
        else:
            with torch.inference_mode():
                actions = self._policy.predict_action_chunk(proc_obs, **predict_kwargs)

        original_actions = actions.squeeze(0)
        postprocessed = self._postprocessor(actions).squeeze(0)

        return original_actions, postprocessed


    def policy_prediction(
        self,
        obs: dict[str, np.ndarray],
        inference_delay: int | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> np.ndarray:
        """Run LeRobot policy to predict an action chunk.

        The base aggregator expects a numpy array with shape (chunk, action_dim).
        Extra RTC kwargs are optional and ignored when RTC is disabled.
        """
        _, postprocessed = self._predict_action_chunk_torch(
            obs,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )

        chunk_np = postprocessed.detach().cpu().numpy().astype(np.float32, copy=False)
        # 简要打印一次 chunk 统计，便于观察是否为常量序列
        if not getattr(self, "_debug_chunk_printed", False):
            var_per_dim = np.var(chunk_np, axis=0)
            log.info(
                f"chunk stats: shape={chunk_np.shape}, first_row={chunk_np[0]}, "
                f"var_per_dim[0:4]={var_per_dim[:4]}"
            )
            self._debug_chunk_printed = True
        return chunk_np

    def _predict_single_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Single-step action using ACT's select_action (internal action queue)."""
        obs_t = prepare_observation_for_inference(
            observation=obs,
            device=self._device,
            task=None,
            robot_type=self._config.get("robot_type", None),
        )
        proc_obs = self._preprocessor(obs_t)

        with torch.inference_mode():
            action = self._policy.select_action(proc_obs)
            action = self._postprocessor(action)

        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return action_np

    def close(self):
        # Nothing specific to cleanup beyond GymApi
        pass


def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml

    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "factory/tasks/inferences_tasks/lerobot/config/act.yaml",
            "help": "Path to the config file",
        }
    }
    args = parse_args("LeRobot inference", arguments)
    cfg = dynamic_load_yaml(args.config)

    runner = Lerobot_Inferencer(cfg)
    runner.start_inference()


if __name__ == "__main__":
    main()
