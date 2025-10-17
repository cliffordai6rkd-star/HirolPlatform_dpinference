#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer

from learning_factory.configs import parser
from learning_factory.configs.train import TrainPipelineConfig
from learning_factory.datasets.factory import make_dataset
from learning_factory.datasets.sampler import EpisodeAwareSampler
from learning_factory.datasets.utils import cycle
from learning_factory.envs.factory import make_env
from learning_factory.envs.utils import close_envs
from learning_factory.optim.factory import make_optimizer_and_scheduler
from learning_factory.policies.factory import make_policy, make_pre_post_processors
from learning_factory.policies.pretrained import PreTrainedPolicy
from learning_factory.rl.wandb_utils import WandBLogger
from learning_factory.scripts.lerobot_eval import eval_policy_all
from learning_factory.utils.logging_utils import AverageMeter, MetricsTracker
from learning_factory.utils.random_utils import set_seed
from learning_factory.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from learning_factory.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def _reduce_output_dict(accelerator: Accelerator, output_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    """Synchronise optional output dictionary across processes.

    Numeric tensors are averaged across all ranks, other entries are left untouched.
    """
    if not output_dict:
        return None

    reduced: dict[str, Any] = {}
    for key, value in output_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.dim() == 0:
                tensor = tensor.reshape(1)
            gathered = accelerator.gather_for_metrics(tensor)
            reduced[key] = gathered.mean().item()
        else:
            reduced[key] = value

    return reduced if accelerator.is_main_process else None


def update_policy(
    accelerator: Accelerator,
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Mixed-precision and distributed coordination are handled through Accelerate.

    Args:
        accelerator: Accelerate `Accelerator` coordinating distributed training.
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        lr_scheduler: An optional learning rate scheduler.
        use_amp: A boolean indicating whether to use automatic mixed precision through Accelerate.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    unwrapped_policy = accelerator.unwrap_model(policy)
    grad_norm_value = 0.0

    with accelerator.accumulate(policy):
        context = accelerator.autocast() if use_amp else nullcontext()
        with context:
            loss, output_dict = policy.forward(batch)

        accelerator.backward(loss)

        if accelerator.sync_gradients:
            grad_norm_value = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

            if lock is not None:
                with lock:
                    optimizer.step()
            else:
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if has_method(unwrapped_policy, "update"):
                unwrapped_policy.update()

        optimizer.zero_grad()

    loss_tensor = loss.detach()
    if loss_tensor.dim() == 0:
        loss_tensor = loss_tensor.reshape(1)

    grad_norm_tensor = torch.tensor([grad_norm_value], device=accelerator.device, dtype=torch.float32)
    update_duration = time.perf_counter() - start_time
    update_duration_tensor = torch.tensor([update_duration], device=accelerator.device, dtype=torch.float32)

    loss_value = accelerator.gather_for_metrics(loss_tensor).mean().item()
    grad_norm_value = accelerator.gather_for_metrics(grad_norm_tensor).mean().item()
    update_duration_value = accelerator.gather_for_metrics(update_duration_tensor).mean().item()

    train_metrics.loss = loss_value
    train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = update_duration_value

    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
    """
    cfg.validate()

    if getattr(cfg.policy, "dtype", None) == "bfloat16":
        mixed_precision = "bf16"
    elif getattr(cfg.policy, "use_amp", False):
        mixed_precision = "fp16"
    else:
        mixed_precision = "no"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[ddp_kwargs])

    if not accelerator.is_local_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if accelerator.is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    accelerator.wait_for_everyone()

    if cfg.seed is not None:
        # Different seeds per rank to avoid identical shuffles when using dataloader workers.
        set_seed(cfg.seed + accelerator.process_index)

    device = accelerator.device
    if device.type == "cuda":
        target_index = device.index if device.index is not None else accelerator.local_process_index
        torch.cuda.set_device(target_index)
        device = torch.device("cuda", target_index)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and accelerator.is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    cfg.policy.device = device.type
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    processor_kwargs: dict[str, Any] = {}
    postprocessor_kwargs: dict[str, Any] = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )

    if lr_scheduler is not None:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
    else:
        policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    dl_iter = cycle(dataloader)
    policy.train()

    global_batch_size = cfg.batch_size * accelerator.num_processes

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        global_batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        dataload_duration = time.perf_counter() - start_time
        dataload_tensor = torch.tensor([dataload_duration], device=accelerator.device, dtype=torch.float32)
        dataload_duration = accelerator.gather_for_metrics(dataload_tensor).mean().item()
        train_tracker.dataloading_s = dataload_duration

        train_tracker, output_dict = update_policy(
            accelerator,
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            lr_scheduler=lr_scheduler,
            use_amp=mixed_precision != "no",
        )

        reduced_output = _reduce_output_dict(accelerator, output_dict)

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            if accelerator.is_main_process:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if reduced_output:
                        wandb_log_dict.update(reduced_output)
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    cfg,
                    accelerator.unwrap_model(policy),
                    optimizer,
                    lr_scheduler,
                    preprocessor,
                    postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and eval_env is not None:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]

                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    global_batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    if eval_info["overall"].get("video_paths"):
                        wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")
            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and eval_env:
        close_envs(eval_env)
    logging.info("End of training")

    if cfg.policy.push_to_hub and accelerator.is_main_process:
        base_policy = accelerator.unwrap_model(policy)
        base_policy.push_model_to_hub(cfg)
        preprocessor.push_to_hub(cfg.policy.repo_id)
        postprocessor.push_to_hub(cfg.policy.repo_id)
    accelerator.wait_for_everyone()


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
# python learning_factory/scripts/lerobot_train.py --config_path=learning_factory/configs/train_dp_local.json
# accelerate launch --num_processes <GPU数> python3 learning_factory/scripts/lerobot_train.py --config_path=...
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2   learning_factory/scripts/lerobot_train.py   --config_path=learning_factory/configs/train_pi05_local.json
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1   learning_factory/scripts/lerobot_train.py   --config_path=learning_factory/configs/train_pi05_local.json
# 恢复训练
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 learning_factory/scripts/lerobot_train.py --config_path=/home/haotian/code/HIROLRobotPlatform/outputs/train/fr3_pi05_local_ee/checkpoints/last/pretrained_model/train_config.json --resume true
