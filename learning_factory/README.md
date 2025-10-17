# Learning Factory

`learning_factory` 为 HIROL 机器人提供基于 Hugging Face LeRobot 生态的端到端学习与推理套件，覆盖数据集接入、策略训练、评估、真机推理，以及结合 RobotMotion 的数据采集与标注流程。

## 概览

- 通过统一的 Dataclass + CLI 配置系统快速切换数据、策略与硬件设定。
- 支持 Hugging Face LeRobot 数据格式、流式加载、时序聚合与视频编码。
- 集成 ACT、Diffusion Policy、PI0.5、SmolVLA 等多种策略，并可扩展到自定义模型。
- 真机推理脚本与 RobotMotion 深度集成，提供键盘控制、回合统计与硬件复位。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `learning_factory/scripts/` | 训练 (`lerobot_train.py`)、推理 (`lerobot_inference.py`)、评估、录制入口脚本 |
| `learning_factory/configs/` | Dataclass 定义、CLI 解析以及示例配置 (`train_act_local.json` 等) |
| `learning_factory/datasets/` | 数据集工厂、流式数据集、采样器、统计与视频处理 |
| `learning_factory/policies/` | 策略实现 (ACT/DP/PI 系列/SmolVLA/VQBeT/TDMPC/SAC) 与工厂 |
| `learning_factory/envs/` | 仿真评估环境工厂与工具函数 |
| `learning_factory/utils/` | 控制工具、日志、随机数种子、Hub 集成等通用组件 |
| `learning_factory/async_inference/` | 异步推理与通信支持模块 |
| `learning_factory/robots/`, `teleoperators/`, `motors/` | 真机接口、遥操作与执行机构抽象 |

## 环境配置

### Python 与 CUDA

- 推荐 Python 3.10，并安装与 GPU 驱动匹配的 PyTorch 版本。
- 快速检测 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 可选：Conda 环境

```bash
conda create -n hirol python=3.10
conda activate hirol
```

### 依赖安装

```bash
pip install -r requirements.txt
pip install -e .
```

### 硬件驱动

- Franka、RealSense、SpaceMouse 等硬件需额外安装厂商 SDK。
- 确认 RobotMotion 所需的驱动、规则与权限已设置完成。

## 快速上手

### 离线训练

```bash
python learning_factory/scripts/lerobot_train.py \
  --config_path=learning_factory/configs/train_act_local.json
```

```bash
python learning_factory/scripts/lerobot_train.py \
  --config_path=learning_factory/configs/train_dp_local.json
```

常用参数：
- 使用点号语法覆盖任意字段：`--policy.device=cuda:1 --dataset.streaming=true`。
- 断点续训：`--config_path=<run_dir>/train_config.json --resume=true`。
- 推送模型到 Hugging Face Hub：`--policy.push_to_hub=true --policy.repo_id=<user/model>`。

输出位置：
- 默认写入 `outputs/train/<timestamp>_<job_name>/`。
- 若 `wandb.enable=true`，训练日志同步到指定项目。

### 仿真评估

```bash
python learning_factory/scripts/lerobot_eval.py \
  --policy.path=outputs/train/fr3_act_local/checkpoints/080000/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --device=cuda
```

### 真机推理

```bash
python learning_factory/scripts/lerobot_inference.py \
  --config learning_factory/configs/robot_motion_inference_act.yaml
```

- YAML 中指定 RobotMotion 配置、数据集根目录与策略权重。
- 默认键盘控制：`ESC` 终止、`→` 结束回合、`s`/`f` 标记成功或失败、`Enter` 进入下一回合。

### 数据采集

详情见HIROLPLATFORM的teleop和dataset

## 配置系统与常用字段

- 所有配置均由 dataclass 定义 (`learning_factory/configs/`)，通过 `draccus` 解析。
- CLI 使用点号覆盖任意嵌套字段，例如：
  - `--dataset.image_transforms.enable=true`
  - `--dataset.image_transforms.tfs.brightness.kwargs='{"brightness": [0.6,1.4]}'`
- 训练配置 (`TrainPipelineConfig`) 关键字段：
  - `dataset.*`：数据集路径、采样、流式加载与增强。
  - `policy.*`：策略类型、设备、预训练路径、优化器预设。
  - `optimizer / scheduler`：自定义优化器和调度器（若关闭策略预设）。
  - `steps / log_freq / eval_freq / save_freq`：训练循环控制。
  - `wandb.*`：Weights & Biases 集成开关。
- 推理配置 (YAML) 中需匹配训练时的 `action_type`、传感器布局与统计文件。

## 策略训练指引

以下示例均可复制到自定义 `train_*.json`，只保留关键字段便于快速调整。

### ACT (Action Chunking Transformer)

文件：`learning_factory/configs/train_act_local.json`

```json
{
  "dataset": {
    "repo_id": "fr3_pick_and_place_3dmouse",
    "root": "/path/to/dataset/root",
    "streaming": false,
    "use_imagenet_stats": true,
    "image_transforms": { "enable": false },
    "video_backend": "torchcodec"
  },
  "policy": {
    "type": "act",
    "device": "cuda",
    "chunk_size": 100,
    "n_action_steps": 100,
    "optimizer_lr": 1e-5,
    "optimizer_weight_decay": 1e-4
  },
  "batch_size": 16,
  "steps": 100000,
  "eval_freq": 1000,
  "save_freq": 20000
}
```

建议：
- `n_action_steps`=1 对应逐步控制；若希望推理阶段加速，可在训练中设为 >1 并配合后处理。
- 如需 VAE 平滑，在 `policy.temporal_ensemble_coeff` 设置指数平滑系数，但需 `n_action_steps=1`。
- 大规模数据可启用 `dataset.streaming=true` 并指定 Hugging Face 缓存路径。

### Diffusion Policy (DP)

文件：`learning_factory/configs/train_dp_local.json`

```json
{
  "policy": {
    "type": "diffusion",
    "device": "cuda",
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "vision_backbone": "resnet18",
    "optimizer_lr": 1e-4
  },
  "steps": 200000,
  "batch_size": 16,
  "save_freq": 20000,
  "use_policy_training_preset": true
}
```

建议：
- `horizon` 决定每次预测的轨迹长度；`n_action_steps` 控制执行窗口。
- 训练时可设 `vision_backbone` 为 `resnet34`/`50` 提升表现，需同步调整学习率。
- 推理 YAML 中确保 `action_type`、`action_ori_type` 与训练一致。

### PI0.5 (OpenPI)

文件：`learning_factory/policies/pi05/configuration_pi05.py`

```json
{
  "policy": {
    "type": "pi05",
    "device": "cuda",
    "chunk_size": 50,
    "n_action_steps": 50,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "optimizer_lr": 2.5e-5,
    "scheduler_warmup_steps": 1000
  },
  "dataset": {
    "image_transforms": { "enable": true }
  },
  "batch_size": 8,
  "steps": 60000
}
```

建议：
- 语言条件任务可在数据集中添加文本 token，并在推理时通过 `task` 字段传入指令。
- `normalization_mapping` 默认使用分位数归一化；确保 `stats.pt` 包含 quantile 信息。
- 如需节约显存，可将 `paligemma_variant` 与 `action_expert_variant` 同设为 `gemma_300m`，并开启 `gradient_checkpointing`。

### SmolVLA

文件：`learning_factory/policies/smolvla/configuration_smolvla.py`

```json
{
  "policy": {
    "type": "smolvla",
    "device": "cuda",
    "chunk_size": 50,
    "n_action_steps": 50,
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "freeze_vision_encoder": true,
    "train_expert_only": true,
    "optimizer_lr": 1e-4
  },
  "batch_size": 4,
  "steps": 80000,
  "use_policy_training_preset": true
}
```

建议：
- 默认冻结视觉编码器，仅训练动作专家；若需端到端 finetune，将 `freeze_vision_encoder=false` 并适当降低学习率。
- `empty_cameras` 可用于补齐缺失视角，保证与预训练权重结构一致。
- `num_steps` 控制推理解码步数，可在资源紧张时设为 4～6。

## 多卡训练（Multi-GPU）

当前脚本默认单进程训练，如需使用多 GPU，可按以下步骤扩展为分布式数据并行 (DDP)：

1. **初始化分布式环境**  
   在 `learning_factory/scripts/lerobot_train.py` 顶部导入并添加初始化逻辑：

   ```python
   import os
   import torch.distributed as dist

   def maybe_init_distributed():
       if "LOCAL_RANK" not in os.environ:
           return None
       dist.init_process_group(backend="nccl")
       torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
       return int(os.environ["LOCAL_RANK"])
   ```

   在 `train` 入口调用 `rank = maybe_init_distributed()`，并在创建 DataLoader 时为 `sampler` 使用 `torch.utils.data.distributed.DistributedSampler`（或为 `EpisodeAwareSampler` 添加分布式包装）。

2. **包裹策略模型**  
   在策略实例化后使用：

   ```python
   if dist.is_initialized():
       policy = torch.nn.parallel.DistributedDataParallel(
           policy,
           device_ids=[torch.cuda.current_device()],
           find_unused_parameters=False,
       )
   ```

3. **使用 `torchrun` 启动**  
   单机多卡示例：

   ```bash
   torchrun --standalone --nproc_per_node=4 \
     learning_factory/scripts/lerobot_train.py \
     --config_path=learning_factory/configs/train_act_local.json \
     --policy.device=cuda
   ```

   - `policy.device` 置为 `cuda`，进程会根据 `LOCAL_RANK` 选择对应 GPU。
   - `batch_size` 为每卡 batch；有效 batch 为 `batch_size × 世界尺寸`。

4. **梯度累积与检查点**  
   - 使用 `policy.optimizer_grad_clip_norm` 或训练循环中的 `grad_clip_norm` 避免梯度爆炸。
   - 如需大 batch，优先考虑梯度累积 (`optimizer.grad_accum`) 而非盲目增加显存占用。
   - 保存与加载检查点时，建议仅在 rank 0 处理权重与日志。

> 若不方便修改代码，可使用多进程各自训练、最终在推理阶段集成结果，或通过 `CUDA_VISIBLE_DEVICES` 手动绑定不同 GPU。

## 数据处理与时间聚合

- `learning_factory/datasets/factory.py`：根据配置构建本地或流式数据集。
- `streaming_dataset.py`：通过 Hugging Face `datasets` 实现流式读取、回看 (look-back) 与前瞻 (look-ahead) 缓冲，支持 `delta_timestamps` 进行多时间步堆叠。
- `learning_factory/datasets/utils.py`：加载 `meta.json`、`stats.pt`，并提供归一化、采样器、视频写入工具。
- `video_backend` 可选 `pyav` 或 `torchcodec`；后者对 GPU 解码友好。
- 推理脚本依据数据集 `fps` 计算循环周期 `target_dt`，可通过更新 `meta.info["fps"]` 或在脚本中覆盖以匹配硬件周期。

## 推理流水线摘要

1. 读取 YAML，加载策略配置与数据统计 (`make_policy`, `make_pre_post_processors`)。
2. 通过 RobotMotion 获取图像、状态，构造 `observation.images.*` 与 `observation.state`。
3. 使用 `predict_action` 前向推理并执行 `_apply_action`，兼容关节/末端姿态等多种动作类型。
4. 键盘事件 (`utils/control_utils.py`) 控制回合、标注成功/失败，并输出统计。

## 故障排查

- **键盘监听失败**：安装 `pynput` 并确保存在 X server；无 GUI 时需自定义事件钩子。
- **RobotMotion 复位报错**：核对 YAML 中 `reset_to_home`、`reset_tool_command`，确认机械臂未处于保护模式。
- **归一化失配**：若 `stats.pt` 缺失或不匹配，使用 `learning_factory/datasets/utils.py` 重新生成统计，或重新训练。
- **控制频率异常**：检查数据集 `fps`、`policy.n_action_steps` 与推理循环的 `target_dt` 是否一致；必要时调整或引入睡眠补偿。

## 后续步骤

- 使用示例配置跑通 ACT/DP 训练与推理，验证数据路径与 RobotMotion 配置。
- 为自定义数据集生成 `meta.json` 与 `stats.pt`，更新 `dataset.repo_id`、`dataset.root`。
- 需要语言条件时参考 PI0.5 或 SmolVLA，结合 `task` 字段或自定义 prompt 完成闭环测试。
