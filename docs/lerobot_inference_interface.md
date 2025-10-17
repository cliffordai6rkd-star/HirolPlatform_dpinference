# LeRobot 推理新接口使用说明

本说明书介绍如何在 HIROLRobotPlatform 中使用全新的 `LeRobotInterface` 完成在线推理。新版接口将原先脚本内的 900 行逻辑抽象为独立组件，实现与数据加载流程的语义对齐，同时支持关节/末端执行器的增量（delta）控制。

## 1. 组件概览

文件路径：`learning_factory/interfaces/lerobot_interface.py`

- **DatasetLayout**  
  读取数据集根目录下的 `meta/info.json`/`meta/stats.json`，解析出 camera 名称、状态维度、动作维度、帧率等信息，并提供给策略构建器使用。

- **ObservationBuilder**  
  从 `RobotMotion` 中获取实时观测，构造与数据集一致的字典：  
  - 支持 `joint_position(_delta)`、`end_effector_pose(_delta)`、`mask` 等配置；  
  - `mask` 会直接返回与数据集声明一致的零向量；  
  - delta 模式会读取历史缓存（ObservationHistory）自动补全差分。

- **CommandTracker**  
  缓存最近一次发送的关节/位姿/夹爪目标，用于 COMMAND_* / *_DELTA 策略推理，保证与数据集中 command 语义对齐。

- **ActionApplier**  
  解析策略输出的向量，根据 `ActionType` 调用 `RobotMotion` 的 `send_joint_command` / `send_pose_command` / `send_gripper_command` 等接口，同时更新 CommandTracker。

- **LeRobotInterface**  
  面向上层脚本的组合对象，提供 `reset()` / `step()` / `close()`：  
  - `reset()` 负责机器人回 home、同步 CommandTracker，并返回第一帧观测；  
  - `step(action)` 应用策略输出，返回最新观测与 info；  
  - `close()` 关闭硬件或仿真句柄。

## 2. 推理脚本（`learning_factory/scripts/lerobot_inference.py`）

新版脚本保留原有命令行和事件循环，主要变化：

1. 启动时通过 `DatasetLayout.from_dataset()` 自动读取数据集元信息，减少重复配置；
2. 通过 `LeRobotInterface` 管理机器人状态、观测构造、动作执行；
3. 继续兼容键盘标记成功/失败、episode 超时等逻辑；
4. 与策略工厂、预处理/后处理保持原接口不变。

运行方式：

```bash
python -m learning_factory.scripts.lerobot_inference \
  --config learning_factory/configs/robot_motion_inference_template.yaml
```

> 说明：`lerobot_inference_bac.py` 已作为旧版备份保留，若需要对比旧实现可直接查看该文件。

## 3. 配置字段说明

配置以 YAML 给出，字段分布如下：

| 字段 | 说明 | 样例 |
| --- | --- | --- |
| `robot_motion_config` | RobotMotion YAML，决定硬件/仿真参数 | `factory/tasks/config/robot_motion_fr3_cfg.yaml` |
| `dataset.repo_id` | 数据集 repo 名称，用于日志与策略校验 | `fr3_pick_and_place_3dmouse` |
| `dataset.root` | 本地数据集根目录（含 `meta/info.json`） | `/home/user/datasets/fr3_pick_and_place_3dmouse` |
| `dataset.action_type` *(可选)* | 优先覆盖策略期望的动作类型 | `command_end_effector_pose_delta` |
| `dataset.observation_type` *(可选)* | 观测类型，默认为 `joint_position` | `mask` |
| `policy.path` | 预训练策略 checkpoint 路径 | `outputs/train/.../pretrained_model` |
| `device` | 运行设备 | `cuda` / `cpu` |
| `enable_hardware` | 是否启用真实控制器 | `true` |
| `gripper.max_position` | 夹爪最大开度（用于归一化） | `0.08` |
| `action_type` *(可选)* | 顶层覆盖动作语义，参考 `dataset/utils.py` | `joint_position_delta` |
| `action_orientation` *(可选)* | 位姿角度表示：`euler`、`quaternion`、`rotvec` | `quaternion` |
| `observation_orientation` *(可选)* | 末端姿态表示方式 | `euler` |
| `max_episodes` | 最多执行多少 episode，缺省为无限 | `10` |
| `episode_timeout_s` | 单次 episode 超时时间，缺省 30 秒 | `120` |
| `task` | 用于策略 prompt 的任务描述 | `collect rice and place in bowl` |

字段查找优先级：脚本参数 > 配置根节点 > `dataset.*` 子项 > 默认值。

## 4. Info 与日志

- `reset()` 与 `step()` 返回的 `info` 包含：
  - `command_tracker`：当前 joint/pose/gripper 的绝对与 command 目标；
  - `robot_state`：最近一次从 `RobotMotion.get_state()` 读取的原始状态；
  - `last_action`：`ActionApplier` 实际发送的目标（含 delta 解析结果）。
- 若观测/动作维度与数据集中声明不一致，接口会抛出 `ValueError` 并在日志中说明。

## 5. 升级注意事项

1. **依赖 SciPy**：`orientation.py` 仍然依赖 `scipy.spatial.transform.Rotation`。部署环境需提供 scipy≥1.10。
2. **RobotMotion 接口**：新版流程假设 `RobotMotion.send_*` 与 `get_state()` 行为与旧版一致，如有自定义实现，请确保兼容。
3. **Mask 与零向量**：若配置或数据集声明 `mask` 观测，接口会直接填充零向量并跳过对 `RobotMotion` 状态的读取。
4. **Delta 策略收敛**：CommandTracker 会在 `reset()` 同步当前状态，如机器人初始化失败需在外层重试，以免 delta 积累错误。
5. **保留旧脚本**：`lerobot_inference_bac.py` 不建议删除，方便回归对比。

## 6. 常见问题

### 指定的数据集没有 `meta/info.json` 怎么办？

`DatasetLayout` 默认依赖该文件。若数据集尚未转换，可先使用 `dataset/lerobot/lerobot_loaderV3.py` 执行一次转换，或在配置中临时写死 `state_dim`/`action_dim` 并补充自定义逻辑。

### 怎样扩展新的 ActionType？

1. 在 `dataset/utils.py` 中注册新的枚举；  
2. 在 `ActionApplier.apply()` 内添加对应分支，确保更新 CommandTracker；  
3. 若涉及新观测，扩展 `ObservationBuilder`；  
4. 同步策略训练数据和 `reader.py` 的解析逻辑。

---

如需进一步帮助，可在 `docs/robot_motion_api.md` 中查看 RobotMotion 功能，或参考已有的 `learning_factory/configs/robot_motion_inference_*.yaml` 配置。
