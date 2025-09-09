# ACT真机推理演示

⚠️ **重要更新**: ACT推理已迁移到统一推理任务架构！

**新的推理入口**: `factory.tasks.inferences_tasks.act.act_inference.ACT_Inferencer`

本目录现在包含示例代码、配置文件和相关工具，支持FR3单臂和Monte01双臂机器人。

## 文件结构

```
learning/demo/
├── example_act_inference.py                      # 新架构使用示例
├── run_monte01_dual_arm.sh                       # Monte01双臂启动脚本（需更新）
├── configs/
│   ├── fr3_inference_config.yaml                 # FR3单臂机器人配置文件（参考用）
│   └── monte01_dual_arm_inference_config.yaml    # Monte01双臂机器人配置文件（参考用）
├── gripper_controller.py                         # 夹爪控制器工具
├── joint_position_plotter.py                     # 关节位置绘图工具
└── README.md                                      # 本文件
```

## 使用方法

### 1. 准备模型和环境

#### FR3单臂机器人
确保你有：
- 训练好的ACT模型检查点目录（8 DOF，包含 `policy_best.ckpt` 和 `dataset_stats.pkl`）
- FR3机器人硬件连接正常
- 2个相机传感器工作正常（ee_cam, third_person_cam）

#### Monte01双臂机器人
确保你有：
- 训练好的双臂ACT模型检查点目录（16 DOF，包含 `policy_best.ckpt` 和 `dataset_stats.pkl`）
- Monte01双臂机器人硬件连接正常（使用duo_arm架构）
- 3个相机传感器工作正常（left_ee_cam, right_ee_cam, third_person_cam）

**注意**：
- Monte01使用duo_arm+duo_tool架构，推理引擎自动处理状态转换：
  - 从duo_arm获取14维关节状态（左臂7 + 右臂7）
  - 从duo_tool获取夹爪真实状态
  - 自动组合为16维完整状态供ACT推理使用
- 左侧夹爪：从corenetic_gripper硬件读取真实位置（0.0-0.074m）
- 右侧夹爪：未安装，自动使用默认值0.0（闭合状态）

### 2. 运行推理 (新架构)

#### 方式1: 使用默认配置文件

```bash
# FR3机器人推理
python -c "
from factory.tasks.inferences_tasks.act.act_inference import ACT_Inferencer
config = {
    'robot_type': 'fr3',
    'checkpoint_path': '/path/to/your/fr3_act_model',
    'frequency': 20.0,
    'max_episode_length': 12000
}
act_inferencer = ACT_Inferencer(config)
act_inferencer.start_inference()
"

```

#### 方式2: 使用YAML配置文件

```bash
# 使用现有配置文件模板
python -c "
from factory.tasks.inferences_tasks.act.act_inference import ACT_Inferencer
from hardware.base.utils import dynamic_load_yaml

config = dynamic_load_yaml('factory/tasks/inferences_tasks/act/config/fr3_act_inference_cfg.yaml')
config['checkpoint_path'] = '/path/to/your/fr3_act_model'
act_inferencer = ACT_Inferencer(config)
act_inferencer.start_inference()
"
```

#### Monte01双臂机器人推理

```bash
# Monte01双臂机器人
python -c "
from factory.tasks.inferences_tasks.act.act_inference import ACT_Inferencer
config = {
    'robot_type': 'monte01',
    'checkpoint_path': '/path/to/monte01_dual_arm_model',
    'frequency': 15.0,
    'max_episode_length': 1500
}
act_inferencer = ACT_Inferencer(config)
act_inferencer.start_inference()
"

# 或使用配置文件
python -c "
from factory.tasks.inferences_tasks.act.act_inference import ACT_Inferencer
from hardware.base.utils import dynamic_load_yaml

config = dynamic_load_yaml('factory/tasks/inferences_tasks/act/config/monte01_act_inference_cfg.yaml')
config['checkpoint_path'] = '/path/to/monte01_dual_arm_model'
act_inferencer = ACT_Inferencer(config)
act_inferencer.start_inference()
"
```

### 3. 参数说明

- `--robot`: 机器人类型（支持 `fr3`, `monte01`）
- `--ckpt_dir`: ACT模型检查点目录路径
- `--config`: 推理配置文件路径
- `--frequency`: 控制频率（Hz），FR3默认10.0，Monte01推荐10.0
- `--max_steps`: 最大episode步数，FR3默认1000，Monte01建议1500
- `--log_level`: 日志级别，默认INFO

### 4. 安全注意事项

⚠️ **重要安全提醒**

#### FR3单臂机器人
- 确保机器人工作空间内没有人员或障碍物
- 始终准备紧急停止按钮
- 首次运行时建议使用较低的控制频率
- 程序支持 Ctrl+C 安全停止

#### Monte01双臂机器人（额外注意事项）
- **双臂工作空间重叠**：左右臂工作空间可能重叠，需额外小心
- **多相机视野**：确保三个相机视野畅通，无遮挡
- **双臂协调**：虽然系统使用简化模式（无碰撞检测），但ACT模型应已学会安全的双臂协调
- **分离控制**：双臂动作独立执行，确保各自的安全性
- **更长任务时间**：双臂任务通常需要更多步数，做好长时间监控准备

### 5. 配置文件说明

#### FR3配置文件 (`fr3_inference_config.yaml`)
- **硬件配置**: FR3机器人和夹爪设置
- **传感器配置**: 2个相机参数和连接信息
- **学习推理配置**: 8 DOF ACT模型参数
- **控制参数**: 控制频率、episode长度等
- **安全参数**: 单臂关节限制、速度限制
- **日志和监控**: 日志级别、数据保存设置

#### Monte01配置文件 (`monte01_dual_arm_inference_config.yaml`)
- **运动配置**: 使用 `duo_xarm_duo_ik.yaml` 双臂控制配置
- **双臂硬件**: Monte01双臂机器人和双夹爪/吸盘设置
- **三相机传感器**: left_ee_cam, right_ee_cam, third_person_cam
- **16 DOF配置**: 双臂ACT模型参数（左臂8 + 右臂8）
- **简化双臂控制**: 无碰撞检测，依赖模型学习的安全行为
- **双臂安全参数**: 16个关节的限制和速度设置

### 6. 故障排除

**FR3单臂常见问题**:

1. **模型加载失败**
   - 检查检查点目录是否包含必要文件
   - 确认模型state_dim与配置一致（FR3应为8）

2. **相机连接失败**
   - 检查相机串口号和连接状态
   - 确认相机配置文件路径正确

3. **机器人连接失败**
   - 检查机器人网络连接
   - 确认机器人处于正确的控制模式

**Monte01双臂特有问题**:

4. **双臂模型维度错误**
   - 确认模型是16 DOF训练（不是8 DOF）
   - 检查数据集统计文件中的状态维度

5. **三相机数据问题**
   - 验证所有三个相机连接正常
   - 检查相机命名是否与配置一致

6. **双臂动作执行失败**
   - 确认Monte01硬件支持独立双臂控制
   - 检查左右臂关节状态获取是否正常

7. **性能问题**
   - 双臂推理计算量更大，可能需要更强的GPU
   - 可以适当降低控制频率到5-8Hz

### 7. 日志和输出

#### FR3单臂日志输出
- 系统初始化状态
- 推理循环进度
- 安全检查结果
- 夹爪状态信息
- 性能统计信息

#### Monte01双臂特有日志
- 双臂协调统计信息
- 左右臂夹爪/吸盘状态
- 16维状态处理信息
- 双臂动作分离执行结果

日志会保存在配置的路径中（FR3: `./logs/trajectories`, Monte01: `./logs/monte01_dual_arm_trajectories`），轨迹数据也会被记录用于后续分析。

### 8. 技术架构

#### 统一的推理框架
- **单一入口**: `run_act_inference.py` 支持两种机器人类型
- **配置驱动**: 通过 `--robot` 参数和配置文件自动适配
- **工厂模式**: 使用HIROLRobotPlatform的工厂模式创建组件
- **状态自适应**: 自动处理8 DOF（FR3）或16 DOF（Monte01）状态

#### Monte01双臂特性
- **16 DOF支持**: 左臂8维 + 右臂8维状态和动作
- **三相机处理**: 并行处理三个相机的图像数据
- **双臂分离执行**: 独立控制左右臂，无碰撞检测依赖
- **简化协调**: 依赖ACT模型学习的安全双臂协调行为

### 9. 扩展支持

本框架设计为可扩展的，如需添加新的机器人类型：
1. 创建对应的数据适配器
2. 添加机器人类型检测逻辑
3. 创建专用的配置文件
4. 更新学习推理工厂