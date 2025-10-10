# RobotMotion API 文档

## 概述

`RobotMotion` 是一个高级运动规划任务类，整合了 `RobotFactory` 和 `MotionFactory`，提供统一的运动控制和数据采集接口。

**主要特性**：
- 运动控制（笛卡尔空间/关节空间）
- 完整数据采集（兼容 LeRobot 格式）
- 双线程架构（控制循环 + 数据采集循环）
- Rerun + OpenCV 可视化
- 键盘交互控制
- 支持单臂/双臂/全身机器人

**文件位置**：`HIROLRobotPlatform/factory/tasks/robot_motion.py`

---

## 目录

- [快速开始](#快速开始)
- [类初始化](#类初始化)
- [运动控制接口](#运动控制接口)
- [状态查询接口](#状态查询接口)
- [数据采集接口](#数据采集接口)
- [系统控制接口](#系统控制接口)
- [键盘交互](#键盘交互)
- [配置文件说明](#配置文件说明)
- [使用示例](#使用示例)
- [常见问题](#常见问题)

---

## 快速开始

### 基础用法

```python
from factory.tasks.robot_motion import RobotMotion

# 初始化
config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
robot_motion = RobotMotion(config_path)

# 运动控制
state = robot_motion.get_state()
target_pose = state['pose'].copy()
target_pose[2] += 0.05  # 向上移动 5cm
robot_motion.send_pose_command(target_pose)

# 关闭系统
robot_motion.close()
```

### 数据采集工作流（API方式，无需键盘）

```python
# 初始化
robot_motion = RobotMotion(config_path)

# 启用硬件执行
robot_motion.enable_hardware()

# 开始数据录制
robot_motion.start_recording()

# 执行运动任务...
robot_motion.send_pose_command(target_pose)
robot_motion.send_gripper_command_simple(0.5)

# 停止录制（数据自动保存）
robot_motion.stop_recording()

# 重置到安全位置
robot_motion.reset_to_home()

# 关闭系统
robot_motion.close()
```

### 数据采集工作流（键盘交互方式）

```python
# 初始化
robot_motion = RobotMotion(config_path)

# 启动主循环（支持键盘控制）
robot_motion.start()

# 通过键盘控制：
# - 按 'h' 启用硬件
# - 按 'r' 开始/停止录制
# - 按 'o' 重置到home位置
# - 按 'q' 退出

# 或混合使用API：
robot_motion.send_pose_command(target_pose)
robot_motion.start_recording()  # 也可以用API
# ...
```

---

## 类初始化

### `__init__(config_path: str, auto_initialize: bool = True)`

初始化 RobotMotion 任务

**参数**：
- `config_path` (str): 配置文件路径（YAML 格式）
- `auto_initialize` (bool, 默认=True): 是否自动初始化硬件连接

**配置文件必须包含**：
- `motion_config`: MotionFactory 配置（包含 RobotFactory 配置）
- `data_collection`: 数据采集设置
- `motion_control`: 运动控制参数

**抛出异常**：
- `RuntimeError`: 如果硬件初始化失败

**示例**：
```python
# 自动初始化（默认）
robot_motion = RobotMotion("config.yaml")

# 延迟初始化
robot_motion = RobotMotion("config.yaml", auto_initialize=False)
robot_motion.initialize()
```

---

### `initialize() -> None`

手动初始化机器人系统

**工作流程**：
1. 创建 MotionFactory 组件（自动创建 RobotFactory）
2. 使能硬件执行
3. 获取末端执行器链接列表
4. 启动数据采集线程
5. 启动键盘监听线程

**抛出异常**：
- `RuntimeError`: 如果硬件初始化失败

**示例**：
```python
robot_motion = RobotMotion(config_path, auto_initialize=False)
robot_motion.initialize()  # 手动初始化
```

---

## 运动控制接口

### `send_pose_command(pose: np.ndarray) -> None`

发送笛卡尔空间位姿命令

**参数**：
- `pose` (np.ndarray): 目标位姿
  - 单臂：`shape=(7,)`, `[x, y, z, qx, qy, qz, qw]`
  - 双臂：`shape=(14,)`, `[left_pose(7), right_pose(7)]`

**注意事项**：
- 命令是否经过 smoother 平滑取决于配置文件中的 `use_smoother` 参数
  - `use_smoother: true` (默认) - 命令被 Ruckig/临界阻尼平滑器平滑到 800Hz
  - `use_smoother: false` - 命令直接下发到机器人
- 用户代码无需关心底层实现，同一套 API 适配两种场景

**抛出异常**：
- `ValueError`: 如果位姿维度与机器人配置不匹配

**示例**：
```python
# 单臂机器人（FR3）
target_pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
robot_motion.send_pose_command(target_pose)

# 双臂机器人
left_pose = np.array([0.3, 0.2, 0.5, 1, 0, 0, 0])
right_pose = np.array([0.3, -0.2, 0.5, 1, 0, 0, 0])
dual_pose = np.hstack([left_pose, right_pose])
robot_motion.send_pose_command(dual_pose)
```

---

### `send_joint_command(joints: np.ndarray) -> None`

发送关节空间位置命令

**参数**：
- `joints` (np.ndarray): 关节位置（弧度）
  - 单臂（FR3）：`shape=(7,)`
  - 双臂（Duo XArm）：`shape=(14,)`
  - 全身（UnitreeG1）：`shape=(29,)`

**注意事项**：
- Smoother 行为由配置文件控制（同 `send_pose_command`）

**抛出异常**：
- `ValueError`: 如果关节维度与机器人 DOF 不匹配

**示例**：
```python
# 单臂机器人
home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
robot_motion.send_joint_command(home_joints)

# 双臂机器人
dual_joints = np.zeros(14)  # 14 DOF
robot_motion.send_joint_command(dual_joints)
```

---

### `send_gripper_command(gripper_commands: Dict[str, float]) -> None`

发送夹爪命令

**参数**：
- `gripper_commands` (Dict[str, float]): 夹爪命令字典
  - 单臂：`{"single": 0.5}`
  - 双臂：`{"left": 0.3, "right": 0.8}`
  - 范围：`[0-1]`（0=关闭，1=打开）

**注意事项**：
- 自动更新 `_latest_tool_action` 用于数据记录
- 超出范围的值会被自动裁剪并发出警告

**抛出异常**：
- `ValueError`: 如果提供的夹爪键无效

**示例**：
```python
# 单臂
robot_motion.send_gripper_command({"single": 1.0})  # 打开

# 双臂
robot_motion.send_gripper_command({
    "left": 0.0,   # 左手关闭
    "right": 1.0   # 右手打开
})
```

---

### `send_gripper_command_simple(width: float) -> None`

发送夹爪命令（简化版，仅适用于单臂）

**参数**：
- `width` (float): 夹爪开口宽度，范围 `[0-1]`

**注意事项**：
- 仅适用于单臂机器人
- 双臂机器人请使用 `send_gripper_command()`

**抛出异常**：
- `RuntimeError`: 如果机器人有多个手臂

**示例**：
```python
# 单臂机器人
robot_motion.send_gripper_command_simple(1.0)  # 打开
robot_motion.send_gripper_command_simple(0.0)  # 关闭
```

---

### `execute_trajectory(waypoints: List[np.ndarray], timing: Optional[List[float]] = None) -> None`

执行轨迹序列（阻塞式）

**参数**：
- `waypoints` (List[np.ndarray]): 位姿序列，每个元素 `shape=(7,)`
- `timing` (Optional[List[float]]): 每个 waypoint 的到达时间（秒）
  - 如果为 `None`，使用默认间隔（2.0 秒）

**工作流程**：
1. 依次发送每个 waypoint
2. 等待到达（通过位置误差判断）
3. 继续下一个 waypoint

**抛出异常**：
- `ValueError`: 如果 waypoints 为空或维度不匹配

**示例**：
```python
# 定义轨迹
waypoints = [
    np.array([0.3, 0.0, 0.5, 1, 0, 0, 0]),
    np.array([0.4, 0.0, 0.5, 1, 0, 0, 0]),
    np.array([0.4, 0.1, 0.5, 1, 0, 0, 0]),
]

# 使用默认时间（每个 waypoint 2 秒）
robot_motion.execute_trajectory(waypoints)

# 自定义时间
robot_motion.execute_trajectory(waypoints, timing=[1.5, 2.0, 1.5])
```

---

## 状态查询接口

### `get_state() -> Dict[str, Any]`

获取当前机器人状态

**返回值**：
字典包含以下字段：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `pose` | np.ndarray | (7,) | TCP 位姿 [x, y, z, qx, qy, qz, qw] |
| `vel` | np.ndarray | (6,) | TCP 速度 [vx, vy, vz, wx, wy, wz] |
| `q` | np.ndarray | (7,) | 关节位置（弧度） |
| `dq` | np.ndarray | (7,) | 关节速度（rad/s） |
| `torque` | np.ndarray | (7,) | 关节力矩（Nm） |
| `gripper_pos` | float | - | 夹爪位置 [0-1] |
| `time_stamp` | float | - | 时间戳 |

**注意事项**：
- 此接口用于外部脚本实时查询状态
- 速度通过雅可比矩阵计算得到
- 单臂机器人返回 7D 数据，双臂返回对应维度

**示例**：
```python
state = robot_motion.get_state()

print(f"当前位置: {state['pose'][:3]}")
print(f"当前关节: {state['q']}")
print(f"夹爪状态: {state['gripper_pos']}")
print(f"时间戳: {state['time_stamp']}")
```

---

## 数据采集接口

### `start_recording() -> None`

开始数据记录

**工作流程**：
1. 如果 `data_recorder` 为 `None`，创建 `EpisodeWriter` 实例
2. 调用 `data_recorder.create_episode()`
3. 使能 `motion_factory` 动作记录
4. 设置 `_enable_recording = True`

**抛出异常**：
- `RuntimeError`: 如果上一个 episode 未保存完成

**采集的数据类型**：
- `colors`: 相机彩色图像
- `depths`: 深度图像
- `joint_states`: 关节状态（位置/速度/加速度/力矩）
- `ee_states`: 末端执行器状态（位姿/速度）
- `tools`: 工具状态（夹爪位置）
- `tactiles`: 触觉传感器数据
- `imus`: IMU 数据
- `actions`: 动作命令（关节目标/末端目标/工具目标）

**数据格式**：LeRobot 兼容格式，详见 [data_collection_specification.md](data_collection_specification.md)

**示例**：
```python
robot_motion.start_recording()
# ... 执行运动 ...
robot_motion.stop_recording()
```

---

### `stop_recording() -> None`

停止数据记录

**工作流程**：
1. 设置 `_enable_recording = False`
2. 调用 `data_recorder.save_episode()`
3. 禁用 `motion_factory` 动作记录

**注意事项**：
- 数据保存是异步的，可能需要几秒完成
- 保存完成前不能创建新 episode

**示例**：
```python
robot_motion.stop_recording()
time.sleep(2.0)  # 等待数据保存完成
```

---

## 系统控制接口

### `enable_hardware() -> None`

启用硬件执行（机器人开始响应运动命令）

**功能**：
- 启用后，`send_pose_command()`, `send_joint_command()` 等命令会控制真实机器人硬件
- 等同于按键盘 'h' 键（当硬件未启用时）
- 幂等操作（重复调用安全）

**使用场景**：
- 自动化任务脚本（无需键盘交互）
- 远程SSH环境
- 集成到大型系统中

**示例**：
```python
robot_motion = RobotMotion(config_path)

# 启用硬件控制
robot_motion.enable_hardware()

# 现在机器人会响应命令
robot_motion.send_pose_command(target_pose)  # 机器人移动
```

---

### `disable_hardware() -> None`

禁用硬件执行（dry-run 模式，用于测试）

**功能**：
- 禁用后，命令被处理但不发送到真实硬件
- 适用于测试运动逻辑而不移动机器人
- 等同于按键盘 'h' 键（当硬件已启用时）
- 幂等操作（重复调用安全）

**使用场景**：
- 测试运动规划逻辑
- 验证数据采集流程（不实际移动机器人）
- 调试任务代码

**示例**：
```python
# 测试模式：不移动真实机器人
robot_motion.disable_hardware()
robot_motion.send_pose_command(target_pose)  # 命令处理但不执行

# 验证逻辑后，启用硬件
robot_motion.enable_hardware()
robot_motion.send_pose_command(target_pose)  # 现在机器人移动
```

---

### `reset_to_home(home_pose: Optional[np.ndarray] = None, space: Robot_Space = Robot_Space.CARTESIAN_SPACE) -> None`

重置机器人到初始位置

**参数**：
- `home_pose` (Optional[np.ndarray]): 目标位姿/关节位置
  - 如果为 `None`，使用配置文件默认值
- `space` (Robot_Space): 运动空间
  - `Robot_Space.CARTESIAN_SPACE`: 笛卡尔空间
  - `Robot_Space.JOINT_SPACE`: 关节空间

**工作流程**：
1. 暂停运动更新（`_update_motion_state = False`）
2. 调用 `motion_factory.reset_robot_system()`
3. 等待到达
4. 恢复运动更新（`_update_motion_state = True`）

**示例**：
```python
# 使用配置文件默认位置
robot_motion.reset_to_home()

# 自定义关节空间目标
from factory.components.motion_factory import Robot_Space
home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
robot_motion.reset_to_home(home_joints, space=Robot_Space.JOINT_SPACE)

# 自定义笛卡尔空间目标
home_pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
robot_motion.reset_to_home(home_pose, space=Robot_Space.CARTESIAN_SPACE)
```

---

### `start() -> None`

启动主循环（非阻塞，用于脚本式交互）

**注意事项**：
- 此方法用于脚本式交互
- 启动后返回，用户可通过 API 发送命令
- 主线程持续运行，用户通过键盘 'q' 或 `close()` 退出

**示例**：
```python
robot_motion = RobotMotion(config_path)
robot_motion.start()

# 用户代码在这里
robot_motion.send_pose_command(target_pose)
robot_motion.start_recording()
# ... 更多命令 ...
robot_motion.stop_recording()

robot_motion.close()
```

---

### `close() -> None`

关闭系统，释放资源

**工作流程**：
1. 停止主线程（`_main_thread_running = False`）
2. 停止数据采集线程
3. 如果正在记录，保存 episode
4. 关闭 `data_recorder`
5. 关闭 `motion_factory`
6. 关闭 `robot_system`
7. 关闭 OpenCV 窗口

**注意事项**：
- 必须在程序退出前调用
- 确保所有资源正确释放

**示例**：
```python
try:
    robot_motion = RobotMotion(config_path)
    # ... 使用机器人 ...
finally:
    robot_motion.close()
```

---

## 键盘交互

RobotMotion 自动启动键盘监听线程，支持以下按键：

| 按键 | 功能 | 说明 |
|------|------|------|
| `h` | 切换硬件执行 | 使能/禁用硬件执行（调试时使用） |
| `r` | 切换数据记录 | 开始/停止数据记录 |
| `o` | 重置机器人 | 重置到 home 位置 |
| `q` | 退出 | 关闭系统并退出 |

**使用方式**：
1. 程序运行后，键盘监听自动启动
2. 按下对应按键触发功能
3. 无需在代码中显式调用

**示例输出**：
```
Keyboard listener started (h=hardware, r=record, o=reset, q=quit)
=============== Hardware execution: True ===============  # 按下 'h'
=============== Data recording started ===============    # 按下 'r'
```

---

## 配置文件说明

### 配置文件结构

完整配置文件见：`factory/tasks/config/robot_motion_fr3_cfg.yaml`

```yaml
# ============= 运动控制配置 =============
motion_config: !include factory/components/motion_configs/fr3_with_franka_hand_ik.yaml

# 覆盖 smoother 设置（可选）
# motion_config:
#   use_smoother: false
#   auto_enable_async_control: false

# ============= 数据采集配置 =============
data_collection:
  save_path_prefix: "robot_motion_test"
  data_record_frequency: 30
  image_visualization: true
  rerun_visualization: true
  task_description: "..."
  task_description_goal: "..."
  task_description_steps: "..."

# ============= 运动控制参数 =============
motion_control:
  control_loop_time: 0.02
  reset_arm_command: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
  reset_space: "joint"  # "joint" 或 "cartesian"
  reset_tool_command:
    single: 1.0
```

### 配置参数说明

#### `motion_config`
继承 MotionFactory 完整配置，包含：
- 机器人硬件配置
- 控制器配置
- Smoother 配置
- 异步控制配置

**Smoother 控制**：
- `use_smoother: true` - 启用平滑（50Hz → 800Hz）
- `use_smoother: false` - 禁用平滑（直接下发）

#### `data_collection`
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_path_prefix` | string | "robot_motion_test" | 数据保存路径前缀 |
| `data_record_frequency` | int | 30 | 数据采集频率（Hz） |
| `image_visualization` | bool | true | OpenCV 实时显示 |
| `rerun_visualization` | bool | true | Rerun 3D 可视化 |
| `task_description` | string | - | 任务描述 |
| `task_description_goal` | string | - | 任务目标 |
| `task_description_steps` | string | - | 任务步骤 |

#### `motion_control`
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `control_loop_time` | float | 0.02 | 控制循环周期（秒，对应50Hz） |
| `reset_arm_command` | List[float] | - | 重置关节位置（7D 数组） |
| `reset_space` | string | "joint" | 重置空间（"joint" 或 "cartesian"） |
| `reset_tool_command` | Dict | - | 重置工具命令 |

---

## 使用示例

### 示例 1：基础运动控制

```python
from factory.tasks.robot_motion import RobotMotion
import numpy as np

# 初始化
config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
robot_motion = RobotMotion(config_path)

try:
    # 使能硬件（或按 'h' 键）
    robot_motion._toggle_hardware()

    # 获取当前状态
    state = robot_motion.get_state()
    print(f"当前位置: {state['pose'][:3]}")

    # 移动到新位置
    target_pose = state['pose'].copy()
    target_pose[2] += 0.1  # 向上移动 10cm
    robot_motion.send_pose_command(target_pose)

    # 等待到达
    import time
    time.sleep(3.0)

    # 控制夹爪
    robot_motion.send_gripper_command_simple(0.0)  # 关闭
    time.sleep(1.0)
    robot_motion.send_gripper_command_simple(1.0)  # 打开

finally:
    robot_motion.close()
```

---

### 示例 2：轨迹执行与数据采集

```python
from factory.tasks.robot_motion import RobotMotion
import numpy as np

config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
robot_motion = RobotMotion(config_path)

try:
    # 使能硬件
    robot_motion._toggle_hardware()

    # 重置到 home
    robot_motion.reset_to_home()

    # 开始录制
    robot_motion.start_recording()

    # 定义轨迹（正方形）
    state = robot_motion.get_state()
    base_pose = state['pose'].copy()

    waypoints = []
    offsets = [[0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]]
    for dx, dy in offsets:
        pose = base_pose.copy()
        pose[0] += dx
        pose[1] += dy
        waypoints.append(pose)

    # 执行轨迹
    robot_motion.execute_trajectory(waypoints, timing=[2.0] * 4)

    # 停止录制
    robot_motion.stop_recording()

    print("数据已保存到:", robot_motion._save_path_dir)

finally:
    robot_motion.close()
```

---

### 示例 3：双臂机器人控制

```python
from factory.tasks.robot_motion import RobotMotion
import numpy as np

# 假设配置为双臂机器人
config_path = "factory/tasks/config/robot_motion_duo_cfg.yaml"
robot_motion = RobotMotion(config_path)

try:
    robot_motion._toggle_hardware()

    # 双臂位姿命令
    left_pose = np.array([0.3, 0.2, 0.5, 1, 0, 0, 0])
    right_pose = np.array([0.3, -0.2, 0.5, 1, 0, 0, 0])
    dual_pose = np.hstack([left_pose, right_pose])

    robot_motion.send_pose_command(dual_pose)

    # 双臂夹爪命令
    robot_motion.send_gripper_command({
        "left": 0.5,
        "right": 0.8
    })

finally:
    robot_motion.close()
```

---

### 示例 4：脚本式交互

```python
from factory.tasks.robot_motion import RobotMotion
import numpy as np
import time

config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
robot_motion = RobotMotion(config_path)

# 启动系统（非阻塞）
robot_motion.start()

# 在这里编写你的控制逻辑
robot_motion._toggle_hardware()

for i in range(10):
    state = robot_motion.get_state()
    target_pose = state['pose'].copy()
    target_pose[2] += 0.01 * ((-1) ** i)  # 上下振荡
    robot_motion.send_pose_command(target_pose)
    time.sleep(0.5)

# 按 'q' 退出，或手动关闭
robot_motion.close()
```

---

## 常见问题

### Q1: 如何禁用 Smoother？

**A**: 在配置文件中覆盖 `motion_config` 的 `use_smoother` 参数：

```yaml
motion_config: !include factory/components/motion_configs/fr3_with_franka_hand_ik.yaml

# 禁用 smoother
motion_config:
  use_smoother: false
  auto_enable_async_control: false
```

---

### Q2: 数据保存在哪里？

**A**: 数据保存在 `HIROLRobotPlatform/dataset/data/<save_path_prefix>/` 目录下：

```
dataset/data/robot_motion_test/
├── episode_0001/
│   ├── data.json
│   ├── colors/
│   ├── depths/
│   └── tactiles/
├── episode_0002/
│   └── ...
```

可以通过配置文件的 `data_collection.save_path_prefix` 修改路径。

---

### Q3: 如何支持双臂机器人？

**A**: RobotMotion 自动检测机器人配置，无需修改代码：

1. 使用双臂机器人的配置文件（如 `duo_xarm_ik.yaml`）
2. API 自动适配：
   - `send_pose_command(pose)`: `pose.shape=(14,)` 而不是 `(7,)`
   - `send_gripper_command({"left": 0.5, "right": 0.8})`

---

### Q4: 键盘监听冲突怎么办？

**A**: 如果快速创建/销毁多个 `RobotMotion` 实例，可能出现键盘监听冲突。解决方法：

```python
# 方法 1：延迟销毁
robot_motion.close()
time.sleep(0.5)  # 等待监听器完全停止

# 方法 2：手动停止监听
from sshkeyboard import stop_listening
stop_listening()
time.sleep(0.5)
```

---

### Q5: 如何调整数据采集频率？

**A**: 修改配置文件的 `data_collection.data_record_frequency`：

```yaml
data_collection:
  data_record_frequency: 50  # 改为 50Hz
```

**注意**：
- 建议频率：30Hz（与图像采集频率匹配）
- 过高频率可能导致磁盘 I/O 瓶颈

---

### Q6: 如何查看采集的数据？

**A**: 数据以 LeRobot 格式存储，可以使用以下方式查看：

```python
import json

# 读取 data.json
with open('dataset/data/robot_motion_test/episode_0001/data.json', 'r') as f:
    data = json.load(f)

# 查看元信息
print("Info:", data['info'])

# 查看任务描述
print("Task:", data['text'])

# 查看数据帧数
print(f"Total frames: {len(data['data'])}")

# 查看第一帧
first_frame = data['data'][0]
print("Frame 0 keys:", first_frame.keys())
```

---

### Q7: 出现 "ValueError: 'JOINT_SPACE' is not a valid Robot_Space" 错误？

**A**: 配置文件中 `reset_space` 应使用枚举值（小写），不是枚举名：

```yaml
# ❌ 错误
reset_space: "JOINT_SPACE"

# ✅ 正确
reset_space: "joint"  # 或 "cartesian"
```

---

### Q8: 如何在没有机器人硬件的情况下测试？

**A**: 使用仿真模式：

1. 修改配置文件：
```yaml
motion_config:
  use_hardware: false
  use_simulation: true
```

2. 运行测试：
```python
robot_motion = RobotMotion(config_path)
# 所有命令会发送到仿真器而不是真实硬件
```

---

## 相关文档

- [数据采集规范](data_collection_specification.md) - 完整数据格式说明
- [MotionFactory API](../factory/components/motion_factory.py) - 运动控制底层接口
- [RobotFactory API](../factory/components/robot_factory.py) - 硬件抽象层接口
- [配置文件模板](../factory/tasks/config/robot_motion_fr3_cfg.yaml) - FR3 配置示例

---

## 版本历史

- **v1.0.0** (2025-09-30): 初版发布
  - 支持单臂/双臂/全身机器人
  - 完整数据采集功能
  - 双线程架构
  - 配置驱动设计

---

## 联系方式

如有问题或建议，请联系 HIROL 团队或提交 Issue。