# HIROLRobotPlatform 数据采集规范

## 目录
- [数据集格式](#数据集格式)
- [数据采集接口](#数据采集接口)
- [数据存储结构](#数据存储结构)
- [数据类型详解](#数据类型详解)
- [实现参考](#实现参考)

---

## 数据集格式

### LeRobot 兼容格式
本项目采用 LeRobot 兼容的数据集格式，便于后续机器人学习任务的训练。

### 目录结构
```
task_dir/                           # 任务根目录
├── episode_0000/                   # 第一个episode
│   ├── data.json                   # 元数据 + 时序数据索引
│   ├── colors/                     # 彩色图像目录
│   │   ├── 000000_cam1_color.jpg
│   │   ├── 000001_cam1_color.jpg
│   │   └── ...
│   ├── depths/                     # 深度图像目录
│   │   ├── 000000_cam1_depth.png
│   │   ├── 000001_cam1_depth.png
│   │   └── ...
│   ├── tactiles/                   # 触觉数据目录
│   │   ├── tactile_000000_sensor1.npy
│   │   ├── tactile_000001_sensor1.npy
│   │   └── ...
│   └── audios/                     # 音频数据目录 (可选)
│       ├── audio_000000_mic1.npy
│       └── ...
├── episode_0001/
│   └── ...
└── episode_NNNN/
    └── ...
```

---

## data.json 格式规范

### 完整结构示例
```json
{
  "info": {
    "version": "1.0.0",
    "date": "2025-09-30",
    "author": "HIROL",
    "image": {
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "depth": {
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "audio": {
      "sample_rate": 16000,
      "channels": 1,
      "format": "PCM",
      "bits": 16
    },
    "joint_names": null,
    "tactile_names": null,
    "sim_state": ""
  },
  "text": {
    "goal": "任务的最终目标（1-2句话）",
    "desc": "任务的详细描述",
    "steps": "step1: 第一步操作. step2: 第二步操作. ..."
  },
  "data": [
    {
      "idx": 0,
      "colors": { ... },
      "depths": { ... },
      "joint_states": { ... },
      "ee_states": { ... },
      "tactiles": { ... },
      "imus": { ... },
      "audios": { ... },
      "tools": { ... },
      "actions": { ... }
    },
    {
      "idx": 1,
      ...
    }
  ]
}
```

### 字段说明

#### `info` 字段（元信息）
| 字段 | 类型 | 说明 |
|-----|------|------|
| `version` | string | 数据格式版本号 |
| `date` | string | 数据采集日期 (YYYY-MM-DD) |
| `author` | string | 数据采集者/机构 |
| `image` | object | 图像元信息 (width, height, fps) |
| `depth` | object | 深度图元信息 (width, height, fps) |
| `audio` | object | 音频元信息 (sample_rate, channels, format, bits) |
| `joint_names` | list/null | 关节名称列表（可选） |
| `tactile_names` | list/null | 触觉传感器名称列表（可选） |
| `sim_state` | string | 仿真器状态（可选） |

#### `text` 字段（任务描述）
| 字段 | 类型 | 说明 |
|-----|------|------|
| `goal` | string | 任务目标（简短描述） |
| `desc` | string | 任务详细描述 |
| `steps` | string | 任务步骤（step1: ... step2: ...） |

#### `data` 字段（时序数据数组）
每个时间步包含以下字段：

---

## 数据类型详解

### 1. Colors（彩色图像）

**数据格式**：
```json
{
  "cam_name_color": {
    "path": "colors/000000_cam_name_color.jpg",
    "time_stamp": 162661.782833385
  }
}
```

**字段说明**：
- `cam_name_color`: 相机名称（多相机用不同key区分）
- `path`: 图像相对路径（相对于episode目录）
- `time_stamp`: 时间戳（秒，高精度浮点数）

**存储格式**：
- 文件格式: `.jpg` (有损压缩)
- 命名规则: `{idx:06d}_{cam_name}_color.jpg`
- 图像尺寸: 配置文件指定（默认640x480）

---

### 2. Depths（深度图像）

**数据格式**：
```json
{
  "cam_name_depth": {
    "path": "depths/000000_cam_name_depth.png",
    "time_stamp": 162661.782833385
  }
}
```

**字段说明**：
- `cam_name_depth`: 深度相机名称
- `path`: 深度图相对路径
- `time_stamp`: 时间戳

**存储格式**：
- 文件格式: `.png` (无损压缩)
- 命名规则: `{idx:06d}_{cam_name}_depth.png`
- 深度编码: 16位灰度图（单位：毫米）

---

### 3. Joint States（关节状态）

**数据格式**：
```json
{
  "single": {
    "position": [0.001, -0.775, 0.003, -2.374, -0.010, 1.598, 0.796],
    "velocity": [-0.0002, 0.0008, 0.0025, -0.0108, 0.0003, 0.0114, 0.0013],
    "acceleration": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "torque": [-0.410, -5.662, -0.737, 23.146, 0.798, 2.407, 0.037],
    "time_stamp": 162661.792109758
  }
}
```

**字段说明**：
- `single` / `left` / `right`: 机器人标识符
  - `single`: 单臂机器人
  - `left` / `right`: 双臂机器人
- `position`: 关节位置（弧度），7D数组
- `velocity`: 关节速度（rad/s），7D数组
- `acceleration`: 关节加速度（rad/s²），7D数组
- `torque`: 关节力矩（Nm），7D数组
- `time_stamp`: 时间戳

**数据来源**：
- `RobotFactory.get_joint_states()` → 原始状态
- `MotionFactory.get_type_joint_state(states, robot_key)` → 按机器人切片

---

### 4. EE States（末端执行器状态）

**数据格式**：
```json
{
  "single": {
    "pose": [0.311, -0.0004, 0.477, 0.9999, -0.003, -0.0001, 0.006],
    "twist": [0.001, 0.0008, -0.004, -0.001, 0.0002, 0.0003],
    "time_stamp": 162661.792109758
  }
}
```

**字段说明**：
- `pose`: 7D位姿 `[x, y, z, qx, qy, qz, qw]`
  - `x, y, z`: 笛卡尔位置（米）
  - `qx, qy, qz, qw`: 四元数姿态
- `twist`: 6D速度 `[vx, vy, vz, wx, wy, wz]`
  - `vx, vy, vz`: 线速度（m/s）
  - `wx, wy, wz`: 角速度（rad/s）
- `time_stamp`: 时间戳

**数据来源**：
```python
cur_ee_pose = motion_factory.get_frame_pose_with_joint_state(
    joint_states, ee_link_name, robot_key, need_vel=True
)
ee_states[robot_key] = {
    "pose": cur_ee_pose[:7].tolist(),
    "twist": cur_ee_pose[7:13].tolist(),
    "time_stamp": joint_states._time_stamp
}
```

---

### 5. Tools（工具状态）

**数据格式**：
```json
{
  "single": {
    "position": 0.080398328602314,
    "time_stamp": 162660.936109839
  }
}
```

**字段说明**：
- `position`: 工具位置（夹爪开口宽度，米）
- `time_stamp`: 时间戳

**数据来源**：
```python
tool_state_dict = robot_factory.get_tool_dict_state()
gripper_state[robot_key] = {
    'position': tool_state_dict[robot_key]._position,
    'time_stamp': tool_state_dict[robot_key]._time_stamp
}
```

---

### 6. Actions（动作命令）

**数据格式**：
```json
{
  "single": {
    "joint": {
      "position": [0.001, -0.781, 0.004, -2.377, -0.012, 1.606, 0.797],
      "time_stamp": 162661.793256789
    },
    "ee": {
      "pose": [0.312, -0.0007, 0.478, 0.9999, -0.003, 0.005, 0.008],
      "time_stamp": 162661.793256789
    },
    "tool": {
      "position": 1.0,
      "time_stamp": 162661.786365185
    }
  }
}
```

**字段说明**：
- `joint`: 关节空间动作命令
  - `position`: 目标关节位置（7D，弧度）
- `ee`: 笛卡尔空间动作命令
  - `pose`: 目标末端位姿（7D，米+四元数）
- `tool`: 工具动作命令
  - `position`: 目标工具位置（标量，归一化或物理单位）

**关键说明**：
- **Actions 不是状态观测，而是发送给机器人的命令**
- 用于模仿学习时作为监督信号
- 时间戳可能略早于对应的状态（控制延迟）

**数据来源**：
```python
# 1. 获取运动动作（关节+末端目标）
motion_action = motion_factory.get_latest_action()

# 2. 获取工具动作（需手动同步）
# 主线程中设置：
with tool_action_lock:
    tool_action[robot_key] = dict(tool=dict(
        position=tool_command,
        time_stamp=time.perf_counter()
    ))

# 3. 数据线程中合并：
with tool_action_lock:
    actions = copy.deepcopy(tool_action)
    for key in actions.keys():
        actions[key]["joint"] = motion_action[key]["joint"]
        actions[key]["ee"] = motion_action[key]["ee"]
```

---

### 7. Tactiles（触觉传感器）

**数据格式**：
```json
{
  "dual_arm_tactile": {
    "path": "tactiles/tactile_000000_dual_arm_tactile.npy",
    "timestamp": 162661.786628891,
    "shape": [2, 120, 3]
  }
}
```

**字段说明**：
- `sensor_name`: 触觉传感器名称
- `path`: NumPy数组文件路径
- `timestamp`: 时间戳
- `shape`: 数据形状（如 `[2, 120, 3]` 表示2个手指，每指120个触点，3维力向量）

**存储格式**：
- 文件格式: `.npy` (NumPy二进制格式)
- 命名规则: `tactile_{idx:06d}_{sensor_name}.npy`
- 数据类型: `np.int32`

**数据来源**：
```python
tactiles = robot_factory.get_tactile_data()
# 返回格式: {sensor_name: {"data": np.ndarray, "timestamp": float}}
```

---

### 8. IMUs（惯性测量单元）

**数据格式**：
```json
{
  "cam_name_imu": {
    "data": [ax, ay, az, gx, gy, gz],
    "time_stamp": 162661.782833385
  }
}
```

**字段说明**：
- `data`: 6D数组 `[ax, ay, az, gx, gy, gz]`
  - `ax, ay, az`: 加速度（m/s²）
  - `gx, gy, gz`: 角速度（rad/s）
- `time_stamp`: 时间戳

**数据来源**：
```python
cameras_data = robot_factory.get_cameras_infos()
for cam_data in cameras_data:
    if 'imu' in cam_data['name']:
        cur_imus[cam_data['name']] = {
            "data": cam_data['imu'],
            "time_stamp": cam_data['time_stamp']
        }
```

---

### 9. Audios（音频数据，可选）

**数据格式**：
```json
{
  "mic_name": "audios/audio_000000_mic_name.npy"
}
```

**存储格式**：
- 文件格式: `.npy`
- 命名规则: `audio_{idx:06d}_{mic_name}.npy`
- 数据类型: `np.int16`（PCM格式）
- 采样率: 16000 Hz（默认）

**数据来源**：
- 当前未实现，预留接口

---

## 数据采集接口

### RobotFactory 接口

#### 传感器数据获取
```python
# 1. 获取所有相机数据（包含colors/depths/imus）
cameras_data = robot_factory.get_cameras_infos()
# 返回: List[Dict], 每个元素包含:
#   - name: str (e.g., "cam1_color", "cam1_depth", "cam1_imu")
#   - img: np.ndarray (如果是图像)
#   - imu: List[float] (如果是IMU)
#   - time_stamp: float

# 2. 获取触觉数据
tactiles = robot_factory.get_tactile_data()
# 返回: Dict[str, Dict]
#   - key: sensor_name
#   - value: {"data": np.ndarray, "timestamp": float}
```

#### 机器人状态获取
```python
# 3. 获取关节状态
joint_states = robot_factory.get_joint_states()
# 返回: JointState对象，包含所有机器人的关节状态

# 4. 获取工具状态
tool_state_dict = robot_factory.get_tool_dict_state()
# 返回: Dict[str, ToolState]
#   - key: robot_key ("single", "left", "right")
#   - value: ToolState(_position, _time_stamp)
```

#### 命令发送
```python
# 5. 发送工具命令
robot_factory.set_tool_command(tool_target)
# 输入: Dict[str, float/List[float]]
#   - key: robot_key
#   - value: 工具命令值（夹爪位置/力等）
```

---

### MotionFactory 接口

#### 运动状态获取
```python
# 1. 获取指定类型的关节状态
sliced_joint_states = motion_factory.get_type_joint_state(
    all_joint_states, robot_key
)
# 返回: JointState对象（仅包含指定机器人的关节）

# 2. 获取末端执行器位姿（带速度）
ee_pose = motion_factory.get_frame_pose_with_joint_state(
    all_joint_states, ee_link_name, robot_key, need_vel=True
)
# 返回: np.ndarray, shape=(13,)
#   - [:7]: pose [x, y, z, qx, qy, qz, qw]
#   - [7:13]: twist [vx, vy, vz, wx, wy, wz]

# 3. 获取坐标系位姿（不带速度）
frame_pose = motion_factory.get_frame_pose(frame_name, robot_key)
# 返回: np.ndarray, shape=(7,) [x, y, z, qx, qy, qz, qw]
```

#### 动作获取
```python
# 4. 获取最新动作命令
motion_action = motion_factory.get_latest_action()
# 返回: Dict[str, Dict]
#   - key: robot_key
#   - value: {
#       "joint": {"position": List[7], "time_stamp": float},
#       "ee": {"pose": List[7], "time_stamp": float}
#     }
```

#### 运动控制
```python
# 5. 更新高级命令（末端位姿目标）
motion_factory.update_high_level_command(high_level_command)
# 输入: np.ndarray
#   - 单臂: shape=(7,) [x, y, z, qx, qy, qz, qw]
#   - 双臂: shape=(14,) [left_pose(7), right_pose(7)]

# 6. 设置下一个位姿目标（带轨迹规划）
motion_factory.set_next_pose_target(target_pose, robot_key)
# 输入: np.ndarray, shape=(7,)
```

---

### EpisodeWriter 接口

#### 初始化
```python
from dataset.lerobot.data_process import EpisodeWriter

recorder = EpisodeWriter(
    task_dir="/path/to/task",          # 任务根目录
    frequency=30,                       # 数据采集频率
    image_size=[640, 480],              # 图像尺寸 [width, height]
    rerun_log=True,                     # 是否启用Rerun可视化
    version="1.0.0",                    # 数据格式版本
    date="2025-09-30",                  # 采集日期
    author="HIROL",                     # 采集者
    task_description="任务描述",
    task_description_goal="任务目标",
    task_description_steps="任务步骤"
)
```

#### 使用流程
```python
# 1. 创建新episode
success = recorder.create_episode()
if not success:
    print("Episode创建失败，请等待上一个episode保存完成")

# 2. 添加数据（循环调用）
recorder.add_item(
    colors=colors_dict,            # Dict[str, Dict]
    depths=depths_dict,            # Dict[str, Dict] or None
    joint_states=joint_states_dict, # Dict[str, Dict]
    ee_states=ee_states_dict,      # Dict[str, Dict]
    tools=gripper_state_dict,      # Dict[str, Dict]
    tactiles=tactiles_dict,        # Dict[str, Dict] or None
    imus=imus_dict,                # Dict[str, Dict] or None
    audios=None,                   # 当前未使用
    actions=actions_dict           # Dict[str, Dict]
)

# 3. 保存episode
recorder.save_episode()

# 4. 关闭recorder（程序退出前）
recorder.close()
```

**注意事项**：
- `add_item()` 内部使用**异步队列+工作线程**处理数据，不会阻塞主线程
- `save_episode()` 仅设置保存标志，实际保存由工作线程完成
- 在 `save_episode()` 完成前，不能调用 `create_episode()`
- 所有数组数据（除图像/音频）需转换为Python list（不能是np.ndarray）

---

## 数据采集最佳实践

### 1. 双线程架构（推荐）

**设计模式**：
```python
class DataCollectionTask:
    def __init__(self):
        self._recording_enabled = False
        self._recording_thread = None

    def start_recording(self):
        self._recording_enabled = True
        self._recording_thread = threading.Thread(
            target=self._data_collection_loop
        )
        self._recording_thread.start()

    def _data_collection_loop(self):
        """独立的数据采集循环（30Hz）"""
        start_time = time.time()
        while self._recording_enabled:
            # 采集所有传感器数据
            self._collect_and_save_data()

            # 频率控制
            elapsed = time.time() - start_time
            sleep_time = (1.0 / 30) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            start_time = time.time()

    def stop_recording(self):
        self._recording_enabled = False
        self._recording_thread.join()
```

**优点**：
- 主线程处理控制逻辑（50Hz）
- 数据线程独立采集传感器（30Hz）
- 避免传感器读取阻塞控制循环

---

### 2. 动作同步机制

**问题**：主线程发送命令，数据线程需记录动作

**解决方案**：线程安全的动作缓存
```python
class DataCollectionTask:
    def __init__(self):
        self._latest_tool_action = {}
        self._tool_action_lock = threading.Lock()

    def send_tool_command(self, tool_target):
        """主线程调用"""
        self.robot_factory.set_tool_command(tool_target)

        # 缓存工具动作
        with self._tool_action_lock:
            for key, command in tool_target.items():
                self._latest_tool_action[key] = {
                    "tool": {
                        "position": command,
                        "time_stamp": time.perf_counter()
                    }
                }

    def _collect_and_save_data(self):
        """数据线程调用"""
        # 获取运动动作
        motion_action = self.motion_factory.get_latest_action()

        # 合并工具动作
        with self._tool_action_lock:
            actions = copy.deepcopy(self._latest_tool_action)
            for key in actions.keys():
                actions[key]["joint"] = motion_action[key]["joint"]
                actions[key]["ee"] = motion_action[key]["ee"]

        # 保存数据
        self.recorder.add_item(..., actions=actions)
```

---

### 3. 频率控制

**推荐配置**：
- **控制循环**：50Hz（MotionFactory 控制器频率）
- **数据采集**：30Hz（图像处理瓶颈）
- **硬件控制**：800Hz（RobotFactory 异步控制，自动）

**实现方式**：
```python
# 精确定时循环（避免累积误差）
target_period = 1.0 / 30  # 30Hz
next_run_time = time.perf_counter()

while running:
    loop_start = time.perf_counter()

    # 数据采集逻辑
    collect_data()

    # 计算下一次运行时间
    next_run_time += target_period
    sleep_time = next_run_time - time.perf_counter()

    if sleep_time > 0:
        time.sleep(sleep_time)
    else:
        # 处理超时
        log.warning(f"Loop overrun: {-sleep_time:.3f}s")
        next_run_time = time.perf_counter()
```

---

### 4. 错误处理

**数据完整性检查**：
```python
def _collect_and_save_data(self):
    # 1. 检查motion_action是否有效
    motion_action = self.motion_factory.get_latest_action()
    if motion_action is None:
        log.warning("Motion action not available, skipping frame")
        return

    # 2. 检查tool_action是否同步
    if len(self._latest_tool_action) == 0:
        log.warning("Tool action not synchronized, skipping frame")
        return

    # 3. 采集数据
    try:
        colors = self._get_colors()
        joint_states = self._get_joint_states()
        # ...

        self.recorder.add_item(
            colors=colors,
            joint_states=joint_states,
            # ...
        )
    except Exception as e:
        log.error(f"Data collection error: {e}")
        # 不中断循环，继续下一帧
```

---

### 5. 可视化

**OpenCV 实时显示**：
```python
# 组合多相机图像
if len(image_list) > 0:
    combined_imgs = image_list[0]
    for img in image_list[1:]:
        combined_imgs = combine_image(combined_imgs, img)
    cv2.imshow('Robot Cameras', combined_imgs)
    cv2.waitKey(1)
```

**Rerun 可视化**：
- `EpisodeWriter` 自动集成 `RerunLogger`
- 设置 `rerun_log=True` 即可启用
- 默认配置：60帧缓冲，300MB内存限制

---

## 实现参考

### 参考代码位置
1. **数据采集主逻辑**: [HIROLRobotPlatform/factory/tasks/robot_teleoperation.py](../factory/tasks/robot_teleoperation.py)
   - `add_teleoperation_data()` 方法 (L262-361)
   - 双线程架构 (L101-102)
   - 动作同步机制 (L228-238)

2. **数据写入器**: [HIROLRobotPlatform/dataset/lerobot/data_process.py](../dataset/lerobot/data_process.py)
   - `EpisodeWriter` 类 (L18-289)
   - `add_item()` 接口 (L142-164)
   - 异步保存机制 (L166-182)

3. **工厂接口**:
   - `RobotFactory`: [HIROLRobotPlatform/factory/components/robot_factory.py](../factory/components/robot_factory.py)
   - `MotionFactory`: [HIROLRobotPlatform/factory/components/motion_factory.py](../factory/components/motion_factory.py)

### 数据样例
- 样例数据集: `/home/hanyu/code/HIROLRobotPlatform/dataset/data/peg_in_hole/episode_0001/`
- 查看 `data.json` 了解完整数据结构

---

## 版本历史
- **v1.0.0** (2025-09-30): 初版，基于 TeleoperationFactory 数据采集实现