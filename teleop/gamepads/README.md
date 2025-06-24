XBOX手柄连接，usb/蓝牙，自行连接好。

test sim
```
python teleop/gamepads/test_xbox_teleop.py --no_camera
```

PS4手柄配对模式：
bash# 1. 关闭手柄（如果已连接其他设备）
# 2. 同时按住 Share + PS按钮 约3-5秒
# 3. 灯条会快速闪烁白光，表示进入配对模式
PS5手柄配对模式：
bash# 1. 关闭手柄
# 2. 同时按住 Create按钮 + PS按钮 约3-5秒
# 3. 灯条会快速闪烁蓝光，表示进入配对模式

# PS5 Robot Teleop

PS5手柄遥控Monte01机器人的实现，支持仿真/真机一键切换。

## 功能特性

- **双臂控制**: 支持左右手臂切换控制
- **精确操作**: 6DOF位置和姿态控制
- **夹爪控制**: 开合夹爪操作
- **安全保护**: 急停功能和控制使能开关
- **Sim2Real**: 一键切换仿真/真机模式

## 控制映射

### 按键控制
- **X**: 切换控制使能/禁用
- **Circle**: 急停（禁用控制）
- **Triangle**: 重置到起始位置
- **Square**: 切换左右手臂
- **L1**: 关闭夹爪
- **R1**: 打开夹爪
- **Share**: 移动到Home位置

### 摇杆控制
- **左摇杆**: X/Y方向平移
- **右摇杆**: Z方向平移 / Z轴旋转
- **L2/R2扳机**: 精细Z方向控制

## 使用方法

### 仿真模式
```bash
cd teleop/gamepads/ps5
python robot_teleop.py
```

### 真机模式
```bash
cd teleop/gamepads/ps5
python robot_teleop.py --use_real_robot
```

## 安全注意事项

1. **首次使用前**，确保在仿真模式下熟悉控制
2. **真机模式下**，确保机器人周围无障碍物
3. **始终准备**按Circle键进行急停
4. **控制前**需按X键启用控制（默认禁用）

## 参数调整

在`robot_teleop.py`中可调整以下参数：
- `movement_scale`: 移动缩放系数（默认2cm）
- `rotation_scale`: 旋转缩放系数（默认0.1弧度）
- `move_cooldown`: 移动间隔（默认50ms）
- `deadzone`: 摇杆死区（默认0.15）

## 依赖要求

- pygame
- numpy
- 项目硬件抽象层（hardware/monte01/）