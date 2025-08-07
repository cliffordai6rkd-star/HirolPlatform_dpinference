# Cartesian Impedance Controller / 笛卡尔阻抗控制器

## 概述 / Overview

Cartesian Impedance Controller 是一个继承自 `controller_base` 的 Python 实现，提供笛卡尔空间的阻抗控制功能。该控制器设计目标是匹配 SERL 的 `cartesian_impedance_controller.cpp` 功能，同时保持解耦设计，易于集成到 HIROLRobotPlatform 中。

The Cartesian Impedance Controller is a Python implementation that inherits from `controller_base` and provides impedance control in Cartesian space. This controller is designed to match the functionality of SERL's `cartesian_impedance_controller.cpp` while being decoupled and easy to integrate with the HIROLRobotPlatform.

## 功能特性 / Features

- **笛卡尔空间控制 / Cartesian Space Control**: 在笛卡尔空间控制机器人末端执行器的位置和姿态
- **阻抗控制 / Impedance Control**: 实现可配置刚度和阻尼的弹簧-阻尼器行为
- **误差裁剪 / Error Clipping**: 裁剪位置和姿态误差以保证稳定性
- **积分控制 / Integral Control**: 可选的积分项用于稳态误差补偿
- **零空间控制 / Nullspace Control**: 通过零空间刚度进行冗余解析
- **力矩速率限制 / Torque Rate Limiting**: 可配置速率限制实现平滑力矩过渡
- **重力补偿 / Gravity Compensation**: 可选的重力补偿
- **科里奥利力补偿 / Coriolis Compensation**: 动力学补偿包含科里奥利和离心力
- **在线参数调整 / Online Parameter Adjustment**: 动态调整刚度和阻尼参数

## 数学原理 / Mathematical Formulation

控制器实现以下控制律：

```
τ = J^T * F_task + τ_nullspace + τ_gravity + τ_coriolis

其中:
F_task = -K_p * x_error - K_d * ẋ_error - K_i * ∫x_error
τ_nullspace = (I - J^T * J_pinv^T) * (k_null * q_error - d_null * q̇)
τ_gravity = g(q)
τ_coriolis = C(q,q̇) * q̇
```

### 误差计算 / Error Computation

1. **位置误差**: `e_pos = x_current - x_desired`
2. **姿态误差**: 使用四元数差分计算，转换为轴角表示
3. **误差裁剪**: 限制误差范围防止不稳定

## 配置说明 / Configuration

控制器通过 YAML 文件配置 (`cartesian_impedance_fr3_cfg.yaml`):

```yaml
cartesian_impedance:
  enable_gravity_compensation: true  # 启用重力补偿
  
  # 刚度参数 / Stiffness parameters
  translational_stiffness: 200.0  # 平移刚度 N/m
  rotational_stiffness: 20.0      # 旋转刚度 Nm/rad
  
  # 阻尼参数 / Damping parameters
  translational_damping: 20.0     # 平移阻尼 Ns/m
  rotational_damping: 5.0         # 旋转阻尼 Nms/rad
  
  # 积分增益参数 / Integral gain parameters
  translational_ki: 0.0           # 平移积分增益
  rotational_ki: 0.0              # 旋转积分增益
  
  # 零空间控制 / Nullspace control
  nullspace_stiffness: 20.0              # 零空间刚度
  joint1_nullspace_stiffness: 20.0       # 第一关节零空间刚度
  
  # 误差裁剪 / Error clipping
  translational_clip_min: [-0.1, -0.1, -0.1]  # 最小平移误差 (m)
  translational_clip_max: [0.1, 0.1, 0.1]      # 最大平移误差 (m)
  rotational_clip_min: [-0.3, -0.3, -0.3]      # 最小旋转误差 (rad)
  rotational_clip_max: [0.3, 0.3, 0.3]        # 最大旋转误差 (rad)
  
  # 力矩速率饱和 / Torque rate saturation
  delta_tau_max: 1.0  # 最大力矩变化率 Nm/s
  
  # 平滑过渡滤波参数 / Filter parameter
  filter_params: 0.005  # 滤波系数 (0-1)
  
  # 关节力矩饱和 / Joint torque saturation
  saturation:
    min: [-40, -40, -40, -40, -40, -40, -40]  # 最小力矩
    max: [40, 40, 40, 40, 40, 40, 40]        # 最大力矩
  
  # 可选：指定手臂关节索引 / Optional: arm joint indices
  # arm_joint_idxes: [0, 1, 2, 3, 4, 5, 6]
```

## 使用方法 / Usage

### 基本使用 / Basic Usage

```python
from controller.cartesian_impedance_controller import CartesianImpedanceController
from motion.pin_model import RobotModel
from hardware.base.utils import RobotJointState
from tools.yaml_loader import load_yaml

# 加载配置 / Load configuration
config = load_yaml("controller/config/cartesian_impedance_fr3_cfg.yaml")
model_config = load_yaml("motion/config/robot_model_fr3_cfg.yaml")
robot_model = RobotModel(model_config["fr3_only"])

# 创建控制器 / Create controller
controller = CartesianImpedanceController(
    config["cartesian_impedance"], 
    robot_model
)

# 设置目标位姿 (7D: [x, y, z, qx, qy, qz, qw])
target = {
    "fr3_ee": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
}

# 获取当前机器人状态 / Get current robot state
robot_state = get_robot_state()  # 返回 RobotJointState 对象

# 计算控制指令 / Compute control command
success, torque_command, mode = controller.compute_controller(
    target, 
    robot_state
)

# 应用力矩到机器人 / Apply torques to robot
if success:
    apply_torques(torque_command)  # mode 将是 'torque'
```

### 动态参数调整 / Dynamic Parameter Adjustment

```python
# 调整刚度 / Adjust stiffness
controller.set_stiffness(
    translational_stiffness=300.0, 
    rotational_stiffness=30.0
)

# 调整阻尼 / Adjust damping
controller.set_damping(
    translational_damping=25.0, 
    rotational_damping=7.0
)

# 重置积分误差累加器 / Reset integral error accumulator
controller.reset_integral_error()
```

### 完整示例 / Complete Example

```python
import numpy as np
from simulation.mujoco.mujoco_sim import MujocoSim

# 初始化仿真和控制器
mujoco_config = load_yaml("simulation/config/mujoco_fr3_scene.yaml")
mujoco = MujocoSim(mujoco_config["mujoco"])

# 控制循环
while True:
    # 获取目标位姿
    target_pose = mujoco.get_site_pose("target_site", "xyzw")
    target = {model.ee_link: target_pose}
    
    # 获取机器人状态
    robot_state = mujoco.get_joint_states()
    
    # 计算控制力矩
    success, torques, mode = controller.compute_controller(
        target, 
        robot_state
    )
    
    # 应用控制
    if success:
        mujoco.set_joint_command([mode] * 7, torques)
```

## 与 C++ 实现的主要差异 / Key Differences from C++ Implementation

1. **基于 Python / Python-based**: 更易与 Python 机器人框架集成
2. **Pinocchio 后端 / Pinocchio Backend**: 使用 Pinocchio 进行运动学/动力学计算，而非 libfranka
3. **解耦设计 / Decoupled Design**: 无直接 ROS 依赖，可与任何机器人接口配合使用
4. **简化接口 / Simplified Interface**: 遵循 controller_base 模式的更清晰 API

## 测试 / Testing

运行测试脚本验证控制器：

```bash
# 基础仿真测试 / Basic simulation test
python test/test_cartesian_impedance_controller.py

# 功能测试 / Feature test
python test/test_cartesian_impedance_controller.py --test-features
```

## 实现细节 / Implementation Details

### 控制流程 / Control Flow

1. **更新运动学**: 使用当前关节状态更新机器人运动学
2. **计算误差**: 计算笛卡尔空间的位置和姿态误差
3. **应用裁剪**: 限制误差范围防止不稳定
4. **计算笛卡尔力**: F = -Kp*e - Kd*ė - Ki*∫e
5. **雅可比转换**: τ_task = J^T * F
6. **零空间控制**: 添加零空间力矩以优化关节配置
7. **动力学补偿**: 添加重力和科里奥利力补偿
8. **力矩饱和**: 应用速率和幅值限制

### 参数调优指南 / Parameter Tuning Guide

1. **刚度 (Kp)**:
   - 较高值：更硬的行为，更快的响应
   - 较低值：更柔顺的行为，更慢的响应
   - 建议：平移 100-1000 N/m，旋转 10-100 Nm/rad

2. **阻尼 (Kd)**:
   - 临界阻尼：Kd = 2*sqrt(Kp*M)，其中 M 是有效质量
   - 过阻尼：减少振荡但响应变慢
   - 欠阻尼：快速响应但可能振荡

3. **积分增益 (Ki)**:
   - 用于消除稳态误差
   - 过高会导致不稳定
   - 建议从 0 开始，逐步增加

## 故障排除 / Troubleshooting

### 常见问题 / Common Issues

1. **控制器不收敛**:
   - 检查刚度/阻尼比
   - 验证目标位姿是否可达
   - 确认运动学模型正确

2. **振荡**:
   - 降低刚度或增加阻尼
   - 检查控制频率是否足够高
   - 验证传感器噪声水平

3. **稳态误差**:
   - 启用积分控制 (设置 Ki > 0)
   - 检查重力补偿是否正确
   - 验证摩擦补偿

4. **力矩限制**:
   - 降低刚度或误差裁剪范围
   - 检查目标轨迹是否过激
   - 调整力矩饱和限制

### 调试建议 / Debug Tips

```python
# 打印调试信息
print(f"Position error: {np.linalg.norm(error[:3]):.4f} m")
print(f"Orientation error: {np.linalg.norm(error[3:]):.4f} rad")
print(f"Max torque: {np.max(np.abs(torque_command)):.2f} Nm")

# 监控积分误差
print(f"Integral error: {controller.error_integral}")

# 检查雅可比条件数
J = robot_model.get_jacobian(frame_name, q)
print(f"Jacobian condition number: {np.linalg.cond(J):.2f}")
```

## API 参考 / API Reference

### CartesianImpedanceController

#### 方法 / Methods

- `__init__(config, robot_model)`: 初始化控制器
- `compute_controller(target, robot_state)`: 计算控制指令
- `set_stiffness(translational_stiffness, rotational_stiffness)`: 设置刚度
- `set_damping(translational_damping, rotational_damping)`: 设置阻尼  
- `reset_integral_error()`: 重置积分误差

#### 属性 / Properties

- `translational_stiffness`: 平移刚度
- `rotational_stiffness`: 旋转刚度
- `translational_damping`: 平移阻尼
- `rotational_damping`: 旋转阻尼
- `error_integral`: 当前积分误差
