# XArm Agent 通信测试脚本

本目录包含基于 `hardware/monte01/agent.py` 接口的 XArm API 通信测试脚本。

## 脚本说明

### 1. test_xarm_agent_communication.py
完整的通信和运动测试脚本，包含详细的测试功能。

**功能特性：**
- 左右臂连接测试
- 躯干通信测试  
- 基本运动测试（可选）
- 双臂同步测试
- 详细的日志输出

**使用方法：**
```bash
# 基本连接测试
python test_xarm_agent_communication.py --verbose

# 包含运动测试
python test_xarm_agent_communication.py --test-movement --verbose

# 仅测试手臂
python test_xarm_agent_communication.py --arms-only --verbose

# 仅测试躯干
python test_xarm_agent_communication.py --trunk-only --verbose

# 仅模拟器模式
python test_xarm_agent_communication.py --simulation --verbose
```

### 2. test_xarm_quick.py
快速验证脚本，用于快速检查基本通信功能。

**功能特性：**
- 快速初始化检查
- 基本连接验证
- 简洁的输出格式

**使用方法：**
```bash
# 快速测试（真实机器人）
python test_xarm_quick.py

# 快速测试（仅模拟器）
python test_xarm_quick.py --simulation

# 自定义配置文件
python test_xarm_quick.py --config path/to/config.yaml
```

## 测试内容

### 左右臂测试
- **连接验证**：检查 XArm API 连接状态
- **关节位置读取**：获取当前关节角度
- **TCP位置计算**：验证运动学计算
- **夹爪状态**：检查夹爪可用性
- **基本运动**：小幅度关节运动测试

### 躯干测试
- **身体关节读取**：获取腰部和膝关节位置
- **坐标变换**：验证世界坐标到胸部坐标的变换
- **关节控制**：测试躯干关节运动

### 双臂同步测试
- **状态同步**：将真实机器人状态同步到模拟器
- **协调控制**：验证双臂协同工作能力

## 配置要求

### 网络配置
- 左臂 XArm API: `192.168.11.11`
- 右臂 XArm API: `192.168.11.12` 
- 躯干 RobotLib: `192.168.11.3:50051`

### 配置文件
测试脚本默认使用 `hardware/monte01/config/agent.yaml` 配置文件。

### 依赖项
- `xarm-python-sdk`: XArm Python SDK
- `pyyaml`: 配置文件解析
- `numpy`: 数值计算
- 项目内部模块：`hardware.monte01.agent`

## 安全注意事项

⚠️ **运动测试安全提醒：**
- 运动测试仅进行小幅度关节移动（0.1 弧度或 0.05 弧度）
- 运动前请确保机器人周围环境安全
- 如需禁用运动测试，不要使用 `--test-movement` 参数
- 建议首次使用时先在仅模拟器模式下测试

## 故障排除

### 常见错误
1. **连接超时**：检查网络连接和 IP 地址配置
2. **XArm SDK 未安装**：`pip install xarm-python-sdk`
3. **配置文件错误**：验证 YAML 语法和路径
4. **初始化超时**：增加 `--timeout` 值

### 调试技巧
- 使用 `--verbose` 参数获取详细日志
- 使用 `--simulation` 参数验证代码逻辑
- 检查防火墙和网络连接设置

## 返回值
- **0**: 所有测试通过
- **1**: 测试失败或发生错误

## 示例输出

```
==============================================================
XArm Agent Quick Communication Test
==============================================================
✓ Config loaded from: hardware/monte01/config/agent.yaml
🔄 Initializing Agent (XArm API, Real Robot: True)...
⏳ Waiting for initialization...
✓ Agent initialized successfully

🔄 Testing left arm...
✓ Left arm: 7 joints, TCP pose: (4, 4)
  Gripper: available

🔄 Testing right arm...
✓ Right arm: 7 joints, TCP pose: (4, 4)
  Gripper: available

🔄 Testing trunk...
✓ Trunk: 3 body joints, transform: (4, 4)

🔄 Testing dual arm sync...
✓ Dual arm sync completed

==============================================================
🎉 All tests passed successfully!
==============================================================
```