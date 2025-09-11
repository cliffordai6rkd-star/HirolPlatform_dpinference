# HIROLRobotPlatform 触觉传感器系统

完整的机器人触觉感知解决方案，支持多厂商硬件、本地/网络通信，以及统一的数据接口。

## 📋 目录

- [系统概述](#系统概述)
- [架构设计](#架构设计)
- [支持硬件](#支持硬件)
- [快速开始](#快速开始)
- [配置系统](#配置系统)
- [API文档](#api文档)
- [集成指南](#集成指南)
- [测试工具](#测试工具)
- [扩展开发](#扩展开发)
- [故障排除](#故障排除)

## 系统概述

### ✨ 核心特性

- 🏗️ **统一架构**：基于TactileBase的模块化设计，支持多厂商扩展
- 🔌 **多通信方式**：串口直连、网络传输
- ⚙️ **智能配置**：自动参数生成、配置验证、场景模板
- 🌐 **网络分布**：ZMQ高性能传输、多传感器并发、自动重连
- 📡 **分控制盒广播**：每个控制盒独立数据流，支持全身多部位传感器
- 🏷️ **智能标识**：自动生成sensor_id，支持灵活命名（left_hand, torso等）
- 🧪 **完整测试**：硬件诊断、性能基准、兼容性验证
- 🤖 **无缝集成**：HIROLRobotPlatform原生支持、工厂模式配置

### 🎯 应用场景

| 场景 | 配置 | 典型用例 |
|------|------|----------|
| **单机器人操作** | 串口直连 | 精密装配、物体识别 |
| **分布式系统** | 网络传输 | 多机器人协作、远程操控 |
| **双手机器人** | 多控制盒 | 双臂协调、复杂抓取 |
| **人形机器人** | 全身覆盖 | 人机交互、环境感知 |

## 架构设计

### 🏗️ 系统架构图

```
                    HIROLRobotPlatform
                    ┌─────────────────────────────────┐
                    │      Robot Factory              │
                    │  ┌─────────────────────────────┐ │
                    │  │     Sensor Manager          │ │
                    │  │  ┌─────────────────────────┐│ │
应用层              │  │  │    TactileBase          ││ │
                    │  │  │  (统一触觉接口)          ││ │
                    │  │  └─────────────────────────┘│ │
                    │  └─────────────────────────────┘ │
                    └─────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
            ┌───────▼──────┐ ┌────────▼────────┐ ┌─────▼─────┐
实现层      │PaxiniSerial  │ │PaxiniNetwork    │ │Future     │
            │Sensor        │ │Sensor           │ │Sensors    │
            └───────┬──────┘ └────────┬────────┘ └───────────┘
                    │                 │
            ┌───────▼──────┐ ┌────────▼────────┐
通信层      │Serial/USB    │ │ZMQ Network      │
            │Protocol      │ │Protocol         │
            └───────┬──────┘ └────────┬────────┘
                    │                 │
            ┌───────▼──────┐ ┌────────▼────────┐
硬件层      │Paxini        │ │Remote Paxini    │
            │Hardware      │ │Server           │
            └──────────────┘ └─────────────────┘
```

### 🚀 分控制盒广播架构（v1.1新功能）

**解决的核心问题：** 网络触觉数据混合传输 → 独立控制盒数据流

**之前（v1.0）：**
```
Server → [所有控制盒混合数据] → 所有客户端接收相同数据 ❌
     PAXINI_04_MODULES (左手+右手混合)
```

**现在（v1.1）：**
```
Server → [左手数据 PAXINI_LEFT_HAND_02_MODULES]  → 左手传感器 ✅
       → [右手数据 PAXINI_RIGHT_HAND_02_MODULES] → 右手传感器 ✅
       → [躯干数据 PAXINI_TORSO_04_MODULES]      → 躯干传感器 ✅
```

**技术实现：**
- `collect_data()` 返回按控制盒分组的数据字典
- `generate_box_sensor_id()` 根据控制盒name生成独立ID
- `pack_box_message()` 为每个控制盒独立打包ZMQ消息
- 支持灵活命名：left_hand, right_hand, torso, left_leg等

**应用效果：**
- ✅ Factory模式可独立管理多个传感器实例
- ✅ 左右手传感器接收独立数据流，无相互干扰
- ✅ 支持扩展到全身多部位触觉感知
- ✅ 99.1 FPS稳定传输性能，接近100Hz服务器频率

### 🧩 模块结构

```
hardware/sensors/paxini_tactile/
├── 📁 config/                    # 配置系统
│   ├── 🏭 factory/               # HIROLRobotPlatform集成配置
│   │   ├── paxini_left_arm.yaml
│   │   ├── paxini_right_arm.yaml
│   │   ├── paxini_network_left_arm.yaml
│   │   ├── paxini_network_right_arm.yaml
│   │   └── test_client_config.yaml
│   ├── 🖥️ server/                # 独立服务器配置
│   │   ├── production_server.yaml
│   │   ├── debug_server.yaml
│   │   ├── lab_dual_hand_server.yaml
│   │   └── full_body_server_example.yaml  # 全身传感器示例
│   ├── 📦 common/                # 通用基础配置
│   │   ├── single_sensor.yaml
│   │   └── network_client.yaml
│   ├── 📚 examples/              # 详细示例配置
│   │   └── paxini_config_examples.yaml
│   └── 📖 README.md              # 配置系统说明
├── 📁 test/                      # 测试工具链
│   ├── test_paxini_connection.py     # 硬件连接诊断
│   ├── test_network_demo.py          # 完整网络演示(服务器+客户端)
│   ├── test_network_client.py        # 网络客户端性能测试(支持--config)
│   ├── test_config_compatibility.py  # 配置兼容性验证
│   └── test_sensor_id_auto_generation.py # ID生成测试
├── 📄 paxini_serial_sensor.py    # 串口传感器实现
├── 📄 paxini_network_sensor.py   # 网络传感器实现  
├── 📄 paxini_standalone_server.py # 独立ZMQ服务器
├── 📄 generate_unified_config.py # 配置生成工具
├── 📄 QUICK_START.md             # 快速入门指南
└── 📄 README.md                  # 本文档

hardware/base/
├── 📄 tactile_base.py            # 触觉传感器基类
└── 📄 utils.py                   # 共享工具类 (PaxiniState等)
```

## 支持硬件

### 🤖 Paxini触觉传感器

**支持型号：**

| 系列 | 型号 | 控制模式 | 接口位置 | 应用场景 |
|------|------|----------|----------|----------|
| **GEN2 (推荐)** |
| GEN2-IP-L5325 | 5 | Port 0 | 指尖精密感知 |
| GEN2-IP-M3025 | 5 | Port 0 | 中型指尖传感器 |
| GEN2-MP-M2324 | 5 | Port 1 | 掌心力感知 |
| GEN2-DP-L3530 | 5 | Port 2 | 手背接触检测 |
| GEN2-DP-M2826 | 5 | Port 2 | 中型手背传感器 |
| GEN2-DP-S2716 | 1 | Port 0 | 特殊应用传感器 |
| **GEN1 (兼容)** |
| GEN1-IP-S2516 | 2 | Port 0 | 第一代指尖传感器 |
| GEN1-DP-S2716 | 2 | Port 1 | 第一代手背传感器 |

**技术规格：**
- 📊 **分辨率**：120触觉点/模块，三轴力检测
- ⚡ **采样率**：最高2KHz，典型1KHz
- 🔗 **通信**：RS485串口，460800波特率
- 🌡️ **工作温度**：-10°C ~ +60°C
- 💧 **防护等级**：IP54（可选IP67）

### 🔌 连接拓扑

**单控制盒配置：**
```
┌─────────────────┐    USB    ┌──────────────┐
│ Control Box     │◄──────────┤ Robot PC     │
│ ├─ CN1: Empty   │           │ /dev/ttyACM0 │
│ ├─ CN2: Empty   │           └──────────────┘
│ ├─ CN3: Empty   │
│ ├─ CN4: Empty   │
│ ├─ CN5: Sensor A│ ← GEN2-IP-L5325
│ └─ CN6: Sensor B│ ← GEN2-IP-L5325
└─────────────────┘
```

**多传感器部署（v1.1新架构）：**
```
┌─────────────────┐    USB    ┌──────────────┐
│ Left Control Box│◄──────────┤ Robot PC     │
│ ├─ CN5: Finger │           │ /dev/ttyACM0 │
│ └─ CN6: Finger │           │              │
└─────────────────┘           │ /dev/ttyACM1 │
     │                         └──────────────┘
     │                                 ▲
     v1.1: 独立传感器实例                    │
     │                                 │
┌─────────────────┐    USB            │
│Right Control Box│◄────────────────┘
│ ├─ CN5: Finger │
│ └─ CN6: Finger │
└─────────────────┘

架构特点：v1.1中每个控制盒对应一个独立的传感器实例
```

## 快速开始

### 🚀 10分钟上手

**1️⃣ 环境检查**
```bash
# 检查Python环境
python3 --version  # >= 3.8

# 检查依赖
pip install numpy pyyaml pyzmq glog

# 检查硬件连接
ls /dev/ttyACM*  # 应显示 /dev/ttyACM0 等
```

**2️⃣ 硬件连接测试**
```bash
cd /path/to/HIROLRobotPlatform/hardware/sensors/paxini_tactile

# 快速连接测试
python3 test/test_paxini_connection.py --port /dev/ttyACM0

# 预期输出：
# ✅ 成功连接到Paxini控制盒
# 📊 传感器信息: 控制盒数量: 1, 总模块数: 2
# 📈 数据质量: 最大值: 3.2, 均值: 0.1
```

**3️⃣ 基础串口使用**
```python
# 直接使用示例
from hardware.sensors.paxini_tactile.paxini_serial_sensor import PaxiniSerialSensor

config = {
    "communication_type": "serial",
    "control_box": {
        "port": "/dev/ttyACM0",
        "baudrate": 460800,
        "control_mode": 5,
        "sensors": [{"model": "GEN2-IP-L5325", "connect_ids": [5, 6]}]
    }
}

sensor = PaxiniSerialSensor(config)
sensor.initialize()

success, data, timestamp = sensor.read_tactile_data()
print(f"数据形状: {data.shape}")  # (2, 120, 3)
print(f"最大力值: {data.max():.2f}")
```

**4️⃣ 网络传感器部署**
```bash
# 服务器端（连接硬件）
cp config/server/production_server.yaml my_server.yaml
# 编辑 my_server.yaml 修改串口路径
python3 paxini_standalone_server.py --config my_server.yaml

# 客户端（应用端）
cp config/factory/test_client_config.yaml my_client.yaml  
# 编辑 my_client.yaml 修改服务器IP
python3 test/test_network_client.py --config my_client.yaml
```

**5️⃣ 分控制盒广播测试（v1.1新功能）**
```bash
# 服务器端：双手传感器分别广播
python3 paxini_standalone_server.py --config config/server/lab_dual_hand_server.yaml

# 预期输出：
# 🚀 Broadcasting 2 separate control box streams
# 📊 Stats - Rate: 50.0Hz | Boxes: 2

# 客户端：左手传感器独立接收
python3 test/test_network_client.py --config config/factory/paxini_network_left_arm.yaml

# 预期输出：
# 🏷️ Sensor ID: PAXINI_LEFT_HAND_02_MODULES  
# 📊 Final Results: Average FPS: 99.1 (接近100Hz服务器频率)
# 📐 Data shape: (2, 120, 3) - 仅左手2个模块，不包含右手数据
```

## 配置系统

### 📁 配置架构原则

**🎯 核心设计：一个场景一个配置文件**

```bash
# ✅ 正确：独立配置文件
config/factory/paxini_fr3_left_hand.yaml    # FR3左手专用
config/server/production_server.yaml        # 生产服务器专用  
config/common/single_sensor.yaml            # 基础测试专用

# ❌ 错误：混合配置文件
old_design.yaml:
  paxini_fr3_left_hand: {...}     # 多个配置混在一起
  production_server: {...}        # 维护困难，容易冲突
```

### 🔧 自动配置机制

**sensor_id 智能生成：**
```python
# 客户端配置 - 无需手动指定ID
communication_type: "network"
ip: "192.168.1.100"
# sensor_id 自动生成：PAXINI_02_MODULES (基于下面的模块数)

control_box:
  sensors:
    - model: "GEN2-IP-L5325"
      connect_ids: [5, 6]  # → 2个模块 → 自动生成ID
```

**配置提取机制：**
```yaml
# HIROLRobotPlatform工厂模式
sensor_dicts:
  tactile:
    - name: "left_tactile"
      type: "paxini_serial_sensor"  # ← 传感器类型
      cfg: !include config/factory/paxini_fr3_left_hand.yaml
      
# config/factory/paxini_fr3_left_hand.yaml内容：
paxini_serial_sensor:  # ← 顶级标题 = 传感器类型，工厂自动提取
  communication_type: "serial"
  # ... 实际配置内容
```

### 📖 配置选择指南

| 使用场景 | 配置文件 | 说明 |
|---------|----------|------|
| **快速测试** | `config/common/single_sensor.yaml` | 最简配置 |
| **网络接收** | `config/common/network_client.yaml` | 基础网络客户端 |
| **网络测试** | `config/factory/test_client_config.yaml` | 测试客户端专用 |
| **生产部署** | `config/server/production_server.yaml` | 稳定服务器 |
| **调试开发** | `config/server/debug_server.yaml` | 低频调试 |
| **实验室服务器** | `config/server/lab_dual_hand_server.yaml` | 双手分控制盒服务器(v1.1) |
| **全身传感器** | `config/server/full_body_server_example.yaml` | 多部位传感器示例(v1.1) |
| **FR3集成** | `config/factory/paxini_fr3_left_hand.yaml` | 工厂模式 |
| **网络左手** | `config/factory/paxini_network_left_hand.yaml` | 分控制盒客户端(v1.1) |
| **网络右手** | `config/factory/paxini_network_right_hand.yaml` | 分控制盒客户端(v1.1) |
| **双臂机器人** | Factory配置多个传感器实例 | 多传感器 |
| **详细示例** | `config/examples/paxini_config_examples.yaml` | 学习参考 |

## API文档

### 🔌 TactileBase 基类

**位置：** `hardware/base/tactile_base.py`

```python
class TactileBase(abc.ABC):
    """触觉传感器统一基类"""
    
    # 核心数据接口
    def read_tactile_data(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """读取触觉数据
        
        Returns:
            success (bool): 读取是否成功
            data (np.ndarray): 触觉数据 shape=(n_modules, n_taxels, 3)
                             - n_modules: 控制盒内模块数量
                             - n_taxels: 每模块触觉点数量(120)
                             - 3: 力向量 [Fx, Fy, Fz] (x/y: -128~127, z: 0~255)
            timestamp (float): 时间戳(秒)
            
        数据格式示例：
            串口传感器: shape=(2, 120, 3) - 单控制盒2个模块
            网络传感器: shape=(1, 120, 3) - 分控制盒广播，每个传感器独立接收
        """
    
    def get_tactile_state(self) -> PaxiniState:
        """获取完整传感器状态"""
    
    # 生命周期管理
    @abc.abstractmethod  
    def initialize(self) -> bool:
        """初始化传感器连接"""
    
    @abc.abstractmethod
    def close(self) -> bool:
        """关闭传感器连接"""
    
    # 状态查询
    def is_sensor_connected(self) -> bool:
        """检查传感器连接状态"""
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """获取传感器详细信息"""
    
    def print_state(self) -> None:
        """打印传感器状态（调试用）"""
```

### 📡 PaxiniSerialSensor

**位置：** `paxini_serial_sensor.py`

```python
class PaxiniSerialSensor(TactileBase):
    """Paxini串口传感器实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                communication_type: "serial"
                control_box: {
                    "port": "/dev/ttyACM0",
                    "baudrate": 460800,
                    "timeout": 1.0,
                    "control_mode": 5,
                    "sensors": [
                        {
                            "model": "GEN2-IP-L5325",
                            "connect_ids": [5, 6]
                        }
                    ]
                }
        """
    
    # 特有方法
    def get_control_box_info(self) -> Dict:
        """获取控制盒信息"""
    
    def _send_command(self, box_id: int, command: List[int]) -> List[int]:
        """发送底层协议命令"""
```

### 🌐 PaxiniNetworkSensor

**位置：** `paxini_network_sensor.py`

```python
class PaxiniNetworkSensor(TactileBase):
    """Paxini网络传感器实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                communication_type: "network"
                ip: "192.168.1.100"
                port: 5556
                topic: "PAXINI_TACTILE_STREAM"
                sensor_id: "PAXINI_LEFT_HAND_01_MODULES" (v1.1支持命名)
                timeout: 5.0
                control_box: {
                    "name": "left_hand",  # v1.1新增：控制盒命名
                    "sensors": [...]
                } # 逻辑配置，必须与服务器端配置一致
        """
    
    # 网络特有方法
    def _attempt_reconnect(self) -> None:
        """尝试重连网络"""
    
    # 自动故障恢复
    # - 连接断开自动重连
    # - 数据超时检测
    # - 错误计数与恢复
```

### 🖥️ PaxiniStandaloneServer

**位置：** `paxini_standalone_server.py`

```python
# 使用方式
python3 paxini_standalone_server.py --config server_config.yaml
python3 paxini_standalone_server.py --port /dev/ttyACM1 --zmq-port 5557

class PaxiniStandaloneServer:
    """独立ZMQ服务器，无需HIROLRobotPlatform依赖"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        从配置文件或默认配置初始化服务器
        自动生成sensor_id，支持多控制盒广播
        """
    
    def run_server(self) -> None:
        """运行主循环，广播触觉数据"""
    
    # 特性：
    # - 单控制盒管理，多实例支持
    # - 统一数据格式  
    # - 性能统计显示
    # - 优雅关闭处理
```

## 集成指南

### 🤖 HIROLRobotPlatform集成

**1️⃣ 添加传感器类型到工厂**

```python
# factory/components/robot_factory.py (未来扩展)
self._tactile_classes = {
    'paxini_serial_sensor': PaxiniSerialSensor,
    'paxini_network_sensor': PaxiniNetworkSensor,
    # 未来可添加其他厂商传感器
}
```

**2️⃣ Motion配置集成 **

```yaml
# motion配置文件 (如 factory/components/motion_configs/fr3_with_tactile.yaml)
sensor_dicts:
  cameras:
    - name: "ee_cam"
      type: "realsense_camera"
      cfg: !include hardware/sensors/cameras/config/realsense_cfg.yaml
      
  tactile:  # ← 触觉传感器配置 
    # 左手触觉传感器实例
    - name: "left_hand_tactile"
      type: "paxini_serial_sensor"
      cfg: !include hardware/sensors/paxini_tactile/config/factory/paxini_left_arm.yaml
      
    # 右手触觉传感器实例  
    - name: "right_hand_tactile"
      type: "paxini_serial_sensor"
      cfg: !include hardware/sensors/paxini_tactile/config/factory/paxini_right_arm.yaml
      
    # 可选：网络触觉传感器
    # - name: "remote_tactile"
    #   type: "paxini_network_sensor"
    #   cfg: !include hardware/sensors/paxini_tactile/config/factory/paxini_network_left_arm.yaml
```

**3️⃣ 应用代码使用**

```python
# 在robot控制代码中
from factory.components.robot_factory import RobotFactory

# 创建机器人系统
config = yaml.load(open('motion_config.yaml'))
robot_system = RobotFactory(config)
robot_system.initialize()

# 获取触觉传感器实例 
tactile_sensors = robot_system.get_tactile_sensors()  # 未来API
left_tactile = tactile_sensors['left_hand_tactile']
right_tactile = tactile_sensors['right_hand_tactile']

# 分别读取左右手数据
success_l, data_l, ts_l = left_tactile.read_tactile_data()   # shape: (2, 120, 3) - 左手控制盒数据
success_r, data_r, ts_r = right_tactile.read_tactile_data()  # shape: (2, 120, 3) - 右手控制盒数据

if success_l and success_r:
    # 组合触觉数据 (类似DuoArm的joint states组合)
    tactile_dict = {
        "left_hand": data_l,    # 左手触觉数据
        "right_hand": data_r    # 右手触觉数据
    }
    
    # 处理双手触觉反馈
    max_force_l = data_l.max()
    max_force_r = data_r.max()
    contact_points_l = (data_l[:, :, 2] > threshold).sum()  # 左手接触点
    contact_points_r = (data_r[:, :, 2] > threshold).sum()  # 右手接触点
    
    # 集成到双臂控制算法
    robot_controller.update_dual_tactile_feedback(tactile_dict)
```


## 测试工具

### 🧪 完整测试工具链

**硬件连接测试：**
```bash
# 基础连接验证
python3 test/test_paxini_connection.py --port /dev/ttyACM0

# 长时间稳定性测试  
python3 test/test_paxini_connection.py --port /dev/ttyACM0 --duration 300

# 多传感器测试 (左右手分别测试)
python3 test/test_paxini_connection.py --port /dev/ttyACM0  # 测试左手
python3 test/test_paxini_connection.py --port /dev/ttyACM1  # 测试右手


**网络传输测试：**
```bash
# 完整网络演示 (推荐用于首次测试)
python3 test/test_network_demo.py --mode server --port /dev/ttyACM0
python3 test/test_network_demo.py --mode client --ip 192.168.1.100

# 客户端性能测试 (连接现有服务器)
python3 test/test_network_client.py --config config/factory/test_client_config.yaml

# 两个测试工具的区别：
# test_network_demo.py    - 完整演示：可运行服务器或客户端模式
# test_network_client.py  - 纯客户端：仅连接现有服务器进行性能测试

# 预期输出：
# 🌐 网络传感器测试结果:
# 📊 10秒测试: 接收帧数: 991, 平均FPS: 99.0
# 📈 网络延迟: <1ms, 数据完整性: 100%
```

**配置系统测试：**
```bash
# 配置兼容性验证
python3 test/test_config_compatibility.py

# sensor_id生成测试
python3 test/test_sensor_id_auto_generation.py

# 预期输出：
# ✅ 配置格式兼容性: 直接格式 ✓, 工厂格式 ✓
# ✅ sensor_id自动生成: PAXINI_02_MODULES
# ✅ 手动指定覆盖: PAXINI_CUSTOM_02_MODULES
```

### 📊 性能基准

**典型性能指标：**
| 配置 | 采样率 | CPU占用 | 内存占用 | 网络带宽 |
|------|--------|---------|----------|----------|
| 单传感器串口 | 1000Hz | 2-5% | 10MB | N/A |
| 双传感器串口 | 500Hz | 5-8% | 15MB | N/A |
| 网络客户端(v1.0) | 100Hz | 1-2% | 5MB | 50KB/s |
| 网络服务器(v1.0) | 100Hz | 3-6% | 12MB | 50KB/s |
| **分控制盒客户端(v1.1)** | **99.1Hz** | **1-2%** | **5MB** | **25KB/s** |
| **分控制盒服务器(v1.1)** | **50.0Hz** | **3-6%** | **12MB** | **100KB/s** |

**v1.1性能特点：**
- ✅ **独立数据流**：每个传感器仅接收自己的数据，带宽减半
- ✅ **高稳定性**：99.1 FPS实测，接近100Hz理论值
- ✅ **低延迟**：网络延迟<1ms，数据完整性100%
- ✅ **线性扩展**：支持扩展到多部位全身传感器

## 扩展开发

### 🔧 添加新传感器厂商

**1️⃣ 继承基类**
```python
# hardware/sensors/new_tactile/new_tactile_sensor.py
from hardware.base.tactile_base import TactileBase

class NewTactileSensor(TactileBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 厂商特定初始化
    
    def initialize(self) -> bool:
        # 实现厂商特定连接逻辑
        pass
    
    def close(self) -> bool:
        # 实现厂商特定关闭逻辑  
        pass
    
    # TactileBase的其他方法会自动继承
```

**2️⃣ 添加配置支持**
```yaml
# config/factory/new_sensor_config.yaml
new_tactile_sensor:  # ← 传感器类型标识
  communication_type: "can_bus"  # 厂商特定通信方式
  device_id: 0x123
  protocol_version: "v2.0"
  # ... 厂商特定配置
```

**3️⃣ 注册到工厂**
```python
# factory/components/robot_factory.py
self._tactile_classes = {
    'paxini_serial_sensor': PaxiniSerialSensor,
    'paxini_network_sensor': PaxiniNetworkSensor,
    'new_tactile_sensor': NewTactileSensor,  # ← 新增
}
```

### 🌐 添加新通信协议

**示例：CAN总线支持**
```python
class PaxiniCANSensor(TactileBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._can_interface = config.get('can_interface', 'can0')
        self._device_id = config.get('device_id', 0x100)
    
    def initialize(self) -> bool:
        import can
        self._bus = can.interface.Bus(
            bustype='socketcan', 
            channel=self._can_interface
        )
        # CAN协议初始化
        return True
    
    def _read_raw_data(self) -> Optional[np.ndarray]:
        # 实现CAN消息接收和解析
        msg = self._bus.recv(timeout=1.0)
        if msg and msg.arbitration_id == self._device_id:
            return self._parse_can_message(msg.data)
        return None
```

### 🧪 添加测试支持

**测试文件模板：**
```python
# test/test_new_sensor.py
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from new_tactile_sensor import NewTactileSensor

def test_new_sensor_connection():
    config = {
        "communication_type": "can_bus",
        "can_interface": "vcan0",  # 使用虚拟CAN接口测试
    }
    
    sensor = NewTactileSensor(config)
    assert sensor.initialize(), "传感器初始化失败"
    
    success, data, timestamp = sensor.read_tactile_data()
    assert success, "数据读取失败"
    assert data is not None, "数据为空"
    
    sensor.close()
    print("✅ 新传感器测试通过")

if __name__ == "__main__":
    test_new_sensor_connection()
```

## 故障排除

### 🔧 常见问题诊断

**1️⃣ 串口连接问题**
```bash
# 问题：权限拒绝
# 错误：Permission denied: '/dev/ttyACM0'
sudo chmod 666 /dev/ttyACM0
# 或添加用户到dialout组
sudo usermod -a -G dialout $USER

# 问题：设备不存在
# 错误：No such file or directory: '/dev/ttyACM0'
ls /dev/ttyACM*          # 查看可用设备
dmesg | grep tty         # 查看系统日志

# 问题：设备被占用
# 错误：Device or resource busy
sudo lsof /dev/ttyACM0   # 查看占用进程
sudo fuser -k /dev/ttyACM0  # 强制释放
```

**2️⃣ 网络连接问题**
```bash
# 问题：无法连接服务器
# 错误：Connection refused
ping 192.168.1.100       # 检查网络连通性
telnet 192.168.1.100 5556  # 检查端口开放

# 问题：防火墙阻挡
sudo ufw status          # 检查防火墙状态
sudo ufw allow 5556      # 开放ZMQ端口

# 问题：数据接收超时
# 检查sensor_id匹配
python3 -c "
import yaml
with open('server_config.yaml') as f: server_cfg = yaml.safe_load(f)
with open('client_config.yaml') as f: client_cfg = yaml.safe_load(f)
print('Server would generate:', 'PAXINI_XX_MODULES')
print('Client expects:', client_cfg.get('sensor_id', 'AUTO-GENERATED'))
"
```

**3️⃣ 配置问题**
```bash
# 配置语法验证
python3 -c "import yaml; yaml.safe_load(open('my_config.yaml'))"

# 配置逻辑验证
python3 test/test_config_compatibility.py

# 对比工作配置
diff my_config.yaml config/common/single_sensor.yaml
```

**4️⃣ 性能问题**
```bash
# CPU占用过高
htop                     # 查看CPU使用情况
python3 test/test_paxini_connection.py --profile  # 性能分析

# 内存泄漏
python3 -c "
import psutil, time
from paxini_serial_sensor import PaxiniSerialSensor
sensor = PaxiniSerialSensor(config)
sensor.initialize()

for i in range(1000):
    sensor.read_tactile_data()
    if i % 100 == 0:
        print(f'Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB')
    time.sleep(0.001)
"

# 网络带宽占用
iftop -i eth0            # 查看网络流量
```

### 📞 支持渠道

**技术文档：**
- 📖 [配置系统详细说明](config/README.md)
- 🚀 [快速入门指南](QUICK_START.md) 
- 🧪 [测试工具使用指南](test/)

**网络测试工具详解：**
- **test_network_demo.py**: 完整的服务器+客户端演示，支持`--mode server/client`参数
- **test_network_client.py**: 专门的客户端性能测试工具，支持`--config`参数指定配置文件


---

## 📊 项目信息

**开发**： Haotian Liang
**项目版本**：1.2.0  
**更新日期**：2025-09-11  
**License**：MIT License

**v1.2.0 更新日志 (2025-09-11)：**
- ⚡ **架构重构**：统一Serial和Network为单控制盒管理模式，提升一致性
- 🔧 **配置简化**：`control_boxes: [...]` → `control_box: {...}`，降低配置复杂度
- 🏭 **Factory增强**：多传感器实例支持，每个实例管理一个控制盒，完美支持双臂
- 📡 **分控制盒广播**：每个控制盒独立数据流，解决网络数据混合问题
- 🏷️ **智能sensor_id**：支持语义化命名(left_hand, torso等)，自动生成对应ID
- 🎯 **性能优化**：99.1 FPS稳定传输，网络带宽优化50%
- 📋 **配置清理**：移除冗余dual配置，采用更清晰的命名规范
- 🧪 **测试验证**：teleoperation系统成功验证新架构兼容性  

**技术栈**：
- **核心**：Python 3.8+, NumPy, PyYAML
- **通信**：PySerial, PyZMQ
- **日志**：glog
- **测试**：pytest, unittest



---