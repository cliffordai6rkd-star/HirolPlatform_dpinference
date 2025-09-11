# Paxini触觉传感器快速开始

## 🚀 30秒上手指南

### 1️⃣ 硬件连接测试
```bash
# 检查硬件连接
ls /dev/ttyACM*

# 直接测试（无需配置）
python3 test/test_paxini_connection.py --port /dev/ttyACM0
```

### 2️⃣ 使用预设配置
```bash
# 查看可用配置
ls config/factory/       # 工厂集成配置
ls config/server/        # 服务器配置
ls config/examples/      # 示例配置

# 复制并使用基础配置
cp config/factory/paxini_single_hand.yaml my_config.yaml
python3 test/test_paxini_connection.py --config my_config.yaml
```

### 3️⃣ 网络传感器部署
```bash
# 服务器端（连接硬件的电脑）
cp config/server/lab_single_hand_server.yaml server_config.yaml
# 编辑server_config.yaml修改串口路径
python3 paxini_standalone_server.py --config server_config.yaml

# 客户端（应用电脑）  
cp config/factory/paxini_network_single_hand.yaml client_config.yaml
# 编辑client_config.yaml修改服务器IP
python3 test/test_network_client.py
```

### 4️⃣ HIROLRobotPlatform集成
```yaml
# 在motion配置文件中添加
sensor_dicts:
  tactile:
    - name: "left_hand_tactile"
      type: "paxini_serial_sensor"
      cfg: !include hardware/sensors/paxini_tactile/config/factory/paxini_left_arm.yaml
```

## 📁 配置文件选择

| 用途 | 配置文件 | 说明 |
|------|---------|------|
| 基础测试 | `config/factory/paxini_single_hand.yaml` | 单手传感器配置 |
| 网络客户端 | `config/factory/paxini_network_single_hand.yaml` | 接收网络数据 |
| 左手网络 | `config/factory/paxini_network_left_arm.yaml` | 左手网络客户端 |
| 右手网络 | `config/factory/paxini_network_right_arm.yaml` | 右手网络客户端 |
| 实验室单手服务器 | `config/server/lab_single_hand_server.yaml` | 单手广播服务器 |
| 实验室双手服务器 | `config/server/lab_dual_hand_server.yaml` | 双手广播服务器 |
| 全身服务器示例 | `config/server/full_body_server_example.yaml` | 全身广播示例 |
| FR3左手 | `config/factory/paxini_left_arm.yaml` | 工厂集成左手 |
| FR3右手 | `config/factory/paxini_right_arm.yaml` | 工厂集成右手 |

## 🔧 常用操作

### 修改串口
```bash
# 编辑配置文件
vim config/factory/paxini_single_hand.yaml
# 修改: port: "/dev/ttyACM1"  # 改为你的串口
```

### 修改网络设置
```bash  
# 编辑网络配置
vim config/factory/paxini_network_single_hand.yaml
# 修改: ip: "192.168.1.100"  # 改为服务器IP
```

### 验证配置
```bash
# 语法检查
python3 -c "import yaml; yaml.safe_load(open('my_config.yaml'))"

# 连接测试
python3 test/test_paxini_connection.py --config my_config.yaml

# 网络连接测试
python3 test/test_network_client.py

# 配置兼容性测试
python3 test/test_config_compatibility.py
```

## 🐛 故障排除

### 串口权限问题
```bash
sudo chmod 666 /dev/ttyACM0
```

### 网络连接问题
```bash
# 检查连通性
ping 192.168.1.100

# 检查端口
telnet 192.168.1.100 5556

# 测试网络演示
python3 test/test_network_demo.py
```

### 配置问题
```bash
# 对比工作配置
diff my_config.yaml config/factory/paxini_single_hand.yaml

# 测试工厂集成
python3 test/test_factory_integration.py

# 检查传感器ID自动生成
python3 test/test_sensor_id_auto_generation.py
```

---

**记住**: 新架构不需要复杂工具，直接使用文件即可！