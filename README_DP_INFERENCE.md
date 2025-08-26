# Diffusion Policy Inference 使用指南

## 概述
本文档介绍如何使用 HIROLRobotPlatform 运行训练好的 Diffusion Policy 模型进行机器人推理测试。

## 快速开始

### 1. 配置检查
确保以下文件存在并正确配置：
- `factory/tasks/inferences_tasks/dp/config/fr3_dp_inference_cfg.yaml` - 主配置文件
- 训练好的 DP checkpoint 文件

### 2. 运行推理
```bash
# 基本运行
python run_dp_inference.py

# 指定配置文件
python run_dp_inference.py --config path/to/your/config.yaml

# 禁用图像显示（服务器环境）
python run_dp_inference.py --no-display

# 强制使用CPU
python run_dp_inference.py --device cpu
```

### 3. 交互控制
推理运行时的键盘控制：
- `r` - 重置机器人到初始位置
- `q` - 退出推理程序

## 配置说明

### 主要配置项
```yaml
# 模型相关
checkpoint_path: "/path/to/your/model.ckpt"  # DP模型checkpoint路径
device: "cuda"                               # 推理设备 ("cuda" or "cpu")
image_size: [84, 84]                        # 图像输入尺寸 [H, W]

# 任务配置
tasks: "peg_in_hole"                        # 任务名称

# 显示配置
enable_display: true                        # 是否显示相机画面

# 推理参数
num_inference_steps: 16                     # DDIM 推理步数
```

## 文件结构
```
factory/tasks/inferences_tasks/dp/
├── dp_inference.py              # 主推理类
├── config/
│   └── fr3_dp_inference_cfg.yaml   # 配置文件
└── eval_real_robot_example.py  # 参考实现

run_dp_inference.py              # 运行脚本
```

## 核心功能

### DP_Inferencer 类
主要方法：
- `__init__(config)` - 初始化，加载模型和配置
- `start_inference()` - 开始推理循环
- `_convert_gym_obs_to_dp_format()` - 观测格式转换
- `_convert_dp_action_to_gym_format()` - 动作格式转换

### 数据流程
1. **观测获取**: HIROLRobotPlatform gym interface → 机器人状态 + 相机图像
2. **格式转换**: gym 格式 → DP 标准格式 (robot_obs + image_0/image_1)
3. **模型推理**: DP 模型预测动作序列
4. **动作执行**: DP 动作 → gym 动作格式 → 机器人执行

## 故障排除

### 常见问题
1. **Import Error**: 确保 diffusion_policy 路径正确
2. **CUDA Error**: 检查 GPU 内存，考虑使用 `--device cpu`
3. **Checkpoint Error**: 验证 checkpoint 路径和文件完整性
4. **Camera Error**: 检查相机连接和权限

### 调试模式
设置环境变量启用详细日志：
```bash
export GLOG_v=2  # 启用调试日志
python run_dp_inference.py
```

## 与 PI0 的区别

| 特性 | PI0 | Diffusion Policy |
|------|-----|------------------|
| 推理方式 | 直接预测 | 扩散去噪过程 |
| 动作输出 | 单步动作 | 动作序列 |
| 图像处理 | 原始分辨率 | 标准化尺寸 |
| 模型加载 | 简单加载 | hydra + workspace |

## 开发说明

### 扩展指南
- 新任务：修改 `tasks` 配置
- 新观测：扩展 `_convert_gym_obs_to_dp_format`
- 新动作：修改 `_convert_dp_action_to_gym_format`

### 测试脚本
- `test_dp_import.py` - 基本导入测试
- `test_dp_inference_basic.py` - 功能测试
- `run_dp_inference.py` - 完整推理测试