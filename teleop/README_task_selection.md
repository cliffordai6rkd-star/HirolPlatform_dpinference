# 遥操作任务选择系统

## 概述

新的任务选择系统允许您通过交互式菜单或命令行参数快速切换遥操作任务，无需手动编辑配置文件。

## 功能特性

- ✅ 交互式任务选择菜单
- ✅ 命令行参数直接选择
- ✅ 预定义6种常见任务
- ✅ 易于扩展新任务
- ✅ 错误处理和用户友好提示

## 使用方法

### 1. 交互式选择（推荐）

```bash
cd /workspace
python teleop/teleoperation.py
```

系统会显示任务列表：
```
============================================================
Available Tasks:
============================================================
  [0] Peg in Hole
      Description: pick up the peg on the table and put it in the hole
      Goal: pick up the peg on the table and place it in the co...

  [1] Block Stacking
      Description: stack colored blocks in the order: red -> blue -> yellow
      Goal: build a tower by stacking blocks in the correct col...

  [2] Solid Transfer
      Description: transfer rice from the box into the bowl
      Goal: move all rice grains from the source box to the tar...

  ... 更多任务
============================================================
Select task (0-5) or 'q' to quit:
```

输入任务编号（0-5）或 'q' 退出。

### 2. 命令行参数选择

直接指定任务ID，跳过交互式选择：

```bash
# 选择任务0 (Peg in Hole)
python teleop/teleoperation.py -t 0

# 选择任务1 (Block Stacking)
python teleop/teleoperation.py -t 1

# 选择任务2 (Solid Transfer)
python teleop/teleoperation.py -t 2
```

### 3. 自定义配置文件

```bash
python teleop/teleoperation.py -c your_config.yaml -t 1
```

## 可用任务列表

| ID | 任务名称 | 描述 | 保存路径前缀 |
|----|----------|------|--------------|
| 0 | Peg in Hole | 拿起桌上的钉子并放入孔中 | peg_in_hole |
| 1 | Block Stacking | 按红→蓝→黄顺序堆叠积木 | block_stacking |
| 2 | Solid Transfer | 将米粒从盒子转移到碗中 | solid_transfer |
| 3 | Liquid Transfer | 将水从左杯倒入右杯 | liquid_transfer |
| 4 | Solid Weighing | 称量固体材料达到目标重量 | solid_weighing |
| 5 | Powder Weighing | 分配粉末达到目标重量 | powder_weighing |

## 添加新任务

### 1. 编辑任务定义文件

编辑 `teleop/config/task_definitions.yaml`：

```yaml
tasks:
  # ... 现有任务 ...

  - id: 6  # 新的ID
    name: "Your New Task"
    save_path_prefix: "new_task"
    task_description: "简短描述您的任务"
    task_description_goal: "任务的最终目标"
    task_description_step: |
      step1: 第一步操作
      step2: 第二步操作
      step3: 完成任务
```

### 2. 字段说明

- **id**: 任务唯一标识符（递增数字）
- **name**: 任务显示名称
- **save_path_prefix**: 数据保存路径前缀
- **task_description**: 任务简要描述
- **task_description_goal**: 任务目标描述
- **task_description_step**: 详细操作步骤（支持多行）

### 3. 验证新任务

运行程序确认新任务出现在列表中：

```bash
python teleop/teleoperation.py
```

## 文件结构

```
teleop/
├── config/
│   ├── task_definitions.yaml    # 任务定义文件
│   └── franka_3d_mouse.yaml    # 主配置文件（清理后）
├── teleoperation.py            # 主程序（增强版）
└── README_task_selection.md    # 本文档
```

## 配置文件变更

### 主配置文件简化

`teleop/config/franka_3d_mouse.yaml` 中的任务相关字段已被移除，现在只包含核心遥操作配置。任务配置将在程序运行时动态加载。

### 任务定义分离

所有任务定义现在集中在 `teleop/config/task_definitions.yaml` 中，便于管理和扩展。

## 故障排除

### 1. 任务定义文件未找到

```
Error: Task definitions file not found: teleop/config/task_definitions.yaml
```

**解决方案**: 确保 `task_definitions.yaml` 文件存在于正确路径。

### 2. YAML格式错误

```
Error: Failed to parse task definitions: ...
```

**解决方案**: 检查YAML文件格式，确保缩进正确。

### 3. 无效任务ID

```
Please enter a number between 0 and 5
```

**解决方案**: 输入有效的任务ID范围内的数字。

## 技术细节

- 使用Python的 `yaml` 模块解析任务定义
- 支持命令行参数解析 (`argparse`)
- 动态配置合并，运行时将任务配置注入主配置
- 完善的错误处理和用户提示
- 向后兼容现有的配置文件结构