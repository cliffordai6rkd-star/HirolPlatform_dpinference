# 多相机场景配置指南

## 场景说明

当您的 RobotMotion 配置中启动了**多个相机**（例如4个相机），但手眼标定时只想使用其中**1个特定相机**。

## 配置步骤

### Step 1: 在 RobotMotion 配置中定义所有相机

编辑您的 motion_config 文件（例如 `factory/components/motion_configs/fr3_with_franka_hand_ik.yaml`）：

```yaml
# sensor infos
sensor_dicts:
  cameras:
    - name: "wrist_camera"           # 相机1：手腕相机
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_ee_1.yaml

    - name: "external_camera_left"   # 相机2：左侧外部相机
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_external_left.yaml

    - name: "external_camera_right"  # 相机3：右侧外部相机
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_external_right.yaml

    - name: "overhead_camera"        # 相机4：顶部相机
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_overhead.yaml
```

**重点**：每个相机的 `name` 字段必须唯一！

### Step 2: 在标定配置中指定使用哪个相机

编辑标定配置文件（例如 `calibration/config/eye_in_hand_fr3_charuco.yaml`）：

```yaml
calibration:
  camera:
    # 精确匹配相机名称（必须与 motion_config 中的 name 完全一致）
    name: "wrist_camera"  # ← 使用手腕相机进行标定
```

### Step 3: 运行标定

```bash
cd /path/to/HIROLRobotPlatform/calibration
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml
```

**日志输出示例**：
```
INFO: Available cameras in RobotMotion: ['wrist_camera', 'external_camera_left', 'external_camera_right', 'overhead_camera']
INFO: Selected camera: 'wrist_camera' (exact match)
```

## 常见场景

### 场景1：Eye-in-Hand (使用手腕相机)

**motion_config**:
```yaml
sensor_dicts:
  cameras:
    - name: "wrist_camera"
      cfg: !include hardware/sensors/cameras/config/d435i_ee_1.yaml
    - name: "external_camera"
      cfg: !include hardware/sensors/cameras/config/d435i_external.yaml
```

**calibration_config**:
```yaml
calibration:
  type: "eye_in_hand"
  camera:
    name: "wrist_camera"  # 使用手腕相机
```

### 场景2：Eye-to-Hand (使用外部相机)

**motion_config**: 同上

**calibration_config**:
```yaml
calibration:
  type: "eye_to_hand"
  camera:
    name: "external_camera"  # 使用外部相机
```

### 场景3：多个同类型相机（需要明确区分）

**motion_config**:
```yaml
sensor_dicts:
  cameras:
    - name: "d435i_serial_123456789"  # 通过序列号区分
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_1.yaml

    - name: "d435i_serial_987654321"  # 另一个相机
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_2.yaml
```

**calibration_config**:
```yaml
calibration:
  camera:
    name: "d435i_serial_123456789"  # 精确指定序列号
```

## 错误排查

### 错误1: "Camera 'xxx' not found"

**症状**:
```
RuntimeError: Camera 'my_camera' not found.
Available cameras: ['wrist_camera', 'external_camera_left']
```

**解决方案**:
1. 检查拼写：`my_camera` 应改为 `wrist_camera`
2. 更新标定配置中的 `camera.name`：
   ```yaml
   camera:
     name: "wrist_camera"  # 修正名称
   ```

### 错误2: "No cameras configured in RobotMotion"

**症状**:
```
RuntimeError: No cameras configured in RobotMotion.
Hint: Add cameras to motion_config's sensor_dicts.cameras
```

**解决方案**:
1. 检查 motion_config 文件中的 `sensor_dicts` 部分
2. 确保相机配置未被注释掉
3. 确认相机配置格式正确

### 错误3: 使用了错误的相机

**症状**: 标定时看到的图像不是预期相机的画面

**解决方案**:
1. 检查日志，确认选中的相机：
   ```
   INFO: Selected camera: 'wrist_camera' (exact match)
   ```
2. 如果不对，修改标定配置中的 `camera.name`
3. 验证相机名称在 motion_config 中的定义

## 最佳实践

### 1. 使用描述性的相机名称

❌ **不推荐**:
```yaml
- name: "camera1"
- name: "camera2"
```

✅ **推荐**:
```yaml
- name: "wrist_camera"
- name: "external_camera_left"
```

### 2. 在名称中包含位置信息

```yaml
- name: "ee_camera_front"     # 末端执行器前方相机
- name: "ee_camera_side"      # 末端执行器侧方相机
- name: "base_camera_left"    # 基座左侧相机
- name: "base_camera_right"   # 基座右侧相机
```

### 3. 对于多个同型号相机，包含序列号

```yaml
- name: "d435i_405622073844"  # RealSense D435i (序列号)
- name: "d435i_405622073855"  # 另一个 D435i
```

### 4. 测试相机选择

在标定前，先运行一次看日志：

```bash
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml

# 查看日志输出
INFO: Available cameras in RobotMotion: [...]
INFO: Selected camera: '...' (exact match)
```

确认选中了正确的相机后，再继续标定。

## 完整示例

### motion_config.yaml
```yaml
use_hardware: true
use_simulation: false

robot: "fr3"
robot_config: !include hardware/fr3/config/fr3_cfg.yaml

gripper: "franka_hand"
gripper_config: !include hardware/fr3/config/franka_hand_cfg.yaml

sensor_dicts:
  cameras:
    - name: "wrist_d435i"
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_ee_1.yaml

    - name: "external_d435i_left"
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_external_left.yaml

    - name: "external_d435i_right"
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_external_right.yaml

    - name: "overhead_d435i"
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_overhead.yaml

# ... 其他配置 ...
```

### calibration_config.yaml (Eye-in-Hand)
```yaml
robot_motion_config: "../factory/tasks/config/robot_motion_fr3_cfg.yaml"

calibration:
  type: "eye_in_hand"

  camera:
    name: "wrist_d435i"  # 使用手腕相机

  board:
    type: "charuco"
    square_length: 0.020
    marker_length: 0.015
    board_size: [14, 9]
    aruco_dict: "DICT_5X5_250"

  # ... 其他配置 ...
```

### 运行结果
```
INFO: Loaded configuration from config/eye_in_hand_fr3_charuco.yaml
INFO: RobotMotion initialized
INFO: Available cameras in RobotMotion: ['wrist_d435i', 'external_d435i_left', 'external_d435i_right', 'overhead_d435i']
INFO: Selected camera: 'wrist_d435i' (exact match)
INFO: Camera intrinsics loaded: fx=616.5, fy=616.6
INFO: Board detector created: charuco
INFO: Calibration type: eye_in_hand
INFO: ============================================================
INFO:  HandEyeCalibration initialized successfully
INFO: ============================================================
```

## 总结

- ✅ **精确匹配**: 相机名称必须完全一致（区分大小写）
- ✅ **查看日志**: 运行时会显示所有可用相机
- ✅ **唯一命名**: 确保每个相机名称在 motion_config 中唯一
- ✅ **描述性命名**: 使用清晰的名称（位置、型号、序列号等）

更多信息请参考 [README.md](README.md) 和 [config/README.md](config/README.md)。
