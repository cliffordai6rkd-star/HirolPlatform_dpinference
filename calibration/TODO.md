# Calibration System - Future Enhancements

## 待定功能 (Pending Features)

### 🔲 多相机并行标定 (Multi-Camera Parallel Calibration)

**需求**：同时标定多个相机（例如：末端的前置相机 + 侧置相机）

**使用场景**：
- 机器人末端安装了多个相机（不同视角）
- 需要同时获得多个相机相对于末端的变换
- 一次数据采集，得到多个标定结果

**设计方案**：

#### 方案A：完全并行（推荐）
```yaml
# 配置示例
calibration:
  camera:
    names: ["ee_cam_front", "ee_cam_side"]  # 相机列表
```

**实现要点**：
- 每个相机独立的检测器 (`List[BoardDetectorBase]`)
- 数据结构：`samples[i]` 包含多个相机的检测结果
  ```python
  {
    'T_base_ee': np.ndarray,
    'cameras': {
      'ee_cam_front': {
        'T_camera_board': np.ndarray,
        'image_path': str,
        'reprojection_error': float
      },
      'ee_cam_side': {...}
    }
  }
  ```
- 独立求解每个相机的标定结果,并可以同时手上手外（机械臂上贴aruco）
- 输出：`calibration_result.json` 包含多个 `T_ee_camera`



---

### 🔲 在线标定验证 (Online Calibration Verification)

**需求**：标定完成后，实时验证标定精度

**功能**：
1. 加载已有标定结果
2. 实时显示：
   - 预测的标定板位姿 vs 检测的位姿
   - 位置误差（mm）
   - 旋转误差（度）
3. 可视化叠加（AR模式）

**使用场景**：
- 验证标定是否正确
- 检测标定结果是否退化



---

### 🔲 标定质量实时评估 (Real-time Quality Assessment)

**需求**：采集过程中实时显示标定质量指标

**功能**：
- 实时更新条件数
- 显示位姿空间覆盖率（3D可视化）
- 建议下一个采集位姿（优化覆盖）


---

### 🔲 支持其他标定板类型 (Additional Board Types)

**需求**：支持更多标定板

**候选类型**：
- AprilTag
- Checkerboard (纯棋盘格)
- Custom pattern

**实现**：添加新的检测器类继承 `BoardDetectorBase`


---



---

### 🔲 数据增强与优化 (Data Augmentation)

**需求**：从有限样本中生成更多训练数据

**功能**：
- 使用已标定结果生成合成数据
- 添加噪声模拟真实误差
- 用于测试标定鲁棒性



---

### 🔲 Web界面 (Web Interface)

**需求**：通过浏览器控制标定流程

**功能**：
- 实时相机预览
- 点击采集样本
- 显示标定结果
- 3D可视化

**技术栈**：Flask + WebSocket + Three.js


---

## 优先级排序

### P0 - 高优先级（核心功能增强）
- [ ] 多相机并行标定
- [ ] 在线标定验证

### P1 - 中优先级（易用性改进）
- [ ] 标定板自动生成工具
- [ ] 标定质量实时评估

### P2 - 低优先级（锦上添花）
- [ ] 支持其他标定板类型
- [ ] 数据增强与优化
- [ ] Web界面

---

## 贡献指南

如果您想实现上述功能，请参考以下步骤：

1. **Fork 仓库**并创建新分支
2. **阅读代码**：理解现有架构（策略模式 + 工厂模式）
3. **设计方案**：先在 Issue 中讨论设计
4. **实现功能**：遵循现有编码规范
5. **添加测试**：确保不破坏现有功能
6. **更新文档**：README + 配置示例
7. **提交 PR**：详细说明修改内容

---

## 相关资源

- **OpenCV Hand-Eye Calibration**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- **ROS Easy Handeye**: https://github.com/IFL-CAMP/easy_handeye
- **Tsai-Lenz Paper**: https://ieeexplore.ieee.org/document/34770

---

*最后更新: 2025-10-09*