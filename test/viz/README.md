# ACT HDF5 数据可视化工具

## 使用方法

### 方式1：Web Viewer（推荐）
```bash
# 使用Web viewer，会在浏览器中自动打开
./test/viz/start_rerun_viewer.sh

# 或直接运行Python脚本
python test/viz/show_hdf5.py
```
Web viewer地址：http://localhost:9090

### 方式2：原生Viewer
```bash
# 使用原生Rerun viewer
./test/viz/start_rerun_viewer.sh --native

# 或直接运行
python test/viz/show_hdf5.py --native
```

### 参数选项

- `--hdf5_path`: 指定HDF5文件路径（默认：`test/viz/fr3_bs_seg_overlap/episode_8.hdf5`）
- `--native`: 使用原生Rerun viewer替代Web viewer

### 示例
```bash
# 可视化特定文件
./test/viz/start_rerun_viewer.sh --hdf5_path /path/to/your/episode.hdf5

# 使用原生viewer可视化特定文件
./test/viz/start_rerun_viewer.sh --hdf5_path /path/to/your/episode.hdf5 --native
```

## 可视化内容

1. **关节位置时间序列**：7个关节的位置变化
2. **夹爪动作分析**：
   - 夹爪位置时间序列
   - 原始动作值vs归一化值
   - 统计信息（min, max, mean, std）
3. **相机图像序列**：
   - ee_cam（末端相机）
   - side_cam（侧面相机）
   - third_person_cam（第三人称相机）

## 故障排除

如果Web viewer没有自动打开浏览器：
1. 手动访问 http://localhost:9090
2. 或使用 `--native` 参数改用原生viewer

如果原生viewer有连接问题：
1. 先手动启动：`rerun --port 9876`
2. 再运行Python脚本