import pyrealsense2 as rs
import numpy as np
import time

# 初始化相机管道
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 开始数据流
profile = pipeline.start(config)

# 给相机硬件和自动曝光稳定时间
print("Warming up the camera for hardware readiness...")
time.sleep(5)

# 获取并打印当前的深度内参
print("Before Calibration:")
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_stream.get_intrinsics()
print(intrinsics)

# 获取设备及自动校准接口
device = profile.get_device()
auto_dev = device.as_auto_calibrated_device()

# 准备校准参数（可根据需求调整）
calib_json = '{"speed":3,"scan":0,"adjust both sides":0,"white_wall_mode":0}'

# 提示用户对准白墙
print("请确保相机对准一面平整的白墙或白色平面以进行深度自校准...")
time.sleep(2)

# 进行On-Chip自校准
print("Running on-chip self-calibration...")
try:
    # timeout 10000ms，可根据需要延长
    table, health = auto_dev.run_on_chip_calibration(calib_json, 10000)
    print(f"Calibration completed. Health: {health}")

    if health[1] < health[0]:  # 后半段RMS更小表示校准有效
        auto_dev.set_calibration_table(table)
        print("Applied new calibration table.")
        auto_dev.write_calibration()
        print("Saved calibration data to firmware.")
    else:
        print("Calibration did not improve depth RMS. Keeping previous calibration.")
except Exception as e:
    print(f"Calibration failed: {e}\n请确认固件版本支持On-Chip校准，并重试。")

# 重新获取内参并打印，验证是否有变化
print("After Calibration:")
intrinsics_after = depth_stream.get_intrinsics()
print(intrinsics_after)

# 停止数据流
pipeline.stop()

# 转换并打印内参矩阵
intrinsics_matrix = np.array([
    [intrinsics_after.fx, 0, intrinsics_after.ppx],
    [0, intrinsics_after.fy, intrinsics_after.ppy],
    [0, 0, 1]
])
print("Calibrated Intrinsics Matrix:")
print(intrinsics_matrix)
