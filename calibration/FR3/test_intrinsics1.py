import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
import time

# 确保安装了: opencv-contrib-python==4.7.0.68

# ---------------------
# ChArUco 板参数
# ---------------------
squares_x = 14             # 格子数 X
squares_y = 9             # 格子数 Y
square_length = 0.02      # 方格边长 (m)
marker_length = 0.015     # 标记边长 (m)

# ---------------------
# ArUco 字典与 CharucoBoard 设置
# ---------------------
dictionary_id = cv2.aruco.DICT_5X5_250  # 根据板子实际型号修改
dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
# board = cv2.aruco.CharucoBoard_create(
#     squaresX=squares_x,
#     squaresY=squares_y,
#     squareLength=square_length,
#     markerLength=marker_length,
#     dictionary=dictionary
# )
board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y), 
    square_length, 
    marker_length, 
    dictionary
)

# ---------------------
# ArUco 检测参数
# ---------------------
params = cv2.aruco.DetectorParameters()

# ---------------------
# RealSense 管道初始化
# ---------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# ---------------------
# 自动采集设置
# ---------------------
capture_duration = 10.0    # 自动采集时长 (秒)
max_views = 20             # 最大采集视角数
capture_interval = capture_duration / max_views

generated = 0
all_corners = []  # 存储 Charuco 角点
all_ids = []      # 存储对应 id
image_size = None
last_capture = time.time()
start_time = last_capture

print(f"开始自动采集 {capture_duration} 秒或最多 {max_views} 视角...")
cv2.namedWindow("Charuco Detection", cv2.WINDOW_NORMAL)

# 自动采集循环
while time.time() - start_time < capture_duration and generated < max_views:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 检测 ArUco 标记
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    vis = img.copy()
    if ids is not None and len(ids) > 0:
        # 2) 插值 Charuco 角点
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
        if retval >= 4 and time.time() - last_capture >= capture_interval:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            generated += 1
            last_capture = time.time()
            print(f"采集视角 {generated}/{max_views}：角点数 {len(charuco_ids)}")
        # 可视化所有检测到角点
        if retval >= 4:
            vis = cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    # 显示检测结果
    cv2.imshow("Charuco Detection", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if image_size is None:
        image_size = gray.shape[::-1]

cv2.destroyAllWindows()
pipeline.stop()

# ---------------------
# 检查采集结果并标定
# ---------------------
min_views = 3
if len(all_corners) < min_views:
    pipeline.stop()
    raise RuntimeError(f"采集视角不足 ({len(all_corners)})，至少需要 {min_views} 视角。")

print("\n开始标定...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("自动标定完成")
print(f"重投影误差: {ret:.4f}")
print("相机内参矩阵:")
print(camera_matrix)
print("畸变系数:")
print(dist_coeffs)

# 保存标定结果至 YAML
data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeffs': dist_coeffs.tolist()
}
with open('realsense_charuco_calib.yaml', 'w') as f:
    yaml.dump(data, f)
print("标定参数已保存至 realsense_charuco_calib.yaml")
