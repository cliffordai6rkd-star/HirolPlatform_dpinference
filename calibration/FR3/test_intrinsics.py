import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# 确保已安装：opencv-contrib-python==4.7.0.68，以支持所有 ArUco/Charuco API

# ---------------------
# ChArUco 板参数
# ---------------------
squares_x = 14             # 格子数 X
squares_y = 9             # 格子数 Y
square_length = 0.02      # 方格边长（m）
marker_length = 0.015     # 标记边长（m）

# ---------------------
# ArUco 字典与 CharucoBoard 设置
# ---------------------
dictionary_id = cv2.aruco.DICT_5X5_250  # 根据板子实际型号修改
dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
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
# 数据采集
# ---------------------
all_corners = []  # 存储 Charuco 角点
all_ids = []      # 存储对应 id
image_size = None
required_views = 20  # 需要采集的视角数（多视角）

print("指令：检测到合格的 Charuco 角点之后，按 'c' 键采集并移动板子到下一个视角。按 'q' 退出")
cv2.namedWindow("Charuco Detection", cv2.WINDOW_NORMAL)

try:
    while len(all_corners) < required_views:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. 检测 ArUco 标记
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        vis = img.copy()
        if ids is not None and len(ids) > 0:
            # 2. 插值 Charuco 角点
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )
            if retval >= 4:
                vis = cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

        cv2.imshow("Charuco Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ids is not None and len(ids) > 0 and retval >= 4:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print(f"已采集视角: {len(all_corners)}/{required_views}")
            # 提示移动板子到新视角
        elif key == ord('q'):
            break

        if image_size is None and gray is not None:
            image_size = gray.shape[::-1]

    cv2.destroyAllWindows()

    # ---------------------
    # 相机内参标定
    # ---------------------
    if len(all_corners) < 3:
        raise RuntimeError("采集视角不足，至少需要 3 个不同视角")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("\n标定完成")
    print(f"重投影误差: {ret:.4f}")
    print("相机内参矩阵:")
    print(camera_matrix)
    print("畸变系数:")
    print(dist_coeffs)

    # 保存结果
    data = {'camera_matrix': camera_matrix.tolist(), 'dist_coeffs': dist_coeffs.tolist()}
    with open('realsense_charuco_calib.yaml', 'w') as f:
        yaml.dump(data, f)
    print("标定参数已保存至 realsense_charuco_calib.yaml")

finally:
    pipeline.stop()
