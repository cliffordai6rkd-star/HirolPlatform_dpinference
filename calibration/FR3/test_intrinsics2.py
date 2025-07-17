import cv2
import numpy as np
import pyrealsense2 as rs
import time
import yaml

# ChArUco 板信息
squares_x = 14
squares_y = 9
square_length = 0.02  # 20 mm
marker_length = 0.015  # 15 mm
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
charuco_board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

# RealSense相机初始化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

all_corners = []
all_ids = []
image_size = None

print("开始捕获标定数据，请移动ChArUco板到不同位置进行拍摄，按 's' 保存图像，拍摄20张后按 'q' 退出")

try:
    while len(all_corners) < 20:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        if corners:
            res = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
            if res[1] is not None and res[2] is not None and len(res[1]) > 10:
                cv2.aruco.drawDetectedCornersCharuco(img, res[1], res[2])

                cv2.imshow('ChArUco Board', img)
                key = cv2.waitKey(1)

                if key == ord('s'):
                    all_corners.append(res[1])
                    all_ids.append(res[2])
                    image_size = gray.shape[::-1]
                    print(f"Captured image {len(all_corners)}")

                elif key == ord('q'):
                    break
        else:
            cv2.imshow('ChArUco Board', img)
            cv2.waitKey(1)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# 标定相机
print("开始进行相机标定，请稍候...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_corners, all_ids, charuco_board, image_size, None, None)

# 保存标定结果
calibration_data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeff': dist_coeffs.tolist()
}

with open("camera_intrinsics.yaml", "w") as f:
    yaml.dump(calibration_data, f)

# 输出内参矩阵
print("Calibration finished successfully!\n")
print("内参矩阵（Intrinsic Matrix）:\n", camera_matrix)
print("\n畸变系数（Distortion Coefficients）:\n", dist_coeffs)
print("\n结果已保存至 camera_intrinsics.yaml")
