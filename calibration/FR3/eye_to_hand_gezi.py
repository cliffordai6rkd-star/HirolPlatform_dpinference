import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import json
import os
import panda_py

# 定义ChArUco板参数（按照CharuCO-300规格）
square_length = 0.02  # 棋盘格方格的边长（20mm -> 0.02m）
marker_length = 0.015  # ArUco标记的边长（15mm -> 0.015m）

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # 假设没有畸变

print(f"Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# 初始化ArUco字典和CharucoBoard（使用DICT_5X5字典和9×14的板）
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# board_size = (9, 14)  # 棋盘格尺寸 (行，列) - CharuCO-300规格
board_size = (14, 9)
charuco_board = aruco.CharucoBoard(board_size, square_length, marker_length, aruco_dict)

# 检测器参数
detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Create a window
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# Initialize variables
data = []  # 标定数据列表
image_counter = 0  # 图片计数器

# 初始化机器人连接
hostname = '192.168.1.100'
panda = panda_py.Panda(hostname)

# 标定板到末端的变换矩阵（如果标定板不是直接固定在末端，需要根据实际情况调整）
# 如果标定板直接固定在末端，这个矩阵就是单位矩阵
marker_to_end = np.eye(4)  # 标定板到末端的变换，根据实际安装情况修改

def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames() 
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 检测ArUco标记
        corners, ids, rejected = detector.detectMarkers(gray)
        # print(f"检测到的ArUco标记数量: {len(ids) if ids is not None else 0}")
        
        # 初始化变换矩阵
        homogeneous_matrix = None
        
        # 如果检测到标记
        if ids is not None and len(ids) > 0:
            # 绘制检测到的标记
            aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # 检测ChArUco角点
            result, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, charuco_board, 
                cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
            # print(f"检测到的ChArUco角点数量: {result if result is not None else 0}")
            
            # 如果检测到足够的角点，进行位姿估计
            if result > 3:
                # 绘制ChArUco角点
                cv2.drawChessboardCorners(color_image, (1, len(charuco_corners)), charuco_corners, True)
                
                # 创建初始的旋转向量和平移向量
                rvec = np.zeros((3, 1), dtype=np.float32)
                tvec = np.zeros((3, 1), dtype=np.float32)

                # 估计位姿 - 注意这里需要传入初始的rvec和tvec
                valid = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, charuco_board, 
                    camera_matrix, dist_coeffs, rvec, tvec)
                # print(f"位姿估计是否成功: {valid}")

                if valid:
                    # 绘制坐标轴
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.2)
                    
                    # 将旋转向量转换为旋转矩阵
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.flatten()
                    
                    # 创建相机到标定板的同质变换矩阵 (camera_T_marker) 在相机坐标系下 marker的位姿
                    homogeneous_matrix = np.eye(4)
                    homogeneous_matrix[:3, :3] = R
                    homogeneous_matrix[:3, 3] = t
                    
                    # 显示平移向量
                    pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
                    cv2.putText(color_image, pose_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Show images
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key == ord('r') and homogeneous_matrix is not None:
            # 获取机器人末端在基座坐标系下的位姿 (end_T_base)
            hostname = '192.168.1.100'
            panda = panda_py.Panda(hostname)

            matrix = panda.get_pose()
            
            print("Robot pose matrix (end to base): ", matrix)
            
            # 保存图像
            image_filename = f'image_{image_counter}.png'
            cv2.imwrite(image_filename, color_image)
            
            # 保存数据
            data.append({
                'image': image_filename,
                'matrix': matrix.tolist(),  # 末端在基座坐标系下的位姿
                'homogeneous_matrix': homogeneous_matrix.tolist(),  # 在相机坐标系下的tag的位姿
            })
            print(f"捕获第 {image_counter+1} 组数据")
            image_counter += 1

        elif key == ord('q'):
            # 获取内参
            intrinsics_data = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
            }
            # 保存数据到JSON文件
            save_to_json('data.json', {'data': data, 'intrinsics': intrinsics_data})
            print("数据已保存至 data.json，程序结束")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()