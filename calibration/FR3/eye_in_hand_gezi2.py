import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import json
import os
import panda_py

# Initialize the robot
# 定义ChArUco板参数（按照CharuCO-300规格）
square_length = 0.02  # 棋盘格方格的边长（20mm -> 0.02m）
marker_length = 0.015  # ArUco标记的边长（15mm -> 0.015m）

# 使用固定的相机内参矩阵和畸变系数
camera_matrix = np.array([
    [901.22840647, 0.0, 648.18440974],
    [0.0, 901.43812443, 379.54714976],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist_coeffs = np.array([
    [0.11428496, -0.24351861, 0.00070993, -0.00222305, 0.06933642]
], dtype=np.float32)

# 从相机矩阵中提取参数
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print(f"使用固定相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# Initialize Intel RealSense pipeline (只用于获取图像流)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

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
image_counter = 0  # Counter for image filenames 计数

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
        print(f"检测到的ArUco标记数量: {len(ids) if ids is not None else 0}")
        
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
            print(f"检测到的ChArUco角点数量: {result if 'result' in locals() else 0}")
            
            # 如果检测到足够的角点，进行位姿估计
            if result > 3:
                # 绘制ChArUco角点
                cv2.drawChessboardCorners(color_image, (1, len(charuco_corners)), charuco_corners, True)
                
                # 创建初始的旋转向量和平移向量
                rvec = np.zeros((3, 1), dtype=np.float32)
                tvec = np.zeros((3, 1), dtype=np.float32)

                # 估计位姿 - 使用初始的rvec和tvec
                valid = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, charuco_board, 
                    camera_matrix, dist_coeffs, rvec, tvec)
                print(f"位姿估计是否成功: {valid}")

                if valid:
                    # 绘制坐标轴
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.2)
                    
                    # 将旋转向量转换为旋转矩阵
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.flatten()
                    print(f'create homogeneous_matrix')
                    # 创建同质变换矩阵
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
            try:
                # 获取末端到基座的变换
                hostname = '192.168.1.100'
                panda = panda_py.Panda(hostname)
                matrix = panda.get_pose()
                print("pose_matrix: ", matrix)
                
                # 保存图像
                image_filename = f'{image_counter}.png'
                cv2.imwrite(image_filename, color_image)
                
                # 保存数据
                data.append({
                    'image': image_filename,
                    'matrix': matrix.tolist(),
                    'homogeneous_matrix': homogeneous_matrix.tolist()
                })
                print(f"Image, matrix, and homogeneous matrix recorded: {image_filename}")
                image_counter += 1
            except Exception as e:
                print(f"记录数据时出错: {e}")

        elif key == ord('q'):
            # 保存固定内参到数据文件
            intrinsics_data = {
                'width': 1280,
                'height': 720,
                'ppx': cx,
                'ppy': cy,
                'fx': fx,
                'fy': fy,
                'distortion_coefficients': dist_coeffs.flatten().tolist()
            }
            # 保存数据到JSON文件
            save_to_json('data.json', {'data': data, 'intrinsics': intrinsics_data})
            print("数据和内参已保存，程序退出。")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()