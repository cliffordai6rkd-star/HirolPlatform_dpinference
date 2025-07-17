# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import json
# import os
# from pupil_apriltags import Detector
# import panda_py

# # 定义标签大小（米）
# tag_size = 0.06  # 60 mm

# # 初始化Intel RealSense相机
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# # 开始流
# profile = pipeline.start(config)

# # 获取相机内参
# intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# fx, fy = intrinsics.fx, intrinsics.fy
# cx, cy = intrinsics.ppx, intrinsics.ppy
# camera_params = [fx, fy, cx, cy]

# print(f"相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# # 初始化AprilTag检测器
# detector = Detector(families="tag25h9")

# # 创建窗口
# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# # 初始化变量
# data = []
# intrinsics = None
# image_counter = 0  # 图像文件名计数器

# def save_to_json(filename, data):
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=4)

# # 连接到Panda机器人
# hostname = '192.168.1.100'
# try:
#     panda = panda_py.Panda(hostname)
#     print("成功连接到Panda机器人")
# except Exception as e:
#     print(f"连接机器人失败: {e}")
#     panda = None

# try:
#     while True:
#         # 等待一帧图像
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             continue

#         # 转换为numpy数组
#         color_image = np.asanyarray(color_frame.get_data())

#         # 转为灰度图
#         gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

#         # 检测AprilTag并估计姿态
#         detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

#         # 绘制检测结果
#         homogeneous_matrix = None
#         for detection in detections:
#             # 绘制标签边界
#             for idx in range(len(detection.corners)):
#                 pt1 = tuple(map(int, detection.corners[idx]))
#                 pt2 = tuple(map(int, detection.corners[(idx + 1) % 4]))
#                 cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

#             # 绘制中心
#             center = tuple(map(int, detection.center))
#             cv2.circle(color_image, center, 5, (0, 0, 255), -1)

#             # 显示ID
#             cv2.putText(color_image, f"ID: {detection.tag_id}",
#                         (center[0] - 10, center[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             # 提取姿态
#             t = detection.pose_t.flatten()  # 平移向量
#             R = detection.pose_R  # 旋转矩阵

#             # 在GUI显示平移
#             pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
#             cv2.putText(color_image, pose_text, (center[0] - 50, center[1] + 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#             # 绘制坐标轴 (X = 红, Y = 绿, Z = 蓝)
#             axis_length = 0.2  # 坐标轴长度
#             axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])  # X, Y, Z
#             axes_img_pts = []
#             for i in range(3):
#                 axis_end = t + R @ axes[i]  # 变换轴端点
#                 x, y = int(fx * axis_end[0] / axis_end[2] + cx), int(fy * axis_end[1] / axis_end[2] + cy)
#                 axes_img_pts.append((x, y))

#             cv2.line(color_image, center, axes_img_pts[0], (0, 0, 255), 2)  # X - 红
#             cv2.line(color_image, center, axes_img_pts[1], (0, 255, 0), 2)  # Y - 绿
#             cv2.line(color_image, center, axes_img_pts[2], (255, 0, 0), 2)  # Z - 蓝

#             # 创建齐次变换矩阵
#             # 相机坐标系到AprilTag坐标系的变换 (c_T_t)
#             homogeneous_matrix = np.eye(4)
#             homogeneous_matrix[:3, :3] = R
#             homogeneous_matrix[:3, 3] = t
            
#             # 添加眼在手上系统的注释
#             cv2.putText(color_image, "Eye-in-Hand System", (10, 30),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#         # 显示图像
#         cv2.imshow('RealSense', color_image)
#         key = cv2.waitKey(1)

#         if key == ord('r') and panda is not None:
#             if homogeneous_matrix is None:
#                 print("未检测到AprilTag，无法记录数据")
#                 continue
                
#             # 获取机器人当前位姿
#             # 机器人基座到末端执行器的变换 (b_T_e)
#             robot_pose = panda.get_pose()
#             print("机器人位姿: ", robot_pose)
            
#             # 保存图像
#             image_filename = f'{image_counter}.png'
#             cv2.imwrite(image_filename, color_image)
            
#             # 保存数据对
#             # 在眼在手上系统中：b_T_e (机器人位姿) 和 c_T_t (相机到标签的变换)
#             # 用于后续计算 e_T_c (末端执行器到相机的变换)
#             data.append({
#                 'image': image_filename,
#                 'robot_pose': robot_pose.tolist(),  # b_T_e
#                 'camera_to_tag': homogeneous_matrix.tolist()  # c_T_t
#             })
#             print(f"已记录图像和变换矩阵: {image_filename}")
#             image_counter += 1  # 增加计数器

#         elif key == ord('q'):
#             # 获取内参
#             intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
#             intrinsics_data = {
#                 'width': intrinsics.width,
#                 'height': intrinsics.height,
#                 'ppx': intrinsics.ppx,
#                 'ppy': intrinsics.ppy,
#                 'fx': intrinsics.fx,
#                 'fy': intrinsics.fy,
#             }
#             # 保存数据到JSON文件
#             save_to_json('eye_in_hand_data.json', {'data': data, 'intrinsics': intrinsics_data})
#             print("内参已保存，程序退出。")
#             break

# finally:
#     # 停止流
#     pipeline.stop()
#     cv2.destroyAllWindows()

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from pupil_apriltags import Detector
import panda_py

# Initialize the robot
# Define tag size in meters
tag_size = 0.0557  # 64 mm 

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics dynamically 内参
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
camera_params = [fx, fy, cx, cy]

print(f"Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# Initialize AprilTag detector
detector = Detector(families="tag25h9")

# Create a window
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# Initialize variables
data = [] # 标定数据列表
intrinsics = None # 相机内参
image_counter = 0  # Counter for image filenames 计数

def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames() 
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data()) # 将图像转为numpy数组

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) # 转为灰度图用于AprilTag检测

        # Detect AprilTags with pose estimation
        detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

        # Draw detection results
        for detection in detections:
            # Draw tag border
            for idx in range(len(detection.corners)):
                pt1 = tuple(map(int, detection.corners[idx]))
                pt2 = tuple(map(int, detection.corners[(idx + 1) % 4]))
                cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

            # Draw center
            center = tuple(map(int, detection.center))
            cv2.circle(color_image, center, 5, (0, 0, 255), -1)

            # Display ID
            cv2.putText(color_image, f"ID: {detection.tag_id}",
                        (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Extract pose
            t = detection.pose_t.flatten()  # Translation vector
            R = detection.pose_R  # Rotation matrix

            # Display translation in GUI
            pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
            cv2.putText(color_image, pose_text, (center[0] - 50, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw coordinate axes (X = red, Y = green, Z = blue)
            axis_length = 0.2  # Reduced axis length for better visualization
            axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])  # X, Y, Z
            axes_img_pts = []
            for i in range(3):
                axis_end = t + R @ axes[i]  # Transform axis endpoints
                x, y = int(fx * axis_end[0] / axis_end[2] + cx), int(fy * axis_end[1] / axis_end[2] + cy)
                axes_img_pts.append((x, y))

            cv2.line(color_image, center, axes_img_pts[0], (0, 0, 255), 2)  # X - Red
            cv2.line(color_image, center, axes_img_pts[1], (0, 255, 0), 2)  # Y - Green
            cv2.line(color_image, center, axes_img_pts[2], (255, 0, 0), 2)  # Z - Blue (pointing outward)

            # Print pose to console
            # print(f"Tag ID: {detection.tag_id}")
            # print(f"Translation (m): x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f}")
            # print(f"Rotation matrix:\n{R}\n")

            # Create homogeneous transformation matrix
            # 相机坐标到AprilTag坐标系的变换
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = R
            homogeneous_matrix[:3, 3] = t

        # Show images
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key == ord('r'):
            # Generate a random 4x4 matrix
            # 末端到基座
            hostname = '192.168.1.100'
            panda = panda_py.Panda(hostname)

            matrix = panda.get_pose()
            print("pose_matrix: ", matrix)
            # Save the image to a file
            image_filename = f'{image_counter}.png'
            cv2.imwrite(image_filename, color_image)
            # Save the image filename, random matrix, and homogeneous transformation matrix as a pair
            data.append({
                'image': image_filename,
                'matrix': matrix.tolist(),
                'homogeneous_matrix': homogeneous_matrix.tolist()
            })
            print(f"Image, matrix, and homogeneous matrix recorded: {image_filename}")
            image_counter += 1  # Increment the counter

        elif key == ord('q'):
            # Get intrinsics
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics_data = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                # 'model': intrinsics.model,
                # 'coeffs': intrinsics.coeffs
            }
            # Save data to JSON file
            save_to_json('data.json', {'data': data, 'intrinsics': intrinsics_data})
            print("Intrinsics saved and exiting.")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()