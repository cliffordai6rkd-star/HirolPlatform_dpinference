import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from pupil_apriltags import Detector
import panda_py
import time

# 从用户获取机器人IP地址
hostname = input("请输入机器人的IP地址: ")
print(f"连接到机器人: {hostname}")
panda = panda_py.Panda(hostname)

# 获取工作空间3D范围
print("\n=== 设定工作空间3D范围 ===")
workspace = {}
try:
    workspace['x_min'] = float(input("请输入X轴最小值 (米):  ，最小0.3M"))
    workspace['x_max'] = float(input("请输入X轴最大值 (米): ，最大0.75M"))
    workspace['y_min'] = float(input("请输入Y轴最小值 (米): ，最小-0.5M"))
    workspace['y_max'] = float(input("请输入Y轴最大值 (米): ，最大0.5M"))
    workspace['z_min'] = float(input("请输入Z轴最小值 (米): "))
    workspace['z_max'] = float(input("请输入Z轴最大值 (米): "))
    
    # 验证输入合法性
    if workspace['x_min'] >= workspace['x_max'] or \
       workspace['y_min'] >= workspace['y_max'] or \
       workspace['z_min'] >= workspace['z_max']:
        raise ValueError("范围值无效：最小值必须小于最大值")
    
    # 新增：获取标定点数量
    num_calibration_points = int(input("\n请输入希望生成的标定点总数 (推荐20-30个): "))
    if num_calibration_points < 15:
        print("警告：点数过少可能影响标定精度，已设置为最小值15")
        num_calibration_points = 15
    elif num_calibration_points > 50:
        print("警告：点数过多可能导致标定时间过长，已设置为最大值50") 
        num_calibration_points = 50
        
    print(f"工作空间设置成功: X: [{workspace['x_min']}, {workspace['x_max']}], " 
          f"Y: [{workspace['y_min']}, {workspace['y_max']}], "
          f"Z: [{workspace['z_min']}, {workspace['z_max']}]")
except ValueError as e:
    print(f"输入错误: {e}. 使用默认工作空间范围。")
    # 设置默认工作空间范围
    workspace = {
        'x_min': 0.3, 'x_max': 0.5,  
        'y_min': -0.3, 'y_max': 0.3,
        'z_min': 0.2, 'z_max': 0.4
    }
    print(f"使用默认工作空间: X: [{workspace['x_min']}, {workspace['x_max']}], " 
          f"Y: [{workspace['y_min']}, {workspace['y_max']}], "
          f"Z: [{workspace['z_min']}, {workspace['z_max']}]")
    num_calibration_points = 24  # 默认点数
    print(f"使用默认标定点数: {num_calibration_points}")

# 生成3D标定网络点
def generate_calibration_grid(workspace, num_points=24):
    # 计算每个轴上的点数
    x_count = int(np.ceil(num_points**(1/3)))
    y_count = int(np.ceil(num_points**(1/3)))
    z_count = int(np.ceil(num_points**(1/3)))
    
    # 生成均匀分布的点calibration_points = generate_calibration_grid(workspace, num_calibration_points)
    x_values = np.linspace(workspace['x_min'], workspace['x_max'], x_count)
    y_values = np.linspace(workspace['y_min'], workspace['y_max'], y_count)
    z_values = np.linspace(workspace['z_min'], workspace['z_max'], z_count)
    
    grid_points = []
    for x in x_values:
        for y in y_values:
            for z in z_values:
                grid_points.append([x, y, z])
    
    print(f"生成了 {len(grid_points)} 个标定网络点")
    return grid_points

# 创建标定网络点
calibration_points = generate_calibration_grid(workspace, num_calibration_points)
current_point_index = 0
calibration_started = False

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

# def move_robot_to_point(point):
#     """移动机器人到指定的3D点位置"""
#     try:
#         # 获取当前位姿
#         current_pose = panda.get_pose()
#         # 创建新位姿（保持原有的旋转，只改变位置）
#         new_pose = current_pose.copy()
#         # 更新位置
#         new_pose[:3, 3] = point
#         # 移动到新位置
#         print(f"移动到点位: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")
#         panda.move_to_pose(new_pose)
#         return True
#     except Exception as e:
#         print(f"移动机器人失败: {e}")
#         return False

def move_robot_to_point(point):
    """移动机器人到指定的3D点位置，直接使用基于初始姿态的随机旋转"""
    global initial_rotation, initial_pose
    
    max_attempts = 5  # 最大尝试次数
    
    try:
        # 如果是第一次移动，保存初始位姿作为参考
        if 'initial_pose' not in globals():
            initial_pose = panda.get_pose().copy()
            initial_rotation = initial_pose[:3, :3].copy()
            print("已保存初始位姿作为参考")
            
        for attempt in range(max_attempts):
            try:
                # 创建新位姿（基于初始位姿）
                new_pose = initial_pose.copy()
                # 更新位置
                new_pose[:3, 3] = point
                
                # 直接生成随机旋转，而不是累积旋转
                # 随着尝试次数增加，减小范围
                max_angle = 0.15 - 0.02 * attempt  # 最大约±9度
                max_angle = max(0.02, max_angle)   # 确保至少有一些随机性
                
                # 生成双向随机旋转角度 (弧度)
                rx = np.random.uniform(-max_angle, max_angle)  
                ry = np.random.uniform(-max_angle, max_angle)
                rz = np.random.uniform(-max_angle*3, max_angle*3)  # z轴范围更大
                
                # 创建旋转矩阵
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]
                ])
                
                Ry = np.array([
                    [np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]
                ])
                
                Rz = np.array([
                    [np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]
                ])
                
                # 组合旋转
                R_random = Rz @ Ry @ Rx
                
                # 直接将随机旋转应用到初始旋转上
                R_new = initial_rotation @ R_random
                
                # 更新位姿矩阵中的旋转部分
                new_pose[:3, :3] = R_new
                
                # 输出信息
                print(f"移动到点位: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")
                print(f"添加随机旋转: rx={np.degrees(rx):.1f}°, ry={np.degrees(ry):.1f}°, rz={np.degrees(rz):.1f}°")
                
                # 移动到新位置和姿态
                panda.move_to_pose(new_pose)
                return True
                
            except Exception as e:
                print(f"尝试 {attempt+1} 失败: {e}")
                if "joint wall" in str(e):
                    print("关节限制错误，减小旋转范围")
                    
        # 如果所有尝试都失败，使用纯初始姿态
        print("所有随机旋转尝试都失败，尝试使用初始姿态...")
        try:
            reset_pose = initial_pose.copy()
            reset_pose[:3, 3] = point  # 只改变位置
            panda.move_to_pose(reset_pose)
            print(f"使用初始姿态移动成功")
            return True
        except Exception as e:
            print(f"使用初始姿态移动也失败: {e}")
            
            # 最后尝试当前姿态的纯位移
            print("尝试当前姿态的纯位移...")
            current_pose = panda.get_pose()
            current_pose[:3, 3] = point
            panda.move_to_pose(current_pose)
            print(f"纯位移成功")
            return True
        
    except Exception as e:
        print(f"移动机器人失败: {e}")
        return False

try:
    print("\n=== 标定说明 ===")
    print("1. 按 'a' 键开始标定过程，机器人将移动到第一个网格点")
    print("2. 在每个点检测到AprilTag后，按 'r' 键记录该点的标定数据")
    print("3. 记录完成后，机器人将自动移动到下一个点")
    print("4. 所有点都完成后，按 'q' 键保存数据并退出")
    
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

        # 添加标定状态信息
        if calibration_started:
            status_text = f"标定中: 点 {current_point_index+1}/{len(calibration_points)}"
        else:
            status_text = "按 'a' 键开始标定"
        
        cv2.putText(color_image, status_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw detection results
        for detection in detections:
            # 绘制标签边界
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

            # Create homogeneous transformation matrix
            # 相机坐标系下的AprilTag的pose
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = R
            homogeneous_matrix[:3, 3] = t

        # Show images
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        # 处理按键事件
        if key == ord('a') and not calibration_started:
            # 开始标定过程
            calibration_started = True
            current_point_index = 0
            print(f"\n开始标定过程! 总共有 {len(calibration_points)} 个标定点")
            # 移动到第一个点
            move_robot_to_point(calibration_points[current_point_index])
            print("现在可以按 'r' 键记录当前点的数据")
        elif key == ord('l') and calibration_started:
            # 跳过当前点，移动到下一个点
            current_point_index += 1
            if current_point_index < len(calibration_points):
                print(f"跳过当前点，移动到点 {current_point_index+1}/{len(calibration_points)}")
                move_robot_to_point(calibration_points[current_point_index])
            else:
                print("\n所有点都已完成! 请按 'q' 键保存数据并退出")
                calibration_started = False

        elif key == ord('r') and calibration_started:
            # 记录当前点的数据
            try:
                # 获取机器人当前位姿
                matrix = panda.get_pose()
                print("pose_matrix: ", matrix)
                
                # 仅当检测到AprilTag时记录
                if len(detections) > 0:
                    # 保存图像
                    image_filename = f'point_{current_point_index}_{image_counter}.png'
                    cv2.imwrite(image_filename, color_image)
                    
                    # # 保存数据
                    # data.append({
                    #     'point_index': current_point_index,
                    #     'calibration_point': calibration_points[current_point_index],
                    #     'image': image_filename,
                    #     'matrix': matrix.tolist(),
                    #     'homogeneous_matrix': homogeneous_matrix.tolist() if len(detections) > 0 else None
                    # })
                     # 保存数据
                    data.append({
                        'image': image_filename,
                        'matrix': matrix.tolist(),
                        'homogeneous_matrix': homogeneous_matrix.tolist() if len(detections) > 0 else None
                    })
                    
                    print(f"记录点 {current_point_index+1}/{len(calibration_points)} 的数据")
                    image_counter += 1
                    
                    # 移动到下一个点
                    current_point_index += 1
                    if current_point_index < len(calibration_points):
                        print(f"移动到下一个点 {current_point_index+1}/{len(calibration_points)}")
                        move_robot_to_point(calibration_points[current_point_index])
                    else:
                        print("\n所有点都已完成! 请按 'q' 键保存数据并退出")
                        calibration_started = False
                else:
                    print("警告: 未检测到AprilTag，请确保标签在相机视野内")
            except Exception as e:
                print(f"记录数据时出错: {e}")

        elif key == ord('q'):
            # 获取相机内参
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics_data = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
            }
            
            # 保存数据到JSON文件
            save_to_json('data.json', {
                'data': data, 
                'intrinsics': intrinsics_data
            })
            
            print("标定数据已保存到 data.json")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()