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

# 定义标签尺寸（米）
tag_size = 0.0557  # 64 毫米 

# 初始化Intel RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 开始数据流
profile = pipeline.start(config)

# 动态获取相机内参
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
camera_params = [fx, fy, cx, cy]

print(f"相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# 初始化AprilTag检测器
detector = Detector(families="tag25h9")

# 创建窗口
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# 初始化变量
data = [] # 标定数据列表
intrinsics = None # 相机内参
image_counter = 0  # 图像文件名计数

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
#         # 等待机器人完全停止运动
#         time.sleep(2)
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
    print("1. 按 'a' 键开始标定过程，机器人将自动移动到各个网格点")
    print("2. 在每个点，系统会等待检测到恰好一个AprilTag后自动记录数据")
    print("3. 所有点都完成后，请按 'q' 键保存数据并退出")
    print("4. 按 's' 键可以跳过当前点位")
    
    # 添加新的变量
    robot_ready = False  # 机器人是否已到达位置并等待记录
    stable_detection_count = 0  # 稳定检测计数
    last_detection_id = -1  # 上一次检测的标签ID
    stability_threshold = 10  # 需要连续检测到相同标签的次数
    
    while True:
        # 等待获取帧：彩色
        frames = pipeline.wait_for_frames() 
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 将图像转为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 转为灰度图用于AprilTag检测
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 检测AprilTags并进行姿态估计
        detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

        # 添加标定状态信息
        if calibration_started:
            if robot_ready:
                # 显示检测状态
                if len(detections) == 0:
                    status_text = f"点 {current_point_index+1}/{len(calibration_points)} - 等待检测AprilTag"
                    stable_detection_count = 0
                elif len(detections) > 1:
                    status_text = f"点 {current_point_index+1}/{len(calibration_points)} - 检测到多个标签，请保留一个"
                    stable_detection_count = 0
                else:
                    current_tag_id = detections[0].tag_id
                    if last_detection_id != current_tag_id:
                        last_detection_id = current_tag_id
                        stable_detection_count = 1
                    else:
                        stable_detection_count += 1
                    
                    status_text = f"点 {current_point_index+1}/{len(calibration_points)} - 稳定检测中 {stable_detection_count}/{stability_threshold}"
            else:
                status_text = f"点 {current_point_index+1}/{len(calibration_points)} - 移动中..."
        else:
            status_text = "按 'a' 键开始标定"
        
        cv2.putText(color_image, status_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 绘制检测结果
        homogeneous_matrix = None
        for detection in detections:
            # 提取标签角点和中心
            corners = detection.corners
            center = detection.center
            corners = corners.astype(int)
            center = center.astype(int)
            
            # 绘制标签轮廓和ID
            cv2.polylines(color_image, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            cv2.putText(color_image, str(detection.tag_id), (center[0], center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 获取姿态信息
            if detection.pose_R is not None and detection.pose_t is not None:
                # 创建齐次变换矩阵
                homogeneous_matrix = np.zeros((4, 4))
                homogeneous_matrix[:3, :3] = detection.pose_R
                homogeneous_matrix[:3, 3] = detection.pose_t.flatten()
                homogeneous_matrix[3, 3] = 1.0
                # 绘制3D坐标轴
                # 定义坐标轴长度
                axis_length = 0.05  # 5厘米
                
                # 定义坐标轴3D点（从原点到X、Y、Z轴）
                axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
                
                # 构建相机内参矩阵
                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                
                # 使用相机内参和位姿将3D点投影到2D图像平面
                image_points, _ = cv2.projectPoints(
                    axis_points, 
                    detection.pose_R, 
                    detection.pose_t, 
                    camera_matrix, 
                    distCoeffs=None
                )
                
                # 转换为整数坐标
                image_points = image_points.reshape(-1, 2).astype(int)
                
                # 绘制坐标轴线（X=红色，Y=绿色，Z=蓝色）
                origin = tuple(image_points[0])
                cv2.line(color_image, origin, tuple(image_points[1]), (0, 0, 255), 2)  # X轴（红色）
                cv2.line(color_image, origin, tuple(image_points[2]), (0, 255, 0), 2)  # Y轴（绿色）
                cv2.line(color_image, origin, tuple(image_points[3]), (255, 0, 0), 2)  # Z轴（蓝色）
                
                # 在坐标轴端点绘制轴标识
                cv2.putText(color_image, "X", tuple(image_points[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(color_image, "Y", tuple(image_points[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(color_image, "Z", tuple(image_points[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 显示图像
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        # 自动记录数据的逻辑
        if calibration_started and robot_ready and len(detections) == 1 and stable_detection_count >= stability_threshold:
            try:
                # 获取机器人当前位姿
                matrix = panda.get_pose()
                print("姿态矩阵: ", matrix)
                
                # 保存图像
                image_filename = f'point_{current_point_index}_{image_counter}.png'
                cv2.imwrite(image_filename, color_image)
                
                # 保存数据
                data.append({
                    'point_index': current_point_index,
                    'calibration_point': calibration_points[current_point_index],
                    'image': image_filename,
                    'matrix': matrix.tolist(),
                    'homogeneous_matrix': homogeneous_matrix.tolist() if len(detections) > 0 else None
                })
                
                print(f"自动记录点 {current_point_index+1}/{len(calibration_points)} 的数据")
                image_counter += 1
                
                # 重置标志
                robot_ready = False
                stable_detection_count = 0
                last_detection_id = -1
                
                # 移动到下一个点
                current_point_index += 1
                if current_point_index < len(calibration_points):
                    print(f"移动到下一个点 {current_point_index+1}/{len(calibration_points)}")
                    if move_robot_to_point(calibration_points[current_point_index]):
                        robot_ready = True
                else:
                    print("\n所有点都已完成! 请按 'q' 键保存数据并退出")
                    calibration_started = False
            except Exception as e:
                print(f"记录数据时出错: {e}")

        # 处理按键事件
        if key == ord('a') and not calibration_started:
            # 开始标定过程
            calibration_started = True
            current_point_index = 0
            print(f"\n开始标定过程! 总共有 {len(calibration_points)} 个标定点")
            # 移动到第一个点
            if move_robot_to_point(calibration_points[current_point_index]):
                robot_ready = True
                stable_detection_count = 0
                last_detection_id = -1
            print("等待检测到一个稳定的AprilTag进行自动记录")
            
        elif key == ord('s') and calibration_started and robot_ready:
            # 跳过当前点
            print(f"跳过点 {current_point_index+1}/{len(calibration_points)}")
            robot_ready = False
            stable_detection_count = 0
            
            # 移动到下一个点
            current_point_index += 1
            if current_point_index < len(calibration_points):
                print(f"移动到下一个点 {current_point_index+1}/{len(calibration_points)}")
                if move_robot_to_point(calibration_points[current_point_index]):
                    robot_ready = True
            else:
                print("\n所有点都已完成! 请按 'q' 键保存数据并退出")
                calibration_started = False


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
    # 停止数据流
    pipeline.stop()
    cv2.destroyAllWindows()