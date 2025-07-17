import pyrealsense2 as rs
import numpy as np
import cv2
import panda_py  # 添加导入机器人控制库
from utilis.IK_solver import IK_solve
from panda_py import libfranka
import time

# 手眼标定矩阵（转换：相机坐标 -> 基坐标）
# T_cam_to_base = np.array([[0.11706393, 0.80613097, -0.58004215, 1.06607572],
#                           [0.99255536, -0.07519943, 0.09580655, -0.06666815],
#                           [0.03361379, -0.58693943, -0.80893276, 0.81717161],
#                           [0., 0., 0., 1.]])

T_cam_to_base = np.array(
[[ 0.03972886 , 0.8599489 , -0.50883152 , 1.02822117],
 [ 0.99920286, -0.032201  ,  0.02359519 ,-0.0026208 ],
 [ 0.00390578, -0.50936332, -0.86054271  ,0.86001323],
 [ 0.  ,        0.    ,      0.     ,     1.        ]]
)

# 控制机器人移动到点击位置
def move_to_clicked_point(p_base):
    # 获取当前末端位姿
    current_pose = panda.get_pose()
    

    # 创建新位姿矩阵，保持当前姿态，修改位置为点击位置
    target_pose = current_pose.copy()

    
    target_pose[:3, 3] = p_base  # 替换位置部分
    print("目标位姿:\n", target_pose)
    
    # 将计算出来的基座系下位置放入矩阵中
    # 创建保护位姿pose1，在目标位置上方5cm
    pose1 = target_pose.copy()
    pose1[2, 3] = p_base[2] + 0.05  # Z轴抬高10cm
    print("保护位姿(抬高5cm):\n", pose1)

    # 控制机器人移动到目标位置
    # 根据panda_py的API调用相应函数
    
    panda.move_to_pose(pose1)
    # q = panda.q
    # q_1 = IK_solve(pose1, q)
    # print("保护位姿的关节角度:", q_1)
    # print("保护位姿:", pose1)
    time.sleep(1)  # 等待1秒，确保机器人到达保护位姿
    # q1 = panda_py.ik(target_pose)
    q_0 = panda.q

    # q_1 = IK_solve(p_base, q_0)
    # print(q_1)
    # print("机器人正在移动到点击位置:", q_1)
    gripper.move(0.0, 0.2)
    panda.move_to_pose(target_pose)
    panda.move_to_pose(target_pose)
    panda.move_to_pose(target_pose)
    
    print(target_pose)

    # if q_1 is not None:
    #     print("逆运动学解算成功，移动到:", p_base)
    #     panda.move_to_joint_position(q_1)
    # else:
    #     print("逆运动学无解，目标位置超出工作空间")

    # panda.move_to_position(p_base)  # 只移动位置
    
    # print("机器人正在移动到点击位置:", q_1)


# -----------------------------
# 连接机器人控制器
hostname = '192.168.1.100'  # 替换为你的FR3机器人IP
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
# 同时启用彩色图和深度图
config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 开始流
profile = pipeline.start(config)

# 获取彩色图像内参
color_profile = profile.get_stream(rs.stream.color)
intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
print(f"Camera intrinsics: fx={intrinsics.fx}, fy={intrinsics.fy}, ppx={intrinsics.ppx}, ppy={intrinsics.ppy}")

# 构造相机内参矩阵（仅用于可选的手动计算）
CAMERA_MATRIX = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]], dtype=np.float32)

# 获取深度传感器并提取深度尺度（将深度值转为米）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale}")

# 全局变量保存当前帧，供鼠标回调使用
current_depth_frame = None
current_color_frame = None


def mouse_callback(event, x, y, flags, param):
    global current_depth_frame, current_color_frame, intrinsics, depth_scale
    if event == cv2.EVENT_LBUTTONDOWN:
        # 点击时获取该像素的深度（单位：米）
        if current_depth_frame is None:
            print("当前没有深度帧数据")
            return

        depth_value = current_depth_frame.get_distance(x, y)
        print(f"像素 ({x}, {y}) 处深度: {depth_value:.3f} m")

        # 利用 RealSense SDK 函数将像素点转换为相机坐标系下的 3D 点
        point_camera = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)
        point_camera = np.array(point_camera)  # [X, Y, Z]（单位：米）
        print("相机坐标系下的 3D 点:", point_camera)

        # 转换到齐次坐标（4x1向量）
        point_camera_hom = np.append(point_camera, 1.0)
        # 使用手眼标定矩阵转换到基坐标系
        point_base_hom = T_cam_to_base @ point_camera_hom
        # 如果最后一个分量不为1，需要归一化
        point_base = point_base_hom[:3] / point_base_hom[3]
        print("基坐标系下的 3D 点:", point_base)

        # 在图像上标记点击位置
        cv2.circle(current_color_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(current_color_frame, "Press 'g' to move, 'c' to cancel", (x+10, y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Color Frame', current_color_frame)
        
        # 确保窗口置顶获取焦点
        cv2.setWindowProperty('Color Frame', cv2.WND_PROP_TOPMOST, 1)
        
        # 询问是否要移动到该位置，使用循环等待有效按键
        print("按 'g' 键移动到点击位置，按 'c' 键取消")
        
        # 循环等待有效按键
        while True:
            key = cv2.waitKey(100) & 0xFF  # 每100ms检查一次按键
            if key == 255:  # 无按键，继续等待
                continue
            print(f"按下的键值: {key}, g键的ASCII码: {ord('g')}, c键的ASCII码: {ord('c')}")
            
            if key == ord('g'):
                try:
                    print("按下G键，准备移动")
                    move_to_clicked_point(point_base)
                except Exception as e:
                    print(f"移动过程中发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                break
            elif key == ord('c') or key == 27:  # c键或ESC键取消
                print("取消移动")
                break
            else:
                print(f"无效按键: {key}，请按'g'移动或'c'取消")


# 设置显示窗口及鼠标回调
cv2.namedWindow('Color Frame', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Color Frame', mouse_callback)

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 更新全局变量，便于鼠标回调获取最新帧数据
        current_depth_frame = depth_frame
        current_color_frame = np.asanyarray(color_frame.get_data())

        # 显示彩色图
        cv2.imshow('Color Frame', current_color_frame)
        key = cv2.waitKey(1)
        if key == 27:  # 按 ESC 退出
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()