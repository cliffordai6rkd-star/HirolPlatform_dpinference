import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from pupil_apriltags import Detector
import panda_py

# Initialize the robot
# Define tag size in meters
tag_size = 0.0557  # 55.7 mm 

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# --- 用固定内参和畸变系数替换动态获取 ---
camera_matrix = np.array([
    [917.02853276,   0.        , 650.12363944],
    [  0.        , 917.91187791, 372.4978344 ],
    [  0.        ,   0.        ,   1.        ]
], dtype=float)
dist_coeffs = np.array([  
    3.06038045e-02,  4.75049482e-01, -8.27964072e-04,
    1.58231256e-03, -1.80318321e+00
], dtype=float)

# 用固定内参设置 AprilTag 检测器的 camera_params (fx, fy, cx, cy)
camera_params = [
    camera_matrix[0,0],
    camera_matrix[1,1],
    camera_matrix[0,2],
    camera_matrix[1,2]
]

print(f"Camera Intrinsics: fx={camera_params[0]}, fy={camera_params[1]}, cx={camera_params[2]}, cy={camera_params[3]}")

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
                x, y = int(camera_params[0] * axis_end[0] / axis_end[2] + camera_params[2]), int(camera_params[1] * axis_end[1] / axis_end[2] + camera_params[3])
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
            # Save data to JSON file (固定内参 + 畸变系数)
            save_to_json('data.json', {
                'data': data,
                'camera_matrix': camera_matrix.tolist(),
                'dist_coeffs': dist_coeffs.tolist()
            })
            print("Intrinsics saved and exiting.")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()