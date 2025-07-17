#!/usr/bin/env python3
"""
Hand-eye calibration node (ROS 2 Humble).

Press  r  in the display window to record a (image, T_cam_tag, T_base_ee) trio.
Press  q  to save all data to data.json and exit.

Author: Haotian Liang
"""

import os
import json
from typing import Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

from xarm.wrapper import XArmAPI

import numpy as np
import math


def euler_to_matrix(position, euler_angles, is_degrees=True,to_meters=True):
    """
    将位置和欧拉角转换为4×4变换矩阵
    
    参数:
        position: [x, y, z] 格式的位置向量，默认单位为毫米
        euler_angles: [roll, pitch, yaw] 格式的欧拉角
        is_degrees: 指示欧拉角是否为角度单位（True）或弧度单位（False）
        to_meters: 是否将位置从毫米转换为米（True）或保持原始单位（False）
    
    返回:
        4×4的numpy数组表示变换矩阵，平移部分根据to_meters参数决定单位
    """
    # 提取位置
    px, py, pz = position
    
    # 如果需要，将毫米转换为米
    if to_meters:
        px /= 1000.0
        py /= 1000.0
        pz /= 1000.0
    
    # 提取欧拉角
    roll, pitch, yaw = euler_angles
    
    # 如果输入是角度，转换为弧度
    if is_degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    
    # 计算三角函数值
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    # 构建旋转矩阵 (ZYX顺序: yaw -> pitch -> roll)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    
    # 创建4×4变换矩阵
    matrix = np.array([
        [r00, r01, r02, px],
        [r10, r11, r12, py],
        [r20, r21, r22, pz],
        [0, 0, 0, 1]
    ])
    
    return matrix

def robot_pose_to_matrix(pose_data, is_degrees=True, to_meters=True):
    """
    将机器人API返回的欧拉角位姿直接转换为4×4变换矩阵
    
    参数:
        pose_data: 机器人位姿列表 [x, y, z, roll, pitch, yaw]
        is_degrees: 指示欧拉角是否为角度单位（True）或弧度单位（False）
    
    返回:
        4×4的numpy数组表示变换矩阵
    """
    position = pose_data[0:3]
    euler_angles = pose_data[3:6]
    return euler_to_matrix(position, euler_angles, is_degrees, to_meters)

class HandEyeCalib(Node):
    def __init__(self):
        super().__init__("hand_eye_calib")

        # ▶ xArm
        ip = "192.168.11.12"
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(2)       # 示教模式
        self.arm.set_state(state=0)
        self.save_dir = os.path.join(os.getcwd(), "hand_eye_data")

        # ▶ Charuco board (DICT_5X5, 14×9 squares, Charuco-300)
        square_len = 0.02   # m
        marker_len = 0.015  # m
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        board_size = (14, 9)     # (cols, rows) per OpenCV
        self.charuco_board = aruco.CharucoBoard(
            board_size, square_len, marker_len, self.aruco_dict)
        det_params = aruco.DetectorParameters()
        det_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        self.detector = aruco.ArucoDetector(self.aruco_dict, det_params)
    

        # ▶ ROS interfaces
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, "/camera_head_front/color/video",
            self.image_cb, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera_head_front/color/camera_info",
            self.info_cb, 10)

        # ▶ Data containers
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.latest_img: Optional[np.ndarray] = None
        self.latest_tag_pose: Optional[np.ndarray] = None  # 4×4 cam-T-tag
        self.data = []     # list of dicts to dump
        self.image_counter = 0

        # ▶ OpenCV window timer (30 Hz)
        self.timer = self.create_timer(1.0 / 30.0, self.display_loop)

        self.get_logger().info("hand_eye_calib node started")

    # ===== ROS callbacks ======================================================

    def info_cb(self, msg: CameraInfo):
        # One-time extract intrinsics
        if self.camera_matrix is None:
            k = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            d = np.array(msg.d, dtype=np.float32).reshape((-1, 1))
            self.camera_matrix, self.dist_coeffs = k, d
            self.intrinsics_msg = msg  # keep full info for JSON
            self.get_logger().info("CameraInfo received")

    def image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_img = cv_img
            if self.camera_matrix is not None:
                self.latest_tag_pose = self.detect_charuco(cv_img)
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")

    # ===== Charuco detection ==================================================

    def detect_charuco(self, img: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, self.charuco_board,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs)
        if retval < 4:
            return None

        # ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
        #     charuco_corners, charuco_ids, self.charuco_board,
        #     self.camera_matrix, self.dist_coeffs)
        try:
            # OpenCV ≥ 4.7：rvec / tvec 必填
            ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.charuco_board,
                self.camera_matrix, self.dist_coeffs,
                None,   # rvec 占位
                None    # tvec 占位
            )
        except TypeError:
            # 旧版 OpenCV（≤ 4.5）依旧用 5 参数
            ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.charuco_board,
                self.camera_matrix, self.dist_coeffs
            )

        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R
        T_cam_tag[:3, 3] = tvec.flatten()
        return T_cam_tag

    # ===== Display / key handling ============================================

    def display_loop(self):
        if self.latest_img is None:
            return

        img_disp = self.latest_img.copy()

        # Draw tag axes if pose available
        if self.latest_tag_pose is not None:
            rvec, _ = cv2.Rodrigues(self.latest_tag_pose[:3, :3])
            tvec = self.latest_tag_pose[:3, 3].reshape((3, 1))
            cv2.drawFrameAxes(img_disp, self.camera_matrix,
                              self.dist_coeffs, rvec, tvec, 0.2)

        cv2.imshow("hand_eye_calib", img_disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            self.try_record_sample(img_disp)
        elif key == ord("q"):
            self.save_and_quit()

    # ===== Record / save ======================================================

    def try_record_sample(self, frame: np.ndarray):
        if self.latest_tag_pose is None:
            self.get_logger().warn("No Charuco pose – sample ignored")
            return

        code, xarm_pose = self.arm.get_position(is_radian=None)
        if code != 0:
            self.get_logger().error(f"xArm get_position failed, code={code}")
            return

        try:
            T_base_ee = robot_pose_to_matrix(xarm_pose)
        except Exception as e:
            self.get_logger().error(f"Pose→matrix failed: {e}")
            return

        # Save frame to disk
        fname = f"image_{self.image_counter:03d}.png"
        fpath = os.path.join(self.save_dir, fname)
        cv2.imwrite(fpath, frame)

        self.data.append({
            "image": fpath,
            "T_base_ee": T_base_ee.tolist(),
            "T_cam_tag": self.latest_tag_pose.tolist(),
        })
        self.image_counter += 1
        self.get_logger().info(
            f"Captured sample #{self.image_counter}")

    def save_and_quit(self):
        if self.camera_matrix is None:
            self.get_logger().error("Never received CameraInfo; aborting save")
            rclpy.shutdown()
            return

        intr = self.intrinsics_msg
        intrinsics_data = {
            "width": intr.width,
            "height": intr.height,
            "ppx": intr.k[2],
            "ppy": intr.k[5],
            "fx": intr.k[0],
            "fy": intr.k[4],
        }

        out = {"data": self.data, "intrinsics": intrinsics_data}
        json_path = os.path.join(self.save_dir, "data.json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=4)

        self.get_logger().info(
            f"Saved {len(self.data)} samples to data.json – shutting down")
        cv2.destroyAllWindows()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = HandEyeCalib()
    rclpy.spin(node)
    # spin exits via save_and_quit

