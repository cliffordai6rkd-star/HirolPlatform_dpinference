'''
Description: 手眼标定计算矩阵函数
Version: V1.0
Author: Gaojing
Date: 2025-05-06
Copyright (C) 2024-2050 Corenetic Technology Inc All rights reserved.
'''
import json
import cv2
import numpy as np
import os

def load_matrices(file_path):
    """
    加载JSON文件中的矩阵数据
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 检查数据结构是否包含期望的字段
    if 'data' in data:
        entries = data['data']
    else:
        entries = data  # 假设数据直接是条目列表
    
    base_ee_matrices = []
    cam_tag_matrices = []
    
    for entry in entries:
        if 'T_base_ee' in entry and 'T_cam_tag' in entry:
            base_ee_matrices.append(np.array(entry['T_base_ee']))
            cam_tag_matrices.append(np.array(entry['T_cam_tag']))
    
    return base_ee_matrices, cam_tag_matrices

def perform_hand_eye_calibration(matrices, homogeneous_matrices):
    # Convert转换 lists to numpy arrays
    # end to base
    R_gripper2base = [np.array(matrix[:3,:3]) for matrix in matrices]
    t_gripper2base = [np.array(matrix[:3, 3]) for matrix in matrices]
    
    # base to end
    R_base2gripper = [R.T for R in R_gripper2base]
    t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base, t_gripper2base)]

    # tag2cam##
    R_target2cam = [np.array(matrix[:3,:3]) for matrix in homogeneous_matrices]
    t_target2cam = [np.array(matrix[:3, 3]) for matrix in homogeneous_matrices]
    
    # Perform hand-eye calibration
    # input: base2end (getpose()-1), tag2cam (json)
    # output: cam2base (calibrateHandEye)
    R_cam2gbase, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper, R_target2cam, t_target2cam
    )

    
    # Create homogeneous transformation matrix
    cam2base_transformation_matrix = np.eye(4)
    cam2base_transformation_matrix[:3, :3] = R_cam2gbase
    cam2base_transformation_matrix[:3, 3] = t_cam2base.flatten()  # Flatten the column vector
            
    return cam2base_transformation_matrix




# Example usage
file_path = os.path.join(os.path.dirname(__file__), 'data.json')
print("Loading matrices from:", file_path)
matrices, homogeneous_matrices = load_matrices(file_path)

cam2base_transformation_matrix = perform_hand_eye_calibration(matrices, homogeneous_matrices)


print("Transformation from camera in base corrdinate system is:")
print(cam2base_transformation_matrix)





