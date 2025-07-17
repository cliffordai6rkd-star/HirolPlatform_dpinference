# import json
# import cv2
# import numpy as np

# def load_matrices(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
    
#     matrices = [] # 在基座坐标系下的末端的位置
#     homogeneous_matrices = [] # 在相机坐标系下的标定板的位置
    
#     for entry in data['data']:
#         matrices.append(np.array(entry['matrix']))
#         homogeneous_matrices.append(np.array(entry['homogeneous_matrix']))
    
#     return matrices, homogeneous_matrices


# def perform_eye_in_hand_calibration(matrices, homogeneous_matrices):
#     # 从末端到基座的变换
#     R_gripper2base = [np.array(matrix[:3,:3]) for matrix in matrices]
#     t_gripper2base = [np.array(matrix[:3, 3]) for matrix in matrices]
    
#     # 从标记到相机的变换
#     R_target2cam = [np.array(matrix[:3,:3]) for matrix in homogeneous_matrices]
#     t_target2cam = [np.array(matrix[:3, 3]) for matrix in homogeneous_matrices]
    
    
#     # 执行手眼标定
#     R_gripper2cam, t_gripper2cam = cv2.calibrateHandEye(
#         R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
#     )
    
#     # 创建同质变换矩阵
#     gripper2cam_transformation_matrix = np.eye(4)
#     gripper2cam_transformation_matrix[:3, :3] = R_gripper2cam
#     gripper2cam_transformation_matrix[:3, 3] = t_gripper2cam.flatten()
            
#     return gripper2cam_transformation_matrix


# # Example usage
# file_path = './data.json'
# matrices, homogeneous_matrices = load_matrices(file_path)

# gripper2cam_transformation_matrix = perform_eye_in_hand_calibration(matrices, homogeneous_matrices)

# print("Transformation from gripper to camera is:")
# print(gripper2cam_transformation_matrix)


import json
import cv2
import numpy as np

def load_matrices(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)['data']
    # 直接返回 list of 4x4 numpy array
    matrices           = [np.array(d['matrix'])             for d in data]  # H_gripper→base
    homogeneous_mats   = [np.array(d['homogeneous_matrix']) for d in data]  # H_tag→cam
    return matrices, homogeneous_mats

def eye_in_hand_calibration(matrices, homogeneous_matrices):
    # ——— 1) 准备第一组：R_gripper2base, t_gripper2base ———
    R_gripper2base = [M[:3, :3]   for M in matrices]
    t_gripper2base = [M[:3,  3:]  for M in matrices]  # shape (3,1)
    
    # ——— 2) 准备第二组：R_target2cam, t_target2cam ———
    R_target2cam = [M[:3, :3]   for M in homogeneous_matrices]
    t_target2cam = [M[:3,  3:]  for M in homogeneous_matrices]  # shape (3,1)
    
    # ——— 3) 调用 OpenCV 标定 ———
    # 返回 R_cam2gripper 和 t_cam2gripper（都是 3×3 / 3×1）
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,    t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # ——— 4) 构造 H_cam2gripper ———
    H_cam2gripper = np.eye(4)
    H_cam2gripper[:3,:3] = R_cam2gripper
    H_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    # ——— 5) 如果你真正需要的是 “末端→相机” (H_gripper2cam)，再取逆 ———
    H_gripper2cam = np.linalg.inv(H_cam2gripper)
    
    return H_cam2gripper, H_gripper2cam

# —— 用法示例 ——  
file_path = './data.json'
matrices, homogeneous_matrices = load_matrices(file_path)
H_cam2gripper, H_gripper2cam = eye_in_hand_calibration(matrices, homogeneous_matrices)

print("Camera → Gripper:\n", H_cam2gripper)
print("Gripper → Camera:\n", H_gripper2cam)


