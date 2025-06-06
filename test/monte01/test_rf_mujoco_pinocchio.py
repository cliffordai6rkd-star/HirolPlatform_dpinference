from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
from motion.kinematics import PinocchioKinematicsModel as KinematicsModel
import os
import glog as log
import numpy as np
import pinocchio as pin
from tools import file_utils
sim = Monte01Mujoco()

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(cur_path, '../../hardware/monte01/config/arm.yaml')
config = file_utils.read_config(robot_config_file)
print(f"Configuration loaded: {config}")
urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
print(f"URDF Path: {urdf_path}")
base_link = 'chest_link'
print(f"Base Link: {base_link}")
end_link = 'left_arm_link_7'
print(f"End Link: {end_link}")
kinematics=None
body_kinematics=None
try:
    kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)
except Exception as e:
    log.error(f"Failed to load URDF: {e}")

# --- 新增：身體運動學模型 ---
# 這個模型描述從機器人底部到胸部的鏈
body_base_link = 'base_link' # URDF 的根
body_end_link = 'chest_link'
try:
    body_kinematics = KinematicsModel(
        urdf_path=urdf_path, 
        base_link=body_base_link, 
        end_effector_link=body_end_link
    )
    # 獲取身體部分的關節名稱列表
    body_joint_names = [body_kinematics.model.names[i] for i in range(1, body_kinematics.model.njoints)]
    log.info(f"載入 BODY 運動學模型成功，關節名稱: {body_joint_names}")
except Exception as e:
    log.error(f"載入 BODY 運動學模型失敗: {e}")
    body_kinematics = None

pin_model = kinematics.model # 這是手臂的縮減模型
pin_data = kinematics.data

# 獲取手臂基座 chest_link 在兩個模型中的世界姿態 (我們已知它們是匹配的)
T_world_to_chest_pin = np.eye(4)
arm_kin = kinematics
body_kin = body_kinematics
# 計算 body 在零位時的姿態
body_q_zero = np.zeros(body_kin.n_joints)
T_world_to_chest_at_zero = body_kin.fk(body_q_zero)

# 計算 arm 在零位時的姿態 (相對於 chest)
arm_q_zero = np.zeros(arm_kin.n_joints)
T_chest_to_hand_at_zero = arm_kin.fk(arm_q_zero)

T_world_to_chest_pin = T_world_to_chest_at_zero
log.info(f"T_world_to_chest_pin:\n{T_world_to_chest_pin}")

chest_id = sim.mj_model.body('chest_link').id
hand_id = sim.mj_model.body('left_arm_link_7').id

# 獲取它們在世界座標系下的姿態矩陣
T_world_to_chest_mujoco = np.eye(4)
T_world_to_chest_mujoco[:3, :3] = sim.mj_data.xmat[chest_id].reshape(3, 3)
T_world_to_chest_mujoco[:3, 3] = sim.mj_data.xpos[chest_id]

# 儲存原始關節位置
original_joint_positions = {}
for joint_name in config['joint_names_left'] + config['joint_names_right']:
    joint = sim.mj_model.joint(joint_name)
    qpos_addr = joint.qposadr[0]
    original_joint_positions[joint_name] = sim.mj_data.qpos[qpos_addr]

# 設定手臂關節位置為零
zero_joint_positions = {}
for joint_name in config['joint_names_left'] + config['joint_names_right']:
    zero_joint_positions[joint_name] = 0.0
sim.set_joint_positions(zero_joint_positions)
sim.forward()

# 從 MuJoCo 獲取 chest 姿態
T_world_to_chest_mujoco = np.eye(4)
chest_id = sim.mj_model.body('chest_link').id
T_world_to_chest_mujoco[:3, :3] = sim.mj_data.xmat[chest_id].reshape(3, 3)
T_world_to_chest_mujoco[:3, 3] = sim.mj_data.xpos[chest_id]

print("\n--- T_world_to_chest_mujoco ---")
print(T_world_to_chest_mujoco)

# 恢復原始關節位置
sim.set_joint_positions(original_joint_positions)
sim.forward()

arm_links_to_check = [
    'left_arm_link_1', 
    'left_arm_link_2',
    'left_arm_link_3',
    'left_arm_link_4'
    'left_arm_link_5'
    'left_arm_link_6'
    'left_arm_link_7'
]

# 零位關節角度
arm_q_zero = np.zeros(pin_model.nq)

# Pinocchio FK
pin.forwardKinematics(pin_model, pin_data, arm_q_zero)
pin.updateFramePlacements(pin_model, pin_data)

print("--- Link-by-Link Pose Comparison (all joints at zero) ---")
for link_name in arm_links_to_check:
    # --- Pinocchio ---
    pin_frame_id = pin_model.getFrameId(link_name)
    T_chest_to_link_pin = pin_data.oMf[pin_frame_id]
    T_world_to_link_pin = T_world_to_chest_pin @ T_chest_to_link_pin.homogeneous
    
    # --- MuJoCo ---
    mjc_body_id = sim.mj_model.body(link_name).id
    T_world_to_link_mjc = np.eye(4)
    with sim.locker:
        T_world_to_link_mjc[:3, :3] = sim.mj_data.xmat[mjc_body_id].reshape(3, 3)
        T_world_to_link_mjc[:3, 3] = sim.mj_data.xpos[mjc_body_id]
        
    print(f"\n--- Checking Link: {link_name} ---")
    print("Pinocchio World Pose:\n", T_world_to_link_pin)
    print("MuJoCo World Pose:\n", T_world_to_link_mjc)

    # 比較兩個矩陣
    if not np.allclose(T_world_to_link_pin, T_world_to_link_mjc, atol=1e-3):
        print(f"***** MISMATCH FOUND AT LINK: {link_name} *****")
        # 找到第一個不匹配的連結後，就可以集中火力解決它
        break
