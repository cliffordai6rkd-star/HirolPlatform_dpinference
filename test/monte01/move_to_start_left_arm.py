import time
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

HOME_JOINT1 = [35.9, 4.5, 17.7,
              85, -271, 91,
              18.7]  # 回到安全位的关节角（deg）

direction = np.array([-0.81405081,  0.58064973 ,-0.01292953])
direction /= np.linalg.norm(direction)

def move_j1(angles):
    """
    angles: 7个轴的角度，单位为 degree
    """
    try:
        code = arm1.set_servo_angle(angle=angles, is_radian=False, wait=True)
        if code != 0:
            raise RuntimeError(f"xArm set_servo_angle failed with code {code}")
    except Exception as e:
        time.sleep(1)
        _, warn = arm1.get_err_warn_code()
        if warn and warn[0] == 31:
            print(f"Detected warning code 31, attempting recovery...")
            arm1.set_state(state=0)  # 清除警告状态
            code, pose = arm1.get_position()
            safe_pose = pose - direction * 100
            arm1.set_position(safe_pose[0], safe_pose[1], safe_pose[2], pose[3], pose[4], pose[5], wait=True)
        arm1.set_state(state=0)
        arm1.set_servo_angle(angle=HOME_JOINT1, is_radian=False, wait=True)
        raise RuntimeError(f"Failed to move joints: {e}")

ip = '192.168.11.11'
arm1 = XArmAPI(ip)
arm1.set_control_modbus_baudrate(921600)
# TCP坐标系偏移
arm1.set_tcp_offset([0, 0, 172, 0, 0, 0])
arm1.motion_enable(enable=True)
arm1.set_mode(0)
arm1.set_state(state=0)
move_j1(HOME_JOINT1)
