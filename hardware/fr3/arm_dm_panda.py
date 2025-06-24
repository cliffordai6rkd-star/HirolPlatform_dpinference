import sys
sys.path.append("../../dependencies/libfranka-python/franka_bindings")

from hardware.base.arm import ArmBase
# from hardware.fr3.gripper import Gripper
import glog as log
import numpy as np

import time, os
from typing import Text, Mapping, Any

from motion.kinematics import PinocchioKinematicsModel as KinematicsModel

import numpy as np

import threading

import dm_env
import numpy as np
from absl import logging

# dm_robotics imports
from dm_robotics.panda import environment
from scipy.spatial.transform import Rotation as R
from hardware.fr3.gripper_panda_py import Gripper

kJointPositionStartData = np.array([
    0.0,
    -np.pi / 4,
    0.0,
    -3 * np.pi / 4,
    0.0,
    np.pi / 2,
    np.pi / 4,
])
ARRIVAL_TOLERANCE = 0.0025 # 此处减小，可以显著提升精度，但到达这个位置需要的时间也增加
class Arm(ArmBase):
    def __init__(self, config: Mapping[Text, Any],
        env: environment.PandaEnvironment,
        kp: float = 1.18,  # ku:=1.8
        ki: float = 0.0,
        kd: float = 0.1):
        super().__init__()

        self.config = config
        log.info(f"Robot instance created with IP: {config['ip']}")
        self.gripper_open = 0
        # self._gripper = Gripper(config['gripper'])

        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
        base_link = config['base_link']
        end_link = config['end_link']

        try:
            self.kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)
        except Exception as e:
            log.error(f"Failed to load URDF: {e}")

        self.flange_t_tcp = np.eye(4)

        self.tcp_t_flange = self.flange_t_tcp.T

        mode = config['control_mode']

        self.mode = mode

        self._env = env
        self._kp = kp
        self._action_spec = env.action_spec()

        # 執行緒安全相關
        self._lock = threading.Lock()
        self._move_done = threading.Event()

        # 目標與狀態
        self._target_joint_positions: np.ndarray | None = None
        self._current_timestep: dm_env.TimeStep | None = None
        self._observation = None
        self.arr_t=[]
        self.arr_r=[]

        self._ki = ki # 新增的積分增益
        self._kd = kd
        # 用於儲存積分誤差
        self._error_integral = np.zeros(self._action_spec.shape, dtype=np.float64)
        self._previous_error = np.zeros(self._action_spec.shape, dtype=np.float64)
        self._last_step_time = None
        
        # 新增目標姿態，用於更精確的到達判斷
        self._target_joint_positions: np.ndarray | None = None
        self._target_pose: np.ndarray | None = None

        self.panda_gripper_width_prev = -1

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """
        每個時間步被 run_loop 呼叫的核心函式。
        計算並返回一個關節速度指令。
        """
        # 儲存最新的時間步資訊，以便其他函式可以取得
        self._current_timestep = timestep

        current_time = timestep.observation["time"][0]
        dt = (current_time - self._last_step_time) if self._last_step_time is not None else 0.001
        self._last_step_time = current_time

        with self._lock:
            if self._target_joint_positions is None:
                self._reset_controller_state()
                # 如果沒有目標，保持靜止
                self._move_done.set()  # 確保事件是設定狀態
                x = np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype)
                x[7] = self.gripper_open
                return x

            # --- P 控制器邏輯 ---
            self._observation = timestep.observation
            current_q = timestep.observation["panda_joint_pos"]
            panda_gripper_width = timestep.observation["panda_gripper_width"]
            # log.info(f"panda_gripper_width: {panda_gripper_width}")
            current_q = np.concatenate([current_q, [timestep.observation["panda_gripper_width"]]])
            error = self._target_joint_positions[:8] - current_q
            # 積分項計算
            self._error_integral += error * dt
            # 積分飽和，防止積分爆炸 (wind-up)
            np.clip(self._error_integral, -1.0, 1.0, out=self._error_integral)
            
            # 2. 微分項 (D)
            # 避免在第一步時 dt 為 0 或過大
            derivative_error = np.zeros_like(error)
            if dt > 1.1e-3:
                derivative_error = (error - self._previous_error) / dt
            else:
                derivative_error = np.zeros_like(error)

            self._previous_error = error
            # 檢查是否到達目標
            # 我們只檢查7個手臂關節，忽略夾爪
            if np.linalg.norm(error[:7]) < ARRIVAL_TOLERANCE:
                if (self.gripper_open > 0.5 and abs(panda_gripper_width-0.08)<1e-3) or (self.gripper_open <= 0.5 and abs(panda_gripper_width)<1e-3) or abs(self.panda_gripper_width_prev-panda_gripper_width)<1e-3:
                    logging.info("Target reached!")
                    self._target_joint_positions = None
                    self._move_done.set()
                    action =  np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype)
                    action[7] = self.gripper_open
                    return action
            self.panda_gripper_width_prev = panda_gripper_width
            # 計算速度指令
            action = self._kp * error + self._ki * self._error_integral + self._kd * derivative_error

            # 限制速度大小，防止抖動和超速
            action = np.clip(
                action, self._action_spec.minimum, self._action_spec.maximum
            )
            
            action[7] = self.gripper_open # binary status only: open/close
            return action
    def _reset_controller_state(self):
        """重置控制器內部狀態，在每次新移動前呼叫"""
        self._error_integral.fill(0)
        self._previous_error.fill(0)
        self._last_step_time = None
        self._target_pose = None

    def get_flange_pose(self) -> np.ndarray:
        """Gets the pose of the flange.
        """
        x: np.ndarray = self.get_joint_positions()
        
        return self.kinematics.fk(x)
    
    def get_joint_positions(self) -> np.ndarray | None:
        """獲取目前的關節角度。"""
        if self._current_timestep:
            return self._current_timestep.observation["panda_joint_pos"]
        return None

    def get_controller_time(self) -> float | None:
        """獲取控制器內部的時間。"""
        if self._current_timestep:
            return self._current_timestep.observation["time"][0]
        return None

    def set_joint_positions(self, target_q: np.ndarray, blocking: bool = True):
        """移動到指定的關節角度。"""
        logging.info(f"Setting new joint target: {target_q[:7]}...")
        with self._lock:
            self._reset_controller_state()
            # 確保輸入的維度正確
            if len(target_q) == 7: # 如果只給了7個關節角度
                full_q = np.zeros(9)
                full_q[:7] = target_q
                self._target_joint_positions = full_q
            else:
                self._target_joint_positions = target_q.copy()
            self._move_done.clear() # 清除事件，表示移動開始
        
        if blocking:
            self._move_done.wait() # 等待移動完成事件

    def get_joint_target_from_pose(self, target):
        """Gets joint configuration via IK.
        """
        flange_target = target @ self.tcp_t_flange
        return self.ik(flange_target)
    
    def getRT(self, pose):
        rotation_matrix = pose[0:3, 0:3]

        # 2. Create a Rotation object from the rotation matrix
        rot = R.from_matrix(rotation_matrix)
        euler_angles_deg = rot.as_euler('zyx', degrees=False)
        for i in range(3):
            if euler_angles_deg[i] < -np.pi/2:
                euler_angles_deg[i]+=np.pi*2
        return euler_angles_deg, pose[0:3, 3]
    
    def move_to_pose(self, target) -> bool:
        """Moves to the target that specifies TCP pose in base frame.
        """
        self.set_joint_positions(
        self.get_joint_target_from_pose(target))

        r1,t1 = self.getRT(target)
        log.info(f"pose1 == \n{target}")
        R = self._observation['panda_tcp_rmat'].reshape(3,3)
        t = self._observation['panda_tcp_pos']
        pose_matrix = np.eye(4)  # Start with a 4x4 identity matrix

        # Place the rotation matrix into the top-left 3x3 block
        pose_matrix[:3, :3] = R

        # Place the translation vector into the top-right 3x1 column
        pose_matrix[:3, 3] = t
        log.info(f"pose2 == \n{pose_matrix}")

        r2,t2 = self.getRT(pose_matrix)
        log.info(f"[move_to_pose] r1== {r1}")
        log.info(f"[move_to_pose] r2== {r2}")

        dt = t2-t1
        dr = r2-r1
        log.info(f"[move_to_pose] dx,dy,dz== {dt}")
        log.info(f"[move_to_pose] dR,dP,dY== {dr}")
        self.arr_t.append(dt)
        self.arr_r.append(dr)

        avg_dt = np.mean(self.arr_t, keepdims=True, axis=0)
        log.info(f"[move_to_pose] average dt ==  {avg_dt}")
        avg_dr = np.mean(self.arr_r, keepdims=True, axis=0)
        log.info(f"[move_to_pose] average dr ==  {avg_dr}")

        div1 = np.var(self.arr_t, ddof=1, keepdims=True, axis=0)
        log.info(f"[move_to_pose] div dt ==  {div1}")

        div2 = np.var(self.arr_r, ddof=1, keepdims=True, axis=0)
        log.info(f"[move_to_pose] div dr ==  {div2}")
      
    def move_to_start(self):
        self.set_joint_positions(kJointPositionStartData, True)

    def get_tcp_pose(self) -> np.ndarray:
        return self.get_flange_pose() @ self.flange_t_tcp
    
    def set_gripper_open(self, open, blocking=True):
        log.info(f"set_gripper_open {open}")
        with self._lock:
            self._reset_controller_state() # 可以選擇是否重置手臂的PID狀態
            if self._target_joint_positions is None:
                # 如果手臂沒有目標，則基於當前手臂位置設定夾爪目標
                current_arm_q = self.get_joint_positions()
                if current_arm_q is None:
                    logging.error("Cannot set gripper target, arm positions unknown.")
                    return
                self._target_joint_positions = np.zeros(8) # 與 move_to_joint_target 一致
                self._target_joint_positions[:7] = current_arm_q

            self.gripper_open = open
            self._move_done.clear()

        if blocking:
            self._move_done.wait()

    def get_gripper(self):
        return self._gripper
    
    def ik(self, pose):
        return self.kinematics.ik(pose)[:7]

    def fk(self, jp):
        return self.kinematics.fk(jp)

    def print_state(self):
        log.info(f"{self._observation}")
        
