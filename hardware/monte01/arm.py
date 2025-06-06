import importlib.util
import os
from .defs import ROBOTLIB_SO_PATH
spec = importlib.util.spec_from_file_location(
    "RobotLib", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), ROBOTLIB_SO_PATH))
)
RobotLib_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RobotLib_module)
RobotLib = RobotLib_module.Robot
from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco

from typing import Text, Mapping, Any, Callable, Sequence, Union

from data_types import se3
from hardware.base.arm import ArmBase

from motion.kinematics import PinocchioKinematicsModel as KinematicsModel

from motion import trajectory_planner, trajectory_executor
from tools import file_utils

import glog as log
import time,math,os
import numpy as np
from scipy.spatial.transform import Rotation as R

from hardware.monte01.gripper_xarm import Gripper
from hardware.monte01.trunk import Trunk

ARM_ENABLE=1
ARM_DENABLE=2

COM_TYPE_LEFT=1
COM_TYPE_RIGHT=2

ARM_MODE_POSITION_CTRL = 0
ARM_MODE_SERVO_MOTION = 1
ARM_MODE_JOINT_TEACHING = 2

GRIPPER_MODE_POSITION_CTRL = 1
GRIPPER_MODE_TORQUE_CTRL = 2

ARM_STATE_SPORT=0
ARM_STATE_PAUSE=1
ARM_STATE_STOP=2

BODY_JOINT_IDS = [1,2,3]
HEAD_JOINT_IDS = [4,5]
LEFT_ARM_JOINT_IDS = [6,7,8,9,10,11,12,]
RIGHT_ARM_JOINT_IDS = [13,14,15,16,17,18,19]

DEFAULT_JP_LEFT = np.array([0.6265732, 1.64933479, -1.2618717, -1.65806019, -6.30063725, 1.58824956, 0.32637656])  # Default joint positions for left arm
class Arm(ArmBase):
    def __init__(self, config: Mapping[Text, Any], hardware_interface: RobotLib, simulator: Monte01Mujoco, isLeft: bool = True):
        super().__init__()
        self.config = config
        self.joint_ids = LEFT_ARM_JOINT_IDS if isLeft else RIGHT_ARM_JOINT_IDS
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
        base_link = 'chest_link'
        end_link = 'left_arm_link_7' if isLeft else 'right_arm_link_7'
        try:
            self.kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)
        except Exception as e:
            log.error(f"Failed to load URDF: {e}")

        self.ctrl_dt = 1.0 / config['control_rate']
        # --- 新增：身體運動學模型 ---
        # 這個模型描述從機器人底部到胸部的鏈
        body_base_link = 'base_link' # URDF 的根
        body_end_link = 'chest_link'
        try:
            self.body_kinematics = KinematicsModel(
                urdf_path=urdf_path, 
                base_link=body_base_link, 
                end_effector_link=body_end_link
            )
            # 獲取身體部分的關節名稱列表
            self.body_joint_names = [self.body_kinematics.model.names[i] for i in range(1, self.body_kinematics.model.njoints)]
            log.info(f"載入 BODY 運動學模型成功，關節名稱: {self.body_joint_names}")
        except Exception as e:
            log.error(f"載入 BODY 運動學模型失敗: {e}")
            self.body_kinematics = None

        ###############################################
        # arm_kin = self.kinematics
        # body_kin = self.body_kinematics
        # # 計算 body 在零位時的姿態
        # body_q_zero = np.zeros(body_kin.n_joints)
        # T_world_to_chest_at_zero = body_kin.fk(body_q_zero)

        # # 計算 arm 在零位時的姿態 (相對於 chest)
        # arm_q_zero = np.zeros(arm_kin.n_joints)
        # T_chest_to_hand_at_zero = arm_kin.fk(arm_q_zero)

        # # 計算 hand 在世界座標系下的理論姿態
        # T_world_to_hand_at_zero_pinocchio = T_world_to_chest_at_zero @ T_chest_to_hand_at_zero

        # print('T_world_to_chest pin===')
        # print(T_world_to_chest_at_zero)

        # print('T_chest_to_hand_at_zero pin===')
        # print(T_chest_to_hand_at_zero)

        # print("--- Pinocchio (URDF) Zero Pose ---")
        # print(T_world_to_hand_at_zero_pinocchio)
        ###############################################

        self.flange_t_tcp = np.eye(4)

        self.tcp_t_flange = self.flange_t_tcp.T

        self.joint_names = config['joint_names_left'] if isLeft else config['joint_names_right']
        self.robot = hardware_interface
        self.simulator = simulator

        trajectory_args = {'rate': config['control_rate'],
                       'time_func': self.get_time,
                       'logging': config.get('log_trajectory', False)}
        if config['control_mode'] == 'position':
            trajectory_args['pos_setter'] = self.set_joint_positions
        elif config['control_mode'] == 'velocity':
            trajectory_args['vel_setter'] = self.set_joint_velocities
        else:
            raise ValueError('control_mode should be one of position or velocity.')
        self.trajectory_executor = trajectory_executor.OpenTrajectoryExecutor(
        **trajectory_args)

        time.sleep(0.1) # 給予一點緩衝時間
        if hardware_interface is not None:
            component_type = COM_TYPE_LEFT if isLeft else COM_TYPE_RIGHT
            hardware_interface.set_arm_enable(component_type, ARM_ENABLE)
            hardware_interface.set_arm_mode(component_type, ARM_MODE_SERVO_MOTION)
            hardware_interface.set_arm_state(component_type, ARM_STATE_SPORT)
            self.component_type = component_type

            gripper_ip = "192.168.11.11" if isLeft else "192.168.11.12"
            self.gripper = Gripper(config=config['gripper'], ip=gripper_ip, simulator=simulator)

            self.trunk = Trunk(config, hardware_interface, simulator)
        else:
            self.gripper = None

        self.qs_prev = self.get_joint_positions()

        self.time_log = []
        self.jp_log = []
    def get_gripper(self):
        return self.gripper
    
    def hold_position_for_duration(self, duration: float):
        """
        在指定的持續時間內，主動保持在當前關節位置。
        """
        log.info(f"Holding current position for {duration} seconds...++++")
        q_hold = self.get_joint_positions()
        log.info(f"q_hold == {q_hold}")
        start_time = self.get_time()
        while self.get_time() - start_time < duration:
            self.set_joint_positions(q_hold)
            time.sleep(self.ctrl_dt)
        log.info("Holding finished.--------------------------------------")
        
    def get_time(self) -> float:
        """獲取當前時間，優先從模擬器獲取，否則使用系統時間。"""
        if self.robot is not None:
            return time.time()
        else:
            return self.simulator.get_time()
    
    def get_joint_positions(self, joint_names: Any = None) -> np.ndarray:
        names = joint_names if joint_names is not None else self.joint_names
        if self.robot is None:
            return self.simulator.get_joint_positions(names)
        success, angles = self.robot.get_arm_servo_angle(self.component_type)
        log.debug(f"get_joint_positions: success={success}, angles={angles}")
        if not success:
            raise RuntimeError("Failed to get joint positions from the robot.")
        return np.array(angles)
    
    def hold_joint_positions(self):
        self.simulator.hold_joint_positions(self.joint_names)

    def set_joint_velocities(self, velocities: np.ndarray):
        raise NotImplementedError("set_joint_velocities is not implemented for this arm.")
    
    def set_joint_positions(self, positions: np.ndarray):
        target_positions = {}
        for i, joint_name in enumerate(self.joint_names):
            target_positions[self.joint_ids[i]] = positions[i]

        if self.robot:
            self.robot.set_arm_mode(self.component_type, ARM_MODE_SERVO_MOTION)
            self.robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
            self.robot.set_arm_servo_angle_j(self.component_type, list(positions), 1, 0, 1)
        if self.simulator:
            self.simulator.set_joint_positions(target_positions)

    def get_flange_pose(self) -> np.matrix:
        """Gets the pose of the flange.
        """
        return self.kinematics.fk(self.get_joint_positions())
    def ik(self, target: np.ndarray, seed: np.ndarray = None) -> np.ndarray:
        """Gets joint configuration via IK.
        
        Args:
            target: The target pose of the flange in base frame.
            seed: Optional seed for the IK solver.
        
        Returns:
            Joint positions that achieve the target pose, or None if no solution is found.
        """
        return self.kinematics.ik(target, seed=seed, max_iter=5000)
    def get_tcp_pose(self) -> np.matrix:
        return self.get_flange_pose() @ self.flange_t_tcp
    
    def move_thru_joint_targets(self, targets: Sequence[np.ndarray],blocking: bool = True) -> bool:
        cur_jp = self.get_joint_positions()
        log.info('before constructing trajectory from ' +
                    ','.join(map(str, cur_jp)) + ' to ' +
                    ','.join(map(str, targets[-1])))
        trajectory = trajectory_planner.TimeOptimalTrajectoryWrapper(
        [cur_jp] + targets, self.config['max_deviation'],
        self.config['jvel_limit'], self.config['jacc_limit'])
        log.info('after constructing trajectory')
        return self.execute_trajectory(
        trajectory, timeout=trajectory.get_duration() + 0.5, blocking=blocking)
    
    def execute_trajectory(
        self, trajectory: trajectory_planner.TimeOptimalTrajectoryWrapper,
        timeout: float = None, blocking: bool = True) -> bool:
        self.trajectory_executor.follow_trajectory(trajectory)
        if blocking:
            return self.wait_for_trajectory_done(timeout)
        else:
            return True
        
    def log_trajectory(self, timestamp: float):
        # self.time_log.append(timestamp)
        # self.jp_log.append(self.get_joint_positions())
        pass

    def wait_for_trajectory_done(self, timeout: float = None) -> bool:
        result = self.trajectory_executor.wait(
        timeout=timeout, callback=self.log_trajectory)
        return result
    
    def hold_joints(self) -> None:
        """Holds current joint position with zero velocity.
        """
        # TODO: check if this is correct
        # cur_jp = self.get_joint_angles()
        # self.set_joint_angles(cur_jp)
        if self.qs_prev is not None:
            self.set_joint_positions(self.qs_prev)
    
    def move_to_joint_target(self, target: np.ndarray, blocking: bool = True) -> bool:
        """Gets current jp and directly moves to target.

        Using the two waypoint trajectory.
        """
        return self.move_thru_joint_targets([target], blocking)

    def get_joint_target_from_pose(self, target: np.matrix,
                                start: np.ndarray = None):
        """Gets joint configuration via IK.
        """
        flange_target = target @ self.tcp_t_flange
        return self.ik(flange_target, seed=start)
    
    def get_body_joint_positions(self) -> np.ndarray:
        if self.robot is not None:
            success,positions,_,_ = self.robot.get_joint_state(BODY_JOINT_IDS)
            log.info(f"get_body_joint_positions from robot: success={success}, positions={positions}")
            if not success:
                log.error(f"Failed to get body joint positions: {success}")
                return np.zeros(len(BODY_JOINT_IDS))
            return np.array(positions)
        else:
            positions = self.simulator.get_joint_positions(self.body_joint_names)
            log.info(f"get_body_joint_positions from simulator: positions={positions}")
            return positions

    def get_world_to_chest_transform(self) -> np.matrix:
        """Gets the transformation matrix from world to chest frame.
        """
        try:
            # 獲取身體關節的即時角度
            body_joint_angles = self.get_body_joint_positions()
            log.info(f"body_joint_angles in get_world_to_chest_transform: {body_joint_angles}")
            # 使用身體運動學模型計算 FK
            return self.body_kinematics.fk(body_joint_angles)
        except Exception as e:
            log.error(f"為真實機器人計算身體 FK 時出錯: {e}")
            return np.eye(4)
    
    def convert_pose_to_world(self, local_pose: np.ndarray) -> np.ndarray:
        world_to_chest_transform = self.get_world_to_chest_transform()
        world_pose = world_to_chest_transform @ local_pose
        return world_pose

    def convert_pose_to_local(self, world_pose: np.ndarray) -> np.ndarray:
        world_to_chest_transform = self.get_world_to_chest_transform()
        try:
            # 計算逆矩陣 T_chest_world = inv(T_world_chest)
            chest_to_world_transform = np.linalg.inv(world_to_chest_transform)
        except np.linalg.LinAlgError:
            log.error("計算逆矩陣失敗。")
            return world_pose # 返回原姿態以避免崩潰
        
        local_pose = chest_to_world_transform @ world_pose
        return local_pose
    
    def move_to_pose(self, target: np.matrix) -> bool:
        """Moves to the target that specifies TCP pose in base frame.
        """
        current_pose = self.get_tcp_pose()
        return self.move_to_joint_target(
        self.get_joint_target_from_pose(target, self.get_joint_positions()))
        

    def print_state(self):
        if self.robot is not None:
            success, pos, vel, effort = self.robot.get_joint_state(self.joint_ids)
            log.info(f"Arm state: \nsuccess: {success}\npos: {pos}\nvel: {vel}\neffort: {effort}\n")

    def move_to_start(self):
        if self.robot is not None and self.component_type == COM_TYPE_LEFT:
            robot = self.robot
            robot.set_arm_enable(self.component_type, ARM_ENABLE)
            robot.set_arm_mode(self.component_type, ARM_MODE_POSITION_CTRL)
            robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
            success = robot.set_arm_servo_angle(1, DEFAULT_JP_LEFT, 1, 0, 1)
            log.info(f"Move to start position success: {success}")
            time.sleep(1)
        else:
            log.error("Robot interface is not available or component type is not left arm.")
    
    def get_tf_transform(self, parent_frame_id, child_frame_id, start_position):
        ret, transform = self.robot.get_tf_transform(parent_frame_id, child_frame_id, start_position)
        if not ret:
            log.error(f"get_tf_transform Failed!")
            return np.eye(4)
        log.info(f"get_tf_transform\n{transform}")
        return transform