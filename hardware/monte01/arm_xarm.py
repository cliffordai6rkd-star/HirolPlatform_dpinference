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
from xarm.wrapper import XArmAPI

ARM_ENABLE=1
ARM_DENABLE=2

COM_TYPE_LEFT=1
COM_TYPE_RIGHT=2

ARM_MODE_POSITION_CTRL = 0
ARM_MODE_SERVO_MOTION = 1
ARM_MODE_JOINT_TEACHING = 2

ARM_STATE_SPORT=0
ARM_STATE_PAUSE=1
ARM_STATE_STOP=2

HARDWARE_WRITE=True

class Arm(ArmBase):
    def __init__(self, config: Mapping[Text, Any], hardware_interface: XArmAPI, simulator: Monte01Mujoco, isLeft: bool = True):
        super().__init__()
        self.config = config
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
        base_link = 'chest_link'
        end_link = 'left_hand_link' if isLeft else 'right_hand_link'
        try:
            self.kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)
        except Exception as e:
            log.error(f"Failed to load URDF: {e}")

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
        self.qs_prev = self.get_joint_positions()
        if hardware_interface:
            hardware_interface.set_control_modbus_baudrate(921600)
            # TCP坐标系偏移
            hardware_interface.set_tcp_offset([0, 0, 172, 0, 0, 0])
            hardware_interface.motion_enable(enable=True)
            hardware_interface.set_mode(0)
            hardware_interface.set_state(state=0)

    def hold_position_for_duration(self, duration: float):
        """
        在指定的持續時間內，主動保持在當前關節位置。
        """
        log.info(f"Holding current position for {duration} seconds...")
        q_hold = self.get_joint_positions()
        log.info(f"Current joint positions q_hold: {q_hold}")
        start_time = self.get_time()
        while self.get_time() - start_time < duration:
            self.set_joint_positions(q_hold)
            # 以一個較高的頻率發送保持指令
            time.sleep(0.005) 
        log.info("Holding finished.")
        
    def get_time(self) -> float:
        """獲取當前時間，優先從模擬器獲取，否則使用系統時間。"""
        # if self.robot:
        #     # 如果您的 RobotLib 提供了獲取時間的 API，請在此處呼叫
        #     # return self.robot.get_time() 
        #     raise NotImplementedError("RobotLib does not provide a time API.")
        # el
        if self.simulator:
            return self.simulator.get_time()
        return time.time()
    
    def get_joint_positions(self, joint_names: Any = None) -> np.ndarray:
        names = joint_names if joint_names is not None else self.joint_names
        if self.robot is None:
            return self.simulator.get_joint_positions(names)
        success, angles = self.robot.get_servo_angle(is_radian=True)
        if 0 != success:
            raise RuntimeError("Failed to get joint positions from the robot.")
        return angles
    
    def hold_joint_positions(self):
        self.simulator.hold_joint_positions(self.joint_names)

    def set_joint_velocities(self, velocities: np.ndarray):
        raise NotImplementedError("set_joint_velocities is not implemented for this arm.")
    def set_joint_positions(self, positions: np.ndarray):
        
        target_positions = {}
        for i, joint_name in enumerate(self.joint_names):
            target_positions[joint_name] = positions[i]

        if self.robot and HARDWARE_WRITE:
            self.robot.set_servo_angle(angle=positions, wait=True)
        if self.simulator:
            self.simulator.set_joint_positions(target_positions)
            
    def get_flange_pose(self) -> se3.Transform:
        """Gets the pose of the flange.
        """
        return self.kinematics.fk(self.get_joint_positions())
        
    def ik(self, target: np.matrix, seed: np.ndarray = None) -> np.ndarray:
        """Gets joint configuration via IK.
        
        Args:
            target: The target pose of the flange in base frame.
            seed: Optional seed for the IK solver.
        
        Returns:
            Joint positions that achieve the target pose, or None if no solution is found.
        """
        return self.kinematics.ik(target, seed=seed, max_iter=5000)
    def get_tcp_pose(self) -> np.matrix:
        """Gets the pose of the TCP in base frame.
        return x,y,z,(mm), roll,pitch,yaw
        """
        # return self.get_flange_pose() @ self.flange_t_tcp
        success, pose6d = self.robot.get_position(is_radian=True)
        if 0 != success:
            raise RuntimeError("Failed to get TCP pose from the robot. Error code: {}".format(success))
        
        log.info(f"TCP pose6d: {pose6d}")
        
        return self.pose6d_to_matrix(pose6d)
    
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
        # log.info(f'.')
        pass

    def wait_for_trajectory_done(self, timeout: float = None) -> bool:
        result = self.trajectory_executor.wait(
        timeout=timeout, callback=self.log_trajectory)
        self.hold_joints()
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
        
        # code, pose = arm1.get_position()
        return self.ik(flange_target, seed=start)
        # jp_cur = start if start is not None else self.get_joint_positions()

        # jp_target_1 = self.kinematics.ik(flange_target, seed=jp_cur)
        # jp_target_2 = self.kinematics.ik(flange_target)

        # if jp_target_1 is not None and jp_target_2 is not None:
        #     dist_1 = np.linalg.norm(jp_target_1 - jp_cur)
        #     dist_2 = np.linalg.norm(jp_target_2 - jp_cur)
        #     if dist_1 < dist_2:
        #         jp_target = jp_target_1
        #         dist_min = dist_1
        #     else:
        #         jp_target = jp_target_2
        #         dist_min = dist_2
        # else:
        #     jp_target = jp_target_1 if jp_target_1 is not None else jp_target_2
        #     if jp_target is not None:
        #         dist_min = np.linalg.norm(jp_target - jp_cur)

        # # for seed in self.kinematics.seeds:
        # #     jp_candidate = self.kinematics.ik(flange_target, seed=seed)
        # #     log.info(f"jp_candidate: {jp_candidate}")
        # #     if jp_candidate is not None:
        # #         dist_candidate = np.linalg.norm(jp_candidate - jp_cur)
        # #         if jp_target is None or dist_candidate < dist_min:
        # #             jp_target = jp_candidate
        # #             dist_min = dist_candidate

        # return jp_target
    def pose6d_to_matrix(self, pose6d) -> np.matrix:
        xyz = np.array(pose6d[:3], dtype=np.float64) / 1000.0  # convert mm to m
        rpy = pose6d[3:]  # rad

        log.info(f"xyz={xyz}, rpy={rpy}")
        roll, pitch, yaw = rpy
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = xyz
        return np.matrix(pose)
    
    def move_to_pose(self, target: np.matrix) -> bool:
        """Moves to the target that specifies TCP pose in base frame.
        """
        if HARDWARE_WRITE:
            # 从4x4 matrix恢复xyz和rpy
            pose_mat = target
            xyz = pose_mat[:3, 3] * 1000.0  # m -> mm
            R = pose_mat[:3, :3]
            # 从旋转矩阵恢复rpy
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                roll = math.atan2(R[2, 1], R[2, 2])
                pitch = math.atan2(-R[2, 0], sy)
                yaw = math.atan2(R[1, 0], R[0, 0])
            else:
                roll = math.atan2(-R[1, 2], R[1, 1])
                pitch = math.atan2(-R[2, 0], sy)
                yaw = 0
            # 组装参数
            xyz = np.asarray(xyz).flatten().tolist()

            log.info(f"Moving to pose: {target}, xyz={xyz}, roll={roll}, pitch={pitch}, yaw={yaw}")
            # return self.robot.set_position(
            #     xyz[0], xyz[1], xyz[2],
            #     roll=roll,
            #     pitch=pitch,
            #     yaw=yaw,
            #     wait=True,
            #     is_radian=True,
            # )
            return True
        else:
            # Convert xyz (mm) and rpy (deg) to 4x4 transformation matrix (np.matrix)
            log.info(f"Moving to pose: {target}")
            return self.move_to_joint_target(
                self.get_joint_target_from_pose(
                    target, self.get_joint_positions()))
    def print_state(self):
        pass
