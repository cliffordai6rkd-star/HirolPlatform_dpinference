from typing import Text, Mapping, Any, Callable, Sequence, Union

from data_types import se3
from hardware.unitreeG1.dex3 import Dex3
from hardware.base.arm import ArmBase
from motion.kinematics_model import KinematicsModel
from motion import trajectory_planner, trajectory_executor
from tools import file_utils

import glog as log
import time,math,os
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Arm(ArmBase):
    def __init__(self, config: Mapping[Text, Any], isLeft: bool = True, time_func: Callable[[], float] = time.time,
                 write_func: Callable[[LowCmd_], None] = None, simulator: bool = False):
        super().__init__()

        self.qs_prev = None

        self.is_sim = simulator
        self.config = config
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path'])) #urdf_path: "assets/unitree_g1/g1_29dof_with_hand.urdf"
        log.info(f"arm urdf path: {urdf_path}")

        base_link = 'torso_link'
        end_link = 'left_wrist_yaw_link' if isLeft else 'right_wrist_yaw_link'
        # end_link = 'left_hand_palm_link' if isLeft else 'right_hand_palm_link'
        joint_names = config['joint_names_left'] if isLeft else config['joint_names_right']

        try:
            self.kinematics = KinematicsModel(urdf=file_utils.read_file(urdf_path), base_link=base_link, ee_link=end_link, joint_names=joint_names)
        except Exception as e:
            log.error(f"Failed to load URDF: {e}")

        trajectory_args = {'rate': config['control_rate'],
                       'time_func': time_func,
                       'logging': config.get('log_trajectory', False)}
        if config['control_mode'] == 'position':
            trajectory_args['pos_setter'] = self.set_joint_positions
        elif config['control_mode'] == 'velocity':
            trajectory_args['vel_setter'] = self.set_joint_velocities
        else:
            raise ValueError('control_mode should be one of position or velocity.')
        self.trajectory_executor = trajectory_executor.OpenTrajectoryExecutor(
        **trajectory_args)
        
        self.jointIndices = [15,16,17,18,19,20,21] if isLeft else [22,23,24,25,26,27,28]

        self._dex3 = Dex3(config['hand'], isLeft)
        self.low_state = None
        self.flange_t_tcp = se3.Transform()
        self.tcp_t_flange = self.flange_t_tcp.inverse()
        self.write_func = write_func


    def LowCmdUpdate(self, low_cmd: LowCmd_, low_state: LowState_, ratio: float):
        for i in self.jointIndices:
            low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
            low_cmd.motor_cmd[i].tau = 0.
            low_cmd.motor_cmd[i].q = (1.0 - ratio) * low_state.motor_state[i].q
            low_cmd.motor_cmd[i].dq = 0.
            low_cmd.motor_cmd[i].kp = Kp[i]
            low_cmd.motor_cmd[i].kd = Kd[i]

    def Enable(self, low_cmd: LowCmd_):
        low_cmd.mode_pr = Mode.PR
        low_cmd.mode_machine = 6
        for i in self.jointIndices:
            low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
            low_cmd.motor_cmd[i].tau = 0.
            low_cmd.motor_cmd[i].q = 0.
            low_cmd.motor_cmd[i].dq = 0.
            low_cmd.motor_cmd[i].kp = Kp[i]
            low_cmd.motor_cmd[i].kd = Kd[i]
        self.low_cmd = low_cmd

    def update_low_state(self, low_state: LowState_):
        self.low_state = low_state
        
    def print_state(self):
        for idx in self.jointIndices:
            log.info(f"motor_state[{idx}]: {self.low_state.motor_state[idx]}")
        self._dex3.print_state()

    def get_joint_positions(self) -> np.ndarray:
        log.info(f"get_joint_angles: {[self.low_state.motor_state[i].q for i in self.jointIndices]}")
        return np.array([self.low_state.motor_state[i].q for i in self.jointIndices])
    
    def set_joint_positions(self, qs: Sequence[float]):
        self.qs_prev = qs
        log.debug(f"set_joint_angles: {qs}")
        for i in range(len(self.jointIndices)):
            self.low_cmd.motor_cmd[self.jointIndices[i]].q = qs[i]
        self.write_func(self.low_cmd)

    def set_joint_velocities(self, dqs: Sequence[float]):
        log.info(f"set_joint_vels: {dqs}")
        for i in range(len(self.jointIndices)):
            self.low_cmd.motor_cmd[self.jointIndices[i]].dq = dqs[i]
        self.write_func(self.low_cmd)

    # def get_model(self):
    #     pass
    
    # def get_tcp_orientation(self):
    #     pass
    
    def get_flange_pose(self) -> se3.Transform:
        """Gets the pose of the flange.
        """
        log.info(f"flange pose: {self.kinematics.fk(self.get_joint_positions())}")
        return self.kinematics.fk(self.get_joint_positions())

    def get_tcp_pose(self) -> se3.Transform:
        return self.get_flange_pose() * self.flange_t_tcp
    
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
        pass
        # raise NotImplementedError("log_trajectory not implemented")
        # self.time_log.append(timestamp)
        # self.jp_log.append(self.arm.get_jp())
        # try:
        #     self.jv_log.append(self.arm.get_jv())
        # except NotImplementedError as e:
        #     log.error(e)

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

    def get_joint_target_from_pose(self, target: se3.Transform,
                                start: np.ndarray = None):
        """Gets joint configuration via IK.
        """
        flange_target = target * self.tcp_t_flange
        jp_cur = start if start is not None else self.get_joint_positions()

        jp_target_1 = self.kinematics.ik(flange_target, seed=jp_cur)
        jp_target_2 = self.kinematics.ik(flange_target)

        if jp_target_1 is not None and jp_target_2 is not None:
            dist_1 = np.linalg.norm(jp_target_1 - jp_cur)
            dist_2 = np.linalg.norm(jp_target_2 - jp_cur)
            if dist_1 < dist_2:
                jp_target = jp_target_1
                dist_min = dist_1
            else:
                jp_target = jp_target_2
                dist_min = dist_2
        else:
            jp_target = jp_target_1 if jp_target_1 is not None else jp_target_2
            if jp_target is not None:
                dist_min = np.linalg.norm(jp_target - jp_cur)

        for seed in self.kinematics.seeds:
            jp_candidate = self.kinematics.ik(flange_target, seed=seed)
            log.info(f"jp_candidate: {jp_candidate}")
            if jp_candidate is not None:
                dist_candidate = np.linalg.norm(jp_candidate - jp_cur)
                if jp_target is None or dist_candidate < dist_min:
                    jp_target = jp_candidate
                    dist_min = dist_candidate

        return jp_target
    
    def move_to_pose(self, target: se3.Transform) -> bool:
        """Moves to the target that specifies TCP pose in base frame.
        """
        return self.move_to_joint_target(
        self.get_joint_target_from_pose(target))
    
    # def get_tcp_position(self):
    #     pass

    def get_state(self):
        pass

    def hand_grasp(self):
        if self.is_sim:
            log.warn(f"hand_grasp not supported in simulation")
            return
        self._dex3.grip_hand()


    #TODO, not ready
    # def hand_rotate_motors(self):
    #     self._dex3.rotate_motors_async()