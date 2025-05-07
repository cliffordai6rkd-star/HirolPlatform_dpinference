from typing import Text, Mapping, Any
from data_types import se3
from hardware.unitreeG1.dex3 import Dex3
from hardware.base.arm import ArmBase
from motion.kinematics_model import KinematicsModel
from tools import file_utils

import glog as log
import time,math
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
    def __init__(self, config: Mapping[Text, Any], isLeft: bool = True):
        super().__init__()

        self.jointIndices=[]
        if isLeft:
            self.jointIndices = [15,16,17,18,19,20,21]
        else:
            self.jointIndices = [22,23,24,25,26,27,28]
        self.isLeft = isLeft

        self._dex3 = Dex3(config['hand'], isLeft)

        # self.urdf = KinematicsModel(urdf=file_utils.read_file(config['urdf']), base_link=config['base_link'], end_link=config['end_link'])

    def LowCmdUpdate(self, low_cmd: LowCmd_, low_state: LowState_, ratio: float):
        for i in self.jointIndices:
            low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
            low_cmd.motor_cmd[i].tau = 0.
            low_cmd.motor_cmd[i].q = (1.0 - ratio) * low_state.motor_state[i].q
            low_cmd.motor_cmd[i].dq = 0.
            low_cmd.motor_cmd[i].kp = Kp[i]
            low_cmd.motor_cmd[i].kd = Kd[i]

    def print_state(self, low_state: LowState_):
        for idx in self.jointIndices:
            print(f"motor_state[{idx}]: {low_state.motor_state[idx]}")
        self._dex3.print_state()

    def get_model(self):
        pass
    
    # def get_tcp_orientation(self):
    #     pass
    
    def get_tcp_pose(self) -> se3.Transform:
        raise NotImplementedError("get_tcp_pose not implemented")

    # def get_tcp_position(self):
    #     pass

    def get_state(self):
        pass

    def hand_grasp(self):
        self._dex3.grip_hand()


    #TODO, not ready
    # def hand_rotate_motors(self):
    #     self._dex3.rotate_motors_async()