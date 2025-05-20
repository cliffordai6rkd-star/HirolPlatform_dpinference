from typing import Text, Mapping, Any
from hardware.base.leg import LegBase

import glog as log
import time,math
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyroki_snippets as pks

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

Q = [
  -0.25, 0,0,0.24,-0.01,0,
  -0.25, 0,0,0.24,-0.01,0,
  0,0,-0.26,
  0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,
]
class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Leg(LegBase):
  
  def __init__(self, config: Mapping[Text, Any], isLeft: bool = True):
    super().__init__()

    self.jointIndices=[0,1,2,3,4,5] if isLeft else [6,7,8,9,10,11]
    self.low_state = None

    urdf = load_robot_description("g1_description")
    target_link_name = "left_ankle_roll_link" if isLeft else "right_ankle_roll_link"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)
    log.info(f"robot: {robot}")

  def LowCmdUpdate(self, low_cmd: LowCmd_):
    for i in self.jointIndices:
      low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
      low_cmd.motor_cmd[i].tau = 0.
      low_cmd.motor_cmd[i].q = Q[i]
      low_cmd.motor_cmd[i].dq = 0.
      low_cmd.motor_cmd[i].kp = Kp[i]
      low_cmd.motor_cmd[i].kd = Kd[i]

  def print_state(self):
    for idx in self.jointIndices:
      print(f"motor_state[{idx}]: {self.low_state.motor_state[idx]}")
  def update_low_state(self, low_state: LowState_):
    self.low_state = low_state
  def get_model(self):
    pass
  
  def get_state(self):
    pass
  


 
