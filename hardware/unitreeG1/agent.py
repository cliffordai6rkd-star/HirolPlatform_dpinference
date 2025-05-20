from typing import Text, Mapping, Any

import time
from hardware.unitreeG1.arm import Arm
from hardware.unitreeG1.leg import Leg
from hardware.unitreeG1.waist import Waist
from hardware.unitreeG1.camera import Camera

from hardware.base.robot import Robot

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, MainBoardState_, BmsState_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

import numpy as np

import glog as log

from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

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

# PR 模式：控制踝关节的 Pitch(P) 和 Roll(R) 电机 (默认模式，对应 URDF 模型)
# AB 模式：直接控制踝关节的 A 和 B 电机 (需要用户自己计算并联机构运动学)
# https://support.unitree.com/home/zh/G1_developer/basic_motion_routine
class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints
class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any]):
        log.info(f"network_interface: {config['network_interface']}")

        self.update_mode_machine_ = False
        self.mode_machine_ = 0
        self.low_state = None
        self.time_ = 0.0
        self.control_dt_ = config['control_dt']  # [2ms]
        self.duration_ = config['duration']    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0

        ChannelFactoryInitialize(config['robot_domain'], config['network_interface'])

        # G1 常用TOPIC 列表
        # TopicName 	Idl 	Info
        # rt/dex3/left/state 	hg/idl/HandState_.idl 	获取左灵巧手反馈状态-低频模式
        # rt/lf/dex3/left/state 	hg/idl/HandState_.idl 	获取左灵巧手反馈状态
        # rt/dex3/left/cmd 	hg/idl/HandCmd_.idl 	控制左灵巧手
        # rt/dex3/right/state 	hg/idl/HandState_.idl 	获取左灵巧手反馈状态
        # rt/lf/dex3/right/state 	hg/idl/HandState_.idl 	获取右灵巧手反馈状态-低频模式
        # rt/dex3/right/cmd 	hg/idl/HandCmd_.idl 	控制右灵巧手
        # rt/lf/mainboardstate 	hg/idl/MainBoardState_.idl 	获取主板反馈信息
        # rt/lowstate 	hg/idl/LowState_.idl 	获取底层反馈信息(IMU、电机等)
        # rt/lf/lowstate 	hg/idl/LowState_.idl 	获取底层反馈信息(IMU、电机等)-低频模式
        # rt/lowcmd 	hg/idl/LowCmd_.idl 	底层控制命令
        # rt/lf/bmsstate 	hg/idl/BmsState_.idl 	获取电池反馈数据
        # rt/odommodestate 	hg/idl/IMUState_.idl 	获取里程计信息
        # rt/lf/odommodestate 	hg/idl/IMUState_.idl 	获取里程计信息-低频模式
        # rt/lf/secondary_imu 	/hg/IMUState_.idl 	获取机身IMU数据-低频模式
        # rt/secondary_imu 	/hg/IMUState_.idl 	获取机身IMU数据
        if config['robot_domain'] == 0:
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)

        crc = CRC()
        low_cmd: LowCmd_ = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd = low_cmd
        self.crc = crc

        is_sim=True if config['robot_domain'] == 1 else False
        self._arm_left = Arm(config['arm'], write_func=self.Write,simulator=is_sim)
        self._arm_right = Arm(config['arm'], False, write_func=self.Write,simulator=is_sim)

        self._arm_left.Enable(self.low_cmd)
        # self._arm_right.Enable(self.low_cmd)

        self._leg_left = Leg(config['leg'])
        self._leg_right = Leg(config['leg'], False)

        self._waist = Waist(config['waist'])

        audio_client = AudioClient()
        audio_client.SetTimeout(10.0)
        audio_client.Init()
        self.audio_client = audio_client

        if config['robot_domain'] == 0:
            self.camera = Camera(config=config['camera'])

        #TODO: add livox support

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        # create subscriber # 
        self.lowstate_subscriber_ = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber_.Init(self.LowStateHandler, 10)

        self.mainboardstate_subscriber_ = ChannelSubscriber("rt/lf/mainboardstate", MainBoardState_)
        self.mainboardstate_subscriber_.Init(self.MainBoardStateHandler, 10)

        self.bms_subscriber_ = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
        self.bms_subscriber_.Init(self.BMSStateHandler, 10)

        # https://support.unitree.com/home/zh/G1_developer/odometer_service_interface
        # self.odom_sub_ = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.odom_sub_ = ChannelSubscriber("rt/lf/odommodestate", SportModeState_)
        self.odom_sub_.Init(self.HighStateHandler, 10)

    def CameraCapture(self):
        return self.camera.capture()

    def SetVolume(self, volume: int):
        self.audio_client.SetVolume(volume)
        time.sleep(7)
        ret, vol = self.audio_client.GetVolume()
        time.sleep(1)

        if 0 == ret:
            log.info(f"debug GetVolume: {vol}")
            self.audio_client.TtsMaker(f"当前音量{vol}",0)
        else:
            self.audio_client.TtsMaker(f"音量获取失败",0)
    
    def HighStateHandler(self, msg: SportModeState_):
        log.debug(f"Position: {msg.position}")
        log.debug(f"Velocity: {msg.velocity}")
        log.debug(f"Yaw velocity: {msg.yaw_speed}")
        log.debug(f"Foot position in body frame: {msg.foot_position_body}")
        log.debug(f"Foot velocity in body frame: {msg.foot_speed_body}")
        
    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        self._arm_left.update_low_state(msg)
        self._arm_right.update_low_state(msg)
        self._leg_left.update_low_state(msg)
        self._leg_right.update_low_state(msg)
        self._waist.update_low_state(msg)

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

    def MainBoardStateHandler(self, msg:MainBoardState_):
        log.debug(f"Main Board State: {msg}")

    def BMSStateHandler(self, msg: BmsState_):
        log.debug(f"Battery State: {msg}")

    def print_state(self):
        log.info(f"IMU State: {self.low_state.imu_state}")
        log.info(f"Version: {self.low_state.version}")
        log.info(f"Mode PR: {self.low_state.mode_pr}")
        log.info(f"Mode Machine: {self.low_state.mode_machine}")
        log.info(f"Tick: {self.low_state.tick}")
        log.info(f"Wireless Remote: {self.low_state.wireless_remote}")
        log.info(f"Reserve: {self.low_state.reserve}")
        log.info(f"CRC: {self.low_state.crc}")
        log.info('------------------left arm--------------------------')
        
        self._arm_left.print_state()
        log.info('------------------right arm--------------------------')
        self._arm_right.print_state()
        log.info('------------------left leg--------------------------')
        self._leg_left.print_state()
        log.info('------------------right leg--------------------------')
        self._leg_right.print_state()
        log.info('------------------waist--------------------------')
        self._waist.print_state()

    def arm_left(self) -> Arm:
        return self._arm_left
    
    def arm_right(self) -> Arm:
        return self._arm_right
    
    def leg_left(self) -> Leg:
        return self._leg_left
    
    def leg_right(self) -> Leg:
        return self._leg_right
    
    def waist(self) -> Waist:
        return self._waist
    
    def Start(self):
      self.lowCmdWriteThreadPtr = RecurrentThread(
          interval=self.control_dt_, target=self.LowCmdWrite, name="control"
      )
      while self.update_mode_machine_ == False:
          time.sleep(1)
      if self.update_mode_machine_ == True:
          self.lowCmdWriteThreadPtr.Start()
  
    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        # [Stage 1]: set robot to zero posture
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
        self._arm_left.LowCmdUpdate(self.low_cmd, self.low_state, ratio)
        self._arm_right.LowCmdUpdate(self.low_cmd, self.low_state, ratio)

        # self.FreezeLegs()
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def Write(self, low_cmd: LowCmd_):
        
        low_cmd.crc = self.crc.Crc(low_cmd)
        self.lowcmd_publisher_.Write(low_cmd)

    def FreezeLegs(self):
        self._leg_left.LowCmdUpdate(self.low_cmd)
        self._leg_right.LowCmdUpdate(self.low_cmd)
        self._waist.LowCmdUpdate(self.low_cmd)

    def PreMove(self):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
        self._arm_left.LowCmdUpdate(self.low_cmd, self.low_state, ratio)
        self._arm_right.LowCmdUpdate(self.low_cmd, self.low_state, ratio)

    def LowCmdApply(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
    

