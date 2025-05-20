from typing import Text, Mapping, Any
from hardware.base.hand import HandBase
import glog as log
import time,math

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
from unitree_sdk2py.utils.thread import RecurrentThread

# Set URDF Limits
maxLimits_left = [1.05, 1.05, 1.75, 0, 0, 0, 0]  # Set max motor value
minLimits_left = [-1.05, -0.724, 0, -1.57, -1.75, -1.57, -1.75]
maxLimits_right = [1.05, 0.742, 0, 1.57, 1.75, 1.57, 1.75]
minLimits_right = [-1.05, -1.05, -1.75, 0, 0, 0, 0]
MOTOR_MAX  = 7
SENSOR_MAX  = 9

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Dex3(HandBase):
    def print_state(self):
        if self.hand_state is None:
            log.info("Hand state is None")
            return
        log.info(f"Motor State: {self.hand_state.motor_state}")
        log.info(f"Press Sensor State: {self.hand_state.press_sensor_state}")
        log.info(f"Power Voltage: {self.hand_state.power_v}")
        log.info(f"Power Current: {self.hand_state.power_a}")
        log.info(f"System Voltage: {self.hand_state.system_v}")
        log.info(f"Device Voltage: {self.hand_state.device_v}")
        log.info(f"Error: {self.hand_state.error}")
        log.info(f"Reserve: {self.hand_state.reserve}")

    def HandStateHandler(self, msg: HandState_):
        self.hand_state = msg

    def __init__(self, config: Mapping[Text, Any], isLeft: bool = True):
        super().__init__()
        self.isLeft = isLeft
        if isLeft:
            self._hand_id = 0
            self._dds_namespace = "rt/dex3/left"
            self._sub_namespace = "rt/lf/dex3/left/state"
        else:
            self._hand_id = 1
            self._dds_namespace = "rt/dex3/right"
            self._sub_namespace = "rt/lf/dex3/right/state"

        self.time_ = 0.0
        self.control_dt_ = config['control_dt']  # [2ms]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_machine_ = 0
        self.hand_cmd = unitree_hg_msg_dds__HandCmd_()  
        self.hand_state = None 
        self.update_mode_machine_ = False

        # create publisher #
        self.handcmd_publisher_ = ChannelPublisher(self._dds_namespace+"/cmd", HandCmd_)
        self.handcmd_publisher_.Init()

        # create subscriber # 
        self.handstate_subscriber = ChannelSubscriber(self._sub_namespace, HandState_)
        self.handstate_subscriber.Init(self.HandStateHandler, 10)

    def grip_hand(self):
        is_left_hand = self.isLeft
        max_limits = maxLimits_left if is_left_hand else maxLimits_right
        min_limits = minLimits_left if is_left_hand else minLimits_right

        for i in range(MOTOR_MAX):
            ris_mode = {
                "id": i,
                "status": 0x01,
                "timeout": 0x00
            }

            mode = 0
            mode |= (ris_mode["id"] & 0x0F)
            mode |= (ris_mode["status"] & 0x07) << 4
            mode |= (ris_mode["timeout"] & 0x01) << 7

            self.hand_cmd.motor_cmd[i].mode = mode
            self.hand_cmd.motor_cmd[i].tau = 0

            mid = (max_limits[i] + min_limits[i]) / 2.0

            self.hand_cmd.motor_cmd[i].q = mid
            self.hand_cmd.motor_cmd[i].dq = 0
            self.hand_cmd.motor_cmd[i].kp = 1.5
            self.hand_cmd.motor_cmd[i].kd = 0.1

        success = self.handcmd_publisher_.Write(self.hand_cmd)
        log.info(f"hand grasp succeed: {success}")
        time.sleep(1)
    
    def rotate_motors_async(self):
        thname = 'ctlL'
        if not self.isLeft:
            thname = 'ctlR'

        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.rotate_motors, name=thname
        )

        self.lowCmdWriteThreadPtr.Start()

    def rotate_motors(self):
        _count = 1
        dir = 1
        is_left_hand = self.isLeft
        max_limits = maxLimits_left if is_left_hand else maxLimits_right
        min_limits = minLimits_left if is_left_hand else minLimits_right

        while True:
            for i in range(MOTOR_MAX):
                ris_mode = {
                    "id": i,
                    "status": 0x01,
                    "timeout": 0x00
                }

                mode = 0
                mode |= (ris_mode["id"] & 0x0F)
                mode |= (ris_mode["status"] & 0x07) << 4
                mode |= (ris_mode["timeout"] & 0x01) << 7

                self.hand_cmd.motor_cmd[i].mode = mode
                self.hand_cmd.motor_cmd[i].tau = 0

                self.hand_cmd.motor_cmd[i].kp = 0.5
                self.hand_cmd.motor_cmd[i].kd = 0.1

                range_val = max_limits[i] - min_limits[i]
                mid = (max_limits[i] + min_limits[i]) / 2.0
                amplitude = range_val / 2.0
                x=0
                if is_left_hand:
                    x=math.pi
                q = mid + amplitude * math.sin(_count / 20000.0 * math.pi + x)
                self.hand_cmd.motor_cmd[i].q = q

            self.handcmd_publisher_.Write(self.hand_cmd)
            _count += dir

            if _count >= 10000:
                dir = -1
            if _count <= -10000:
                dir = 1

            time.sleep(0.0001)  # 100 microseconds
 