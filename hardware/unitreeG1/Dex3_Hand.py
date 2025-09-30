from hardware.base.tool_base import ToolBase, ToolControlMode, ToolType
import glog as log
import time, threading, copy
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

maxLimits_left =  [  1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0     , 0   ]
minLimits_left = [ -1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75]
maxLimits_right = [  1.05 , 0.742  ,   0  ,  1.57 , 1.75  , 1.57  , 1.75]
minLimits_right = [ -1.05 , -1.05  , -1.75,    0  ,  0    ,   0   ,0    ]

JOINT_ID_MAPPING = {
    0: 5, 1: 6, 2: 3, 3: 4, 4: 0, 5: 1, 6: 2
}

class Dex3Hand(ToolBase):
    _tool_type: ToolType = ToolType.HAND
    _MOTOR_MAX: int = 7
    def __init__(self, config):
        self._last_command = 1.0
        self._state_lock = threading.Lock()
        
        self._network_interface = config["network_interface"]
        self._prefix = config["hand_prefix"]
        if self._prefix == "left":
            self._min_values = minLimits_left
            self._max_values = maxLimits_left
        else:
            self._min_values = minLimits_right
            self._max_values = maxLimits_right
        topic_prefix = "rt/dex3/" + self._prefix
        
        # dds
        ChannelFactoryInitialize(0, self._network_interface) # dds channel initialization
        self._hand_cmd_publisher = ChannelPublisher(topic_prefix+"/cmd", HandCmd_)
        self._hand_state_subscriber = ChannelSubscriber(topic_prefix+"/state", HandState_)
        
        super().__init__(config)
        self._state._tool_type = self._tool_type
        self._state._position = np.zeros(7)
        self._state._force = np.zeros(7)

    def initialize(self):
        if self._is_initialized:
            return True
        
        self._hand_cmd_publisher.Init()
        self._hand_state_subscriber.Init(self._hand_state_handler, 1)
        
        log.info(f'{self._prefix} Dex3 hand successfully is initialized!!!!')
        
    def get_tool_state(self):
        with self._state_lock:
            state = copy.deepcopy(self._state)
        return state
    
    def set_hardware_command(self, command):
        hand_cmd_msg = unitree_hg_msg_dds__HandCmd_()
        
        if self._control_mode == ToolControlMode.BINARY:
            command = np.clip(command, 0.0, 1.0)
            if command > self._binary_threshold:
                posi = self._max_values
            else:
                posi = (self._min_values + self._max_values) / 2.0
            command = posi
            
        for i in range(self._MOTOR_MAX):
            cur_motor_id = JOINT_ID_MAPPING[i]
            ris_mode = self._RIS_Mode(id=cur_motor_id, status=0x01, timeout=0)
            mode = ris_mode._mode_to_uint8()
            
            hand_cmd_msg.motor_cmd[cur_motor_id].mode = mode
            hand_cmd_msg.motor_cmd[cur_motor_id].tau = 0
            cur_command = np.clip(command[i], self._min_values[i], self._max_values[i])
            hand_cmd_msg.motor_cmd[cur_motor_id].q = cur_command
            hand_cmd_msg.motor_cmd[cur_motor_id].dq = 0
            hand_cmd_msg.motor_cmd[cur_motor_id].kp = 1.5
            hand_cmd_msg.motor_cmd[cur_motor_id].kd = 0.2
        
        log.info(f'dds write hand: {hand_cmd_msg}')
        self._hand_cmd_publisher.Write(hand_cmd_msg)
        time.sleep(0.001)
    
    def stop_tool(self):
        self._hand_cmd_publisher.Close()
        self._hand_state_subscriber.Close()
    
    def get_tool_type_dict(self):
        tool_type_dict = {'single': self._tool_type}
        return tool_type_dict

    def _hand_state_handler(self, msg: HandState_):
        with self._state_lock:
            for i in range(self._MOTOR_MAX):
                self._state._position[i] = msg.motor_state[JOINT_ID_MAPPING[i]].q
                self._state._force[i] = msg.motor_state[JOINT_ID_MAPPING[i]].tau_est
            self._state._time_stamp = time.perf_counter()
        log.info(f'dds read hand: {self._state._position}')
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode
        