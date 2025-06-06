from xarm.wrapper import XArmAPI
from hardware.base.gripper import GripperBase
from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
from typing import Text, Mapping, Any
import glog as log
GRIPPER_CLOSE_POSITION=0
GRIPPER_OPEN_POSITION=800
class Gripper(GripperBase):
    def __init__(self, config: Mapping[Text, Any], ip:str, simulator: Monte01Mujoco):
        super().__init__()
        self.hardware = XArmAPI(ip, default_gripper_baud=921600)

        code = self.hardware.set_control_modbus_baudrate(921600)
        print('set gripper baudrate: 921600, code={}'.format(code))

        # 夹爪初始化
        code = self.hardware.set_gripper_mode(config['ctrl_mode'])
        print('set gripper mode: location mode, code={}'.format(code))
        # 设置夹爪波特率为 921600
        code = self.hardware.set_control_modbus_baudrate(921600)
        print('set gripper baudrate: 921600, code={}'.format(code))

        code = self.hardware.set_gripper_enable(True)
        print('set gripper enable, code={}'.format(code))

        code = self.hardware.set_gripper_speed(5000)
        print('set gripper speed, code={}'.format(code))

        self.print_state()

    def print_state(self) -> None:
        try:
            # 获取夹爪状态来验证通信是否正常
            code, pos = self.hardware.get_gripper_position()
            if code == 0:
                print(f'gripper position: {pos}, baudrate setting successful')
            else:
                print(f'gripper communication test failed, code: {code}, pos {pos}')
        except Exception as e:
            print(f'gripper communication error: {e}')

    def gripper_close(self):
        self.gripper_move(GRIPPER_CLOSE_POSITION)

    def gripper_open(self):
        self.gripper_move(GRIPPER_OPEN_POSITION)

    def gripper_move(self, position = 200):
        if position < GRIPPER_CLOSE_POSITION or position > GRIPPER_OPEN_POSITION:
            log.error(f"[gripper_move] Invalid input: need a 0~800 value.")
        code = self.hardware.set_gripper_position(position, wait=False)
        if code != 0:
            log.error(f"gripper move FAILED!")

        