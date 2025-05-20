from typing import Text, Mapping, Any

from hardware.fr3.arm import Arm
from hardware.base.robot import Robot
import glog as log

class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any]):
        ip = config['ip']
        self._arm = Arm(config=config['arm'])

    def print_state(self):
        # log.info(f"Has realtime kernel: {libfranka.has_realtime_kernel()}")
        self._arm.print_state()

    def get_arm(self):
        return self._arm
    
    # def move_to_start(self):
    #     self._arm.move_to_start()

    # def get_pose(self):
    #     return self._arm.get_tcp_pose()
    
    # def get_position(self):
    #     return self._arm.get_ee_position()
    
    # def get_orientation(self):
    #     return self._arm.get_ee_orientation()

    # def move_to_joint_position(self, q):
    #     self._arm.move_to_joint_position(q)

    # def move_to_pose(self, pose):
    #     self._arm.move_to_pose(pose)

    # def grasp(self):
    #     self._gripper.grasp(width=0, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)
    
    # def get_controller_time(self):
    #     return self._arm.get_controller_time()
    
    # def set_impedance_control(self, position, orientation):
    #     return self._arm.set_impedance_control(position=position, orientation=orientation)
    
    # def start_controller(self):
    #     self._arm.start_controller()

    # def create_context(self, frequency=1e3, max_runtime=1):
    #     return self._arm.create_context(frequency, max_runtime)