import sys, os, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import glog as log
log.setLevel("INFO")

from hardware.unitreeG1.agent import Agent

from tools import file_utils
import time 

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

#okay
r.Start()

#okay
arm_l = r.arm_left()
log.info(f"arm_l.get_tcp_pose before grasp: {arm_l.get_tcp_pose()}")
arm_l.hand_grasp()
log.info(f"arm_l.get_tcp_pose after grasp: {arm_l.get_tcp_pose()}")

arm_r = r.arm_right()
log.info(f"arm_r.get_tcp_pose before grasp: {arm_r.get_tcp_pose()}")
arm_r.hand_grasp()
log.info(f"arm_r.get_tcp_pose after grasp: {arm_r.get_tcp_pose()}")

r.SetVolume(85)

while True:
    log.info("---------------------------------")
    time.sleep(5)
    r.print_state()
    rgb,d = r.CameraCapture()
    pass