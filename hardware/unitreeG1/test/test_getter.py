import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

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
arm_l.hand_grasp()

arm_r = r.arm_right()
arm_r.hand_grasp()

r.SetVolume(85)

while True:
    log.info("---------------------------------")
    time.sleep(5)
    r.print_state()
    rgb,d = r.CameraCapture()
    pass