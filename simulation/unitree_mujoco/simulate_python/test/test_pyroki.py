import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent))

import glog as log
log.setLevel("INFO")

from hardware.unitreeG1.agent import Agent

from tools import file_utils
import time
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

#okay
leg_l = r.leg_left()