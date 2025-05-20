import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
from panda_py import ik
import numpy as np
from panda_py import constants
from tools import file_utils

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent_impedance.yaml')
config = file_utils.read_config(robot_config_file)
r = Agent(config)

r.move_to_start()

x0 = r.get_position()
q0 = r.get_orientation()
runtime = np.pi * 4.0
r.start_controller()

with r.create_context(frequency=1e3, max_runtime=runtime) as ctx:
    while ctx.ok():
        x_d = x0.copy()
        x_d[1] += 0.1 * np.sin(r.get_controller_time())
        r.set_impedance_control(x_d, q0)