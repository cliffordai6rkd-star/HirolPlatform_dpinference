import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
import numpy as np
from tools import file_utils

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

a = r.get_arm()

g = a.get_gripper()
# g.homing()

g.grasp(width=0.02, speed=0.04, force=10, epsilon_inner=0.04, epsilon_outer=0.04)

g.move(width=0.06, speed=0.2)

a.print_state()
#TODO: 将sim验证过的代码放到这里

# r.move_to_start()

# pose = r.get_pose()
# pose[2,3] -= .1
# q = ik(pose)
# r.move_to_joint_position(q)

# log.info(f"controller time: {r.get_controller_time()}")
# pose = r.get_pose()
# pose[1,3] -= .1
# r.move_to_pose(pose)
# log.info(f"controller time: {r.get_controller_time()}")


# T_0 = panda_py.fk(constants.JOINT_POSITION_START)
# T_0[1, 3] = 0.25
# T_1 = T_0.copy()
# T_1[1, 3] = -0.25

# r.move_to_pose(T_0)

# r.move_to_start()

