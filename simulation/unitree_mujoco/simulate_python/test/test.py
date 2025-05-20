import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent))

import glog as log
log.setLevel("INFO")

from hardware.unitreeG1.agent import Agent

from tools import file_utils
import time
import numpy as np
from data_types import se3

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

#okay
# r.Start()

#okay
arm_l = r.arm_left()
log.info(f"arm_l.get_tcp_pose before grasp: {arm_l.get_tcp_pose()}")
log.info(f"arm_l.get_tcp_pose after grasp: {arm_l.get_tcp_pose()}")
start_jp = arm_l.get_joint_positions()

goal_jp = start_jp + np.array([-1.57, 0, 0, 0, 0, 0, 0])
goal_pose = arm_l.kinematics.fk(goal_jp)
arm_l.move_to_pose(goal_pose)

t = goal_pose.translation.tolist()
t[0] -= 0.1
t[1] += 0.04
t[2] -= 0.15
t3 = [t[0],t[1],t[2]]
r = goal_pose.rpy.tolist()
arm_l.move_to_pose(se3.Transform(xyz=[t[0],t[1],t[2]], rot=[r[0],r[1],r[2]]))
arm_l.hand_grasp()

# trajectory = trajectory_planner.TimeOptimalTrajectoryWrapper(
# [start_jp, goal_jp], 0.01, [3.14] * 6, [1.57] * 6)
# assert abs(trajectory.get_duration() - 2.006) < 1e-2
# t = np.linspace(0, trajectory.get_duration(), 100)
# ps = [trajectory.get_position(tt)[1] for tt in t]
# vs = [trajectory.get_velocity(tt)[1] for tt in t]
# cur_sim_time = simulator.simulated_time()
# robot.execute_trajectory(trajectory, timeout=trajectory.get_duration() + 0.5)
# assert abs(simulator.simulated_time() - cur_sim_time -
#             trajectory.get_duration()) < 0.1

# assert robot.move_to_joint_target(start_jp)
# assert max(np.abs(robot.arm.get_jp() - start_jp)) < 0.05



# arm_r = r.arm_right()
# log.info(f"arm_r.get_tcp_pose before grasp: {arm_r.get_tcp_pose()}")
# arm_r.hand_grasp()
# log.info(f"arm_r.get_tcp_pose after grasp: {arm_r.get_tcp_pose()}")

c = 0
while True:
    time.sleep(0.002)
    # c+=1
    # if c % 2500 == 0:
    #     log.info("---------------------------------")
    #     r.print_state()

    arm_l.hold_joints()
    pass