import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

sys.path.append("../../dependencies/libfranka-python/franka_bindings")
from franka_bindings import JointPositions

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
import numpy as np
from tools import file_utils

from panda_py import ik_full, ik

# from motion.ik import ik
from data_types import se3


cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, './config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

a = r.get_arm()

g = a.get_gripper()
g.homing()

g.grasp(width=0.02, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)

g.move(width=0.06, speed=0.2)

a.print_state()

initial_position = a.get_joint_positions()

amplitude = np.pi / 8.0
frequency = 0.2  # Hz
run_time = 5.0  # seconds
elapsed_time = 0.0

desired_position=None
while elapsed_time < run_time:

    try:
        duration = a.get_duration()
    except Exception as e:
        print(f"Error reading state: {e}")
        break

    elapsed_time += duration
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Calculate desired position
    desired_position = initial_position.copy()
    delta_angle = amplitude * (0.0 - np.sin(2.0 * np.pi * frequency * elapsed_time))
    desired_position[3] += delta_angle  # Move joint 4
    desired_position[0] += delta_angle  # Move joint 4
    
    joint_positions = JointPositions(desired_position)
    try:
        a.set_joint_positions(joint_positions)
    except Exception as e:
        print(f"Error writing joint positions: {e}")
        continue

a.print_state()

joint_positions = JointPositions(desired_position)

pose = a.get_pose()

pose[2,3] -= .1

tt = se3.Transform(matrix=pose)
position = tt.translation
orientation = tt.quaternion

q = a.kinematics.ik(se3.Transform(matrix=pose))
log.info(f"ik: {q}")
joint_positions = JointPositions(q)

# log.info(f"pose: \n{pose}")
# # from panda_py import ik_full, ik
# x = ik(pose)
# log.info(f"ik x: {x}")

# print(f"position: {position}")
# print(f"orientation: {orientation}")
# x = ik(position=position, orientation=orientation)
# log.info(f"ik x: {x}")
# joint_positions = JointPositions(x)



# joint_positions.motion_finished = True
try:
    a.set_joint_positions(joint_positions)
except Exception as e:
    print(f"Error writing joint positions: {e}")


a.stop()
print("Robot stopped.")

g.stop()
log.info("Gripper stop done")

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

