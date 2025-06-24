import threading
import time

import numpy as np
from absl import logging

from dm_robotics.panda import arm_constants, environment, run_loop, utils
from dm_robotics.panda import parameters as params
from dm_robotics.panda.parameters import CollisionBehavior

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent_dm_panda import Agent
from hardware.fr3.arm_dm_panda import Arm
import numpy as np
from tools import file_utils
from dm_robotics.moma.effectors import default_gripper_effector
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85,robot_hand
from dm_robotics.moma.sensors import robotiq_gripper_sensor

kJointPositionStartData = np.array([
    0.0,
    -np.pi / 4,
    0.0,
    -3 * np.pi / 4,
    0.0,
    np.pi / 2,
    np.pi / 4,
])
xx = np.array([
        0.0,
        -np.pi / 4,
        0.0,
        -3 * np.pi / 4,
        0.0,
        np.pi / 2 + np.pi / 8,
        np.pi / 4,
    ])

def run_command_sequence(controller: Arm):
    logging.info("Waiting for controller to initialize...")
    time.sleep(2)

    try:
        logging.info("Executing user command sequence...")

        controller.move_to_start()
        time.sleep(5)
        controller.set_gripper_open(1)
        controller.set_joint_positions(xx)
        controller.set_gripper_open(0)

        controller.print_state()

        pose = controller.get_tcp_pose()
        if pose is not None:
            pose[2, 3] -= 0.1  # 向下移動 10cm
            success, q = controller.ik(pose)
            if success:
                controller.set_joint_positions(q)

        logging.info(f"Controller time: {controller.get_controller_time()}")
        pose = controller.get_tcp_pose()
        if pose is not None:
            pose[1, 3] -= 0.1  # 沿 Y 軸負向移動 10cm
            controller.move_to_pose(pose)
        logging.info(f"Controller time: {controller.get_controller_time()}")

        T_0 = controller.fk(kJointPositionStartData)
        T_0[1, 3] = 0.25
        T_1 = T_0.copy()
        T_1[1, 3] = -0.25

        controller.move_to_pose(T_0)

        # controller.set_gripper_open(1)

        controller.move_to_pose(T_1)
        # gripper.close()
        
        # controller.set_gripper_open(0)

        controller.move_to_start()
        controller.set_gripper_open(1)
        
        logging.info("Command sequence finished.")
        time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        logging.info("Exiting.")

if __name__ == "__main__":
    utils.init_logging()
    parser = utils.default_arg_parser()
    args = parser.parse_args()

    my_lower_torque_thresholds = [10.0] * 7
    my_upper_torque_thresholds = [20.0] * 7
    my_lower_force_thresholds = [10.0] * 6
    my_upper_force_thresholds = [20.0] * 6

    custom_collision_behavior = CollisionBehavior(
        lower_torque_thresholds=my_lower_torque_thresholds,
        upper_torque_thresholds=my_upper_torque_thresholds,
        lower_force_thresholds=my_lower_force_thresholds,
        upper_force_thresholds=my_upper_force_thresholds
    )

    robot_params = params.RobotParams(
        robot_ip=args.robot_ip, actuation=arm_constants.Actuation.JOINT_VELOCITY,
        joint_stiffness= [600, 600, 600, 600, 250, 150, 50],
        joint_damping= [50, 50, 50, 20, 20, 20, 10],
        collision_behavior=custom_collision_behavior
    )
    panda_env = environment.PandaEnvironment(robot_params, physics_timestep=0.001)

    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(
    cur_path, '../../hardware/fr3/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)

    print(config)
    
    with panda_env.build_task_environment() as env:
        r = Agent(config, env)

        controller = r.get_arm()
        
        command_thread = threading.Thread(
            target=run_command_sequence,
            args=(controller,),
            daemon=True
        )
        command_thread.start()

        if args.gui:
            logging.info("Launching GUI application...")
            app = utils.ApplicationWithPlot()
            app.launch(env, policy=controller.step)
        else:
            logging.info("Running in headless mode...")
            run_loop.run(env, controller, [], max_steps=1_000_000, real_time=True)

        logging.info("Main loop finished. Exiting.")