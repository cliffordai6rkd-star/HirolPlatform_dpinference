import os
import time
import numpy as np
from threading import Thread
from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
from hardware.monte01.agent_xarm import Agent
from hardware.monte01.arm_xarm import Arm
from tools import file_utils
import glog as log
from data_types import se3

log.setLevel("INFO")
# --------------------桌面法向量--------------------------#
direction = np.array([-0.81405081,  0.58064973 ,-0.01292953])
direction /= np.linalg.norm(direction)

def control_loop(agent):
    sim = agent.sim
    """A simple control loop that moves the arm and gripper."""
    print("Control loop started.")
    
    # Wait for the viewer to be initialized
    while sim.viewer is None:
        time.sleep(0.1)

    # Wait for the viewer to start running
    while not sim.viewer.is_running():
        time.sleep(0.1)

    default_jp_l = np.array([1.38, 2.22, -1.61, -1.65, -6.33, 1.73, 1.57])
    default_jp_r = np.array([4.54, 2.22, -1.45, -1.89, -3.5, 1.34, 0.0])
    try:
        # --- Thread-safe simulation interaction ---
        current_time = sim.get_time()

        arm_left: Arm = agent.arm_left()
        arm_right: Arm = agent.arm_right()

        jp_l = np.array([35.9, 4.5, 17.7,
            85, -271, 91,
            18.7])*np.pi/180
        jp_r = np.array([-41.92195942729991, 41.37143593491313, 
                            -30.859276198454094, 94.9338511231394, 
                            -85.46737285804181, 57.61186700749217,
                            112.32257018388641])*np.pi/180
        # print("--- Step 1: Moving left arm to default position (blocking)... ---")
        # # 使用 blocking=True 來等待移動完成
        # success = arm_left.move_to_joint_target(default_jp_l, blocking=True)
        # if not success:
        #     log.error("Step 1 failed to complete.")
        #     return # 如果失敗，可以提前結束

        # print("Step 1 finished. Current time:", sim.get_time())
        # arm_left.hold_position_for_duration(1.0) # 暫停一秒，便於觀察

        print("\n--- Step 2: Getting pose and moving to a sine-wave modified pose... ---")
        # todo: 這個空檔，jp已經改變，如何保持jp不變
        pose = arm_left.get_tcp_pose()
        jp0 = arm_left.get_joint_positions()
        print(f"Current pose after step 1: \n{pose}")
        
        #CAUSION: this will damage monte's head, TODO: re-impl this
        # if pose is not None:
        #     # 這裡的 current_time 只是用來計算 sin，可以即時獲取
        #     current_time = sim.get_time()

        #     # t = pose.translation
        #     # t[2] +=0.005
        #     # pose = se3.Transform(xyz=t, rot=pose.rpy)
        #     # pose[:3] = pose[:3] - direction * 0.01
        #     pose[2, 3] += 0.005
        #     # print("Calculating IK for the new pose...")
        #     # q = arm_left.get_joint_target_from_pose(pose) # 使用您的 ik 封裝函式
        #     # log.info(f"Gap between current and target joint positions: {jp0 - q}")
        #     # if q is not None:
        #     #     print("IK solution found. Moving to new joint target (blocking)...")
        #     #     # 再次使用 blocking=True 等待移動完成
        #     #     arm_left.move_to_joint_target(q, blocking=True)
        #     # else:
        #     #     log.error("Step 2 failed: IK solution not found.")
        #     arm_left.move_to_pose(pose)
        # else:
        #     log.error("Step 2 failed: Could not get TCP pose.")

        print("Step 2 finished.")

        arm_left.hold_position_for_duration(1.0)

        print("\n--- Holding final position and exiting control loop. ---")
        
        # Optional: Print current joint positions, with thread-safe access to last_print_time
        
        if sim.should_print():
            positions = arm_left.get_joint_positions()
            print(f"Time: {current_time:.2f}s, Positions: {positions}")

        arm_left.hold_position_for_duration(float('inf'))  # Hold the position indefinitely

    except Exception as e:
        print(f"Error in control loop: {e}")

if __name__ == "__main__":
    # Initialize the simulation
    # sim = Monte01Mujoco()
    
    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(cur_path, '../../hardware/monte01/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)
    print(f"Configuration loaded: {config}")
    
    agent = Agent(config=config, use_real_robot=True)

    # Start the simulation in a separate thread
    # sim_thread = Thread(target=sim.start)
    # sim_thread.start()

    # Start the control loop in another thread
    control_thread = Thread(target=control_loop, args=(agent,))
    control_thread.start()

    # Wait for the simulation to end
    # sim_thread.join()
    agent.sim_thread.join()
    control_thread.join()

    print("Simulation finished.")
