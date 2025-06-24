import os
import time
import numpy as np
from threading import Thread
from hardware.monte01.agent import Agent
from hardware.monte01.arm import Arm
from tools import file_utils
import glog as log
import argparse
import cv2, rclpy
from hardware.monte01.gripper_xarm import Gripper
log.setLevel("INFO")

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

def proc2(arm_left: Arm):
    pose = arm_left.get_tcp_pose()
    pose_before = pose.copy()
    pose[AXIS_Y, 3] -= 0.1
    arm_left.move_to_pose(pose)
    pose1 = arm_left.get_tcp_pose()
    print(f"pose1\n{pose1}")
    print(f"pose1[:3, 3]\n{pose1[:3, 3]}")

    delta1 = pose1[:3, 3] - pose_before[:3, 3]
    print(f"Moved after first move: {delta1}")

    gripper: Gripper = arm_left.get_gripper()
    if gripper is not None:
        gripper.gripper_close()
        # NOTE: sleep will cause sudden move
        arm_left.hold_position_for_duration(1.5)

        gripper.gripper_open()

        arm_left.hold_position_for_duration(1.5)
        
    pose[AXIS_Y, 3] += 0.2
    arm_left.move_to_pose(pose)
    pose2 = arm_left.get_tcp_pose()
    delta2 = pose2[:3, 3] - pose1[:3, 3]
    print(f"Moved after second move: {delta2}")

# def proc1(arm_left):
#     # 1. 獲取當前在局部座標系下的姿態
#     local_pose_now = arm_left.get_tcp_pose()
#     if local_pose_now is None:
#         log.error("Could not get current pose.")
#         return
    
#     # 2. 將其轉換到世界座標系，作為我們操作的基準
#     world_pose_base = arm_left.convert_pose_to_world(local_pose_now)

#     print(f"Base pose in WORLD frame acquired: \n{world_pose_base}")
    
#     # --- 第一次移動：沿世界 Y 軸負向移動 10cm ---
#     # 使用副本進行操作，避免修改原始基準姿態
#     world_pose_target1 = world_pose_base.copy()
#     world_pose_target1[AXIS_Y, 3] -= 0.1
    
#     # 將世界目標姿態轉換回手臂基座的局部座標系，以便 IK 求解器使用
#     local_pose_target1 = arm_left.convert_pose_to_local(world_pose_target1)

#     print("Moving sideways (World Y-)...")
#     arm_left.move_to_pose(local_pose_target1)

#     # pose1 = arm_left.get_tcp_pose()

#     # 比较 local_pose_now 和 pose1
#     # print("Comparing initial and final local poses:")
#     # print("local_pose_now:")
#     # print(local_pose_now)
#     # print("pose1:")
#     # print(pose1)
#     # if pose1 is not None:
#     #     diff = pose1 - local_pose_now
#     #     print("Difference (pose1 - local_pose_now):")
#     #     print(diff)
#     # else:
#     #     print("pose1 is None, cannot compare.")

#     arm_left.hold_position_for_duration(1.5)

#     arm_left.print_state()

#     gripper: Gripper = arm_left.get_gripper()
#     if gripper is not None:
#         gripper.gripper_close()
#         # NOTE: sleep will cause sudden move
#         arm_left.hold_position_for_duration(1.5)

#         gripper.gripper_open()

#         arm_left.hold_position_for_duration(1.5)

#     # --- 第二次移動：沿世界 Y 軸正向移回原位 ---
#     # 再次使用原始基準姿態的副本
#     world_pose_target2 = world_pose_base.copy()
#     # 您可以設定新的目標，例如移回原位，或從原位向另一側移動
#     world_pose_target2[AXIS_Y, 3] += 0.2
    
#     local_pose_target2 = arm_left.convert_pose_to_local(world_pose_target2)

#     print("Returning to original sideways position...")
#     arm_left.move_to_pose(local_pose_target2)

def control_loop(agent: Agent):
    agent.wait_for_ready()

    sim = agent.sim
    """A simple control loop that moves the arm and gripper."""
    print("Control loop started.")
    
    # Wait for the viewer to be initialized
    while sim.viewer is None:
        time.sleep(0.1)

    # Wait for the viewer to start running
    while not sim.viewer.is_running():
        time.sleep(0.1)

    try:
        arm_left: Arm = agent.arm_left()
        
        # --- Step 1: 移動到起始位置並穩定 ---
        
        arm_left.move_to_start()

        arm_left.hold_position_for_duration(0.2) # 穩定

        # --- Step 2: 規劃基於世界座標系的移動 ---
        print("\n--- Step 2: Planning moves in WORLD coordinate frame... ---")
        
        proc2(arm_left)

        arm_left.hold_position_for_duration(1.0)

        arm_left.move_to_start()
        
        print("Step 2 finished.")
        arm_left.hold_position_for_duration(float('inf'))

    except Exception as e:
        log.error(f"Error in control loop: {e}", exc_info=True)

if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(cur_path, '../../hardware/monte01/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)
    print(f"Configuration loaded: {config}")

    parser = argparse.ArgumentParser(description='Test sim2real with option to use real robot.')
    parser.add_argument('--use_real_robot', action='store_true', help='Enable real robot mode.')
    args = parser.parse_args()
    
    agent = Agent(config=config, use_real_robot=args.use_real_robot)

    # Start the control loop in another thread
    control_thread = Thread(target=control_loop, args=(agent,))
    control_thread.start()

    # Start the ROS2 node in another thread
    def ros2_spin():
        try:
            rclpy.spin(agent.head_front_camera())
        except KeyboardInterrupt:
            print("偵測到 Ctrl+C，正在關閉節點...")
        finally:
            if rclpy.ok():
                agent.head_front_camera().destroy_node()
                rclpy.shutdown()
            cv2.destroyAllWindows()
            print("程式已乾淨地關閉。")

    ros2_thread = Thread(target=ros2_spin)
    ros2_thread.start()
    ros2_thread.join()

    # Wait for the simulation to end
    agent.sim_thread.join()
    control_thread.join()

    print("Simulation finished.")
