import os
import time
import numpy as np
from threading import Thread
import threading
from hardware.monte01.agent import Agent
from hardware.monte01.arm import Arm
from hardware.monte01.dual_arm_controller import (
    DualArmController, 
    create_demo_left_arm_actions, 
    create_demo_right_arm_actions
)
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
    # Use world coordinate frame for consistent behavior between sim and real
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
        gripper.gripper_move(0.5)
        arm_left.hold_position_for_duration(1.5)

        gripper.gripper_close()
        # NOTE: sleep will cause sudden move
        arm_left.hold_position_for_duration(1.5)

        gripper.gripper_open()

        arm_left.hold_position_for_duration(1.5)
        
    pose[AXIS_Z, 3] += 0.2  # This should now move along world Z-axis
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

def improved_control_loop(agent: Agent):
    """改进的双臂控制循环"""
    log.info("Starting improved dual arm control loop...")
    
    # 创建双臂控制器
    controller = DualArmController(agent)
    controller.initialize()
    
    sim = agent.sim
    # Wait for the viewer to be initialized and running
    while sim.viewer is None or not sim.viewer.is_running():
        time.sleep(0.1)
    
    try:
        log.info("\n=== 演示1: 顺序执行不同手臂动作 ===")
        
        # 左臂执行动作，右臂保持位置
        log.info("Left arm executing sequence, right arm holding position...")
        left_actions = create_demo_left_arm_actions()
        controller.execute_arm_sequence("left", left_actions, hold_other_arm=True)
        
        log.info("Left arm sequence completed")
        # time.sleep(1.0)
        
        # 右臂执行动作，左臂保持位置
        log.info("Right arm executing sequence, left arm holding position...")
        right_actions = create_demo_right_arm_actions()
        controller.execute_arm_sequence("right", right_actions, hold_other_arm=True)
        
        log.info("Right arm sequence completed")
        # time.sleep(1.0)
        
        # 返回起始位置
        log.info("Returning both arms to start positions...")
        controller.concurrent_move_to_start()
        
        log.info("Demo completed, holding positions...")
        
        def hold_left():
            while sim.viewer.is_running():
                controller.arm_left.hold_position_for_duration(float('inf'))

        def hold_right():
            while sim.viewer.is_running():
                controller.arm_right.hold_position_for_duration(float('inf'))

        threading.Thread(target=hold_left).start()
        threading.Thread(target=hold_right).start()
        
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received, stopping...")
        controller.emergency_stop()
    except Exception as e:
        log.error(f"Error in improved control loop: {e}", exc_info=True)
        controller.emergency_stop()
    finally:
        controller.shutdown()
        log.info("Dual arm controller shutdown completed")


def legacy_control_loop(agent: Agent):
    """保留的原始控制循环，用于对比"""
    agent.wait_for_ready()

    sim = agent.sim
    print("Legacy control loop started.")
    
    # Wait for the viewer to be initialized
    while sim.viewer is None:
        time.sleep(0.1)

    # Wait for the viewer to start running
    while not sim.viewer.is_running():
        time.sleep(0.1)

    try:
        arm_left: Arm = agent.arm_left()
        arm_right: Arm = agent.arm_right()
        
        # 顺序移动到起始位置 (效率低)
        log.info("Moving arms to start positions sequentially...")
        arm_right.move_to_start()
        arm_left.move_to_start()
        arm_left.hold_position_for_duration(0.2)

        # 只控制左臂，右臂不受控制
        log.info("Executing left arm actions (right arm uncontrolled)...")
        proc2(arm_left)

        arm_left.hold_position_for_duration(1.0)
        arm_left.move_to_start()
        
        print("Legacy demo finished.")
        arm_left.hold_position_for_duration(float('inf'))

    except Exception as e:
        log.error(f"Error in legacy control loop: {e}", exc_info=True)

if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    robot_config_file = os.path.join(cur_path, '../../hardware/monte01/config/agent.yaml')
    config = file_utils.read_config(robot_config_file)
    print(f"Configuration loaded: {config}")

    parser = argparse.ArgumentParser(description='Test sim2real with option to use real robot.')
    parser.add_argument('--use_real_robot', action='store_true', help='Enable real robot mode.')
    parser.add_argument('--use_legacy', action='store_true', help='Use legacy single-arm control (for comparison).')
    args = parser.parse_args()
    
    agent = Agent(config=config, use_real_robot=args.use_real_robot)

    # Choose control loop based on argument
    if args.use_legacy:
        log.info("Using legacy control loop for comparison...")
        control_function = legacy_control_loop
    else:
        log.info("Using improved dual arm control loop...")
        control_function = improved_control_loop

    # Start the control loop in another thread
    control_thread = Thread(target=control_function, args=(agent,))
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
