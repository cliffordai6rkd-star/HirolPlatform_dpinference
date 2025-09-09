#!/usr/bin/env python3
"""
使用 RobotController 进行真正的运动控制测试
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from a2d_sdk.robot import RobotController, RobotDds
import numpy as np
import glog as log


def test_robot_controller_motion():
    """使用 RobotController 进行运动控制"""
    
    log.info("=== RobotController 运动控制测试 ===")
    
    # 初始化两个接口
    robot_controller = RobotController()
    robot_dds = RobotDds()
    
    time.sleep(2.0)  # 等待初始化
    
    try:
        # 1. 检查当前运动状态
        log.info("检查运动控制状态...")
        motion_status = robot_controller.get_motion_status()
        if motion_status:
            log.info(f"当前模式: {motion_status['mode']['name']} (value: {motion_status['mode']['value']})")
            log.info(f"错误状态: {motion_status['error']['has_error']} - {motion_status['error']['message']}")
        else:
            log.warning("无法获取运动状态")
        
        # 2. 确保进入 servo 模式
        log.info("设置为 SERVO 模式...")
        robot_controller.set_motion_control_mode(1)  # 1 = servo 模式
        time.sleep(0.5)
        
        # 再次检查状态
        motion_status = robot_controller.get_motion_status()
        if motion_status:
            log.info(f"设置后模式: {motion_status['mode']['name']}")
        
        # 3. 获取当前关节状态
        arm_states, _ = robot_dds.arm_joint_states()
        head_states, _ = robot_dds.head_joint_states()
        waist_states, _ = robot_dds.waist_joint_states()
        
        log.info(f"当前arm状态 (前3关节): {arm_states[:3]}")
        
        # 4. 使用关节位置控制进行小幅运动
        log.info("开始关节位置控制测试...")
        
        for i in range(10):
            # 计算轻微的正弦运动
            t = i * 0.2
            amplitude = np.deg2rad(2.0)  # 2度幅度
            sine_offset = amplitude * np.sin(2 * np.pi * 0.2 * t)  # 0.2Hz
            
            # 创建目标位置（只改变左臂第一个关节）
            target_arm_states = list(arm_states)
            target_arm_states[0] = arm_states[0] + sine_offset
            
            # 准备 joint_group 字典
            joint_group = {
                'left_arm': target_arm_states[:7],   # 左臂7个关节
                'right_arm': target_arm_states[7:14], # 右臂7个关节
            }
            
            # 使用关节位置控制
            lifetime = 0.5  # 命令有效时间0.5秒
            robot_controller.set_joint_position_control(lifetime, joint_group)
            
            log.info(f"步骤 {i+1}: 目标偏移 {sine_offset:.4f} ({np.rad2deg(sine_offset):.2f}°)")
            
            time.sleep(0.2)
            
            # 每3步检查实际位置
            if (i + 1) % 3 == 0:
                current_arm_states, _ = robot_dds.arm_joint_states()
                actual_movement = current_arm_states[0] - arm_states[0]
                log.info(f"  实际移动: {actual_movement:.4f} ({np.rad2deg(actual_movement):.2f}°)")
        
        log.info("关节位置控制测试完成")
        
    except Exception as e:
        log.error(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 停止运动
        try:
            robot_controller.set_motion_stop()
            log.info("运动已停止")
        except:
            pass
        
        robot_dds.shutdown()
        log.info("连接已关闭")


def test_trajectory_tracking_control():
    """使用轨迹跟踪控制"""
    
    log.info("=== 轨迹跟踪控制测试 ===")
    
    robot_controller = RobotController()
    robot_dds = RobotDds()
    
    time.sleep(2.0)
    
    try:
        # 设置 servo 模式
        robot_controller.set_motion_control_mode(1)
        time.sleep(0.5)
        
        # 获取当前状态
        arm_states, _ = robot_dds.arm_joint_states()
        head_states, _ = robot_dds.head_joint_states()
        waist_states, _ = robot_dds.waist_joint_states()
        
        log.info("开始轨迹跟踪控制...")
        
        for i in range(5):
            # 当前时间戳
            infer_timestamp = int(time.time() * 1e9)
            
            # 计算目标位置
            amplitude = np.deg2rad(3.0)
            sine_offset = amplitude * np.sin(2 * np.pi * 0.1 * i)
            
            target_arm = list(arm_states)
            target_arm[0] = arm_states[0] + sine_offset
            
            # 机器人状态
            robot_states = {
                "head": head_states,
                "waist": waist_states,
                "arm": arm_states,  # 参考状态使用初始状态
            }
            
            # 机器人动作
            robot_actions = [{
                "left_arm": {
                    "action_data": target_arm[:7],
                    "control_type": "ABS_JOINT"
                },
                "right_arm": {
                    "action_data": target_arm[7:14],
                    "control_type": "ABS_JOINT"
                }
            }]
            
            # 执行轨迹跟踪
            robot_controller.trajectory_tracking_control(
                infer_timestamp,
                robot_states,
                robot_actions,
                robot_link="base_link",
                trajectory_reference_time=1.0
            )
            
            log.info(f"轨迹步骤 {i+1}: 偏移 {sine_offset:.4f}")
            time.sleep(1.0)
        
        log.info("轨迹跟踪控制完成")
        
    except Exception as e:
        log.error(f"轨迹跟踪测试出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        robot_controller.set_motion_stop()
        robot_dds.shutdown()


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 关节位置控制测试")
    print("2. 轨迹跟踪控制测试")
    
    choice = input("输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_robot_controller_motion()
    elif choice == "2":
        test_trajectory_tracking_control()
    else:
        print("无效选择，运行关节位置控制测试...")
        test_robot_controller_motion()