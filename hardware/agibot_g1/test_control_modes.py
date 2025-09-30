#!/usr/bin/env python3
"""
测试不同的控制模式以找到能让机器人实际运动的方法
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from a2d_sdk.robot import RobotController, RobotDds
import numpy as np
import glog as log


def test_planning_mode():
    """测试 PLANNING 模式"""
    
    log.info("=== 测试 PLANNING 模式 ===")
    
    robot_controller = RobotController()
    robot_dds = RobotDds()
    
    time.sleep(2.0)
    
    try:
        # 检查当前状态
        motion_status = robot_controller.get_motion_status()
        log.info(f"初始模式: {motion_status['mode']['name']} (value: {motion_status['mode']['value']})")
        
        # 切换到 PLANNING 模式 (value: 2)
        log.info("切换到 PLANNING 模式...")
        robot_controller.set_motion_control_mode(2)
        time.sleep(1.0)
        
        # 确认模式切换
        motion_status = robot_controller.get_motion_status()
        log.info(f"当前模式: {motion_status['mode']['name']}")
        
        if motion_status['mode']['value'] == 2:
            # 获取当前状态
            arm_states, _ = robot_dds.arm_joint_states()
            log.info(f"当前位置 (前3关节): {arm_states[:3]}")
            
            # 尝试关节位置控制
            log.info("在 PLANNING 模式下尝试关节位置控制...")
            
            for i in range(5):
                amplitude = np.deg2rad(5.0)
                sine_offset = amplitude * np.sin(2 * np.pi * 0.2 * i * 0.5)
                
                target_arm = list(arm_states)
                target_arm[0] = arm_states[0] + sine_offset
                
                joint_group = {
                    'left_arm': target_arm[:7],
                    'right_arm': target_arm[7:14],
                }
                
                robot_controller.set_joint_position_control(0.5, joint_group)
                log.info(f"步骤 {i+1}: 偏移 {sine_offset:.4f} ({np.rad2deg(sine_offset):.2f}°)")
                
                time.sleep(0.5)
                
                # 检查实际位置
                current_arm, _ = robot_dds.arm_joint_states()
                movement = current_arm[0] - arm_states[0]
                log.info(f"  实际移动: {movement:.4f} ({np.rad2deg(movement):.2f}°)")
                
        else:
            log.error(f"无法切换到 PLANNING 模式，当前模式: {motion_status['mode']['name']}")
            
    except Exception as e:
        log.error(f"PLANNING 模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        robot_controller.set_motion_stop()
        robot_dds.shutdown()
        log.info("测试结束")


def test_direct_joint_velocity():
    """测试直接速度控制"""
    
    log.info("=== 测试直接速度控制 ===")
    
    robot_controller = RobotController()
    robot_dds = RobotDds()
    
    time.sleep(2.0)
    
    try:
        # 设置为 SERVO 模式
        robot_controller.set_motion_control_mode(1)
        time.sleep(0.5)
        
        # 获取当前状态
        arm_states, _ = robot_dds.arm_joint_states()
        initial_pos = arm_states[0]
        log.info(f"初始位置: {initial_pos:.4f}")
        
        # 尝试速度控制
        log.info("尝试关节速度控制...")
        
        velocity_rad_per_sec = np.deg2rad(5.0)  # 5度/秒
        
        joint_group = {
            'left_arm': [velocity_rad_per_sec, 0, 0, 0, 0, 0, 0],
            'right_arm': [0, 0, 0, 0, 0, 0, 0],
        }
        
        # 持续发送速度命令
        for i in range(20):  # 2秒
            robot_controller.set_joint_velocity_control(0.2, joint_group)
            
            if i % 5 == 0:
                current_arm, _ = robot_dds.arm_joint_states()
                movement = current_arm[0] - initial_pos
                log.info(f"时间 {i*0.1:.1f}s: 移动 {movement:.4f} ({np.rad2deg(movement):.2f}°)")
            
            time.sleep(0.1)
        
        # 停止
        robot_controller.set_motion_stop()
        
        # 最终位置
        final_arm, _ = robot_dds.arm_joint_states()
        total_movement = final_arm[0] - initial_pos
        log.info(f"总移动: {total_movement:.4f} ({np.rad2deg(total_movement):.2f}°)")
        
    except Exception as e:
        log.error(f"速度控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        robot_controller.set_motion_stop()
        robot_dds.shutdown()


def test_impedance_control():
    """测试阻抗控制模式"""
    
    log.info("=== 测试阻抗控制 ===")
    
    robot_controller = RobotController()
    robot_dds = RobotDds()
    
    time.sleep(2.0)
    
    try:
        # 设置为 SERVO 模式
        robot_controller.set_motion_control_mode(1)
        time.sleep(0.5)
        
        # 获取当前状态
        arm_states, _ = robot_dds.arm_joint_states()
        log.info(f"当前位置 (前3关节): {arm_states[:3]}")
        
        # 尝试阻抗控制
        log.info("尝试关节阻抗控制...")
        
        # 设置阻抗参数
        stiffness = [100, 100, 100, 50, 50, 50, 50]  # N·m/rad
        damping = [10, 10, 10, 5, 5, 5, 5]  # N·m·s/rad
        
        joint_group = {
            'left_arm': {
                'stiffness': stiffness,
                'damping': damping,
                'position': list(arm_states[:7])
            },
            'right_arm': {
                'stiffness': stiffness,
                'damping': damping,
                'position': list(arm_states[7:14])
            }
        }
        
        # 尝试设置阻抗
        robot_controller.set_joint_impedance_control(0.5, joint_group)
        log.info("阻抗控制已设置")
        
        time.sleep(1.0)
        
        # 改变目标位置
        for i in range(10):
            amplitude = np.deg2rad(3.0)
            sine_offset = amplitude * np.sin(2 * np.pi * 0.2 * i * 0.2)
            
            target_left = list(arm_states[:7])
            target_left[0] += sine_offset
            
            joint_group['left_arm']['position'] = target_left
            
            robot_controller.set_joint_impedance_control(0.5, joint_group)
            log.info(f"步骤 {i+1}: 目标偏移 {sine_offset:.4f}")
            
            time.sleep(0.2)
            
            if i % 3 == 0:
                current_arm, _ = robot_dds.arm_joint_states()
                movement = current_arm[0] - arm_states[0]
                log.info(f"  实际移动: {movement:.4f}")
                
    except Exception as e:
        log.error(f"阻抗控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        robot_controller.set_motion_stop()
        robot_dds.shutdown()


def test_whole_body_control():
    """测试全身控制方法"""
    
    log.info("=== 测试全身控制 ===")
    
    robot_controller = RobotController()
    robot_dds = RobotDds()
    
    time.sleep(2.0)
    
    try:
        # 检查全身状态
        whole_body_status = robot_controller.get_whole_body_status()
        if whole_body_status:
            log.info(f"左臂控制: {whole_body_status.get('left_arm_control', False)}")
            log.info(f"右臂控制: {whole_body_status.get('right_arm_control', False)}")
            log.info(f"左臂急停: {whole_body_status.get('left_arm_estop', False)}")
            log.info(f"右臂急停: {whole_body_status.get('right_arm_estop', False)}")
            
            # 如果有急停，尝试解除
            if whole_body_status.get('left_arm_estop') or whole_body_status.get('right_arm_estop'):
                log.warning("检测到急停状态，尝试解除...")
                robot_controller.set_motion_control_mode(0)  # STOP mode
                time.sleep(0.5)
                robot_controller.set_motion_control_mode(1)  # SERVO mode
                time.sleep(0.5)
                
                # 重新检查
                whole_body_status = robot_controller.get_whole_body_status()
                log.info(f"解除后左臂急停: {whole_body_status.get('left_arm_estop', False)}")
                log.info(f"解除后右臂急停: {whole_body_status.get('right_arm_estop', False)}")
        
        # 尝试全身协调控制
        arm_states, _ = robot_dds.arm_joint_states()
        head_states, _ = robot_dds.head_joint_states()
        waist_states, _ = robot_dds.waist_joint_states()
        
        log.info("尝试全身协调控制...")
        
        for i in range(5):
            amplitude = np.deg2rad(2.0)
            sine_offset = amplitude * np.sin(2 * np.pi * 0.1 * i)
            
            target_arm = list(arm_states)
            target_arm[0] += sine_offset
            
            # 使用轨迹跟踪控制
            robot_states = {
                "head": head_states,
                "waist": waist_states,
                "arm": arm_states,
            }
            
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
            
            timestamp = int(time.time() * 1e9)
            robot_controller.trajectory_tracking_control(
                timestamp,
                robot_states,
                robot_actions,
                "base_link",
                0.5  # 更快的参考时间
            )
            
            log.info(f"轨迹步骤 {i+1}: 偏移 {sine_offset:.4f}")
            time.sleep(0.5)
            
            # 检查实际运动
            current_arm, _ = robot_dds.arm_joint_states()
            movement = current_arm[0] - arm_states[0]
            log.info(f"  实际移动: {movement:.4f} ({np.rad2deg(movement):.2f}°)")
            
    except Exception as e:
        log.error(f"全身控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        robot_controller.set_motion_stop()
        robot_dds.shutdown()


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. PLANNING 模式")
    print("2. 速度控制")
    print("3. 阻抗控制")
    print("4. 全身协调控制")
    print("5. 运行所有测试")
    
    choice = input("输入选择 (1-5): ").strip()
    
    if choice == "1":
        test_planning_mode()
    elif choice == "2":
        test_direct_joint_velocity()
    elif choice == "3":
        test_impedance_control()
    elif choice == "4":
        test_whole_body_control()
    elif choice == "5":
        test_planning_mode()
        print("\n" + "="*60 + "\n")
        test_direct_joint_velocity()
        print("\n" + "="*60 + "\n")
        test_impedance_control()
        print("\n" + "="*60 + "\n")
        test_whole_body_control()
    else:
        print("无效选择，运行全身控制测试...")
        test_whole_body_control()