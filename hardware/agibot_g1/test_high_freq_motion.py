#!/usr/bin/env python3
"""
高频小幅度运动测试 - 直接调用 move_arm
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from hardware.agibot_g1.agibot_g1 import AgibotG1
import numpy as np
import glog as log


def test_high_frequency_small_movement():
    """测试高频小幅度运动"""
    
    config = {
        'dof': [7, 7], 
        'robot_name': 'HighFreqTest',
        'control_head': False,
        'control_waist': False,
        'control_wheel': False,
        'control_gripper': False,
        'control_hand': False,
    }
    
    log.info("=== 高频小幅度运动测试 ===")
    
    robot = AgibotG1(config)
    
    try:
        # 检查状态
        status = robot._robot.whole_body_status()[0]
        log.info(f"左臂控制: {status['left_arm_control']}")
        log.info(f"右臂控制: {status['right_arm_control']}")
        log.info(f"颈部错误: {status['neck_error']}")
        
        if status['left_arm_control'] and status['right_arm_control']:
            # 获取初始位置
            states = robot.get_joint_states()
            initial_pos = states._positions.copy()
            log.info(f"初始位置 (前3关节): {initial_pos[:3]}")
            
            # 高频小幅度运动参数
            control_freq = 100  # 100Hz
            dt = 1.0 / control_freq
            amplitude_rad = np.deg2rad(1.0)  # 减小到1度
            test_duration = 5.0  # 5秒测试
            num_steps = int(test_duration / dt)
            
            log.info(f"开始高频测试: {control_freq}Hz, 幅度={np.rad2deg(amplitude_rad):.1f}度, 持续{test_duration}秒")
            
            start_time = time.time()
            
            for i in range(num_steps):
                loop_start = time.time()
                
                # 计算当前时间和正弦偏移
                t = i * dt
                sine_offset = amplitude_rad * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz正弦波
                
                # 只移动第一个关节，幅度很小
                target_pos = initial_pos.copy()
                target_pos[0] = initial_pos[0] + sine_offset
                
                # 方法1: 尝试直接调用 move_arm (只发送前14个关节)
                try:
                    robot._robot.move_arm(target_pos[:14])
                    direct_success = True
                except Exception as e:
                    direct_success = False
                    if i == 0:  # 只在第一次失败时打印错误
                        log.error(f"直接调用 move_arm 失败: {e}")
                
                # 方法2: 如果直接调用失败，使用原来的方法
                if not direct_success:
                    success = robot.set_joint_command(['position'], target_pos)
                else:
                    success = True
                
                # 每秒打印一次进度
                if i % control_freq == 0:
                    elapsed = time.time() - start_time
                    current_states = robot.get_joint_states()
                    actual_pos = current_states._positions
                    movement = actual_pos[0] - initial_pos[0]
                    
                    log.info(f"时间: {elapsed:.1f}s, 目标偏移: {sine_offset:.4f}, "
                            f"实际移动: {movement:.4f}, 直接调用: {direct_success}")
                
                # 精确的时间控制
                loop_time = time.time() - loop_start
                if loop_time < dt:
                    time.sleep(dt - loop_time)
                elif loop_time > 1.5 * dt:
                    log.warning(f"控制循环过慢: {loop_time*1000:.1f}ms > {dt*1000:.1f}ms")
            
            log.info("高频测试完成")
            
            # 检查最终位置
            final_states = robot.get_joint_states()
            final_pos = final_states._positions
            total_movement = final_pos[0] - initial_pos[0]
            log.info(f"最终位置变化: {total_movement:.4f} rad ({np.rad2deg(total_movement):.2f} 度)")
            
        else:
            log.error("手臂控制状态异常")
            
    except Exception as e:
        log.error(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        robot.close()


def test_incremental_movement():
    """测试递增式运动 - 每次只移动很小的增量"""
    
    config = {
        'dof': [7, 7], 
        'robot_name': 'IncrementalTest',
        'control_head': False,
        'control_waist': False, 
        'control_wheel': False,
        'control_gripper': False,
        'control_hand': False,
    }
    
    log.info("=== 递增式运动测试 ===")
    
    robot = AgibotG1(config)
    
    try:
        # 获取初始位置
        states = robot.get_joint_states()
        current_pos = states._positions.copy()
        log.info(f"初始位置: {current_pos[0]:.4f}")
        
        # 很小的增量步进
        increment = np.deg2rad(0.1)  # 0.1度增量
        num_steps = 50  # 总共5度
        
        log.info(f"开始递增测试: 每步{np.rad2deg(increment):.2f}度, 共{num_steps}步")
        
        for i in range(num_steps):
            # 计算新的目标位置
            target_pos = current_pos.copy()
            target_pos[0] = current_pos[0] + increment
            
            # 直接调用 move_arm
            try:
                robot._robot.move_arm(target_pos[:14])
                log.info(f"步骤 {i+1}: 目标 {target_pos[0]:.4f}")
                
                # 更新当前位置
                current_pos = target_pos
                
                time.sleep(0.05)  # 50ms间隔
                
                # 每10步检查实际位置
                if (i + 1) % 10 == 0:
                    actual_states = robot.get_joint_states()
                    actual_pos = actual_states._positions[0]
                    log.info(f"第{i+1}步后实际位置: {actual_pos:.4f}")
                    
            except Exception as e:
                log.error(f"步骤 {i+1} 失败: {e}")
                break
        
        log.info("递增测试完成")
        
    except Exception as e:
        log.error(f"递增测试出错: {e}")
        
    finally:
        robot.close()


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 高频小幅度运动 (100Hz, 1度幅度)")
    print("2. 递增式运动 (0.1度步进)")
    
    choice = input("输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_high_frequency_small_movement()
    elif choice == "2":
        test_incremental_movement()
    else:
        print("无效选择，运行高频测试...")
        test_high_frequency_small_movement()