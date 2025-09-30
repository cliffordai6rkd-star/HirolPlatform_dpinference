#!/usr/bin/env python3
"""
测试仅双臂运动，忽略头部错误
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from hardware.agibot_g1.agibot_g1 import AgibotG1
import numpy as np
import glog as log

def test_arm_only_movement():
    """仅测试双臂运动，不涉及头部"""
    
    # 明确禁用所有非手臂组件
    config = {
        'dof': [7, 7], 
        'robot_name': 'ArmOnlyTest',
        'control_head': False,      # 明确禁用头部
        'control_waist': False,     # 禁用腰部
        'control_wheel': False,     # 禁用轮子
        'control_gripper': False,   # 禁用夹爪
        'control_hand': False,      # 禁用手部
    }
    
    log.info("=== 仅双臂运动测试 ===")
    
    robot = AgibotG1(config)
    
    try:
        # 检查状态但忽略头部错误
        status = robot._robot.whole_body_status()[0]
        log.info(f"左臂控制: {status['left_arm_control']}")
        log.info(f"右臂控制: {status['right_arm_control']}")
        log.info(f"左臂急停: {status['left_arm_estop']}")
        log.info(f"右臂急停: {status['right_arm_estop']}")
        log.info(f"颈部错误 (忽略): {status['neck_error']}")
        
        # 如果手臂控制正常，尝试运动
        if status['left_arm_control'] and status['right_arm_control']:
            log.info("手臂控制状态正常，开始测试运动...")
            
            # 获取当前位置
            states = robot.get_joint_states()
            current_pos = states._positions.copy()
            log.info(f"当前位置 (前6关节): {current_pos[:6]}")
            
            # 创建小幅度正弦运动
            for i in range(20):
                t = i * 0.1  # 时间步长
                
                # 生成正弦波偏移 (仅前6个关节，幅度5度)
                amplitude_rad = np.deg2rad(5.0)
                sine_offset = amplitude_rad * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz
                
                target_pos = current_pos.copy()
                # 仅移动左臂前3个关节
                target_pos[0] = current_pos[0] + sine_offset
                target_pos[1] = current_pos[1] + sine_offset * 0.5
                target_pos[2] = current_pos[2] + sine_offset * 0.3
                
                # 发送命令
                success = robot.set_joint_command(['position'], target_pos)
                print(f"发送位置命令: {target_pos[:6]}, 成功={success}")
                if i % 5 == 0:  # 每5步打印一次
                    current_states = robot.get_joint_states()
                    actual_pos = current_states._positions
                    movement = actual_pos[0] - current_pos[0]
                    log.info(f"步骤 {i}: 目标偏移={sine_offset:.3f}, 实际移动={movement:.3f}, 成功={success}")
                
                time.sleep(0.1)
        else:
            log.error("手臂控制状态异常，无法进行运动测试")
            
    except Exception as e:
        log.error(f"测试过程中出错: {e}")
        
    finally:
        robot.close()

if __name__ == "__main__":
    test_arm_only_movement()