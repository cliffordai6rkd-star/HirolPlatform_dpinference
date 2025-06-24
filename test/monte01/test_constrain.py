import mujoco
import numpy as np

def test_single_gripper(model, data, gripper_name, actuator_id, drive_joint_idx, follower_joints, target):
    """测试单个夹爪的约束效果"""
    print(f'\n=== {gripper_name}夹爪测试 ===')
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 获取参考位置
    drive_ref = data.qpos[drive_joint_idx]
    
    print(f'设置{gripper_name}驱动关节目标: {target}')
    data.ctrl[actuator_id] = target
    
    joint_names = [model.joint(i).name for i in follower_joints]
    
    print(f'{gripper_name}仿真进行中...')
    for i in range(1500):
        mujoco.mj_step(model, data)
        
        if i % 300 == 0:
            drive_pos = data.qpos[drive_joint_idx]
            
            print(f'  步骤 {i}: 驱动关节 = {drive_pos:.6f}')
            
            total_error = 0
            for j, joint_idx in enumerate(follower_joints):
                actual = data.qpos[joint_idx]
                # 正确的约束公式: y = y_ref + 0.1259 * (x - x_ref)
                follower_ref = data.qpos0[joint_idx] if hasattr(data, 'qpos0') else 0.0
                expected = follower_ref + 0.1259 * (drive_pos - drive_ref)
                error = abs(actual - expected)
                total_error += error
                print(f'    {joint_names[j]}: {actual:.6f} (期望: {expected:.6f}, 误差: {error:.6f})')
            
            avg_error = total_error / len(follower_joints)
            print(f'    平均约束误差: {avg_error:.6f}')
    
    # 最终评估
    print(f'\n{gripper_name}最终评估:')
    drive_pos = data.qpos[drive_joint_idx]
    
    errors = []
    for j, joint_idx in enumerate(follower_joints):
        actual = data.qpos[joint_idx]
        follower_ref = 0.0  # 初始配置中所有关节都是0
        expected = follower_ref + 0.1259 * (drive_pos - drive_ref)
        error = abs(actual - expected)
        errors.append(error)
        print(f'{joint_names[j]}: 实际={actual:.6f}, 期望={expected:.6f}, 误差={error:.6f}')
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    print(f'\n{gripper_name}约束性能:')
    print(f'平均误差: {avg_error:.6f}')
    print(f'最大误差: {max_error:.6f}')
    
    if avg_error < 0.05:
        result = '✓ 良好'
    elif avg_error < 0.1:
        result = '△ 可接受'
    else:
        result = '✗ 需调整'
    
    print(f'评估结果: {result}')
    return avg_error

def test_both_grippers(model, data):
    """测试双手夹爪同时工作"""
    print(f'\n=== 双手夹爪同时测试 ===')
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 左右手不同目标
    left_actuator = 12   # left_drive_gear_joint_ctrl
    right_actuator = 20  # right_drive_gear_joint_ctrl
    
    left_target = 0.8   # 左手张开
    right_target = 0.2  # 右手略开
    
    print(f'左手目标: {left_target}')
    print(f'右手目标: {right_target}')
    
    data.ctrl[left_actuator] = left_target
    data.ctrl[right_actuator] = right_target
    
    # 关键关节监控
    left_key_joint = 14   # left_right_finger_joint
    right_key_joint = 28  # right_right_finger_joint
    
    print('\n双手仿真中...')
    for i in range(2000):
        mujoco.mj_step(model, data)
        
        if i % 400 == 0:
            left_drive = data.qpos[12]
            right_drive = data.qpos[26]  # right_drive_gear_joint
            
            left_actual = data.qpos[left_key_joint]
            right_actual = data.qpos[right_key_joint]
            
            # 使用正确的约束公式
            left_expected = 0.0 + 0.1259 * (left_drive - 0.0)
            right_expected = 0.0 + 0.1259 * (right_drive - 0.0)
            
            left_error = abs(left_actual - left_expected)
            right_error = abs(right_actual - right_expected)
            
            print(f'  步骤 {i}:')
            print(f'    左手: 驱动={left_drive:.4f}, 手指={left_actual:.4f}, 误差={left_error:.4f}')
            print(f'    右手: 驱动={right_drive:.4f}, 手指={right_actual:.4f}, 误差={right_error:.4f}')

def main():
    model = mujoco.MjModel.from_xml_path('assets/monte_01/urdf/robot_description.xml')
    data = mujoco.MjData(model)
    
    print('=== 双手夹爪约束测试 ===')
    print(f'约束数量: {model.neq}')
    print(f'时间步长: {model.opt.timestep}')
    
    # 左手夹爪参数
    left_actuator = 12  # left_drive_gear_joint_ctrl
    left_drive_joint = 12  # left_drive_gear_joint (0-based)
    left_followers = [13, 14, 15, 16, 17, 18]  # 左手所有从动关节
    
    # 右手夹爪参数  
    right_actuator = 20  # right_drive_gear_joint_ctrl
    right_drive_joint = 26  # right_drive_gear_joint (0-based)
    right_followers = [27, 28, 29, 30, 31, 32]  # 右手所有从动关节
    
    # 测试左手
    left_error = test_single_gripper(
        model, data, "左手", 
        left_actuator, left_drive_joint, left_followers, 
        0.5
    )
    
    # 测试右手
    right_error = test_single_gripper(
        model, data, "右手",
        right_actuator, right_drive_joint, right_followers,
        0.7
    )
    
    # 测试双手同时
    test_both_grippers(model, data)
    
    # 总结
    print(f'\n=== 最终总结 ===')
    print(f'左手平均误差: {left_error:.4f}')
    print(f'右手平均误差: {right_error:.4f}')
    
    if left_error < 0.1 and right_error < 0.1:
        print('✓ 双手夹爪约束整体可接受')
    else:
        print('△ 需要进一步优化约束参数')

if __name__ == "__main__":
    main()