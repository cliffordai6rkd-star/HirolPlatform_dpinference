import mujoco
import numpy as np

def test_constraint_formula():
    """测试约束公式的正确性"""
    model = mujoco.MjModel.from_xml_path('assets/monte_01/urdf/robot_description.xml')
    data = mujoco.MjData(model)
    
    print('=== 约束公式验证测试 ===')
    
    # 重置到参考配置
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 获取参考位置
    drive_ref = data.qpos[12]    # left_drive_gear_joint参考位置
    finger_ref = data.qpos[14]   # left_right_finger_joint参考位置
    
    print(f'参考配置:')
    print(f'  驱动关节参考位置 (x0): {drive_ref:.6f}')
    print(f'  从动关节参考位置 (y0): {finger_ref:.6f}')
    
    # 约束参数: polycoef="0 0.1259 0 0 0" 意味着 a0=0, a1=0.1259, a2=a3=a4=0
    # 所以约束是: y - y0 = 0.1259 * (x - x0)
    # 即: y = y0 + 0.1259 * (x - x0)
    
    print(f'\n约束公式: y = {finger_ref:.6f} + 0.1259 * (x - {drive_ref:.6f})')
    
    # 测试不同的驱动关节位置
    test_positions = [0.0, 0.5, 1.0, 2.0]
    
    for x in test_positions:
        # 手动设置驱动关节位置
        data.qpos[12] = x
        
        # 根据约束公式计算期望的从动关节位置
        expected_y = finger_ref + 0.1259 * (x - drive_ref)
        
        # 强制执行约束
        mujoco.mj_forward(model, data)
        
        # 检查实际从动关节位置
        actual_y = data.qpos[14]
        
        error = abs(actual_y - expected_y)
        
        print(f'\n测试 x = {x:.3f}:')
        print(f'  期望 y = {expected_y:.6f}')
        print(f'  实际 y = {actual_y:.6f}')
        print(f'  误差 = {error:.6f}')
        
        if error < 0.001:
            print(f'  ✓ 约束公式正确')
        else:
            print(f'  ✗ 约束公式不匹配')

def test_dynamic_constraints():
    """测试动态仿真中的约束效果"""
    model = mujoco.MjModel.from_xml_path('assets/monte_01/urdf/robot_description.xml')
    data = mujoco.MjData(model)
    
    print(f'\n=== 动态约束测试 ===')
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 获取参考位置
    drive_ref = data.qpos[12]
    finger_ref = data.qpos[14]
    
    # 设置控制目标
    target = 1.0
    actuator_id = 12  # left_drive_gear_joint_ctrl
    data.ctrl[actuator_id] = target
    
    print(f'设置驱动关节控制目标: {target}')
    print(f'期望最终从动位置: {finger_ref + 0.1259 * (target - drive_ref):.6f}')
    
    # 运行仿真
    for i in range(1000):
        mujoco.mj_step(model, data)
        
        if i % 200 == 0:
            x = data.qpos[12]
            y = data.qpos[14]
            expected_y = finger_ref + 0.1259 * (x - drive_ref)
            error = abs(y - expected_y)
            
            print(f'步骤 {i}: x={x:.4f}, y={y:.4f}, 期望={expected_y:.4f}, 误差={error:.4f}')

if __name__ == "__main__":
    test_constraint_formula()
    test_dynamic_constraints()