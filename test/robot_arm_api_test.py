#!/usr/bin/env python3
"""
7-DOF机械臂API转换关系测试脚本
假设：两个API都使用弧度制，关节顺序相同
主要检测：零点偏移、方向反转、坐标系差异
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import List, Tuple, Dict, Optional
import time
import json
from dataclasses import dataclass
from enum import Enum

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class TransformType(Enum):
    """转换类型枚举"""
    OFFSET = "Fixed Offset"
    SIGN_FLIP = "Sign Flip" 
    OFFSET_AND_FLIP = "Offset+Flip"
    RANGE_SHIFT = "Range Shift"
    COMPLEX = "Complex Relation"

@dataclass
class JointTransform:
    """单个关节的转换关系"""
    joint_id: int
    transform_type: TransformType
    scale: float
    offset: float
    r_squared: float
    formula: str
    
class RobotArmAPIAnalyzer:
    def __init__(self, api_a, api_b, num_joints=7):
        """
        初始化分析器
        
        Args:
            api_a: API A的接口对象
            api_b: API B的接口对象  
            num_joints: 关节数量，默认7
        """
        self.api_a = api_a
        self.api_b = api_b
        self.num_joints = num_joints
        self.test_data = []
        self.joint_transforms = []
        
    def collect_movement_data(self, movement_commands: Optional[List] = None):
        """
        收集运动过程中的数据，确保覆盖各关节的运动范围
        
        Args:
            movement_commands: 可选的运动指令列表
        """
        print("开始收集运动数据...")
        
        if movement_commands is None:
            # 生成默认的测试运动序列
            movement_commands = self._generate_test_movements()
        
        for idx, cmd in enumerate(movement_commands):
            try:
                # 发送运动指令（如果API支持）
                if hasattr(self.api_a, 'move_to'):
                    self.api_a.move_to(cmd)
                    time.sleep(0.5)  # 等待运动完成
                
                # 读取当前位置
                angles_a = self.api_a.get_joint_angles()
                angles_b = self.api_b.get_joint_angles()
                
                self.test_data.append({
                    'command': cmd,
                    'angles_a': np.array(angles_a),
                    'angles_b': np.array(angles_b),
                    'timestamp': time.time()
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"已收集 {idx + 1}/{len(movement_commands)} 组数据")
                    
            except Exception as e:
                print(f"数据收集失败 {idx}: {e}")
                
        print(f"数据收集完成，共{len(self.test_data)}组")
    
    def collect_movement_data_teach_mode(self):
        """
        使用teach mode收集数据 - 同时读取两套API
        """
        print("\n🤖 启动 Teach Mode 数据收集")
        print(f"API A: {self.api_a.get_robot_info()['type']}")
        print(f"API B: {self.api_b.get_robot_info()['type']}")
        
        # 检查硬件连接
        try:
            angles_a = self.api_a.get_joint_angles()
            angles_b = self.api_b.get_joint_angles()
            print(f"✅ 双API连接正常")
            print(f"API A 当前角度: {[f'{a:.3f}' for a in angles_a]}")
            print(f"API B 当前角度: {[f'{a:.3f}' for a in angles_b]}")
        except Exception as e:
            print(f"❌ API连接失败: {e}")
            return
        
        # 启用teach mode（如果支持）
        self._enable_teach_mode_for_both()
        
        print(f"\n📋 数据收集指南:")
        print(f"1. 手动移动机械臂到不同位置")
        print(f"2. 按 [ENTER] 同时记录两套API的当前位置")
        print(f"3. 按 [q] + [ENTER] 结束收集")
        print(f"4. 尽量覆盖各关节的运动范围")
        
        target_samples = 50
        sample_count = 0
        
        while sample_count < target_samples:
            try:
                # 同时读取两套API的当前状态
                angles_a = self.api_a.get_joint_angles()
                angles_b = self.api_b.get_joint_angles()
                
                print(f"\n[{sample_count+1}/{target_samples}]")
                print(f"API A: {[f'{a:.3f}' for a in angles_a]}")
                print(f"API B: {[f'{a:.3f}' for a in angles_b]}")
                
                # 等待用户输入
                user_input = input("按 [ENTER] 记录位置, [q] 退出: ").strip().lower()
                
                if user_input == 'q':
                    print("用户主动结束数据收集")
                    break
                elif user_input == '':
                    # 同时记录两套API的位置
                    self.test_data.append({
                        'command': None,  # teach mode下无指令
                        'angles_a': np.array(angles_a),
                        'angles_b': np.array(angles_b),
                        'timestamp': time.time()
                    })
                    sample_count += 1
                    print(f"✅ 已记录样本 {sample_count} (双API同步)")
                else:
                    print("无效输入，请按 [ENTER] 记录或 [q] 退出")
                    
            except KeyboardInterrupt:
                print("\n用户中断，结束数据收集")
                break
            except Exception as e:
                print(f"❌ 读取角度失败: {e}")
                break
        
        print(f"\n📊 Teach Mode 数据收集完成!")
        print(f"总样本数: {len(self.test_data)}")
    
    def _enable_teach_mode_for_both(self):
        """为两套API启用teach mode（如果支持）"""
        try:
            # API A teach mode
            api_a_info = self.api_a.get_robot_info()
            if api_a_info['type'] == 'xarm7':
                if hasattr(self.api_a, 'set_teaching_mode'):
                    success = self.api_a.set_teaching_mode()
                    if success:
                        print("✅ API A (xArm7) Teach Mode 已启用")
                    else:
                        print("⚠️  API A (xArm7) Teach Mode 启用失败")
            else:
                print(f"ℹ️  API A ({api_a_info['type']}) 不支持自动teach mode")
            
            # API B teach mode
            api_b_info = self.api_b.get_robot_info()
            if api_b_info['type'] == 'xarm7':
                if hasattr(self.api_b, 'set_teaching_mode'):
                    success = self.api_b.set_teaching_mode()
                    if success:
                        print("✅ API B (xArm7) Teach Mode 已启用")
                    else:
                        print("⚠️  API B (xArm7) Teach Mode 启用失败")
            else:
                print(f"ℹ️  API B ({api_b_info['type']}) 不支持自动teach mode")
                
        except Exception as e:
            print(f"⚠️  Teach Mode设置失败: {e}")
    
    def collect_movement_data_static(self):
        """
        静态数据收集（读取当前位置）
        """
        print("使用静态位置读取模式")
        
        for i in range(20):  # 减少样本数
            try:
                angles_a = self.api_a.get_joint_angles()
                angles_b = self.api_b.get_joint_angles()
                
                self.test_data.append({
                    'command': None,
                    'angles_a': np.array(angles_a),
                    'angles_b': np.array(angles_b),
                    'timestamp': time.time()
                })
                
                print(f"已收集 {i+1}/20 组静态数据")
                time.sleep(1)  # 等待1秒
                
            except Exception as e:
                print(f"数据收集失败 {i}: {e}")
        
        print(f"静态数据收集完成，共{len(self.test_data)}组")
        
    def _generate_test_movements(self):
        """生成测试运动序列，覆盖各关节的运动范围"""
        movements = []
        
        # 1. 各关节单独运动
        for joint in range(self.num_joints):
            for angle in np.linspace(-np.pi, np.pi, 5):
                cmd = np.zeros(self.num_joints)
                cmd[joint] = angle
                movements.append(cmd)
        
        # 2. 随机组合位置
        for _ in range(20):
            cmd = np.random.uniform(-np.pi, np.pi, self.num_joints)
            movements.append(cmd)
            
        return movements
    
    def analyze_joint_relationship(self, joint_idx: int) -> JointTransform:
        """
        分析单个关节的转换关系
        """
        a_values = np.array([d['angles_a'][joint_idx] for d in self.test_data])
        b_values = np.array([d['angles_b'][joint_idx] for d in self.test_data])
        
        # 1. 检查是否为简单偏移
        offset = np.mean(b_values - a_values)
        offset_std = np.std(b_values - a_values)
        
        if offset_std < 0.001:  # 几乎恒定的偏移
            return JointTransform(
                joint_id=joint_idx,
                transform_type=TransformType.OFFSET,
                scale=1.0,
                offset=offset,
                r_squared=1.0,
                formula=f"b = a + {offset:.4f}"
            )
        
        # 2. 检查是否为符号反转
        if np.corrcoef(a_values, b_values)[0, 1] < -0.99:
            # 强负相关，可能是反转
            slope, intercept, r_value, _, _ = stats.linregress(a_values, b_values)
            
            if abs(slope + 1) < 0.01:  # 斜率接近-1
                return JointTransform(
                    joint_id=joint_idx,
                    transform_type=TransformType.SIGN_FLIP,
                    scale=-1.0,
                    offset=intercept,
                    r_squared=r_value**2,
                    formula=f"b = -a + {intercept:.4f}"
                )
        
        # 3. 检查是否为范围偏移（如 [0, 2π] vs [-π, π]）
        if (np.min(a_values) >= 0 and np.max(b_values) <= 0) or \
           (np.min(b_values) >= 0 and np.max(a_values) <= 0):
            # 可能存在2π的偏移
            for shift in [2*np.pi, -2*np.pi, np.pi, -np.pi]:
                shifted = a_values + shift
                if np.allclose(b_values, shifted, rtol=0.01):
                    return JointTransform(
                        joint_id=joint_idx,
                        transform_type=TransformType.RANGE_SHIFT,
                        scale=1.0,
                        offset=shift,
                        r_squared=1.0,
                        formula=f"b = a + {shift/np.pi:.2f}π"
                    )
        
        # 4. 一般线性关系
        slope, intercept, r_value, _, _ = stats.linregress(a_values, b_values)
        
        if r_value**2 > 0.95:  # 良好的线性关系
            transform_type = TransformType.OFFSET_AND_FLIP if abs(slope + 1) < 0.5 else TransformType.OFFSET
            return JointTransform(
                joint_id=joint_idx,
                transform_type=transform_type,
                scale=slope,
                offset=intercept,
                r_squared=r_value**2,
                formula=f"b = {slope:.4f}*a + {intercept:.4f}"
            )
        
        # 5. 复杂关系
        return JointTransform(
            joint_id=joint_idx,
            transform_type=TransformType.COMPLEX,
            scale=slope,
            offset=intercept,
            r_squared=r_value**2,
            formula="需要更复杂的模型"
        )
    
    def analyze_all_joints(self):
        """分析所有关节的转换关系"""
        print("\n=== 关节转换关系分析 ===")
        print("="*60)
        
        self.joint_transforms = []
        
        for joint in range(self.num_joints):
            transform = self.analyze_joint_relationship(joint)
            self.joint_transforms.append(transform)
            
            print(f"\n关节 {joint + 1}:")
            print(f"  转换类型: {transform.transform_type.value}")
            print(f"  转换公式: {transform.formula}")
            print(f"  拟合度 R²: {transform.r_squared:.6f}")
            
            # 额外的诊断信息
            a_values = np.array([d['angles_a'][joint] for d in self.test_data])
            b_values = np.array([d['angles_b'][joint] for d in self.test_data])
            
            print(f"  API_A 范围: [{np.min(a_values):.3f}, {np.max(a_values):.3f}]")
            print(f"  API_B 范围: [{np.min(b_values):.3f}, {np.max(b_values):.3f}]")
            
    def check_coupling_effects(self):
        """检查关节之间是否存在耦合效应"""
        print("\n=== 耦合效应检查 ===")
        print("="*60)
        
        coupling_found = False
        
        for target_joint in range(self.num_joints):
            b_values = np.array([d['angles_b'][target_joint] for d in self.test_data])
            
            # 用所有关节的API_A值来预测这个关节的API_B值
            A_matrix = np.column_stack([
                [d['angles_a'][j] for d in self.test_data]
                for j in range(self.num_joints)
            ])
            
            # 添加常数项
            A_matrix = np.column_stack([A_matrix, np.ones(len(self.test_data))])
            
            # 最小二乘拟合
            coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, b_values, rcond=None)
            
            # 计算R²
            ss_res = residuals[0] if len(residuals) > 0 else 0
            ss_tot = np.sum((b_values - np.mean(b_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # 检查是否有显著的交叉影响
            significant_couplings = []
            for j in range(self.num_joints):
                if j != target_joint and abs(coeffs[j]) > 0.1:
                    significant_couplings.append((j+1, coeffs[j]))
            
            if significant_couplings:
                coupling_found = True
                print(f"\n关节 {target_joint + 1} 存在耦合:")
                for source_joint, coeff in significant_couplings:
                    print(f"  来自关节 {source_joint} 的影响系数: {coeff:.4f}")
                print(f"  多元回归 R² = {r_squared:.6f}")
        
        if not coupling_found:
            print("未检测到显著的关节耦合效应")
            
    def visualize_analysis(self):
        """可视化分析结果"""
        # 创建2x4的子图布局（7个关节 + 1个汇总）
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        colors = {
            TransformType.OFFSET: 'blue',
            TransformType.SIGN_FLIP: 'red',
            TransformType.OFFSET_AND_FLIP: 'orange',
            TransformType.RANGE_SHIFT: 'green',
            TransformType.COMPLEX: 'purple'
        }
        
        for joint in range(self.num_joints):
            ax = axes[joint]
            transform = self.joint_transforms[joint]
            
            a_values = [d['angles_a'][joint] for d in self.test_data]
            b_values = [d['angles_b'][joint] for d in self.test_data]
            
            # 散点图
            color = colors.get(transform.transform_type, 'gray')
            ax.scatter(a_values, b_values, alpha=0.6, s=20, c=color)
            
            # 拟合线
            a_range = np.linspace(min(a_values), max(a_values), 100)
            b_fit = transform.scale * a_range + transform.offset
            ax.plot(a_range, b_fit, 'k-', linewidth=2, alpha=0.8)
            
            # 1:1参考线
            ax.plot(a_range, a_range, 'gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('API_A (rad)')
            ax.set_ylabel('API_B (rad)')
            ax.set_title(f'Joint {joint+1}: {transform.transform_type.value}')
            ax.grid(True, alpha=0.3)
            
            # 添加公式
            ax.text(0.05, 0.95, transform.formula, 
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 汇总图
        ax = axes[7]
        ax.axis('off')
        summary_text = "Transform Summary\n" + "="*20 + "\n"
        for i, t in enumerate(self.joint_transforms):
            summary_text += f"J{i+1}: {t.transform_type.value}\n"
            summary_text += f"     R² = {t.r_squared:.4f}\n"
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', family='monospace')
        
        plt.suptitle('7-DOF Robot Arm API Transform Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('joint_transform_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def generate_conversion_code(self):
        """生成优化的转换代码"""
        print("\n=== 生成转换代码 ===")
        print("="*60)
        
        code = '''import numpy as np

class JointAngleConverter:
    """7-DOF机械臂关节角度转换器"""
    
    def __init__(self):
        # 转换参数（基于分析结果）
        self.transforms = {
'''
        
        for t in self.joint_transforms:
            code += f'            {t.joint_id}: {{"scale": {t.scale:.6f}, "offset": {t.offset:.6f}, "type": "{t.transform_type.value}"}},\n'
        
        code += '''        }
    
    def api_a_to_b(self, angles_a):
        """将API_A的角度转换为API_B的角度
        
        Args:
            angles_a: API_A的7个关节角度（弧度）
            
        Returns:
            angles_b: API_B的7个关节角度（弧度）
        """
        angles_b = []
        for i, angle in enumerate(angles_a):
            t = self.transforms[i]
            angle_b = t["scale"] * angle + t["offset"]
            angles_b.append(angle_b)
        return angles_b
    
    def api_b_to_a(self, angles_b):
        """将API_B的角度转换为API_A的角度
        
        Args:
            angles_b: API_B的7个关节角度（弧度）
            
        Returns:
            angles_a: API_A的7个关节角度（弧度）
        """
        angles_a = []
        for i, angle in enumerate(angles_b):
            t = self.transforms[i]
            if abs(t["scale"]) > 0.001:  # 避免除零
                angle_a = (angle - t["offset"]) / t["scale"]
            else:
                angle_a = angle  # 无法转换时保持原值
            angles_a.append(angle_a)
        return angles_a
    
    def validate_conversion(self, test_data):
        """验证转换精度"""
        errors_a2b = []
        errors_b2a = []
        
        for data in test_data:
            # A to B
            predicted_b = self.api_a_to_b(data["angles_a"])
            error_a2b = np.mean(np.abs(np.array(predicted_b) - np.array(data["angles_b"])))
            errors_a2b.append(error_a2b)
            
            # B to A
            predicted_a = self.api_b_to_a(data["angles_b"])
            error_b2a = np.mean(np.abs(np.array(predicted_a) - np.array(data["angles_a"])))
            errors_b2a.append(error_b2a)
        
        print(f"A→B 平均误差: {np.mean(errors_a2b):.6f} rad")
        print(f"B→A 平均误差: {np.mean(errors_b2a):.6f} rad")
        
        return np.mean(errors_a2b), np.mean(errors_b2a)

# 使用示例
if __name__ == "__main__":
    converter = JointAngleConverter()
    
    # 测试转换
    test_angles_a = [0.5, -0.3, 1.2, -0.8, 0.0, 1.5, -0.5]
    angles_b = converter.api_a_to_b(test_angles_a)
    print(f"API_A: {test_angles_a}")
    print(f"API_B: {angles_b}")
    
    # 反向转换
    angles_a_recovered = converter.api_b_to_a(angles_b)
    print(f"恢复的API_A: {angles_a_recovered}")
'''
        
        print(code)
        
        # 保存到文件
        with open('joint_converter.py', 'w') as f:
            f.write(code)
        
        print("\n转换代码已保存到 joint_converter.py")
        
        return code
    
    def save_analysis_report(self, filename='analysis_report.json'):
        """保存详细的分析报告"""
        report = {
            'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': len(self.test_data),
            'num_joints': self.num_joints,
            'transforms': [
                {
                    'joint': t.joint_id + 1,
                    'type': t.transform_type.value,
                    'scale': t.scale,
                    'offset': t.offset,
                    'r_squared': t.r_squared,
                    'formula': t.formula
                }
                for t in self.joint_transforms
            ],
            'statistics': [
                {
                    'joint': joint + 1,
                    'api_a_range': [
                        float(np.min([d['angles_a'][joint] for d in self.test_data])),
                        float(np.max([d['angles_a'][joint] for d in self.test_data]))
                    ],
                    'api_b_range': [
                        float(np.min([d['angles_b'][joint] for d in self.test_data])),
                        float(np.max([d['angles_b'][joint] for d in self.test_data]))
                    ]
                } for joint in range(self.num_joints)
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n分析报告已保存到 {filename}")


# ============ 测试用的模拟API ============

class MockAPIWithOffset:
    """模拟具有零点偏移的API"""
    def __init__(self, base_api=None):
        self.base_api = base_api
        # 每个关节的零点偏移（弧度）
        self.zero_offsets = [0, np.pi/4, -np.pi/6, np.pi/2, 0, -np.pi/3, np.pi/8]
        # 某些关节方向相反
        self.sign_flips = [1, -1, 1, 1, -1, 1, 1]
        
    def get_joint_angles(self):
        if self.base_api:
            base_angles = self.base_api.get_joint_angles()
        else:
            base_angles = np.random.uniform(-np.pi, np.pi, 7)
            
        # 应用变换
        transformed = []
        for i, angle in enumerate(base_angles):
            t_angle = self.sign_flips[i] * angle + self.zero_offsets[i]
            # 确保在[-π, π]范围内
            t_angle = np.arctan2(np.sin(t_angle), np.cos(t_angle))
            transformed.append(t_angle)
        
        return transformed


def main():
    """主测试流程 - 支持真实API"""
    
    print("="*60)
    print("7-DOF 机械臂 API 转换关系分析 (真实API版)")
    print("="*60)
    
    # 导入新的适配器
    from robot_test_factory import RobotTestFactory
    from robot_api_adapters import TeachModeGuide
    
    # 配置API类型（根据实际情况修改）
    print("🔧 配置机器人API类型:")
    print("支持的类型: xarm7_left, xarm7_right, monte01_left, monte01_right, mock")
    
    api_a_type = input("输入 API A 类型 (默认 mock): ").strip() or "mock"
    api_b_type = input("输入 API B 类型 (默认 mock): ").strip() or "mock"
    
    print(f"\n创建适配器: {api_a_type} -> {api_b_type}")
    
    # 创建适配器
    api_a = RobotTestFactory.create_adapter(api_a_type)
    api_b = RobotTestFactory.create_adapter(api_b_type)
    
    print(f"API A: {api_a.get_robot_info()}")
    print(f"API B: {api_b.get_robot_info()}")
    
    # 创建分析器
    analyzer = RobotArmAPIAnalyzer(api_a, api_b, num_joints=7)
    
    # 选择数据收集模式
    print("\n📊 数据收集模式选择:")
    print("1. Teach Mode (手动移动收集)")
    print("2. 静态读取模式")
    print("3. Mock模拟模式")
    
    mode = input("请选择模式 (1/2/3, 默认3): ").strip() or "3"
    
    if mode == "1":
        # Teach Mode数据收集
        analyzer.collect_movement_data_teach_mode()
    elif mode == "2":
        # 静态读取模式
        analyzer.collect_movement_data_static()
    else:
        # Mock模拟模式
        analyzer.collect_movement_data()
    
    if len(analyzer.test_data) == 0:
        print("❌ 未收集到数据，程序退出")
        return
    
    # 分析关节关系
    analyzer.analyze_all_joints()
    
    # 检查耦合效应
    analyzer.check_coupling_effects()
    
    # 可视化结果
    analyzer.visualize_analysis()
    
    # 生成转换代码
    analyzer.generate_conversion_code()
    
    # 保存报告
    analyzer.save_analysis_report()
    
    # 关闭连接
    api_a.close()
    api_b.close()
    
    print("\n" + "="*60)
    print("分析完成！")
    print("生成的文件：")
    print("  - joint_transform_analysis.png (可视化图表)")
    print("  - joint_converter.py (转换代码)")
    print("  - analysis_report.json (详细报告)")
    print("="*60)


if __name__ == "__main__":
    main()
