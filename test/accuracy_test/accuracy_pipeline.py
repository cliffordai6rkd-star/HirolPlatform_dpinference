import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import List, Dict, Tuple
import logging
import time
import os
from fr3_interface import FR3Interface

class RobotPoseAccuracyTester:
    """机器人位姿精度测试系统"""
    
    def __init__(self, robot_interface=None):
        """
        初始化测试系统
        
        Args:
            robot_interface: 机器人接口实例，如果为None则创建新实例
        """
        if robot_interface is None:
            self.robot = FR3Interface()
        else:
            self.robot = robot_interface
        self.test_data = []
        self.statistics = {}
        self.reference_poses = []  # 初始化reference_poses
        
        # 配置日志
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def phase1_collect_reference_poses(self, k: int = 10, save_path: str = "reference_poses.json") -> List[Dict]:
        """
        阶段1：采集k个参考位置姿态
        
        Args:
            k: 要采集的位姿数量
            save_path: 保存参考位姿的JSON文件路径
            
        Returns:
            参考位姿列表
        """
        # 检查是否存在已保存的参考位姿
        if os.path.exists(save_path):
            self.logger.info(f"发现已保存的参考位姿文件: {save_path}")
            user_choice = input("是否加载已保存的参考位姿？(y/n): ").strip().lower()
            
            if user_choice == 'y':
                try:
                    reference_poses = self._load_reference_poses(save_path)
                    self.reference_poses = reference_poses
                    self.logger.info(f"成功加载{len(reference_poses)}个参考位姿")
                    return reference_poses
                except Exception as e:
                    self.logger.error(f"加载参考位姿失败: {e}")
                    self.logger.info("将重新采集参考位姿")
        
        self.logger.info(f"阶段1：开始采集{k}个参考位姿")
        reference_poses = []
        self.robot.move_to_start()
        
        for i in range(k):
            self.logger.info(f"采集第{i+1}个参考位姿")
            
            # 手动示教或自动生成测试位姿
                # 进入示教模式
            self.robot.enter_teach_mode()
            input(f"请手动移动机器人到第{i+1}个参考位置，然后按Enter继续...")
            self.robot.exit_teach_mode()
            
            # 记录当前位姿
            current_pose = self.robot.get_current_pose()

            
            pose_data = {
                'id': f'ref_{i+1}',
                'commanded_pose': current_pose,
                'timestamp': datetime.now().isoformat()
            }
            
            reference_poses.append(pose_data)
            self.logger.info(f"参考位姿{i+1}采集完成")
        
        self.reference_poses = reference_poses
        
        # 自动保存参考位姿
        try:
            self._save_reference_poses(reference_poses, save_path)
            self.logger.info(f"参考位姿已保存到: {save_path}")
        except Exception as e:
            self.logger.error(f"保存参考位姿失败: {e}")
        
        self.logger.info(f"阶段1完成：共采集{len(reference_poses)}个参考位姿")
        return reference_poses
    
    def phase2_absolute_accuracy_test(self, target_poses: List[Dict] = None) -> Dict:
        """
        阶段2：绝对位姿精度测试
        
        Args:
            target_poses: 目标位姿列表，如果为None则从参考位姿中选择10个
            
        Returns:
            绝对精度测试结果
        """
        self.logger.info("阶段2：开始绝对位姿精度测试")
        
        if target_poses is None:
            # 从参考位姿中选择10个作为测试目标
            if len(self.reference_poses) >= 20:
                target_poses = self.reference_poses[:20]
            else:
                raise ValueError("参考位姿数量不足10个")
        
        absolute_test_data = []
        
        stiffness = {"translational": 1500.0, "rotational": 250.0}
        damping = {"translational": 89.0, "rotational": 9.0}
        
        for target_idx, target_pose in enumerate(target_poses):
            self.logger.info(f"测试目标位姿 {target_idx+1}/20")
            
            # 移动到起始位置
            self.robot.move_to_start()
            
            # 移动到目标位置
            target_commanded = target_pose['commanded_pose']
            # self.robot.move_to_pose_traj(target_commanded,3)
            self.robot.move_cartesian_impedance(
                target_commanded, 
                finish_time=None,
                stiffness=stiffness,
                damping=damping
            )
            
            # 测量实际到达位置
            actual_pose = self.robot.get_current_pose()
            
            test_record = {
                'test_type': 'absolute_accuracy',
                'target_id': target_pose['id'],
                'commanded_pose': target_commanded,
                'measured_pose': actual_pose,
                'timestamp': datetime.now().isoformat()
            }
            
            absolute_test_data.append(test_record)
        
        # 计算绝对精度统计
        abs_stats = self._calculate_absolute_accuracy_stats(absolute_test_data)
        
        self.absolute_test_data = absolute_test_data
        self.absolute_accuracy_stats = abs_stats
        
        self.logger.info("阶段2完成：绝对位姿精度测试")
        return abs_stats
    
    def phase3_repeatability_test(self, target_pose_id: str = None, n: int = 30) -> Dict:
        """
        阶段3：重复位姿精度测试
        
        Args:
            target_pose_id: 目标位姿ID，如果为None则选择第一个参考位姿
            n: 重复测试次数
            
        Returns:
            重复精度测试结果
        """
        self.logger.info(f"阶段3：开始重复位姿精度测试，重复{n}次")
        
        # 选择测试目标
        if target_pose_id is None:
            target_pose = self.reference_poses[0]
        else:
            target_pose = next((p for p in self.reference_poses if p['id'] == target_pose_id), 
                             self.reference_poses[0])
        
        
        repeatability_data = []
        
        for i in range(n):
            self.logger.info(f"重复测试 {i+1}/{n}")
            
            # 返回起始位置
            self.robot.move_to_start()
            
            # 短暂停顿以消除动态影响
            time.sleep(1)
            
            # 移动到目标位置
            # self.robot.move_to_pose_traj(target_pose['commanded_pose'],3)
            stiffness = {"translational": 1500.0, "rotational": 250.0}
            damping = {"translational": 89.0, "rotational": 9.0}
            self.robot.move_cartesian_impedance(
                target_pose['commanded_pose'], 
                finish_time=None,
                stiffness=stiffness,
                damping=damping
            )
            
            # 测量实际位置
            actual_pose = self.robot.get_current_pose()
            
            test_record = {
                'test_type': 'repeatability',
                'target_id': target_pose['id'],
                'repetition': i + 1,
                'commanded_pose': target_pose['commanded_pose'],
                'measured_pose': actual_pose,
                'timestamp': datetime.now().isoformat()
            }
            
            repeatability_data.append(test_record)
        
        # 计算重复精度统计
        repeat_stats = self._calculate_repeatability_stats(repeatability_data)
        
        self.repeatability_data = repeatability_data
        self.repeatability_stats = repeat_stats
        
        self.logger.info("阶段3完成：重复位姿精度测试")
        return repeat_stats
    
    # def _generate_random_pose(self) -> Dict:
    #     """生成工作空间内的随机位姿"""
    #     # 这里需要根据具体机器人的工作空间定义
    #     x = np.random.uniform(-500, 500)  # mm
    #     y = np.random.uniform(-500, 500)  # mm
    #     z = np.random.uniform(100, 800)   # mm
    #     rx = np.random.uniform(-180, 180)  # degrees
    #     ry = np.random.uniform(-180, 180)  # degrees
    #     rz = np.random.uniform(-180, 180)  # degrees
        
    #     return {
    #         'x': x, 'y': y, 'z': z,
    #         'rx': rx, 'ry': ry, 'rz': rz
    #     }
    
    # def _generate_start_positions(self, num_starts: int = 6) -> List[Dict]:
    #     """生成多个起始位置"""
    #     start_positions = []
    #     for i in range(num_starts):
    #         pos = self._generate_random_pose()
    #         start_positions.append(pos)
    #     return start_positions
    
    # def _get_home_position(self) -> Dict:
    #     """获取机器人的原点位置"""
    #     return {
    #         'x': 0, 'y': 0, 'z': 300,
    #         'rx': 0, 'ry': 0, 'rz': 0
    #     }
    
    def _quaternion_to_euler(self, qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
        """将四元数转换为欧拉角（弧度）"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        rx = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if np.abs(sinp) >= 1:
            ry = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            ry = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        rz = np.arctan2(siny_cosp, cosy_cosp)
        
        return rx, ry, rz
    
    def _save_reference_poses(self, poses: List[Dict], save_path: str) -> None:
        """保存参考位姿到JSON文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(poses, f, indent=2, ensure_ascii=False)
    
    def _load_reference_poses(self, load_path: str) -> List[Dict]:
        """从JSON文件加载参考位姿"""
        with open(load_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _convert_pose_format(self, pose: Dict[str, float]) -> Dict[str, float]:
        """统一位姿格式，确保包含欧拉角表示"""
        # 复制原始数据
        converted = pose.copy()
        
        # 检查是否包含四元数
        if 'qx' in pose and 'qy' in pose and 'qz' in pose and 'qw' in pose:
            # 转换四元数到欧拉角
            rx, ry, rz = self._quaternion_to_euler(
                pose['qx'], pose['qy'], pose['qz'], pose['qw']
            )
            converted['rx'] = rx
            converted['ry'] = ry
            converted['rz'] = rz
        elif 'rx' not in pose or 'ry' not in pose or 'rz' not in pose:
            raise ValueError(f"无法识别的位姿格式: {list(pose.keys())}")
            
        return converted
    
    def _calculate_pose_error(self, commanded: Dict, measured: Dict) -> Dict:
        """计算位姿误差"""
        # 转换位姿格式
        commanded = self._convert_pose_format(commanded)
        measured = self._convert_pose_format(measured)
        
        pos_error = np.sqrt(
            (commanded['x'] - measured['x'])**2 +
            (commanded['y'] - measured['y'])**2 +
            (commanded['z'] - measured['z'])**2
        ) * 1000  # Convert m to mm
        
        orientation_error = np.sqrt(
            (commanded['rx'] - measured['rx'])**2 +
            (commanded['ry'] - measured['ry'])**2 +
            (commanded['rz'] - measured['rz'])**2
        )
        
        return {
            'position_error': pos_error,
            'orientation_error': orientation_error,
            'x_error': (commanded['x'] - measured['x']) * 1000,  # m to mm
            'y_error': (commanded['y'] - measured['y']) * 1000,  # m to mm
            'z_error': (commanded['z'] - measured['z']) * 1000,  # m to mm,
            'rx_error': commanded['rx'] - measured['rx'],
            'ry_error': commanded['ry'] - measured['ry'],
            'rz_error': commanded['rz'] - measured['rz']
        }
    
    def _calculate_absolute_accuracy_stats(self, test_data: List[Dict]) -> Dict:
        """计算绝对精度统计"""
        position_errors = []
        orientation_errors = []
        
        for record in test_data:
            error = self._calculate_pose_error(
                record['commanded_pose'], 
                record['measured_pose']
            )
            position_errors.append(error['position_error'])
            orientation_errors.append(error['orientation_error'])
        
        stats = {
            'position_accuracy': {
                'mean_error': np.mean(position_errors),
                'std_error': np.std(position_errors),
                'max_error': np.max(position_errors),
                'rms_error': np.sqrt(np.mean(np.array(position_errors)**2))
            },
            'orientation_accuracy': {
                'mean_error': np.mean(orientation_errors),
                'std_error': np.std(orientation_errors),
                'max_error': np.max(orientation_errors),
                'rms_error': np.sqrt(np.mean(np.array(orientation_errors)**2))
            },
            'test_count': len(test_data)
        }
        
        return stats
    
    def _calculate_repeatability_stats(self, test_data: List[Dict]) -> Dict:
        """计算重复精度统计"""
        # 首先转换所有测量数据的格式
        converted_data = []
        for record in test_data:
            converted_record = record.copy()
            converted_record['measured_pose'] = self._convert_pose_format(record['measured_pose'])
            converted_data.append(converted_record)
        
        # 计算所有测量点的平均位置
        x_values = [record['measured_pose']['x'] for record in converted_data]
        y_values = [record['measured_pose']['y'] for record in converted_data]
        z_values = [record['measured_pose']['z'] for record in converted_data]
        rx_values = [record['measured_pose']['rx'] for record in converted_data]
        ry_values = [record['measured_pose']['ry'] for record in converted_data]
        rz_values = [record['measured_pose']['rz'] for record in converted_data]
        
        mean_pos = {
            'x': np.mean(x_values),
            'y': np.mean(y_values),
            'z': np.mean(z_values),
            'rx': np.mean(rx_values),
            'ry': np.mean(ry_values),
            'rz': np.mean(rz_values)
        }
        
        # 计算每次测量相对于平均位置的偏差
        position_deviations = []
        orientation_deviations = []
        
        for record in converted_data:
            measured = record['measured_pose']
            pos_dev = np.sqrt(
                (measured['x'] - mean_pos['x'])**2 +
                (measured['y'] - mean_pos['y'])**2 +
                (measured['z'] - mean_pos['z'])**2
            )
            
            orient_dev = np.sqrt(
                (measured['rx'] - mean_pos['rx'])**2 +
                (measured['ry'] - mean_pos['ry'])**2 +
                (measured['rz'] - mean_pos['rz'])**2
            )
            
            position_deviations.append(pos_dev)
            orientation_deviations.append(orient_dev)
        
        stats = {
            'position_repeatability': {
                'mean_deviation': np.mean(position_deviations),
                'std_deviation': np.std(position_deviations),
                'repeatability_3sigma': 3 * np.std(position_deviations),
                'max_deviation': np.max(position_deviations),
                'mean_position': mean_pos
            },
            'orientation_repeatability': {
                'mean_deviation': np.mean(orientation_deviations),
                'std_deviation': np.std(orientation_deviations),
                'repeatability_3sigma': 3 * np.std(orientation_deviations),
                'max_deviation': np.max(orientation_deviations)
            },
            'test_count': len(test_data)
        }
        
        return stats
    
    def generate_report(self, save_path: str = None) -> Dict:
        """生成完整的测试报告"""
        report = {
            'test_info': {
                'test_date': datetime.now().isoformat(),
                'reference_poses_count': len(getattr(self, 'reference_poses', [])),
                'absolute_tests_count': len(getattr(self, 'absolute_test_data', [])),
                'repeatability_tests_count': len(getattr(self, 'repeatability_data', []))
            },
            'absolute_accuracy': getattr(self, 'absolute_accuracy_stats', {}),
            'repeatability': getattr(self, 'repeatability_stats', {}),
            'raw_data': {
                'reference_poses': getattr(self, 'reference_poses', []),
                'absolute_test_data': getattr(self, 'absolute_test_data', []),
                'repeatability_data': getattr(self, 'repeatability_data', [])
            }
        }
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"测试报告已保存到: {save_path}")
        
        return report
    
    def plot_results(self, save_figures: bool = True, figure_prefix: str = "accuracy_test"):
        """
        绘制测试结果图表
        
        Args:
            save_figures: 是否保存图片
            figure_prefix: 图片文件名前缀
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绝对精度散点图
        if hasattr(self, 'absolute_test_data'):
            pos_errors = []
            for record in self.absolute_test_data:
                error = self._calculate_pose_error(
                    record['commanded_pose'], 
                    record['measured_pose']
                )
                pos_errors.append(error['position_error'])
            
            axes[0, 0].hist(pos_errors, bins=20, alpha=0.7)
            axes[0, 0].set_title('Absolute Position Accuracy Distribution')
            axes[0, 0].set_xlabel('Position Error (mm)')
            axes[0, 0].set_ylabel('Frequency')
        
        # 重复精度散点图
        if hasattr(self, 'repeatability_data'):
            x_vals = [r['measured_pose']['x'] * 1000 for r in self.repeatability_data]  # m to mm
            y_vals = [r['measured_pose']['y'] * 1000 for r in self.repeatability_data]  # m to mm
            
            axes[0, 1].scatter(x_vals, y_vals, alpha=0.6)
            axes[0, 1].set_title('Repeatability Scatter Plot (X-Y Plane)')
            axes[0, 1].set_xlabel('X Position (mm)')
            axes[0, 1].set_ylabel('Y Position (mm)')
            axes[0, 1].axis('equal')
        
        # 位置偏差趋势图
        if hasattr(self, 'repeatability_data'):
            deviations = []
            mean_pos = self.repeatability_stats['position_repeatability']['mean_position']
            
            for record in self.repeatability_data:
                measured = record['measured_pose']
                dev = np.sqrt(
                    (measured['x'] - mean_pos['x'])**2 +
                    (measured['y'] - mean_pos['y'])**2 +
                    (measured['z'] - mean_pos['z'])**2
                ) * 1000  # m to mm
                deviations.append(dev)
            
            axes[1, 0].plot(range(1, len(deviations)+1), deviations, 'o-')
            axes[1, 0].set_title('Repeatability Deviation Trend')
            axes[1, 0].set_xlabel('Test Count')
            axes[1, 0].set_ylabel('Position Deviation (mm)')
        
        # 精度对比柱状图
        if hasattr(self, 'absolute_accuracy_stats') and hasattr(self, 'repeatability_stats'):
            categories = ['Absolute Position\nAccuracy (RMS)', 'Repeatability\n(3σ)']
            values = [
                self.absolute_accuracy_stats['position_accuracy']['rms_error'],
                self.repeatability_stats['position_repeatability']['repeatability_3sigma']
            ]
            
            axes[1, 1].bar(categories, values, color=['blue', 'red'], alpha=0.7)
            axes[1, 1].set_title('Accuracy Metrics Comparison')
            axes[1, 1].set_ylabel('Error/Deviation (mm)')
        
        plt.tight_layout()
        
        if save_figures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{figure_prefix}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"测试结果图表已保存到: {filename}")
        
        # plt.show()

# 使用示例
def run_complete_test():
    """运行完整的测试流程"""
    
    # 初始化机器人接口
    robot_interface = FR3Interface()
    
    # 创建测试器实例
    tester = RobotPoseAccuracyTester(robot_interface)
    
    print("机器人位姿精度测试Pipeline")
    print("=" * 50)
    
    # 阶段1：采集参考位姿
    print("阶段1：采集参考位姿")
    reference_poses = tester.phase1_collect_reference_poses(k=20)
    
    # 阶段2：绝对精度测试
    print("阶段2：绝对精度测试")
    abs_stats = tester.phase2_absolute_accuracy_test()
    
    # 阶段3：重复精度测试
    print("阶段3：重复精度测试")
    repeat_stats = tester.phase3_repeatability_test(n=30)
    
    # 生成报告
    print("生成测试报告")
    report = tester.generate_report('robot_accuracy_test_report.json')
    
    # 绘制结果
    print("绘制测试结果")
    tester.plot_results()
    
    print("测试完成！")

if __name__ == "__main__":
    run_complete_test()