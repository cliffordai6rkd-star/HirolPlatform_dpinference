#!/usr/bin/env python3
"""
关节位置实时绘图器

该模块提供机器人关节和夹爪位置的实时可视化功能，
用于调试和分析推理过程中的运动轨迹。

作者: HIROLRobotPlatform
日期: 2025-09-02
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from typing import List, Dict, Any, Optional
import threading
import time
import glog as log
from pathlib import Path


class JointPositionPlotter:
    """关节位置实时绘图器"""
    
    def __init__(self, config: Dict[str, Any], joint_names: Optional[List[str]] = None):
        """
        初始化关节位置绘图器
        
        Args:
            config: 绘图配置字典
            joint_names: 关节名称列表，默认为 ['J1', 'J2', ..., 'J7', 'Gripper']
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            log.info("📊 关节位置可视化已禁用")
            return
            
        # 关节配置
        self.joint_names = joint_names or [f'Joint_{i+1}' for i in range(7)] + ['Gripper']
        self.n_joints = len(self.joint_names)
        
        # 绘图配置
        self.window_size = config.get('window_size', [1200, 800])
        self.update_frequency = config.get('update_frequency', 5)  # Hz
        self.history_length = config.get('history_length', 1000)
        self.save_data = config.get('save_data', True)
        self.save_interval = config.get('save_interval', 300)  # 每300秒保存一次
        
        # 数据存储
        self.timestamps = deque(maxlen=self.history_length)
        self.joint_positions = [deque(maxlen=self.history_length) for _ in range(self.n_joints)]
        self.data_lock = threading.Lock()
        
        # 绘图状态
        self.is_plotting = False
        self.plot_thread = None
        self.fig = None
        self.axes = None
        self.lines = []
        
        # 自动保存
        self.last_save_time = time.time()
        
        log.info(f"📊 关节位置绘图器初始化完成")
        log.info(f"   - 关节数量: {self.n_joints}")
        log.info(f"   - 更新频率: {self.update_frequency} Hz")
        log.info(f"   - 历史长度: {self.history_length} 点")
        log.info(f"   - 窗口大小: {self.window_size}")
    
    def update_positions(self, joint_positions: np.ndarray, gripper_position: float, timestamp: Optional[float] = None):
        """
        更新关节和夹爪位置数据
        
        Args:
            joint_positions: 7维关节位置数组 (弧度)
            gripper_position: 夹爪位置 (米)
            timestamp: 时间戳，默认使用当前时间
        """
        if not self.enabled:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        # 验证数据维度
        if len(joint_positions) != 7:
            log.warning(f"⚠️ 关节位置维度错误: 期望7维，实际{len(joint_positions)}维")
            return
            
        with self.data_lock:
            # 添加时间戳
            self.timestamps.append(timestamp)
            
            # 添加关节位置 (转换为度数)
            for i, position in enumerate(joint_positions):
                self.joint_positions[i].append(np.degrees(position))
            
            # 添加夹爪位置 (米转换为毫米)
            self.joint_positions[7].append(gripper_position * 1000)  # 转换为毫米显示
            
        # 定期保存数据
        if self.save_data and (timestamp - self.last_save_time) > self.save_interval:
            self._auto_save_data()
            self.last_save_time = timestamp
    
    def start_plotting(self):
        """启动实时绘图"""
        if not self.enabled:
            log.info("📊 关节位置可视化已禁用，跳过绘图启动")
            return
            
        if self.is_plotting:
            log.warning("⚠️ 绘图器已在运行")
            return
            
        self.is_plotting = True
        self.plot_thread = threading.Thread(target=self._plotting_loop, daemon=True)
        self.plot_thread.start()
        log.info("📊 关节位置实时绘图已启动")
    
    def stop_plotting(self):
        """停止实时绘图"""
        if not self.enabled or not self.is_plotting:
            return
            
        self.is_plotting = False
        if self.plot_thread and self.plot_thread.is_alive():
            self.plot_thread.join(timeout=2.0)
        
        if self.fig:
            plt.close(self.fig)
        
        log.info("📊 关节位置实时绘图已停止")
    
    def _plotting_loop(self):
        """绘图主循环"""
        try:
            # 设置matplotlib后端
            plt.ion()
            
            # 创建图形和子图
            self._setup_plots()
            
            # 绘图循环
            update_interval = 1.0 / self.update_frequency
            
            while self.is_plotting:
                start_time = time.time()
                
                try:
                    self._update_plots()
                    plt.pause(0.001)  # 让matplotlib更新显示
                    
                except Exception as e:
                    log.warning(f"⚠️ 绘图更新异常: {e}")
                
                # 控制更新频率
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            log.error(f"❌ 绘图循环异常: {e}")
        finally:
            plt.ioff()
    
    def _setup_plots(self):
        """设置绘图窗口和子图"""
        # 创建主图形
        self.fig, self.axes = plt.subplots(2, 4, figsize=(self.window_size[0]/100, self.window_size[1]/100))
        self.fig.suptitle('机器人关节和夹爪位置实时监控', fontsize=16, fontweight='bold')
        
        # 设置窗口标题
        manager = plt.get_current_fig_manager()
        manager.set_window_title('Joint Position Monitor')
        
        # 展平axes数组便于索引
        axes_flat = self.axes.flatten()
        
        # 为每个关节/夹爪创建子图
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_joints))
        
        for i in range(self.n_joints):
            ax = axes_flat[i]
            
            # 设置标题和标签
            if i < 7:  # 关节
                ax.set_title(f'{self.joint_names[i]} (度)', fontsize=12, fontweight='bold')
                ax.set_ylabel('位置 (度)')
            else:  # 夹爪
                ax.set_title(f'{self.joint_names[i]} (毫米)', fontsize=12, fontweight='bold')
                ax.set_ylabel('位置 (毫米)')
            
            ax.set_xlabel('时间 (秒)')
            ax.grid(True, alpha=0.3)
            
            # 创建空的线条对象
            line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.8)
            self.lines.append(line)
            
            # 设置初始范围
            if i < 7:  # 关节角度范围
                ax.set_ylim(-180, 180)
            else:  # 夹爪位置范围  
                ax.set_ylim(0, 80)  # 0-80毫米
        
        # 调整子图布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def _update_plots(self):
        """更新所有子图的数据"""
        with self.data_lock:
            if len(self.timestamps) == 0:
                return
                
            # 获取时间轴数据 (相对时间)
            timestamps_array = np.array(self.timestamps)
            if len(timestamps_array) > 1:
                time_relative = timestamps_array - timestamps_array[0]
            else:
                time_relative = np.array([0])
            
            # 更新每个子图
            for i in range(self.n_joints):
                if len(self.joint_positions[i]) == 0:
                    continue
                    
                positions_array = np.array(self.joint_positions[i])
                
                # 更新线条数据
                self.lines[i].set_data(time_relative, positions_array)
                
                # 自动调整X轴范围
                ax = self.lines[i].axes
                if len(time_relative) > 1:
                    ax.set_xlim(time_relative[0], time_relative[-1])
                
                # 自动调整Y轴范围 (关节)
                if i < 7 and len(positions_array) > 0:
                    y_min, y_max = np.min(positions_array), np.max(positions_array)
                    y_margin = max(10, (y_max - y_min) * 0.1)
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)
                elif i == 7 and len(positions_array) > 0:  # 夹爪
                    y_min, y_max = np.min(positions_array), np.max(positions_array)
                    y_margin = max(5, (y_max - y_min) * 0.1)
                    ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)
        
        # 重新绘制
        self.fig.canvas.draw_idle()
    
    def save_data(self, filename: Optional[str] = None):
        """
        保存位置数据到文件
        
        Args:
            filename: 保存文件名，默认使用时间戳命名
        """
        if not self.enabled or len(self.timestamps) == 0:
            return
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"joint_positions_{timestamp}.csv"
            
        # 确保保存目录存在
        save_path = Path(filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with self.data_lock:
                # 准备数据
                timestamps_array = np.array(self.timestamps)
                positions_data = []
                
                for i in range(self.n_joints):
                    if len(self.joint_positions[i]) > 0:
                        positions_data.append(np.array(self.joint_positions[i]))
                    else:
                        positions_data.append(np.zeros(len(timestamps_array)))
                
                # 创建CSV内容
                header = ['timestamp'] + self.joint_names
                data_array = np.column_stack([timestamps_array] + positions_data)
                
                # 保存到CSV
                np.savetxt(save_path, data_array, delimiter=',', header=','.join(header), comments='')
                
                log.info(f"📊 关节位置数据已保存: {save_path}")
                log.info(f"   - 数据点数: {len(timestamps_array)}")
                log.info(f"   - 时间范围: {timestamps_array[-1] - timestamps_array[0]:.1f} 秒")
                
        except Exception as e:
            log.error(f"❌ 保存关节位置数据失败: {e}")
    
    def _auto_save_data(self):
        """自动保存数据 (后台线程安全)"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/joint_positions_auto_{timestamp}.csv"
        
        # 在后台线程中保存，避免阻塞主流程
        def save_worker():
            try:
                self.save_data(filename)
                log.debug(f"📊 自动保存关节位置数据: {filename}")
            except Exception as e:
                log.warning(f"⚠️ 自动保存失败: {e}")
        
        threading.Thread(target=save_worker, daemon=True).start()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取当前数据统计信息
        
        Returns:
            Dict: 包含各关节位置统计的字典
        """
        if not self.enabled or len(self.timestamps) == 0:
            return {}
            
        stats = {}
        
        with self.data_lock:
            stats['data_points'] = len(self.timestamps)
            stats['time_span'] = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
            
            for i, joint_name in enumerate(self.joint_names):
                if len(self.joint_positions[i]) > 0:
                    positions = np.array(self.joint_positions[i])
                    stats[joint_name] = {
                        'mean': float(np.mean(positions)),
                        'std': float(np.std(positions)),
                        'min': float(np.min(positions)),
                        'max': float(np.max(positions)),
                        'range': float(np.max(positions) - np.min(positions))
                    }
        
        return stats
    
    def print_statistics(self):
        """打印当前数据统计信息"""
        stats = self.get_statistics()
        
        if not stats:
            log.info("📊 暂无关节位置数据统计")
            return
            
        log.info("📊 关节位置数据统计:")
        log.info(f"   - 数据点数: {stats['data_points']}")
        log.info(f"   - 时间跨度: {stats['time_span']:.1f} 秒")
        
        for joint_name in self.joint_names:
            if joint_name in stats:
                joint_stats = stats[joint_name]
                unit = '度' if 'Joint' in joint_name else '毫米'
                log.info(f"   - {joint_name}: 均值={joint_stats['mean']:.2f}{unit}, "
                        f"标准差={joint_stats['std']:.2f}{unit}, "
                        f"范围=[{joint_stats['min']:.2f}, {joint_stats['max']:.2f}]{unit}")