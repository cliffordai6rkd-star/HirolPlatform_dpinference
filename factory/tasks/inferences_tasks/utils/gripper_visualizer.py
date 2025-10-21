#!/usr/bin/env python3
"""
夹爪动作值实时可视化器

完全解耦的独立模块，通过日志文件或消息队列监听夹爪动作值，
实时绘制曲线图，不影响现有推理系统。

使用方式：
1. 独立运行：python gripper_visualizer.py
2. 或集成到其他系统中作为可选组件
"""

import threading
import time
import queue
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib
# Check if we're in the main thread before setting GUI backend
if threading.current_thread() is not threading.main_thread():
    matplotlib.use('Agg')  # Use non-GUI backend for background threads
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import glog as log

@dataclass
class GripperDataPoint:
    """夹爪数据点"""
    timestamp: float
    raw_value: float           # 原始ACT输出值
    normalized_value: float    # 归一化后的值
    trend: str                 # 趋势 (opening/closing/stable)
    action_taken: str          # 执行的动作 (闭合/张开/保持)
    stable_counter: int        # 稳定计数器
    state: str                 # 夹爪状态 (OPEN/CLOSED/HOLDING)


class GripperVisualizer:
    """
    夹爪动作值实时可视化器

    支持多种数据源：
    - 日志文件监听
    - 消息队列
    - 直接API调用
    """

    def __init__(self,
                 max_points: int = 500,
                 update_interval: float = 0.1,
                 save_data: bool = True,
                 save_path: str = "./logs/gripper_data.csv"):
        """
        初始化可视化器

        Args:
            max_points: 最大显示数据点数
            update_interval: 更新间隔(秒)
            save_data: 是否保存数据到文件
            save_path: 数据保存路径
        """
        self.max_points = max_points
        self.update_interval = update_interval
        self.save_data = save_data
        self.save_path = Path(save_path)

        # 数据存储
        self.data_queue = queue.Queue(maxsize=1000)
        self.data_points: deque[GripperDataPoint] = deque(maxlen=max_points)

        # 可视化组件
        self.fig = None
        self.ax = None
        self.lines = {}
        self.legend_elements = []

        # 运行控制
        self._running = False
        self._threads = []

        # 统计信息
        self.stats = {
            'total_points': 0,
            'close_actions': 0,
            'open_actions': 0,
            'last_update': None
        }

        # 自动保存图形功能
        self.last_save_time = 0.0
        self.save_interval = 5.0  # 每5秒保存一次
        self.plot_save_dir = Path("./logs/gripper_plots")
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)

        # 确保保存目录存在
        if self.save_data:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def add_data_point(self,
                      raw_value: float,
                      normalized_value: float,
                      trend: str = "stable",
                      action_taken: str = "保持",
                      stable_counter: int = 0,
                      state: str = "OPEN") -> None:
        """
        添加数据点（外部API）

        Args:
            raw_value: 原始ACT输出值
            normalized_value: 归一化后的值
            trend: 趋势
            action_taken: 执行的动作
            stable_counter: 稳定计数器
            state: 夹爪状态
        """
        try:
            data_point = GripperDataPoint(
                timestamp=time.time(),
                raw_value=raw_value,
                normalized_value=normalized_value,
                trend=trend,
                action_taken=action_taken,
                stable_counter=stable_counter,
                state=state
            )

            self.data_queue.put_nowait(data_point)

        except queue.Full:
            log.debug("⚠️ 数据队列已满，丢弃数据点")

    def start_log_monitor(self, log_file: str) -> None:
        """
        启动日志文件监听线程

        Args:
            log_file: 日志文件路径
        """
        def monitor_log():
            """监听日志文件变化"""
            log_path = Path(log_file)

            # 等待日志文件存在
            while not log_path.exists() and self._running:
                time.sleep(1)

            if not self._running:
                return

            # 打开文件并跳到末尾
            with open(log_path, 'r', encoding='utf-8') as f:
                f.seek(0, 2)  # 跳到文件末尾

                while self._running:
                    line = f.readline()
                    if line:
                        self._parse_log_line(line.strip())
                    else:
                        time.sleep(0.1)

        if self._running:
            thread = threading.Thread(target=monitor_log, daemon=True)
            thread.start()
            self._threads.append(thread)
            print(f"📖 开始监听日志文件: {log_file}")

    def _parse_log_line(self, line: str) -> None:
        """
        解析日志行并提取夹爪数据

        Args:
            line: 日志行内容
        """
        try:
            # 匹配夹爪相关日志的正则表达式
            patterns = [
                # 匹配: "夹爪动作: 原始值=0.02345, 归一化值=0.234, 趋势=closing, 动作=闭合"
                r'夹爪动作.*?原始值=([0-9.-]+).*?归一化值=([0-9.-]+).*?趋势=(\w+).*?动作=([^,\s]+)',

                # 匹配: "🤏 检测到闭合信号" 等
                r'检测到(闭合|张开)信号.*?值=([0-9.-]+)',

                # 匹配ACT推理输出
                r'ACT.*?夹爪.*?([0-9.-]+)',
            ]

            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) >= 4:  # 完整信息
                        raw_val = float(match.group(1))
                        norm_val = float(match.group(2))
                        trend = match.group(3)
                        action = match.group(4)

                        self.add_data_point(raw_val, norm_val, trend, action)
                        break
                    elif len(match.groups()) >= 2:  # 部分信息
                        action = match.group(1)
                        value = float(match.group(2))

                        self.add_data_point(value, value, "unknown", action)
                        break

        except (ValueError, AttributeError) as e:
            # 忽略解析错误，继续处理下一行
            pass

    def init_plot(self) -> None:
        """初始化绘图界面"""
        # Check if GUI is available
        if matplotlib.get_backend() == 'Agg':
            print("⚠️ 使用非GUI后端，跳过图形界面初始化")
            return

        plt.style.use('default')

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('Gripper Action Real-time Monitor', fontsize=16, fontweight='bold')

        # 设置子图
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Gripper Action Value')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(-0.05, 1.1)  # 适应归一化值范围 0-1

        # 初始化绘图线条
        # self.lines['raw'] = self.ax.plot([], [], 'b-', linewidth=2, label='Raw Value (0-GRIPPER_OPEN)', alpha=0.8)[0]
        self.lines['normalized'] = self.ax.plot([], [], 'g-', linewidth=2, label='Normalized Value (0-1)', alpha=0.8)[0]

        # 添加阈值线（同时显示原始值和归一化值的阈值）
        # self.lines['close_threshold'] = self.ax.axhline(y=0.03, color='r', linestyle='--', alpha=0.6, label='Close Threshold(0.03)')
        # self.lines['open_threshold'] = self.ax.axhline(y=0.035, color='orange', linestyle='--', alpha=0.6, label='Open Threshold(0.035)')
        # 归一化阈值线
        # self.lines['close_norm_threshold'] = self.ax.axhline(y=0.03/GRIPPER_OPEN, color='r', linestyle=':', alpha=0.4, label='Close Norm(0.375)')
        # self.lines['open_norm_threshold'] = self.ax.axhline(y=0.035/GRIPPER_OPEN, color='orange', linestyle=':', alpha=0.4, label='Open Norm(0.438)')

        # 新增条件阈值线
        # self.lines['close_condition_threshold'] = self.ax.axhline(y=0.1, color='darkred', linestyle='-', alpha=0.7, linewidth=2, label='Close Condition(<0.1)')
        # self.lines['open_condition_threshold'] = self.ax.axhline(y=0.55, color='darkorange', linestyle='-', alpha=0.7, linewidth=2, label='Open Condition(>0.55)')

        # 动作标记点（原始值）
        # self.lines['close_actions'] = self.ax.plot([], [], 'ro', markersize=8, label='Close Actions (Raw)', alpha=0.8)[0]
        # self.lines['open_actions'] = self.ax.plot([], [], 'go', markersize=8, label='Open Actions (Raw)', alpha=0.8)[0]

        # 动作标记点（归一化值）
        self.lines['close_actions_norm'] = self.ax.plot([], [], 'r^', markersize=8, label='Close Actions (Norm)', alpha=0.6)[0]
        self.lines['open_actions_norm'] = self.ax.plot([], [], 'g^', markersize=8, label='Open Actions (Norm)', alpha=0.6)[0]

        # 图例
        self.ax.legend(loc='upper right')

        # 状态文本
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       verticalalignment='top', fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

    def update_plot(self, frame) -> List:
        """更新绘图数据"""
        # 处理队列中的新数据
        while not self.data_queue.empty():
            try:
                data_point = self.data_queue.get_nowait()
                self.data_points.append(data_point)
                self.stats['total_points'] += 1
                self.stats['last_update'] = datetime.now()

                # 统计动作
                if '闭合' in data_point.action_taken:
                    self.stats['close_actions'] += 1
                elif '张开' in data_point.action_taken:
                    self.stats['open_actions'] += 1

                # 保存数据
                if self.save_data:
                    self._save_data_point(data_point)

            except queue.Empty:
                break

        if not self.data_points:
            return list(self.lines.values())

        # 准备绘图数据
        times = [dp.timestamp for dp in self.data_points]
        raw_values = [dp.raw_value for dp in self.data_points]
        norm_values = [dp.normalized_value for dp in self.data_points]

        # 转换为相对时间
        if times:
            start_time = times[0]
            relative_times = [(t - start_time) for t in times]

            # 更新曲线
            # self.lines['raw'].set_data(relative_times, raw_values)
            self.lines['normalized'].set_data(relative_times, norm_values)

            # 更新动作标记
            close_times, close_values = [], []
            open_times, open_values = [], []
            close_times_norm, close_values_norm = [], []
            open_times_norm, open_values_norm = [], []

            for i, dp in enumerate(self.data_points):
                if '闭合' in dp.action_taken:
                    close_times.append(relative_times[i])
                    close_values.append(dp.raw_value)
                    close_times_norm.append(relative_times[i])
                    close_values_norm.append(dp.normalized_value)
                elif '张开' in dp.action_taken:
                    open_times.append(relative_times[i])
                    open_values.append(dp.raw_value)
                    open_times_norm.append(relative_times[i])
                    open_values_norm.append(dp.normalized_value)

            # 更新原始值动作标记
            # self.lines['close_actions'].set_data(close_times, close_values)
            # self.lines['open_actions'].set_data(open_times, open_values)

            # 更新归一化值动作标记
            self.lines['close_actions_norm'].set_data(close_times_norm, close_values_norm)
            self.lines['open_actions_norm'].set_data(open_times_norm, open_values_norm)

            # 自适应坐标轴
            if relative_times:
                self.ax.set_xlim(max(0, relative_times[-1] - 600), relative_times[-1] + 5)  # 显示最近600秒

                # 保持固定Y轴范围以正确显示原始值和归一化值
                # self.ax.set_ylim(-0.05, 1.1) 已在init_plot中设置

            # 更新状态信息
            if self.data_points:
                last_dp = self.data_points[-1]
                status_info = (
                    f"Current State: {last_dp.state}\n"
                    f"Raw Value: {last_dp.raw_value:.5f}\n"
                    f"Normalized: {last_dp.normalized_value:.5f}\n"
                    f"Trend: {last_dp.trend}\n"
                    f"Stable Count: {last_dp.stable_counter}\n"
                    f"Total Points: {self.stats['total_points']}\n"
                    f"Close Count: {self.stats['close_actions']}\n"
                    f"Open Count: {self.stats['open_actions']}"
                )
                self.status_text.set_text(status_info)

        # 自动保存图形（每1秒）
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self._auto_save_plot()
            self.last_save_time = current_time

        return list(self.lines.values())

    def _save_data_point(self, data_point: GripperDataPoint) -> None:
        """保存数据点到CSV文件"""
        try:
            # 检查文件是否存在，不存在则写入标题行
            write_header = not self.save_path.exists()

            with open(self.save_path, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write("timestamp,raw_value,normalized_value,trend,action_taken,stable_counter,state\n")

                f.write(f"{data_point.timestamp},{data_point.raw_value},{data_point.normalized_value},"
                       f"{data_point.trend},{data_point.action_taken},{data_point.stable_counter},{data_point.state}\n")

        except Exception as e:
            print(f"⚠️ 保存数据失败: {e}")

    def _auto_save_plot(self) -> None:
        """自动保存当前图形"""
        try:
            if self.fig is None:
                return

            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_filename = f"gripper_plot_{timestamp}.png"
            timestamp_filepath = self.plot_save_dir / timestamp_filename

            # 固定的最新图片文件名
            latest_filename = "gripper_plot_latest.png"
            latest_filepath = self.plot_save_dir / latest_filename

            # 保存带时间戳的图形
            self.fig.savefig(timestamp_filepath, dpi=150, bbox_inches='tight')

            # 保存最新图形（覆盖之前的）
            self.fig.savefig(latest_filepath, dpi=150, bbox_inches='tight')

        except Exception as e:
            print(f"⚠️ 自动保存图形失败: {e}")

    def start_visualization(self, log_file: Optional[str] = None) -> None:
        """
        启动可视化

        Args:
            log_file: 可选的日志文件路径，用于自动监听
        """
        self._running = True

        # Check if we're in the main thread and adjust backend accordingly
        self.is_main_thread = threading.current_thread() is threading.main_thread()
        if not self.is_main_thread:
            print("⚠️ 检测到在后台线程中启动可视化，将尝试使用线程安全的GUI模式")
            # For non-main threads, we'll handle matplotlib backend later
            # but still allow GUI if properly configured

        try:
            # 初始化绘图
            print("🎨 初始化绘图界面...")
            self.init_plot()

            # 启动日志监听（如果提供了日志文件）
            if log_file:
                print(f"📖 启动日志监听: {log_file}")
                self.start_log_monitor(log_file)

            # 启动实时绘图
            print("⏱️  启动动画更新...")
            anim = animation.FuncAnimation(
                self.fig, self.update_plot, interval=int(self.update_interval * 1000),
                blit=False, cache_frame_data=False
            )

            print("🎯 夹爪动作值可视化已启动")
            print("📊 实时曲线图显示中...")
            if log_file:
                print(f"📖 监听日志文件: {log_file}")
            if self.save_data:
                print(f"💾 数据保存至: {self.save_path}")
            print("💡 显示窗口应该已经弹出")
            print("按 Ctrl+C 退出")

            # 非阻塞显示窗口
            plt.show(block=False)
            plt.pause(0.001)  # 让matplotlib有时间处理事件

            # 保持动画运行但不阻塞
            try:
                while self._running:
                    plt.pause(0.1)  # 处理matplotlib事件，但不阻塞
            except KeyboardInterrupt:
                print("\n🛑 用户中断可视化")
                self.stop()

        except KeyboardInterrupt:
            print("\n🛑 用户中断")
            self.stop()
        except Exception as e:
            print(f"❌ 可视化启动失败: {e}")
            import traceback
            traceback.print_exc()
            self.stop()

    def stop(self) -> None:
        """停止可视化"""
        self._running = False
        print("\n🛑 正在停止可视化...")

        # 等待线程结束
        for thread in self._threads:
            thread.join(timeout=1)

        if plt.get_fignums():
            plt.close('all')

        print("✅ 可视化已停止")


def main():
    """主函数 - 独立运行可视化器"""
    import argparse

    parser = argparse.ArgumentParser(description='夹爪动作值实时可视化器')
    parser.add_argument('--log-file', type=str,
                       help='监听的日志文件路径')
    parser.add_argument('--max-points', type=int, default=500,
                       help='最大显示数据点数 (默认: 500)')
    parser.add_argument('--update-interval', type=float, default=0.1,
                       help='更新间隔秒数 (默认: 0.1)')
    parser.add_argument('--save-path', type=str, default='./logs/gripper_data.csv',
                       help='数据保存路径 (默认: ./logs/gripper_data.csv)')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存数据到文件')

    args = parser.parse_args()

    # 创建可视化器
    visualizer = GripperVisualizer(
        max_points=args.max_points,
        update_interval=args.update_interval,
        save_data=not args.no_save,
        save_path=args.save_path
    )

    # 启动可视化
    try:
        visualizer.start_visualization(log_file=args.log_file)
    except KeyboardInterrupt:
        visualizer.stop()


if __name__ == "__main__":
    main()