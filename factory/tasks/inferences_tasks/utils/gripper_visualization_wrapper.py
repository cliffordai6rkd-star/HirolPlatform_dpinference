#!/usr/bin/env python3
"""
夹爪可视化包装器

提供简单的API接口，让现有推理系统可以选择性地启用可视化功能，
不影响原有代码结构。

使用方式：
```python
from factory.tasks.inferences_tasks.utils.gripper_visualization_wrapper import GripperVisualizationWrapper

# 在推理系统中添加（可选）
viz = GripperVisualizationWrapper(enabled=True)  # 或 False 禁用

# 在夹爪动作处理时调用
viz.log_gripper_action(raw_value, normalized_value, trend, action, state)
```
"""

import threading
import time
from typing import Optional
from pathlib import Path

try:
    from .gripper_visualizer import GripperVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ 可视化模块不可用，将跳过可视化功能")


class GripperVisualizationWrapper:
    """
    夹爪可视化包装器

    提供轻量级接口，让现有系统可以选择性地集成可视化功能
    """

    def __init__(self,
                 enabled: bool = False,
                 max_points: int = 500,
                 save_data: bool = True,
                 save_path: str = "./logs/gripper_visualization_data.csv",
                 auto_start_plot: bool = False):
        """
        初始化可视化包装器

        Args:
            enabled: 是否启用可视化
            max_points: 最大显示数据点数
            save_data: 是否保存数据
            save_path: 数据保存路径
            auto_start_plot: 是否自动启动实时绘图窗口
        """
        self.enabled = enabled and VISUALIZATION_AVAILABLE
        self.auto_start_plot = auto_start_plot

        if not self.enabled:
            return

        try:
            self.visualizer = GripperVisualizer(
                max_points=max_points,
                save_data=save_data,
                save_path=save_path
            )

            self._plot_thread = None
            self._plot_started = False

            if self.auto_start_plot:
                self.start_plot()

            print("✅ 夹爪可视化包装器已启用")

        except Exception as e:
            print(f"⚠️ 可视化初始化失败: {e}")
            self.enabled = False

    def log_gripper_action(self,
                          raw_value: float,
                          normalized_value: Optional[float] = None,
                          trend: str = "stable",
                          action_taken: str = "保持",
                          stable_counter: int = 0,
                          state: str = "OPEN") -> None:
        """
        记录夹爪动作数据（主要API）

        Args:
            raw_value: 原始ACT输出值
            normalized_value: 归一化后的值（可选，默认使用raw_value）
            trend: 趋势
            action_taken: 执行的动作
            stable_counter: 稳定计数器
            state: 夹爪状态
        """
        if not self.enabled:
            return

        try:
            if normalized_value is None:
                normalized_value = raw_value

            self.visualizer.add_data_point(
                raw_value=raw_value,
                normalized_value=normalized_value,
                trend=trend,
                action_taken=action_taken,
                stable_counter=stable_counter,
                state=state
            )

            # 自动启动绘图（首次调用时）
            if self.auto_start_plot and not self._plot_started:
                self.start_plot()

        except Exception as e:
            print(f"⚠️ 记录夹爪数据失败: {e}")

    def start_plot(self) -> None:
        """启动实时绘图窗口（在后台线程中）"""
        if not self.enabled or self._plot_started:
            return

        try:
            print("📊 启动夹爪可视化绘图（后台线程）...")

            def visualization_thread():
                """可视化线程函数"""
                try:
                    time.sleep(0.1)  # 短暂延迟确保初始化完成
                    self.visualizer.start_visualization()
                except Exception as e:
                    print(f"⚠️ 可视化线程失败: {e}")

            # 在后台线程中启动可视化
            self._plot_thread = threading.Thread(target=visualization_thread, daemon=True)
            self._plot_thread.start()
            print("✅ 可视化已在后台线程中启动")

        except Exception as e:
            print(f"⚠️ 启动绘图失败: {e}")

        self._plot_started = True

    def save_current_data(self, filepath: Optional[str] = None) -> None:
        """
        手动保存当前数据

        Args:
            filepath: 保存文件路径（可选）
        """
        if not self.enabled:
            return

        try:
            if hasattr(self.visualizer, 'data_points') and self.visualizer.data_points:
                if filepath:
                    # 这里可以添加自定义保存逻辑
                    print(f"💾 数据保存功能开发中: {filepath}")
                else:
                    print(f"💾 数据已保存至默认路径: {self.visualizer.save_path}")
            else:
                print("ℹ️ 暂无数据可保存")

        except Exception as e:
            print(f"⚠️ 保存数据失败: {e}")

    def get_statistics(self) -> dict:
        """获取统计信息"""
        if not self.enabled:
            return {"enabled": False}

        try:
            return {
                "enabled": True,
                "total_points": self.visualizer.stats.get('total_points', 0),
                "close_actions": self.visualizer.stats.get('close_actions', 0),
                "open_actions": self.visualizer.stats.get('open_actions', 0),
                "last_update": self.visualizer.stats.get('last_update'),
                "plot_started": self._plot_started
            }
        except Exception:
            return {"enabled": True, "error": "统计信息获取失败"}

    def stop(self) -> None:
        """停止可视化"""
        if not self.enabled:
            return

        try:
            if hasattr(self, 'visualizer'):
                self.visualizer.stop()
            print("🛑 夹爪可视化已停止")
        except Exception as e:
            print(f"⚠️ 停止可视化失败: {e}")


# 全局可视化实例（单例模式）
_global_visualizer: Optional[GripperVisualizationWrapper] = None


def get_gripper_visualizer(enabled: bool = True, **kwargs) -> GripperVisualizationWrapper:
    """
    获取全局夹爪可视化实例（单例模式）

    Args:
        enabled: 是否启用可视化
        **kwargs: 其他初始化参数

    Returns:
        GripperVisualizationWrapper: 可视化包装器实例
    """
    global _global_visualizer

    if _global_visualizer is None:
        _global_visualizer = GripperVisualizationWrapper(enabled=enabled, **kwargs)

    return _global_visualizer


def quick_log_gripper(raw_value: float,
                     normalized_value: Optional[float] = None,
                     trend: str = "stable",
                     action: str = "保持",
                     **kwargs) -> None:
    """
    快速记录夹爪动作的便捷函数

    Args:
        raw_value: 原始值
        normalized_value: 归一化值
        trend: 趋势
        action: 动作
        **kwargs: 其他参数
    """
    visualizer = get_gripper_visualizer()
    visualizer.log_gripper_action(
        raw_value=raw_value,
        normalized_value=normalized_value,
        trend=trend,
        action_taken=action,
        **kwargs
    )


# 示例用法演示
def demo_usage():
    """演示如何使用可视化包装器"""
    print("🎯 夹爪可视化包装器使用演示")

    # 方式1：使用全局实例
    viz = get_gripper_visualizer(enabled=True, auto_start_plot=True)

    # 模拟一些夹爪数据
    import numpy as np
    import time

    print("📊 生成模拟数据...")

    for i in range(50):
        # 模拟夹爪动作序列
        if i < 20:
            raw_val = 0.025 + np.random.normal(0, 0.002)
            action = "保持"
            trend = "stable"
        elif i < 35:
            raw_val = 0.025 - (i - 20) * 0.0015 + np.random.normal(0, 0.001)
            action = "准备闭合" if i == 30 else "保持"
            trend = "closing"
        else:
            raw_val = 0.002 + np.random.normal(0, 0.0005)
            action = "闭合" if i == 35 else "保持"
            trend = "stable"

        norm_val = max(0, min(1, raw_val))

        viz.log_gripper_action(
            raw_value=raw_val,
            normalized_value=norm_val,
            trend=trend,
            action_taken=action,
            stable_counter=i,
            state="OPEN" if i < 35 else "CLOSED"
        )

        time.sleep(0.1)  # 模拟实时数据

    print("📈 数据生成完成，查看统计信息:")
    stats = viz.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("⏱️ 等待10秒后停止演示...")
    time.sleep(10)
    viz.stop()


if __name__ == "__main__":
    demo_usage()