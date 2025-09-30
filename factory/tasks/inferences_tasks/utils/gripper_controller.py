#!/usr/bin/env python3
"""
智能夹爪控制器
包含传统控制器和基于任务类型的新架构控制器
"""

import numpy as np
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any
import glog as log

# 导入新的任务类型系统
try:
    from .task_types import TaskType, TaskTypeFactory
    from .gripper_strategies import GripperStrategyFactory
    TASK_SYSTEM_AVAILABLE = True
except ImportError as e:
    log.warning(f"⚠️ 任务类型系统不可用: {e}")
    TASK_SYSTEM_AVAILABLE = False




class TaskGripperController:
    """基于任务类型的统一夹爪控制器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化任务感知的夹爪控制器

        Args:
            config: 完整配置字典，包含任务类型信息

        Raises:
            ImportError: 任务类型系统不可用
            ValueError: 无法确定任务类型或创建策略
        """
        if not TASK_SYSTEM_AVAILABLE:
            raise ImportError("任务类型系统不可用，请检查模块导入")

        # 获取任务类型
        self.task_type = TaskType.from_config(config)
        log.info(f"🎯 检测到任务类型: {self.task_type.display_name}")

        # 加载并合并任务特定配置
        try:
            from .config_loader import ConfigLoader
            config_loader = ConfigLoader()

            # 获取任务特定配置并合并到基础配置
            merged_config = config_loader.merge_with_task_config(config, self.task_type.value)
            self.config = merged_config

        except ImportError:
            log.warning("⚠️ 配置加载器不可用，使用原始配置")
            self.config = config
        except Exception as e:
            log.warning(f"⚠️ 任务配置合并失败: {e}，使用原始配置")
            self.config = config

        # 验证任务配置
        try:
            TaskTypeFactory.validate_task_config(self.task_type, self.config)
        except Exception as e:
            log.error(f"❌ 任务配置验证失败: {e}")
            raise

        # 创建对应的夹爪控制策略（使用合并后的配置）
        gripper_config = self.config.get('gripper_postprocess', {})
        self.strategy = GripperStrategyFactory.create(self.task_type.value, gripper_config)

        # 设置任务特定参数（使用合并后的配置）
        self.max_step_nums = self.config.get('max_step_nums', self.strategy.get_max_steps())

        # 向后兼容的属性
        self.state = self.strategy.get_state()
        self.min_grasp_width = self.strategy.min_grasp_width
        self.max_grasp_width = self.strategy.max_grasp_width
        self.grasp_check_enabled = getattr(self.strategy, 'grasp_check_enabled', True)

        # 初始化可视化器相关状态
        self.visualizer = None
        self.last_logged_state = None  # 用于状态变化检测
        self.visualization_step_counter = 0  # 用于定期记录
        self._init_visualizer(config)

        log.info(f"✅ 任务夹爪控制器初始化完成:")
        log.info(f"   - 任务类型: {self.task_type.display_name}")
        log.info(f"   - 控制策略: {self.strategy.__class__.__name__}")
        log.info(f"   - 最大步数: {self.max_step_nums}")

    def _init_visualizer(self, config: Dict[str, Any]) -> None:
        """初始化可视化器"""
        try:
            from .gripper_visualization_wrapper import GripperVisualizationWrapper

            if 'gripper_visualization' in config:
                viz_config = config['gripper_visualization']
                if viz_config.get('enabled', False):
                    self.visualizer = GripperVisualizationWrapper(
                        enabled=True,
                        max_points=viz_config.get('max_points', 500000),
                        save_data=viz_config.get('save_data', True),
                        save_path=viz_config.get('save_path', './logs/gripper_actions.csv'),
                        auto_start_plot=viz_config.get('auto_start_plot', True)
                    )
                    log.info(f"   - 可视化功能: 已启用 (保存至: {viz_config.get('save_path', './logs/gripper_actions.csv')})")
                else:
                    log.info(f"   - 可视化功能: 未启用")
            else:
                log.warning(f"⚠️ gripper_visualization配置未找到，当前配置键: {list(config.keys())}")
        except ImportError:
            log.info(f"   - 可视化功能: 模块不可用")

    def process(self, action_value: float, safety_failed: bool = False,
                end_effector_pose: Optional[np.ndarray] = None) -> Tuple[bool, Optional[float], bool]:
        """
        处理夹爪动作（统一接口）

        Args:
            action_value: 夹爪动作值
            safety_failed: 是否安全检查失败
            end_effector_pose: 末端执行器位姿

        Returns:
            Tuple[bool, Optional[float], bool]:
                (是否需要执行命令, 命令值, 是否需要重新推理)
        """
        # 构建上下文信息
        context = {
            'safety_failed': safety_failed,
            'end_effector_pose': end_effector_pose,
            'task_type': self.task_type
        }

        # 委托给具体策略处理
        result = self.strategy.process(action_value, context)

        # 更新状态（向后兼容）
        self.state = self.strategy.get_state()

        # 记录可视化数据（每次都记录用于绘制曲线，但动作标记仅在状态变化时）
        if hasattr(self, 'visualizer') and self.visualizer is not None:
            try:
                self.visualization_step_counter += 1

                # 判断是否是状态变化
                state_changed = (
                    self.last_logged_state is None or  # 第一次
                    self.state != self.last_logged_state  # 状态发生变化
                )

                # 动作描述：仅在状态变化时设置具体动作，否则为空
                action_desc = ""
                if state_changed:
                    # 基于实际控制器状态描述动作
                    if self.state == 'CLOSED':
                        action_desc = "闭合"
                    elif self.state == 'OPEN':
                        action_desc = "张开"
                    elif self.state == 'HOLDING':
                        action_desc = "保持闭合"
                    elif self.state == 'CHECKING':
                        action_desc = "检查抓取"
                    else:
                        action_desc = f"状态：{self.state}"

                # 每次都记录数据点（用于绘制连续曲线）
                self.visualizer.log_gripper_action(
                    raw_value=action_value,
                    normalized_value=action_value,
                    trend="stable",  # TaskGripperController默认为稳定
                    action_taken=action_desc,  # 动作标记仅在状态变化时有值
                    stable_counter=0,
                    state=self.state
                )

                # 更新上次记录的状态
                if state_changed:
                    self.last_logged_state = self.state

            except Exception:
                # 静默处理可视化错误，不影响主流程
                pass

        return result

    def get_gripper_position(self) -> float:
        """
        获取夹爪位置值（用于状态向量）

        Returns:
            float: 夹爪位置值 (0.0=闭合, 0.08=打开)
        """
        if self.state in ['CLOSED', 'CHECKING', 'HOLDING']:
            return 0.0
        else:
            return 0.08

    def reset(self):
        """重置控制器状态"""
        self.strategy.reset()
        self.state = self.strategy.get_state()

        # 重置可视化状态
        self.last_logged_state = None
        self.visualization_step_counter = 0

        # 重置可视化器
        if hasattr(self, 'visualizer') and self.visualizer is not None:
            try:
                self.visualizer.reset()
            except:
                pass  # 忽略可视化器重置失败

        log.info(f"🔄 任务夹爪控制器已重置 (任务: {self.task_type.display_name})")

    def get_debug_info(self) -> Dict[str, Any]:
        """
        获取调试信息

        Returns:
            Dict: 调试信息字典
        """
        info = self.strategy.get_debug_info()
        info.update({
            'task_type': self.task_type.value,
            'task_display_name': self.task_type.display_name,
            'max_step_nums': self.max_step_nums
        })
        return info


def create_gripper_controller(config: Dict[str, Any],
                             full_config: Optional[Dict[str, Any]] = None) -> Any:
    """
    工厂函数：根据配置创建合适的夹爪控制器

    Args:
        config: 夹爪配置
        full_config: 完整配置（包含任务类型信息）

    Returns:
        夹爪控制器实例
    """
    # 使用新的任务类型系统
    if TASK_SYSTEM_AVAILABLE and full_config:
        return TaskGripperController(full_config)
    else:
        raise ValueError("Task system not available and legacy controllers have been removed")


class GripperStateLogger:
    """Gripper state logging utilities moved from ACT_Inferencer"""

    @staticmethod
    def log_gripper_state(gym_robot, gripper_controller=None, context: str = "") -> None:
        """Log current gripper state for debugging"""
        log.info(f"🔍 Gripper State Check ({context}):")

        # Log controller state
        if gripper_controller and hasattr(gripper_controller, 'state'):
            controller_state = gripper_controller.state
            log.info(f"   - Controller State: {controller_state}")

            # Log controller position
            if hasattr(gripper_controller, 'get_gripper_position'):
                controller_pos = gripper_controller.get_gripper_position()
                log.info(f"   - Controller Position: {controller_pos:.4f}")

        if hasattr(gym_robot, '_robot') and hasattr(gym_robot._robot, '_tool'):
            tool = gym_robot._robot._tool
            if hasattr(tool, 'get_tool_state'):
                tool_state = tool.get_tool_state()
                log.info(f"   - Hardware Tool State: position={tool_state._position:.4f}, grasped={tool_state._is_grasped}")
            elif hasattr(tool, '_state'):
                log.info(f"   - Hardware Tool Position: {tool._state._position:.4f}")