#!/usr/bin/env python3
"""
夹爪控制策略基类和工厂
定义夹爪控制的抽象接口和策略工厂
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import glog as log
from collections import deque


class GripperStrategy(ABC):
    """夹爪控制策略抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化夹爪控制策略

        Args:
            config: 策略配置字典
        """
        self.config = config
        self.task_type = config.get('task_type', 'unknown')  # 保存任务类型信息
        self.state = 'OPEN'  # 初始状态
        self.last_state = 'OPEN'
        self.action_history = deque(maxlen=config.get('history_size', 10))
        self.stable_counter = 0

        # 通用配置参数
        self.min_grasp_width = config.get('min_grasp_width', 0.005)
        self.max_grasp_width = config.get('max_grasp_width', 0.070)
        self.stability_threshold = config.get('stability_threshold', 0.01)
        self.min_stable_steps = config.get('min_stable_steps', 5)

        log.info(f"🔧 初始化夹爪策略: {self.__class__.__name__}")

    @abstractmethod
    def should_close(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该闭合夹爪

        Args:
            action_value: 当前动作值（原始值）
            context: 上下文信息（包含传感器数据、状态等）

        Returns:
            bool: 是否应该闭合夹爪
        """
        pass

    @abstractmethod
    def should_open(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该打开夹爪

        Args:
            action_value: 当前动作值（原始值）
            context: 上下文信息

        Returns:
            bool: 是否应该打开夹爪
        """
        pass

    @abstractmethod
    def get_max_steps(self) -> int:
        """
        获取任务特定的最大步数

        Returns:
            int: 最大执行步数
        """
        pass

    def process(self, action_value: float, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[float], bool]:
        """
        处理夹爪动作的通用流程

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            Tuple[bool, Optional[float], bool]:
                (是否执行命令, 命令值, 是否需要重新预测)
        """
        if context is None:
            context = {}

        # 记录历史
        self.action_history.append(action_value)

        # 检查稳定性
        self._check_stability(context.get('end_effector_pose'))

        execute_command = False
        command_value = None
        force_repredict = False

        # 状态机逻辑
        if self.state == 'OPEN':
            if self.should_close(action_value, context):
                self.state = 'CLOSED'
                command_value = 0.0
                execute_command = True
                force_repredict = True
                log.info(f"🤏 策略 {self.__class__.__name__} 触发闭合: {action_value:.4f} → 0.0")

        elif self.state == 'CLOSED':
            # 子类可以覆盖此行为
            self.state = self._handle_closed_state(context)

        elif self.state == 'HOLDING':
            if self.should_open(action_value, context):
                self.state = 'OPEN'
                command_value = 1.0
                execute_command = True
                force_repredict = True
                log.info(f"✋ 策略 {self.__class__.__name__} 触发打开")

        # 记录状态变化
        if self.state != self.last_state:
            log.info(f"🔄 夹爪状态变化: {self.last_state} → {self.state}")
            self.last_state = self.state

        return execute_command, command_value, force_repredict

    def _handle_closed_state(self, context: Dict[str, Any]) -> str:
        """
        处理闭合状态的后续逻辑

        Args:
            context: 上下文信息

        Returns:
            str: 新的状态
        """
        # 默认行为：直接转入保持状态
        return 'HOLDING'

    def _check_stability(self, end_effector_pose: Optional[np.ndarray]) -> bool:
        """
        检查末端执行器的稳定性

        Args:
            end_effector_pose: 末端位姿

        Returns:
            bool: 是否稳定
        """
        if end_effector_pose is None:
            return False

        # 简化的稳定性检测逻辑
        # 子类可以覆盖实现更复杂的检测
        return self.stable_counter >= self.min_stable_steps

    def reset(self) -> None:
        """重置策略状态"""

        self.state = 'OPEN'
        if self.state != self.last_state:
            log.info(f"🔄 夹爪状态变化: {self.last_state} → {self.state}")
            
        self.last_state = 'OPEN'
        self.action_history.clear()
        self.stable_counter = 0
        log.info(f"🔄 策略 {self.__class__.__name__} 已重置")

    def get_state(self) -> str:
        """获取当前状态"""
        return self.state

    def get_debug_info(self) -> Dict[str, Any]:
        """
        获取调试信息

        Returns:
            Dict: 调试信息字典
        """
        return {
            'strategy': self.__class__.__name__,
            'state': self.state,
            'stable_counter': self.stable_counter,
            'history_size': len(self.action_history),
            'last_action': self.action_history[-1] if self.action_history else None
        }


class GripperStrategyFactory:
    """夹爪策略工厂"""

    _strategies = {}  # 策略注册表

    @classmethod
    def register(cls, task_type: str, strategy_class: type):
        """
        注册策略类

        Args:
            task_type: 任务类型字符串
            strategy_class: 策略类
        """
        cls._strategies[task_type] = strategy_class
        log.info(f"📝 注册策略: {task_type} -> {strategy_class.__name__}")

    @classmethod
    def create(cls, task_type: str, config: Dict[str, Any]) -> GripperStrategy:
        """
        创建策略实例

        Args:
            task_type: 任务类型
            config: 策略配置

        Returns:
            GripperStrategy: 策略实例

        Raises:
            ValueError: 未知的任务类型
        """
        # 导入具体策略类（延迟导入避免循环依赖）
        if not cls._strategies:
            cls._register_default_strategies()

        strategy_class = cls._strategies.get(task_type)
        if not strategy_class:
            raise ValueError(f"未知的任务类型: {task_type}")

        # 将任务类型信息传递给配置
        enhanced_config = config.copy()
        enhanced_config['task_type'] = task_type

        strategy = strategy_class(enhanced_config)
        return strategy

    @classmethod
    def _register_default_strategies(cls):
        """注册默认策略"""
        # 延迟导入避免循环依赖
        from .grasp_release_strategy import GraspReleaseStrategy
        from .liquid_transfer_strategy import LiquidTransferStrategy
        from .solid_transfer_strategy import SolidTransferStrategy

        cls.register("peg_in_hole", GraspReleaseStrategy)
        cls.register("block_stacking", GraspReleaseStrategy)
        cls.register("liquid_transfer", LiquidTransferStrategy)
        cls.register("solid_transfer", SolidTransferStrategy)