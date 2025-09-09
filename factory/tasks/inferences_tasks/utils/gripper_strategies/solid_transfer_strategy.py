#!/usr/bin/env python3
"""
固体转移任务夹爪控制策略
夹爪全程保持闭合状态
"""

from typing import Dict, Any
import glog as log

from .base_strategy import GripperStrategy


class SolidTransferStrategy(GripperStrategy):
    """固体转移任务策略：夹爪全程保持闭合状态"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 固体转移任务直接设置为闭合状态
        self.state = 'HOLDING'  # 初始状态就是保持闭合
        self.always_closed = True

        # 禁用所有检测功能
        self.grasp_check_enabled = False
        self.completion_detection_enabled = False

        log.info(f"📦 固体转移策略初始化")
        log.info(f"   - 初始状态: 闭合")
        log.info(f"   - 行为: 全程保持闭合")
        log.info(f"   - 抓取检测: 禁用")
        log.info(f"   - 任务完成检测: 禁用")

    def should_close(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该闭合夹爪
        固体转移任务：始终不需要闭合动作（因为已经闭合）

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            bool: 始终返回False（不需要执行闭合动作）
        """
        # 固体转移任务已经闭合，不需要额外的闭合动作
        return False

    def should_open(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该打开夹爪
        固体转移任务：始终不打开夹爪

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            bool: 始终返回False（不打开夹爪）
        """
        # 固体转移任务全程保持闭合，不打开夹爪
        return False

    def get_max_steps(self) -> int:
        """获取固体转移任务的最大步数"""
        return self.config.get('max_step_nums', 500)

    def process(self, action_value: float, context: Dict[str, Any] = None) -> tuple[bool, float | None, bool]:
        """
        处理固体转移任务的夹爪控制逻辑

        Args:
            action_value: 当前动作值（忽略）
            context: 上下文信息

        Returns:
            Tuple: (执行命令, 命令值, 强制重新预测)
        """
        if context is None:
            context = {}

        # 记录历史（用于调试）
        self.action_history.append(action_value)

        # 固体转移任务始终保持闭合状态
        execute_command = True
        command_value = 0.0  # 始终发送闭合命令
        force_repredict = False

        # 每100步记录一次状态
        if len(self.action_history) % 100 == 0:
            log.debug(f"📦 固体转移任务保持闭合状态 (已执行{len(self.action_history)}步)")

        return execute_command, command_value, force_repredict

    def reset(self):
        """重置策略状态"""
        # 固体转移任务重置后仍然保持闭合状态
        self.state = 'HOLDING'
        self.last_state = 'HOLDING'
        self.action_history.clear()
        self.stable_counter = 0
        log.info("📦 固体转移策略已重置（保持闭合状态）")

    def get_debug_info(self) -> Dict[str, Any]:
        """获取固体转移策略的调试信息"""
        info = super().get_debug_info()
        info.update({
            'always_closed': self.always_closed,
            'behavior': 'always_closed',
            'command_value': 0.0
        })
        return info