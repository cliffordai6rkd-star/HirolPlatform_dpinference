#!/usr/bin/env python3
"""
倒水任务夹爪控制策略
检测到闭合操作后，保持夹爪闭合直到episode结束
"""

import time
import numpy as np
from typing import Dict, Any
import glog as log

from .base_strategy import GripperStrategy


class LiquidTransferStrategy(GripperStrategy):
    """倒水任务策略：检测到闭合后保持闭合直到episode结束"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 倒水任务特定配置
        self.close_value_threshold = config.get('close_value_threshold', 0.02)
        self.required_close_duration = config.get('required_close_duration', 1.0)  # 更长的确认时间

        # 闭合检测
        self.close_start_time = None
        self.close_value_counter = 0
        self.has_closed_once = False  # 是否已经闭合过

        # 倒水任务禁用某些功能
        self.grasp_check_enabled = False  # 禁用抓取检测

        log.info(f"🌊 倒水策略初始化: 闭合阈值={self.close_value_threshold}, 确认时长={self.required_close_duration}s")
        log.info(f"   - 抓取检测: 禁用")
        log.info(f"   - 任务完成检测: 禁用")
        log.info(f"   - 闭合后行为: 保持闭合直到episode结束")

    def should_close(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该闭合夹爪
        倒水任务需要更严格的闭合检测，避免误操作

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            bool: 是否应该闭合
        """
        if self.state != 'OPEN':
            return False

        # 如果已经闭合过，不再闭合
        if self.has_closed_once:
            return False

        # 持续低值检测（比抓放任务更严格）
        if action_value < self.close_value_threshold:
            if self.close_start_time is None:
                self.close_start_time = time.time()
                self.close_value_counter = 1
                log.info(f"🌊 倒水任务开始检测闭合信号: 值={action_value:.3f}")
            else:
                self.close_value_counter += 1
                elapsed_time = time.time() - self.close_start_time

                if elapsed_time >= self.required_close_duration:
                    log.info(f"🤏 倒水任务确认闭合: 低值持续{elapsed_time:.1f}s, 值={action_value:.3f}")
                    self.has_closed_once = True
                    self._reset_close_detection()

                    return True

                # 移除中间进度日志，减少输出冗余
        else:
            # 值不在低值范围，重置计数器
            if self.close_start_time is not None:
                elapsed = time.time() - self.close_start_time
                log.info(f"🔄 倒水任务闭合信号中断: 值={action_value:.3f} (持续了{elapsed:.1f}s)")
                self._reset_close_detection()

        return False

    def should_open(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该打开夹爪
        倒水任务：一旦闭合，不再自动打开（保持闭合直到episode结束）

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            bool: 是否应该打开（倒水任务始终返回False）
        """
        # 倒水任务：一旦闭合，保持闭合状态直到episode结束
        if self.state == 'HOLDING' and self.has_closed_once:
            # 每100步记录一次状态
            if len(self.action_history) % 100 == 0:
                log.info(f"🌊 倒水任务保持闭合状态 (已执行{len(self.action_history)}步)")
            return False

        return False

    def get_max_steps(self) -> int:
        """获取倒水任务的最大步数（通常需要更多步数）"""
        return self.config.get('max_step_nums', 800)

    def _handle_closed_state(self, context: Dict[str, Any]) -> str:
        """
        处理闭合状态的后续逻辑
        倒水任务：直接进入保持状态，不进行抓取检测

        Args:
            context: 上下文信息

        Returns:
            str: 新的状态
        """
        # 倒水任务不需要抓取检测，直接进入保持状态
        log.info(f"🌊 倒水任务进入保持状态，将保持闭合直到episode结束")
        return 'HOLDING'

    def process(self, action_value: float, context: Dict[str, Any] = None) -> tuple[bool, float | None, bool]:
        """
        处理倒水任务的夹爪控制逻辑

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            Tuple: (执行命令, 命令值, 强制重新预测)
        """
        if context is None:
            context = {}

        # 调用基类处理通用逻辑
        execute_command, command_value, force_repredict = super().process(action_value, context)

        return execute_command, command_value, force_repredict

    def _reset_close_detection(self):
        """重置闭合检测状态"""
        self.close_start_time = None
        self.close_value_counter = 0

    def reset(self):
        """重置策略状态"""
        super().reset()
        self._reset_close_detection()
        self.has_closed_once = False
        log.info("🌊 倒水策略已重置")

    def get_debug_info(self) -> Dict[str, Any]:
        """获取倒水策略的调试信息"""
        info = super().get_debug_info()
        info.update({
            'has_closed_once': self.has_closed_once,
            'close_detection_active': self.close_start_time is not None,
            'required_close_duration': self.required_close_duration,
            'behavior': 'hold_closed_until_episode_end'
        })
        return info