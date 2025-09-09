#!/usr/bin/env python3
"""
抓放任务夹爪控制策略
适用于插孔任务和叠方块任务，依赖ACT模型输出控制夹爪开闭
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Any
import glog as log

from .base_strategy import GripperStrategy


class GraspReleaseStrategy(GripperStrategy):
    """抓放任务策略：插孔任务、叠方块任务"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 抓放任务特定配置
        self.close_value_threshold = config.get('close_value_threshold', 0.02)
        self.open_value_threshold = config.get('open_value_threshold', 0.95)
        self.trend_window_size = config.get('trend_window_size', 8)
        self.min_holding_steps = config.get('min_holding_steps', 30)

        # 波形检测参数
        self.waveform_history = deque(maxlen=100)
        self.peaks_detected = []
        self.valleys_detected = []
        self.target_peak_count = config.get('target_peak_count', 3)
        self.peak_threshold = config.get('peak_threshold', 0.95)
        self.valley_threshold = config.get('valley_threshold', 0.02)

        # 闭合持续时间检测
        self.close_start_time = None
        self.close_value_counter = 0
        self.required_close_duration = config.get('required_close_duration', 0.5)

        # 保持计数器
        self.holding_counter = 0

        # 抓取检测
        self.grasp_check_enabled = config.get('grasp_check_enabled', True)
        self.grasp_check_delay = config.get('grasp_check_delay', 20)
        self.grasp_check_counter = 0
        self.max_grasp_retries = config.get('max_grasp_retries', 3)
        self.grasp_retry_count = 0

        log.info(f"🔧 抓放策略初始化: 闭合阈值={self.close_value_threshold}, 打开阈值={self.open_value_threshold}")

    def should_close(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该闭合夹爪
        基于低值持续时间检测

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            bool: 是否应该闭合
        """
        if self.state != 'OPEN':
            return False

        # 持续低值检测
        if action_value < self.close_value_threshold:
            if self.close_start_time is None:
                self.close_start_time = time.time()
                self.close_value_counter = 1
                log.debug(f"⏱️ 开始检测低值持续时间: {action_value:.3f}")
            else:
                self.close_value_counter += 1
                elapsed_time = time.time() - self.close_start_time

                if elapsed_time >= self.required_close_duration:
                    log.info(f"🤏 低值持续{elapsed_time:.1f}s确认闭合: 值={action_value:.3f}")
                    self._reset_close_detection()
                    return True
        else:
            # 值不在低值范围，重置计数器
            if self.close_start_time is not None:
                elapsed = time.time() - self.close_start_time
                log.debug(f"🔄 低值中断，重置计数器: 值={action_value:.3f} (持续了{elapsed:.1f}s)")
                self._reset_close_detection()

        return False

    def should_open(self, action_value: float, context: Dict[str, Any]) -> bool:
        """
        判断是否应该打开夹爪
        基于波形检测在特定波峰处打开

        Args:
            action_value: 当前动作值
            context: 上下文信息

        Returns:
            bool: 是否应该打开
        """
        if self.state != 'HOLDING':
            return False

        # 保持计数器必须归零
        if self.holding_counter > 0:
            return False

        if action_value > 0.079:
            log.info(f"✋ 动作值接近最大，强制打开: 值={action_value:.3f}")
            return True
        
        return False

    def get_max_steps(self) -> int:
        """获取抓放任务的最大步数"""
        # 优先使用配置中的显式值
        if 'max_step_nums' in self.config:
            return self.config['max_step_nums']

        # 根据任务类型推断默认值（如果有任务类型信息）
        if hasattr(self, 'task_type'):
            if self.task_type == 'peg_in_hole':
                return 600
            elif self.task_type == 'block_stacking':
                return 550

        # 默认值
        return self.config.get('max_step_nums', 550)

    def _handle_closed_state(self, context: Dict[str, Any]) -> str:
        """
        处理闭合状态的后续逻辑
        包含抓取检测和状态转换

        Args:
            context: 上下文信息

        Returns:
            str: 新的状态
        """
        if self.grasp_check_enabled:
            # 进入检查状态
            self.grasp_check_counter = self.grasp_check_delay
            log.info(f"🔍 进入抓取检查状态，等待{self.grasp_check_delay}步")
            return 'CHECKING'
        else:
            # 直接进入保持状态
            self.holding_counter = self.min_holding_steps
            log.info(f"🔒 直接进入保持状态 ({self.min_holding_steps}步)")
            return 'HOLDING'

    def process(self, action_value: float, context: Dict[str, Any] = None) -> tuple[bool, float | None, bool]:
        """
        处理抓放任务的夹爪控制逻辑

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

        # 处理特定状态
        if self.state == 'CHECKING':
            execute_command, command_value, force_repredict = self._handle_checking_state(context)

        elif self.state == 'HOLDING':
            # 保持计数器倒计时
            if self.holding_counter > 0:
                self.holding_counter -= 1
                if self.holding_counter % 20 == 0 and self.holding_counter > 0:
                    log.debug(f"🔒 保持中 (剩余{self.holding_counter}步)")
                elif self.holding_counter == 0:
                    log.info(f"✅ 保持时间结束，开始检测打开信号")

        return execute_command, command_value, force_repredict

    def _handle_checking_state(self, context: Dict[str, Any]) -> tuple[bool, float | None, bool]:
        """
        处理检查状态的逻辑

        Args:
            context: 上下文信息

        Returns:
            Tuple: (执行命令, 命令值, 强制重新预测)
        """
        if self.grasp_check_counter > 0:
            self.grasp_check_counter -= 1
        else:
            # 检查完成，进入保持状态
            self.state = 'HOLDING'
            self.holding_counter = self.min_holding_steps
            self.grasp_retry_count = 0
            log.info(f"✅ 抓取检查完成，进入保持状态 ({self.min_holding_steps}步)")

        return False, None, False

    def _detect_wave_pattern(self, current_value: float) -> str:
        """
        检测波形模式

        Args:
            current_value: 当前值

        Returns:
            str: 波形模式类型
        """
        self.waveform_history.append(current_value)

        if len(self.waveform_history) < 5:
            return 'stable'

        # 获取最近5个值
        recent = list(self.waveform_history)[-5:]
        mid_idx = 2
        mid_val = recent[mid_idx]

        # 波峰检测
        is_peak = (mid_val >= recent[mid_idx-1] and
                   mid_val >= recent[mid_idx+1] and
                   mid_val >= recent[mid_idx-2] and
                   mid_val >= recent[mid_idx+2] and
                   mid_val > self.peak_threshold)

        # 波谷检测
        is_valley = (mid_val <= recent[mid_idx-1] and
                     mid_val <= recent[mid_idx+1] and
                     mid_val <= recent[mid_idx-2] and
                     mid_val <= recent[mid_idx+2] and
                     mid_val < self.valley_threshold)

        if is_peak:
            self.peaks_detected.append((len(self.waveform_history), mid_val))
            if len(self.peaks_detected) > 10:
                self.peaks_detected.pop(0)
            return 'peak'
        elif is_valley:
            self.valleys_detected.append((len(self.waveform_history), mid_val))
            if len(self.valleys_detected) > 10:
                self.valleys_detected.pop(0)
            return 'valley'

        return 'stable'

    def _reset_close_detection(self):
        """重置闭合检测状态"""
        self.close_start_time = None
        self.close_value_counter = 0

    def reset(self):
        """重置策略状态"""
        super().reset()
        self._reset_close_detection()
        self.holding_counter = 0
        self.grasp_check_counter = 0
        self.grasp_retry_count = 0
        self.waveform_history.clear()
        self.peaks_detected = []
        self.valleys_detected = []
        log.info("🔄 抓放策略已重置")

    def get_debug_info(self) -> Dict[str, Any]:
        """获取抓放策略的调试信息"""
        info = super().get_debug_info()
        info.update({
            'holding_counter': self.holding_counter,
            'grasp_check_counter': self.grasp_check_counter,
            'peaks_detected': len(self.peaks_detected),
            'close_detection_active': self.close_start_time is not None
        })
        return info