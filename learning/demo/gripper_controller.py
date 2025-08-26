#!/usr/bin/env python3
"""
智能夹爪控制器
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any
import glog as log


class IncrementalGripperController:
    """Monte01增量式夹爪控制器 - 用于连续值控制"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化增量式夹爪控制器
        
        Args:
            config: 夹爪控制配置字典
        """
        # 简单的滤波参数
        self.smoothing_enabled = config.get('smoothing_enabled', True)
        self.smoothing_factor = config.get('smoothing_factor', 0.3)  # 低通滤波系数
        self.last_command = 0.0
        
        # 变化率限制
        self.max_change_per_step = config.get('max_change_per_step', 0.02)  # 每步最大变化 2cm
        
        # 死区设置
        self.dead_zone = config.get('dead_zone', 0.001)  # 1mm死区
        
        # 夹爪范围
        self.gripper_range = config.get('gripper_range', [0.0, 0.074])
        
        log.info("🔧 增量式夹爪控制器初始化完成")
        log.info(f"   - 平滑滤波: {'启用' if self.smoothing_enabled else '禁用'} (系数={self.smoothing_factor})")
        log.info(f"   - 变化率限制: {self.max_change_per_step*1000:.1f}mm/step")
        log.info(f"   - 死区: {self.dead_zone*1000:.1f}mm")
        log.info(f"   - 夹爪范围: {self.gripper_range[0]*1000:.1f}-{self.gripper_range[1]*1000:.1f}mm")
    
    def process(self, action_value: float, safety_failed: bool = False, 
                tool_interface=None, end_effector_pose: np.ndarray = None) -> Tuple[bool, Optional[float], bool]:
        """
        处理夹爪动作 - 增量式连续控制
        
        Args:
            action_value: 夹爪动作值 (规范化 0-1)
            safety_failed: 是否安全检查失败 (增量模式下忽略)
            tool_interface: 夹爪硬件接口 (增量模式下不需要)
            end_effector_pose: 末端位姿 (增量模式下不需要)
            
        Returns:
            Tuple[bool, Optional[float], bool]: 
                (是否需要执行命令, 命令值, 是否需要重新推理)
        """
        # 限制输入范围
        action_value = np.clip(action_value, 0.0, 1.0)
        
        # 检查死区
        if abs(action_value - self.last_command) < self.dead_zone:
            return False, None, False
        
        # 变化率限制
        max_change = self.max_change_per_step
        if abs(action_value - self.last_command) > max_change:
            if action_value > self.last_command:
                action_value = self.last_command + max_change
            else:
                action_value = self.last_command - max_change
        
        # 平滑滤波
        if self.smoothing_enabled and self.last_command > 0:
            action_value = (1 - self.smoothing_factor) * self.last_command + self.smoothing_factor * action_value
        
        # 最终范围限制
        action_value = np.clip(action_value, 0.0, 1.0)
        
        # 更新记录
        self.last_command = action_value
        
        # 增量模式始终执行命令，不需要重新推理
        return True, action_value, False
    
    def reset(self):
        """重置增量式夹爪控制器状态"""
        self.last_command = 0.0  # 重置为初始开启状态
        log.info("🔄 增量式夹爪控制器已重置")


class SmartGripperController:
    """基于趋势检测的智能夹爪控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化夹爪控制器
        
        Args:
            config: 夹爪控制配置字典
        """
        # 状态管理
        self.state = 'OPEN'  # OPEN, CLOSED, HOLDING, CHECKING
        self.holding_counter = 0
        self.action_history = deque(maxlen=config.get('trend_window_size', 5))
        
        # 位置稳定性检测
        self.joint_position_history = deque(maxlen=10)  # 存储最近10个关节位置
        self.stability_threshold = config.get('stability_threshold', 0.01)  # 位置变化阈值
        self.min_stable_steps = config.get('min_stable_steps', 5)  # 需要稳定的最小步数
        self.stable_counter = 0  # 稳定计数器
        
        # 趋势检测参数
        self.trend_window_size = config.get('trend_window_size', 5)
        self.closing_threshold = config.get('closing_trend_threshold', -0.001)
        self.opening_threshold = config.get('opening_trend_threshold', 0.001)
        
        # 动作阈值
        self.close_action_threshold = config.get('close_action_threshold', 0.065)
        self.open_action_threshold = config.get('open_action_threshold', 0.070)
        
        # 保持参数
        self.min_holding_steps = config.get('min_holding_steps', 30)
        
        # 安全重开机制
        self.safety_reopen_enabled = config.get('safety_reopen_enabled', True)
        self.max_safety_failures = config.get('max_safety_failures', 5)
        self.consecutive_safety_failures = 0
        
        # 抓取失败检测
        self.grasp_check_enabled = config.get('grasp_check_enabled', True)
        self.grasp_width_threshold = config.get('grasp_width_threshold', 0.001)  # 已弃用，保留向后兼容
        self.min_grasp_width = config.get('min_grasp_width', 0.005)  # 新增：最小抓取宽度
        self.max_grasp_width = config.get('max_grasp_width', 0.070)  # 新增：最大抓取宽度
        self.grasp_check_delay = config.get('grasp_check_delay', 20)  # 闭合后等待20步再检查
        self.grasp_check_counter = 0
        self.grasp_retry_count = 0
        self.max_grasp_retries = config.get('max_grasp_retries', 3)
        self.retry_wait_counter = 0  # 重试等待计数器
        self.retry_wait_time = config.get('retry_wait_time', 15)  # 重新打开后等待时间
        
        # 记录上一个状态用于检测变化
        self.last_state = 'OPEN'
        
        # 任务完成检测参数
        self.task_completion_enabled = config.get('task_completion_enabled', True)
        self.task_completion_stable_time = config.get('task_completion_stable_time', 20)
        self.task_completion_position_threshold = config.get('task_completion_position_threshold', 0.005)
        self.task_completion_max_holding = config.get('task_completion_max_holding', 150)
        self.task_completion_force_open = config.get('task_completion_force_open', True)
        self.task_stable_counter = 0  # 任务稳定计数器
        self.task_completion_triggered = False  # 是否已触发任务完成
        
        # 备用强制夹取机制
        self.force_grasp_enabled = config.get('force_grasp_enabled', True)
        self.force_grasp_stable_time = config.get('force_grasp_stable_time', 50)  # 50步后强制夹取
        self.force_grasp_triggered = False
        
        log.info("🔧 智能夹爪控制器初始化完成")
        self._log_config()
    
    def _log_config(self):
        """记录配置信息"""
        log.info(f"   - 初始状态: {self.state}")
        log.info(f"   - 趋势阈值: 闭合={self.closing_threshold}, 打开={self.opening_threshold}")
        log.info(f"   - 动作阈值: 闭合<{self.close_action_threshold}, 打开>{self.open_action_threshold}")
        log.info(f"   - 保持步数: {self.min_holding_steps}")
        if self.safety_reopen_enabled:
            log.info(f"   - 安全重开: 启用 (最大失败={self.max_safety_failures})")
        if self.grasp_check_enabled:
            log.info(f"   - 抓取检测: 启用 (宽度阈值={self.grasp_width_threshold}m, 重试={self.max_grasp_retries}次)")
        if self.task_completion_enabled:
            log.info(f"   - 任务完成检测: 启用 (稳定时间={self.task_completion_stable_time}步, 最大保持={self.task_completion_max_holding}步)")
    
    def get_gripper_width(self, tool_interface) -> float:
        """
        获取夹爪实际宽度
        
        Args:
            tool_interface: 夹爪硬件接口
            
        Returns:
            float: 夹爪宽度（米）
        """
        if tool_interface and hasattr(tool_interface, 'get_tool_state'):
            try:
                gripper_state = tool_interface.get_tool_state()
                # 正确读取夹爪宽度：优先_position属性
                if hasattr(gripper_state, '_position'):
                    width = float(gripper_state._position)
                    log.debug(f"🔍 读取到夹爪宽度: {width:.6f}m (来源: _position)")
                    return width
                else:
                    raise AttributeError("_position属性不存在")
            except Exception as e:
                log.debug(f"无法获取夹爪宽度: {e}")
        return 0.08  # 返回默认值（全开）
    
    def check_grasp_success(self, tool_interface) -> bool:
        """
        检查是否成功抓取物体
        
        Args:
            tool_interface: 夹爪硬件接口
            
        Returns:
            bool: 是否成功抓取
        """
        width = self.get_gripper_width(tool_interface)
        
        # 修正的抓取成功检查逻辑：使用配置化的范围检查
        # - 夹爪宽度应该在合理范围内（既不能完全关闭，也不能完全打开）
        min_grasp_width = getattr(self, 'min_grasp_width', 0.005)  # 使用实例属性或默认值5mm
        max_grasp_width = getattr(self, 'max_grasp_width', 0.070)  # 使用实例属性或默认值70mm
        
        success = min_grasp_width <= width <= max_grasp_width
        
        if not success:
            if width < min_grasp_width:
                log.warning(f"⚠️ 抓取失败！夹爪过紧，宽度={width:.4f}m < 最小阈值={min_grasp_width:.4f}m（可能夹空或损坏）")
            elif width > max_grasp_width:
                log.warning(f"⚠️ 抓取失败！夹爪过松，宽度={width:.4f}m > 最大阈值={max_grasp_width:.4f}m（完全打开，未夹住物体）")
        else:
            log.info(f"✅ 抓取成功！夹爪宽度={width:.4f}m 在合理范围内 [{min_grasp_width:.3f}m - {max_grasp_width:.3f}m]")
            
        return success
    
    def check_stability(self, end_effector_pose: np.ndarray = None) -> bool:
        """
        检查机器人末端是否稳定（基于末端位置变化）
        
        Args:
            end_effector_pose: 当前末端位姿 [x, y, z, rx, ry, rz]
            
        Returns:
            bool: 是否稳定
        """
        if end_effector_pose is not None:
            try:
                # 确保是 numpy 数组
                if not isinstance(end_effector_pose, np.ndarray):
                    end_effector_pose = np.array(end_effector_pose)
                
                # 只使用位置信息 [x, y, z]
                if len(end_effector_pose) >= 3:
                    pose = end_effector_pose[:3].copy()
                    self.joint_position_history.append(pose)
            except Exception as e:
                log.debug(f"末端位置稳定性检测跳过: {e}")
                pass
        
        if len(self.joint_position_history) < 2:
            return False
        
        # 计算末端位置变化
        recent_positions = list(self.joint_position_history)
        if len(recent_positions) >= 2:
            # 计算末端位置变化（米）
            pos_diff = np.linalg.norm(recent_positions[-1] - recent_positions[-2])
            
            # 末端位置稳定阈值更严格（1cm）
            pose_threshold = 0.003
            
            # 如果位置变化很小，增加稳定计数
            if pos_diff < pose_threshold:
                self.stable_counter += 1
            elif pos_diff < pose_threshold * 2:  # 小幅晃动，不重置计数
                pass  # 保持当前计数
            else:
                self.stable_counter = max(0, self.stable_counter - 1)  # 缓慢减少而不是立即重置
            
            # 需要连续稳定多步才认为稳定
            is_stable = self.stable_counter >= self.min_stable_steps
            
            # 调试信息
            if self.stable_counter > 0 and self.stable_counter % 10 == 0:
                log.debug(f"🎯 末端位置稳定性: 计数={self.stable_counter}/{self.min_stable_steps}, 变化={pos_diff:.5f}m (阈值={pose_threshold:.3f}m)")
            
            return is_stable
        
        return False
    
    def check_task_completion(self, end_effector_pose: np.ndarray = None) -> bool:
        """
        检测任务是否完成（末端稳定在目标位置）
        
        Args:
            end_effector_pose: 当前末端位姿
            
        Returns:
            bool: 任务是否完成
        """
        if not self.task_completion_enabled:
            return False
        
        # 只在HOLDING状态检测
        if self.state != 'HOLDING':
            self.task_stable_counter = 0
            return False
        
        # 初始化实际保持时间计数器
        if not hasattr(self, 'actual_holding_time'):
            self.actual_holding_time = 0
            
        # 在HOLDING状态时增加实际保持时间计数
        if self.state == 'HOLDING':
            self.actual_holding_time += 1
            
        # 智能处理安全检查与任务完成检测的冲突
        safety_interference = hasattr(self, 'recent_safety_failed') and self.recent_safety_failed
        
        if safety_interference:
            # 记录连续安全失败次数
            if not hasattr(self, 'consecutive_safety_failures_count'):
                self.consecutive_safety_failures_count = 0
            self.consecutive_safety_failures_count += 1
            
            # 如果连续安全失败但保持时间较长，逐步忽略安全检查
            if self.actual_holding_time >= self.task_completion_max_holding * 0.5:  # 50%时间后开始忽略
                log.debug(f"🛡️ 保持时间较长({self.actual_holding_time}步)，开始忽略部分安全检查进行任务完成检测")
                # 不直接返回，继续执行后续检测
            elif self.consecutive_safety_failures_count >= 50:  # 连续失败50次后也开始忽略
                log.debug(f"🛡️ 连续安全失败{self.consecutive_safety_failures_count}次，开始忽略安全检查")
                # 不直接返回，继续执行后续检测  
            else:
                # 早期阶段仍然受安全检查影响，但不完全阻塞
                if self.task_stable_counter > 0:
                    self.task_stable_counter = max(0, self.task_stable_counter - 1)  # 缓慢递减而非重置
                    log.debug(f"🚫 早期安全失败，稳定计数缓慢递减: {self.task_stable_counter}")
                return False
        else:
            # 安全检查通过，重置连续失败计数
            if hasattr(self, 'consecutive_safety_failures_count'):
                self.consecutive_safety_failures_count = 0
        
        # 如果已经触发过任务完成，不再重复检测
        if self.task_completion_triggered:
            return False
        
        # 使用稳定性检测的历史数据
        if len(self.joint_position_history) < 2:
            return False
        
        # 计算最近位置的稳定性
        recent_positions = list(self.joint_position_history)
        if len(recent_positions) >= self.task_completion_stable_time:
            # 检查最近N步的位置变化
            positions_to_check = recent_positions[-self.task_completion_stable_time:]
            max_deviation = 0.0
            
            for i in range(1, len(positions_to_check)):
                deviation = np.linalg.norm(positions_to_check[i] - positions_to_check[i-1])
                max_deviation = max(max_deviation, deviation)
            
            # 如果所有位置变化都在阈倿内，认为任务完成
            if max_deviation < self.task_completion_position_threshold:
                self.task_stable_counter += 1
                
                # 定期输出调试信息
                if self.task_stable_counter % 5 == 0:
                    log.debug(f"🎯 任务完成检测: 稳定计数={self.task_stable_counter}/{self.task_completion_stable_time}, 最大偏差={max_deviation:.5f}m（阈值={self.task_completion_position_threshold}m）")
                
                if self.task_stable_counter >= self.task_completion_stable_time:
                    log.info(f"✅ 检测到任务完成！末端稳定{self.task_stable_counter}步，最大偏差={max_deviation:.5f}m")
                    self.task_completion_triggered = True
                    return True
            else:
                self.task_stable_counter = 0
                log.debug(f"📊 任务完成检测: 稳定计数重置，最大偏差={max_deviation:.5f}m > 阈值={self.task_completion_position_threshold}m")
        
        # 分层任务完成检测机制
        
        # 第一层：标准位置稳定检测（主要机制）
        if len(recent_positions) >= self.task_completion_stable_time:
            # 检查最近N步的位置变化
            positions_to_check = recent_positions[-self.task_completion_stable_time:]
            max_deviation = 0.0
            
            for i in range(1, len(positions_to_check)):
                deviation = np.linalg.norm(positions_to_check[i] - positions_to_check[i-1])
                max_deviation = max(max_deviation, deviation)
            
            # 如果所有位置变化都在阈值内，认为任务完成
            if max_deviation < self.task_completion_position_threshold:
                self.task_stable_counter += 1
                
                # 定期输出调试信息
                if self.task_stable_counter % 5 == 0:
                    log.debug(f"🎯 第一层检测: 稳定计数={self.task_stable_counter}/{self.task_completion_stable_time}, 最大偏差={max_deviation:.5f}m（阈值={self.task_completion_position_threshold}m）")
                
                if self.task_stable_counter >= self.task_completion_stable_time:
                    log.info(f"✅ 第一层检测成功：位置稳定{self.task_stable_counter}步，最大偏差={max_deviation:.5f}m")
                    self.task_completion_triggered = True
                    return True
            else:
                self.task_stable_counter = 0
                log.debug(f"📊 第一层检测: 稳定计数重置，最大偏差={max_deviation:.5f}m > 阈值={self.task_completion_position_threshold}m")
        
        # 第二层：运动状态检测（次级机制）- 基于连续无安全失败
        if not hasattr(self, 'stable_motion_counter'):
            self.stable_motion_counter = 0
            
        # 如果连续一段时间没有安全失败，认为运动趋于稳定
        if not safety_interference:
            self.stable_motion_counter += 1
        else:
            self.stable_motion_counter = 0
            
        # 连续稳定运动达到阈值且保持时间足够长
        stable_motion_threshold = 30  # 连续30步无安全失败
        min_holding_for_motion_detect = self.task_completion_max_holding * 1.5  # 需要保持更长时间
        
        if (self.stable_motion_counter >= stable_motion_threshold and 
            self.actual_holding_time >= min_holding_for_motion_detect):
            log.info(f"✅ 第二层检测成功：运动稳定{self.stable_motion_counter}步，保持时间{self.actual_holding_time}步")
            self.task_completion_triggered = True
            return True
            
        # 第三层：超时保护机制（兜底机制）
        max_total_holding_time = self.task_completion_max_holding * 4  # 4倍时间作为绝对上限(约6秒)
        if self.actual_holding_time >= max_total_holding_time:
            log.warning(f"⚠️ 第三层检测：超时保护触发！保持时间{self.actual_holding_time}步 >= 上限{max_total_holding_time}步，强制完成任务")
            self.task_completion_triggered = True
            return True
        
        return False
    
    def detect_trend(self, current_value: float) -> str:
        """
        检测夹爪动作趋势
        
        Args:
            current_value: 当前夹爪动作值
            
        Returns:
            str: 'closing'/'opening'/'stable'
        """
        self.action_history.append(current_value)
        
        if len(self.action_history) < 3:
            return 'stable'
        
        # 计算移动平均变化率
        values = list(self.action_history)
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        avg_diff = np.mean(diffs)
        
        # 更严格的趋势判断
        if avg_diff < self.closing_threshold and abs(avg_diff) > 0.0005:  # 避免噪声
            return 'closing'
        elif avg_diff > self.opening_threshold and abs(avg_diff) > 0.0005:  # 避免噪声
            return 'opening'
        else:
            return 'stable'
    
    def should_close(self, action_value: float, trend: str, is_stable: bool = False) -> bool:
        """
        判断是否应该闭合夹爪
        
        Args:
            action_value: 当前动作值
            trend: 当前趋势
            is_stable: 机器人是否稳定
            
        Returns:
            bool: 是否应该闭合
        """
        if self.state != 'OPEN':
            return False
        
        # 如果在重试等待期间，不闭合
        if self.retry_wait_counter > 0:
            self.retry_wait_counter -= 1
            if self.retry_wait_counter % 5 == 0:
                log.debug(f"⏳ 重试等待中...剩余{self.retry_wait_counter}步")
            return False
            
        if len(self.action_history) < 5:  # 需要合理的历史数据
            return False
            
        # 额外检查：如果夹爪值明显过高，不闭合
        if action_value > self.close_action_threshold + 0.010:  # 超出阈值10个单位
            if len(self.action_history) % 50 == 0:
                log.debug(f"🚫 夹爪值过高，不闭合: {action_value:.4f} > {self.close_action_threshold + 0.010:.3f}")
            return False
            
        recent_values = list(self.action_history)[-5:]
        avg_recent = np.mean(recent_values)
        
        # 主要闭合条件：动作值低于阈值 + 关节稳定
        joint_stable_required = 60  # 3秒 * 20Hz = 60步
        if (avg_recent < self.close_action_threshold and 
            self.stable_counter >= joint_stable_required and
            trend != 'opening'):
            log.info(f"🤏 检测到闭合信号（阈值+稳定）: 值={avg_recent:.4f} < 阈值={self.close_action_threshold}, 稳定={self.stable_counter}步, 趋势={trend}")
            return True
        
        # 备用机制：智能强制夹取（但需要确保安全且合理）
        if (self.force_grasp_enabled and not self.force_grasp_triggered and 
            self.stable_counter >= self.force_grasp_stable_time and
            not (hasattr(self, 'recent_safety_failed') and self.recent_safety_failed) and
            action_value < self.close_action_threshold + 0.005):  # 只有接近阈值才强制夹取
            log.warning(f"⚠️ 机器人安全稳定{self.stable_counter}步且夹爪值接近阈值，强制尝试夹取！当前值={action_value:.4f} < {self.close_action_threshold + 0.005:.3f}")
            self.force_grasp_triggered = True
            return True
        
        # 方法2：相对检测 - 检测从高值下降
        if len(self.action_history) >= 10:
            older_values = list(self.action_history)[-10:-5]
            avg_older = np.mean(older_values)
            
            # 如果值下降了超过0.003，认为是闭合信号
            if avg_older - avg_recent > 0.003 and trend in ['closing', 'stable']:
                log.info(f"🤏 检测到闭合信号（稳定后相对下降）: 从{avg_older:.4f}降到{avg_recent:.4f}")
                return True
                
        return False
    
    def should_open(self, action_value: float, trend: str) -> bool:
        """
        判断是否应该打开夹爪
        
        Args:
            action_value: 当前动作值
            trend: 当前趋势
            
        Returns:
            bool: 是否应该打开
        """
        if self.state != 'HOLDING':
            return False
        
        # 保持计数器必须归零
        if self.holding_counter > 0:
            return False
        
        # 需要明确的打开信号
        should_open = trend == 'opening' and action_value > self.open_action_threshold
        if should_open:
            log.debug(f"✋ 准备打开: 值={action_value:.4f}, 趋势={trend}")
        return should_open
    
    def handle_safety_failure(self, tool_interface) -> bool:
        """
        处理安全失败情况
        
        Args:
            tool_interface: 夹爪硬件接口
            
        Returns:
            bool: 是否触发了安全重开
        """
        if not self.safety_reopen_enabled:
            return False
        
        self.consecutive_safety_failures += 1
        
        if self.consecutive_safety_failures >= self.max_safety_failures:
            if self.state in ['CLOSED', 'HOLDING']:
                log.warning(f"⚠️ 连续{self.consecutive_safety_failures}次安全失败，强制打开夹爪！")
                self.state = 'OPEN'
                self.consecutive_safety_failures = 0
                
                # 执行打开命令
                if tool_interface:
                    tool_interface._set_binary_command(1.0)
                return True
        return False
    
    def process(self, action_value: float, safety_failed: bool = False, 
                tool_interface=None, end_effector_pose: np.ndarray = None) -> Tuple[bool, Optional[float], bool]:
        """
        处理夹爪动作
        
        Args:
            action_value: 夹爪动作值
            safety_failed: 是否安全检查失败
            tool_interface: 夹爪硬件接口
            end_effector_pose: 末端位姿（用于稳定性检测）
            
        Returns:
            Tuple[bool, Optional[float], bool]: 
                (是否需要执行命令, 命令值, 是否需要重新推理)
        """
        # 记录安全状态用于稳定性检测
        self.recent_safety_failed = safety_failed
        
        # 处理安全失败
        if safety_failed and self.handle_safety_failure(tool_interface):
            return True, 1.0, True
        elif not safety_failed:
            self.consecutive_safety_failures = 0
        
        # 检测稳定性
        is_stable = self.check_stability(end_effector_pose)
        
        # 检测趋势
        trend = self.detect_trend(action_value)
        
        # 每50步输出一次调试信息
        if len(self.action_history) > 0 and len(self.action_history) % 50 == 0:
            recent = list(self.action_history)[-5:] if len(self.action_history) >= 5 else list(self.action_history)
            log.debug(f"📊 夹爪调试: 状态={self.state}, 稳定={is_stable}({self.stable_counter}步), 当前值={action_value:.4f}, 趋势={trend}, 最近5值={[f'{v:.4f}' for v in recent]}")
        
        execute_command = False
        command_value = None
        force_repredict = False
        
        # 状态机逻辑
        if self.state == 'OPEN':
            if self.should_close(action_value, trend, is_stable):
                self.state = 'CLOSED'
                command_value = 0.0
                execute_command = True
                force_repredict = True
                log.info(f"🤏 闭合夹爪！值={action_value:.4f}, 趋势={trend}")
                
        elif self.state == 'CLOSED':
            # 进入检查状态
            if self.grasp_check_enabled:
                self.state = 'CHECKING'
                self.grasp_check_counter = self.grasp_check_delay
                log.info(f"🔍 进入检查状态，等待{self.grasp_check_delay}步后检查抓取")
            else:
                # 如果禁用检查，直接进入保持状态
                self.state = 'HOLDING'
                self.holding_counter = self.min_holding_steps
                log.info(f"🔒 进入保持状态 ({self.min_holding_steps}步)")
        
        elif self.state == 'CHECKING':
            # 等待一段时间让夹爪稳定
            if self.grasp_check_counter > 0:
                self.grasp_check_counter -= 1
            else:
                # 检查是否成功抓取
                if self.check_grasp_success(tool_interface):
                    # 成功，进入保持状态
                    self.state = 'HOLDING'
                    self.holding_counter = self.min_holding_steps
                    self.grasp_retry_count = 0  # 重置重试计数
                    log.info(f"✅ 抓取成功！进入保持状态，保持{self.min_holding_steps}步，任务完成检测已启动")
                else:
                    # 失败，重新打开夹爪
                    self.grasp_retry_count += 1
                    self.state = 'OPEN'
                    command_value = 1.0
                    execute_command = True
                    force_repredict = True
                    self.retry_wait_counter = self.retry_wait_time  # 设置等待时间
                    self.stable_counter = 0  # 重置稳定计数
                    log.warning(f"❌ 抓取失败！重新打开夹爪（重试 {self.grasp_retry_count}/{self.max_grasp_retries}），等待{self.retry_wait_time}步")
            
        elif self.state == 'HOLDING':
            # 倒计时
            if self.holding_counter > 0:
                self.holding_counter -= 1
                if self.holding_counter % 20 == 0 and self.holding_counter > 0:
                    log.debug(f"🔒 保持中 (剩余{self.holding_counter}步), 任务完成检测: {self.task_stable_counter}/{self.task_completion_stable_time}")
                elif self.holding_counter == 0:
                    log.info(f"✅ 保持时间结束（{self.min_holding_steps}步），开始检测任务完成或打开信号")
            
            # 检查任务是否完成
            task_completed = self.check_task_completion(end_effector_pose)
            if task_completed:
                if self.task_completion_force_open:
                    self.state = 'OPEN'
                    command_value = 1.0
                    execute_command = True
                    force_repredict = True
                    log.info(f"🎯 任务完成！自动打开夹爪")
                else:
                    log.info(f"🎯 任务完成检测触发，但未启用强制打开")
            # 检查是否应该打开（基于动作值）
            elif self.should_open(action_value, trend):
                self.state = 'OPEN'
                command_value = 1.0
                execute_command = True
                force_repredict = True
                log.info(f"✋ 打开夹爪！值={action_value:.4f}, 趋势={trend}")
        
        # 记录状态变化并重置计数器
        if self.state != self.last_state:
            log.info(f"🔄 夹爪状态: {self.last_state} → {self.state}")
            self.last_state = self.state
            # 状态变化时重置相关计数器
            if self.state != 'HOLDING':
                if hasattr(self, 'actual_holding_time'):
                    self.actual_holding_time = 0
                if hasattr(self, 'stable_motion_counter'):
                    self.stable_motion_counter = 0
        
        # 执行命令
        if execute_command and command_value is not None and tool_interface:
            success = tool_interface._set_binary_command(command_value)
            if not success:
                log.warning("⚠️ 夹爪命令执行失败")
                return False, None, force_repredict
        
        return execute_command, command_value, force_repredict
    
    def get_position(self) -> float:
        """
        获取夹爪位置值（用于状态向量）
        
        Returns:
            float: 0.0（闭合）或 0.08（打开）
        """
        return 0.0 if self.state in ['CLOSED', 'CHECKING', 'HOLDING'] else 0.08
    
    def reset(self):
        """重置控制器状态"""
        self.state = 'OPEN'
        self.last_state = 'OPEN'
        self.holding_counter = 0
        self.action_history.clear()
        self.joint_position_history.clear()
        self.stable_counter = 0
        self.consecutive_safety_failures = 0
        self.grasp_check_counter = 0
        self.grasp_retry_count = 0
        self.retry_wait_counter = 0
        self.task_stable_counter = 0
        self.task_completion_triggered = False
        self.force_grasp_triggered = False
        log.info("🔄 夹爪控制器已重置")