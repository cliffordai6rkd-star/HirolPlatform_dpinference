#!/usr/bin/env python3
"""
任务类型定义和检测逻辑
提供统一的任务类型抽象，支持从checkpoint路径自动推断任务类型
"""

import re
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path
import glog as log


class TaskType(Enum):
    """任务类型枚举"""
    PEG_IN_HOLE = "peg_in_hole"
    BLOCK_STACKING = "block_stacking"
    LIQUID_TRANSFER = "liquid_transfer"
    SOLID_TRANSFER = "solid_transfer"
    PICK_AND_PLACE = "pick_and_place"
    INSERT_TUBE = "insert_tube"

    def __str__(self) -> str:
        return self.value

    @property
    def display_name(self) -> str:
        """获取任务类型的显示名称"""
        display_names = {
            TaskType.PEG_IN_HOLE: "插孔任务",
            TaskType.BLOCK_STACKING: "叠方块任务",
            TaskType.LIQUID_TRANSFER: "倒水任务",
            TaskType.SOLID_TRANSFER: "固体转移任务",
            TaskType.PICK_AND_PLACE: "抓放任务",
            TaskType.INSERT_TUBE: "插管任务"
        }
        return display_names[self]

    @staticmethod
    def from_checkpoint(checkpoint_path: str) -> 'TaskType':
        """
        根据checkpoint路径推断任务类型

        Args:
            checkpoint_path: checkpoint目录路径

        Returns:
            TaskType: 推断出的任务类型

        Raises:
            ValueError: 无法推断任务类型时抛出
        """
        if not checkpoint_path:
            raise ValueError("checkpoint_path cannot be empty")

        # 规范化路径
        path_str = str(checkpoint_path).lower()

        # 任务类型匹配模式
        patterns = {
            TaskType.PEG_IN_HOLE: [
                r'peg.*in.*hole',
                r'pih',
                r'plug.*in',
                r'insertion'
            ],
            TaskType.BLOCK_STACKING: [
                r'block.*stack',
                r'stack.*block',
                r'bs_',  # fr3_bs_xxx
                r'blockstack',
                r'lego.*tower',
                r'tower'
            ],
            TaskType.LIQUID_TRANSFER: [
                r'liquid.*transfer',
                r'pour.*liquid',
                r'water.*pour',
                r'liquid.*pour',
                r'pour.*water',
                r'water'  # 匹配包含water的路径
            ],
            TaskType.SOLID_TRANSFER: [
                r'solid.*transfer',
                r'object.*transfer',
                r'st_'  # fr3_st_xxx
            ],
            TaskType.PICK_AND_PLACE: [
                r'pick.*and.*place',
                r'pick.*place',
                r'pp_',  # fr3_pp_xxx
                r'pickplace',
                r'pickandplace'
            ],
            TaskType.INSERT_TUBE: [
                r'insert.*tube',
                r'tube.*insert',
                r'it_',  # fr3_it_xxx
                r'inserttube'
            ]
        }

        # 按优先级匹配（避免误判）
        for task_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, path_str):
                    log.info(f"🔍 检测到任务类型: {task_type.display_name} (匹配模式: '{pattern}')")
                    return task_type

        # 如果无法匹配，默认为抓放任务
        log.warning(f"⚠️ 无法从路径'{checkpoint_path}'推断任务类型，默认使用固体转移任务")
        return TaskType.SOLID_TRANSFER

    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'TaskType':
        """
        从配置中获取任务类型

        Args:
            config: 配置字典

        Returns:
            TaskType: 配置中指定的任务类型，若未指定则从checkpoint推断
        """
        # 优先使用显式配置的任务类型
        explicit_type = config.get("task_type")
        if explicit_type:
            try:
                return TaskType(explicit_type)
            except ValueError:
                log.warning(f"⚠️ 无效的任务类型配置: '{explicit_type}'")

        # 从checkpoint路径推断
        checkpoint_path = config.get("checkpoint_path", "")
        return TaskType.from_checkpoint(checkpoint_path)

    def get_config_file_name(self) -> str:
        """
        获取任务类型对应的配置文件名

        Returns:
            str: 配置文件名
        """
        return f"{self.value}_cfg.yaml"

    def get_config_path(self, base_dir: str = None) -> Path:
        """
        获取任务类型对应的配置文件路径

        Args:
            base_dir: 基础配置目录，默认使用标准路径

        Returns:
            Path: 配置文件完整路径
        """
        if base_dir is None:
            base_dir = "/workspace/factory/tasks/inferences_tasks/act/config"

        return Path(base_dir) / "tasks" / self.get_config_file_name()


class TaskConfigurationError(Exception):
    """任务配置错误异常"""
    pass


class TaskTypeFactory:
    """任务类型工厂，提供任务类型相关的工具方法"""

    @staticmethod
    def get_default_max_steps(task_type: TaskType) -> int:
        """
        获取任务类型的默认最大步数

        Args:
            task_type: 任务类型

        Returns:
            int: 默认最大步数
        """
        default_steps = {
            TaskType.PEG_IN_HOLE: 600,      # 插孔任务需要精确控制
            TaskType.BLOCK_STACKING: 550,   # 叠方块任务步数适中
            TaskType.LIQUID_TRANSFER: 800,  # 倒水任务需要更多步数
            TaskType.SOLID_TRANSFER: 500,   # 固体转移任务步数较少
            TaskType.PICK_AND_PLACE: 500,   # 抓放任务步数适中
            TaskType.INSERT_TUBE: 600       # 插管任务需要精确控制
        }
        return default_steps.get(task_type, 500)

    @staticmethod
    def get_task_characteristics(task_type: TaskType) -> Dict[str, Any]:
        """
        获取任务类型的特征参数

        Args:
            task_type: 任务类型

        Returns:
            Dict: 任务特征参数字典
        """
        characteristics = {
            TaskType.PEG_IN_HOLE: {
                "requires_precision": True,
                "gripper_control_type": "act_driven",
                "completion_detection": True,
                "force_open_after_completion": True
            },
            TaskType.BLOCK_STACKING: {
                "requires_precision": True,
                "gripper_control_type": "act_driven",
                "completion_detection": True,
                "force_open_after_completion": True
            },
            TaskType.LIQUID_TRANSFER: {
                "requires_precision": False,
                "gripper_control_type": "hold_after_close",
                "completion_detection": False,
                "force_open_after_completion": False
            },
            TaskType.SOLID_TRANSFER: {
                "requires_precision": False,
                "gripper_control_type": "always_closed",
                "completion_detection": False,
                "force_open_after_completion": False
            },
            TaskType.PICK_AND_PLACE: {
                "requires_precision": True,
                "gripper_control_type": "act_driven",
                "completion_detection": True,
                "force_open_after_completion": True
            },
            TaskType.INSERT_TUBE: {
                "requires_precision": True,
                "gripper_control_type": "act_driven",
                "completion_detection": True,
                "force_open_after_completion": True
            }
        }
        return characteristics.get(task_type, {})

    @staticmethod
    def validate_task_config(task_type: TaskType, config: Dict[str, Any]) -> bool:
        """
        验证任务配置的合理性

        Args:
            task_type: 任务类型
            config: 配置字典

        Returns:
            bool: 配置是否合理

        Raises:
            TaskConfigurationError: 配置不合理时抛出
        """
        characteristics = TaskTypeFactory.get_task_characteristics(task_type)

        # 检查必要的配置项
        if not config.get("checkpoint_path"):
            raise TaskConfigurationError(f"任务类型 {task_type.display_name} 缺少 checkpoint_path 配置")

        # 检查checkpoint路径是否存在
        checkpoint_path = Path(config["checkpoint_path"])
        if not checkpoint_path.exists():
            raise TaskConfigurationError(f"Checkpoint路径不存在: {checkpoint_path}")

        # 验证任务特定配置
        gripper_config = config.get("gripper_postprocess", {})
        if not gripper_config.get("enabled", True):
            log.warning(f"⚠️ 任务类型 {task_type.display_name} 的夹爪后处理被禁用")

        return True