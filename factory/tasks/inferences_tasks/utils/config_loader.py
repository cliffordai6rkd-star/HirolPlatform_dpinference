#!/usr/bin/env python3
"""
配置加载工具模块
支持YAML配置文件的加载、深度合并和任务特定配置覆盖
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import glog as log
from copy import deepcopy


class ConfigLoader:
    """配置加载器，支持任务特定配置的优先级合并"""

    def __init__(self, base_config_dir: str = None):
        """
        初始化配置加载器

        Args:
            base_config_dir: 基础配置目录路径
        """
        self.base_config_dir = base_config_dir or "/workspace/factory/tasks/inferences_tasks/act/config"

    def load_yaml_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载YAML配置文件，支持!include指令

        Args:
            config_path: 配置文件路径

        Returns:
            Dict: 加载的配置字典

        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 处理!include指令
        content = self._process_include_directives(content, config_path.parent)

        try:
            config = yaml.safe_load(content)
            log.debug(f"✅ 成功加载配置文件: {config_path}")
            return config or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML解析错误 ({config_path}): {e}")

    def _process_include_directives(self, content: str, base_dir: Path) -> str:
        """
        处理YAML中的!include指令

        Args:
            content: YAML文件内容
            base_dir: 基础目录路径

        Returns:
            str: 处理后的YAML内容
        """
        import re

        def replace_include(match):
            include_path = match.group(1).strip()
            full_path = base_dir / include_path

            if not full_path.exists():
                log.warning(f"⚠️ Include文件不存在: {full_path}")
                return "{}"

            with open(full_path, 'r', encoding='utf-8') as f:
                include_content = f.read()

            # 递归处理嵌套的!include
            return self._process_include_directives(include_content, full_path.parent)

        # 匹配!include指令
        pattern = r'!include\s+([^\n\r]+)'
        return re.sub(pattern, replace_include, content)

    def deep_merge_configs(self, base_config: Dict[str, Any],
                          override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个配置字典，override_config优先

        Args:
            base_config: 基础配置字典
            override_config: 覆盖配置字典

        Returns:
            Dict: 合并后的配置字典
        """
        merged = deepcopy(base_config)

        def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """递归深度合并字典"""
            for key, value in source.items():
                if (key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)):
                    # 递归合并嵌套字典
                    _deep_merge(target[key], value)
                else:
                    # 直接覆盖值
                    target[key] = deepcopy(value)

        _deep_merge(merged, override_config)
        return merged

    def load_task_specific_config(self, task_type: str) -> Optional[Dict[str, Any]]:
        """
        加载任务特定的配置文件

        Args:
            task_type: 任务类型字符串

        Returns:
            Dict: 任务特定配置，如果文件不存在返回None
        """
        config_file = f"{task_type}_cfg.yaml"
        config_path = Path(self.base_config_dir) / "tasks" / config_file

        if not config_path.exists():
            log.debug(f"📝 任务特定配置不存在: {config_path}")
            return None

        try:
            config = self.load_yaml_config(config_path)
            log.info(f"✅ 成功加载任务特定配置: {config_file}")
            return config
        except Exception as e:
            log.warning(f"⚠️ 加载任务特定配置失败 ({config_file}): {e}")
            return None

    def merge_with_task_config(self, base_config: Dict[str, Any],
                              task_type: str) -> Dict[str, Any]:
        """
        将基础配置与任务特定配置合并

        Args:
            base_config: 基础配置字典
            task_type: 任务类型

        Returns:
            Dict: 合并后的配置字典
        """
        # 加载任务特定配置
        task_config = self.load_task_specific_config(task_type)

        if not task_config:
            log.debug(f"📋 使用基础配置 (无任务特定配置): {task_type}")
            return deepcopy(base_config)

        # 深度合并配置
        merged_config = self.deep_merge_configs(base_config, task_config)

        # 记录关键覆盖参数
        self._log_config_overrides(base_config, task_config, task_type)

        return merged_config

    def _log_config_overrides(self, base_config: Dict[str, Any],
                             task_config: Dict[str, Any],
                             task_type: str) -> None:
        """
        记录配置覆盖信息

        Args:
            base_config: 基础配置
            task_config: 任务配置
            task_type: 任务类型
        """
        overrides = []

        # 检查常见的覆盖参数
        common_params = [
            'max_step_nums',
            ('gripper_postprocess', 'target_peak_count'),
            ('gripper_postprocess', 'control_mode'),
            ('gripper_postprocess', 'grasp_check_enabled'),
        ]

        for param in common_params:
            if isinstance(param, tuple):
                # 嵌套参数
                section, key = param
                base_value = base_config.get(section, {}).get(key)
                task_value = task_config.get(section, {}).get(key)
                param_name = f"{section}.{key}"
            else:
                # 顶级参数
                base_value = base_config.get(param)
                task_value = task_config.get(param)
                param_name = param

            if task_value is not None and task_value != base_value:
                overrides.append(f"{param_name}: {base_value} → {task_value}")

        if overrides:
            log.info(f"🔧 任务配置覆盖 ({task_type}):")
            for override in overrides:
                log.info(f"   - {override}")


# 全局配置加载器实例
_config_loader = ConfigLoader()


def load_merged_config(base_config_path: Union[str, Path],
                      task_type: str) -> Dict[str, Any]:
    """
    加载基础配置并与任务特定配置合并

    Args:
        base_config_path: 基础配置文件路径
        task_type: 任务类型

    Returns:
        Dict: 合并后的配置字典

    Raises:
        FileNotFoundError: 基础配置文件不存在
        yaml.YAMLError: 配置解析错误
    """
    # 加载基础配置
    base_config = _config_loader.load_yaml_config(base_config_path)

    # 与任务特定配置合并
    return _config_loader.merge_with_task_config(base_config, task_type)


def get_task_config_path(task_type: str) -> Optional[Path]:
    """
    获取任务特定配置文件路径

    Args:
        task_type: 任务类型

    Returns:
        Path: 配置文件路径，不存在则返回None
    """
    config_file = f"{task_type}_cfg.yaml"
    config_path = Path(_config_loader.base_config_dir) / "tasks" / config_file

    return config_path if config_path.exists() else None


def validate_config_structure(config: Dict[str, Any]) -> bool:
    """
    验证配置结构的完整性

    Args:
        config: 配置字典

    Returns:
        bool: 配置结构是否有效

    Raises:
        ValueError: 配置结构无效
    """
    required_keys = ['checkpoint_path', 'robot_type']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置缺少必需键: {key}")

    # 验证checkpoint路径
    checkpoint_path = Path(config['checkpoint_path'])
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint路径不存在: {checkpoint_path}")

    log.debug("✅ 配置结构验证通过")
    return True