#!/usr/bin/env python3
"""
机器人测试工厂 - 简化版
为不同类型的机器人创建统一的测试适配器
"""

from typing import Dict, Any
import glog as log
from robot_api_adapters import RobotTestInterface, Xarm7Adapter, Monte01Adapter, MockAdapter


class RobotTestFactory:
    """简化的机器人测试工厂"""
    
    @staticmethod
    def create_adapter(robot_type: str, config: Dict[str, Any] = None) -> RobotTestInterface:
        """
        创建机器人适配器
        
        Args:
            robot_type: 'xarm7', 'monte01_left', 'monte01_right', 'mock'
            config: 配置字典，为None时使用默认配置
            
        Returns:
            RobotTestInterface: 适配器实例（自动降级到Mock）
        """
        
        # 使用默认配置
        if config is None:
            config = RobotTestFactory._get_default_config(robot_type)
        
        # 创建适配器（自动降级到Mock）
        try:
            if robot_type == 'xarm7' or robot_type == 'xarm7_left':
                # 为左臂添加side配置
                if robot_type == 'xarm7_left':
                    config['side'] = 'left'
                return Xarm7Adapter(config)
            elif robot_type == 'xarm7_right':
                # 为右臂添加side配置
                config['side'] = 'right'
                return Xarm7Adapter(config)
            elif robot_type == 'monte01_left':
                return Monte01Adapter(config, 'left')
            elif robot_type == 'monte01_right':
                return Monte01Adapter(config, 'right')
            elif robot_type == 'mock':
                return MockAdapter()
            else:
                raise ValueError(f"Unsupported robot type: {robot_type}")
                
        except Exception as e:
            log.warning(f"Hardware adapter failed for {robot_type}: {e}")
            log.info(f"Using mock adapter for {robot_type}")
            return MockAdapter(robot_type)
    
    @staticmethod
    def _get_default_config(robot_type: str) -> Dict[str, Any]:
        """获取默认配置"""
        if robot_type == 'xarm7' or robot_type == 'xarm7_left':
            return {
                "ip": "192.168.11.11",
                "dof": 7,
                "init_joint_positions": [0.0] * 7
            }
        elif robot_type == 'xarm7_right':
            return {
                "ip": "192.168.11.12",
                "dof": 7,
                "init_joint_positions": [0.0] * 7
            }
        elif robot_type.startswith('monte01'):
            return {
                "ip": "192.168.11.3:50051",
                "dof": [7, 7],
                "control_body": False,
                "control_chassis": False,
                "comm_freq": 500.0
            }
        else:
            return {}
    
    @staticmethod
    def create_dual_apis(api_a_type: str, api_b_type: str,
                        config_a: Dict[str, Any] = None,
                        config_b: Dict[str, Any] = None) -> tuple[RobotTestInterface, RobotTestInterface]:
        """
        创建一对API用于转换关系测试
        
        Args:
            api_a_type: API A的类型
            api_b_type: API B的类型
            config_a: API A的配置
            config_b: API B的配置
            
        Returns:
            tuple: (api_a, api_b)
        """
        
        api_a = RobotTestFactory.create_adapter(api_a_type, config_a)
        api_b = RobotTestFactory.create_adapter(api_b_type, config_b)
        
        log.info(f"Created dual APIs: {api_a_type} -> {api_b_type}")
        
        return api_a, api_b