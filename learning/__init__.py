"""学习模块：提供机器人学习推理功能.

该模块集成了多种机器人学习算法，包括ACT、PPO等，
为HIROLRobotPlatform提供统一的学习推理接口。

主要组件:
    - inference: 通用推理接口和具体算法实现
    - policies: 学习策略模型定义
    - models: 深度学习模型架构
    - config: 学习组件配置文件
"""

__version__ = "0.1.0"
__author__ = "HIROLRobotPlatform Team"