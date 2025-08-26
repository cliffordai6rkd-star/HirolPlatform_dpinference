#!/usr/bin/env python3
"""学习推理工厂：统一创建和管理学习推理组件.

该模块提供学习推理引擎的工厂方法，支持多种学习算法的统一接口，
与HIROLRobotPlatform的工厂模式保持一致。
"""

import os
from typing import Dict, Any, Optional, Type

import glog as log
try: 
    from learning.inference.policy_inference import PolicyInference, ACTInference
    from learning.inference.data_adapter import (
    RobotDataAdapter, 
    FR3DataAdapter, 
    Monte01DataAdapter,
    create_data_adapter)
    import_ok = True
except:
    class MockLearningFactory:
        def create_inference_engine(cls, algorithm: str, ckpt_dir: str, 
                                    config: Dict[str, Any]):
            return None
        def create_data_adapter(cls, robot_type: str):
            return None
        def create_learning_pipeline(cls, algorithm: str, robot_type: str, 
                ckpt_dir: str,config: Dict[str, Any]):
            return None, None
        def _validate_config(cls, algorithm: str, config: Dict[str, Any]) -> None:
            return None
        def get_supported_algorithms(cls) -> list[str]:
            return None
        @classmethod
        def get_supported_robots(cls) -> list[str]:
            return None
        @classmethod
        def validate_checkpoint_directory(cls, ckpt_dir: str, algorithm: str) -> bool:
            return False
        
    LearningInferenceFactory = MockLearningFactory
    import_ok = False

if import_ok:
    class LearningInferenceFactory:
        """学习推理组件工厂.
        
        提供统一的学习推理引擎创建接口，支持多种学习算法和机器人平台的组合配置。
        遵循HIROLRobotPlatform的工厂设计模式。
        """
        
        # 支持的学习算法映射
        SUPPORTED_ALGORITHMS = {
            "ACT": ACTInference,
            "act": ACTInference,  # 小写别名
            # 未来可扩展: "PPO": PPOInference, "SAC": SACInference
        }
        
        # 支持的机器人类型
        SUPPORTED_ROBOTS = {
            "fr3": FR3DataAdapter,
            "monte01": Monte01DataAdapter,
            "generic": RobotDataAdapter,
        }
        
        @classmethod
        def create_inference_engine(
            cls,
            algorithm: str,
            ckpt_dir: str,
            config: Dict[str, Any]
        ) -> PolicyInference:
            """创建学习推理引擎实例.
            
            Args:
                algorithm: 学习算法名称 ("ACT", "PPO", etc.)
                ckpt_dir: 模型检查点目录路径
                config: 算法配置参数字典
                
            Returns:
                PolicyInference: 推理引擎实例
                
            Raises:
                ValueError: 当算法不支持或配置无效时抛出
                FileNotFoundError: 当检查点目录不存在时抛出
            """
            # 验证算法支持性
            if algorithm not in cls.SUPPORTED_ALGORITHMS:
                supported = list(cls.SUPPORTED_ALGORITHMS.keys())
                raise ValueError(f"不支持的学习算法: {algorithm}. 支持的算法: {supported}")
            
            # 验证检查点目录
            if not os.path.exists(ckpt_dir):
                raise FileNotFoundError(f"检查点目录不存在: {ckpt_dir}")
            
            # 验证必要的配置参数
            cls._validate_config(algorithm, config)
            
            # 创建推理引擎
            inference_class = cls.SUPPORTED_ALGORITHMS[algorithm]
            
            try:
                if algorithm.upper() == "ACT":
                    # ACT特定参数
                    state_dim = config.get("state_dim", 7)
                    camera_names = config.get("camera_names", ["front_camera"])
                    
                    # 【新增】自动检测Monte01双臂模式
                    robot_type = config.get("robot_type")
                    if robot_type == "monte01":
                        # Monte01固定为16 DOF双臂模式
                        state_dim = 16
                        camera_names = config.get("camera_names", ["left_ee_cam", "right_ee_cam", "third_person_cam"])
                        log.info(f"🤖 检测到Monte01双臂模式，自动设置为16 DOF，三相机配置")
                    elif robot_type == "fr3":
                        # FR3保持单臂7 DOF + 夹爪 = 8 DOF
                        state_dim = config.get("state_dim", 8) 
                        camera_names = config.get("camera_names", ["ee_cam", "third_person_cam"])
                        log.info(f"🤖 检测到FR3单臂模式，使用{state_dim} DOF配置")
                    
                    # 提取ACT参数
                    act_kwargs = {
                        key: config[key] for key in config
                        if key not in ["state_dim", "camera_names", "robot_type"]
                    }
                    
                    inference_engine = inference_class(
                        ckpt_dir=ckpt_dir,
                        state_dim=state_dim,
                        camera_names=camera_names,
                        **act_kwargs
                    )
                else:
                    # 其他算法的通用创建方式
                    inference_engine = inference_class(
                        ckpt_dir=ckpt_dir,
                        policy_config=config
                    )
                
                log.info(f"✅ 学习推理引擎创建成功: {algorithm}")
                log.info(f"   - 检查点目录: {ckpt_dir}")
                log.info(f"   - 配置参数: {len(config)} 个")
                
                return inference_engine
                
            except Exception as e:
                log.error(f"❌ 学习推理引擎创建失败: {algorithm}")
                log.error(f"   - 错误信息: {str(e)}")
                raise
        
        @classmethod
        def create_data_adapter(cls, robot_type: str) -> RobotDataAdapter:
            """创建对应机器人的数据适配器.
            
            Args:
                robot_type: 机器人类型 ("fr3", "monte01", etc.)
                
            Returns:
                RobotDataAdapter: 数据适配器实例
                
            Raises:
                ValueError: 当机器人类型不支持时抛出
            """
            try:
                adapter = create_data_adapter(robot_type)
                log.info(f"✅ 数据适配器创建成功: {robot_type}")
                return adapter
            except ValueError as e:
                supported = list(cls.SUPPORTED_ROBOTS.keys())
                log.error(f"❌ 不支持的机器人类型: {robot_type}")
                log.error(f"   - 支持的类型: {supported}")
                raise
        
        @classmethod
        def create_learning_pipeline(
            cls,
            algorithm: str,
            robot_type: str,
            ckpt_dir: str,
            config: Dict[str, Any]
        ) -> tuple[PolicyInference, RobotDataAdapter]:
            """创建完整的学习推理流水线.
            
            Args:
                algorithm: 学习算法名称
                robot_type: 机器人类型
                ckpt_dir: 检查点目录
                config: 算法配置
                
            Returns:
                tuple: (推理引擎, 数据适配器)
            """
            log.info(f"🚀 创建学习推理流水线: {algorithm} + {robot_type}")
            
            # 创建推理引擎
            inference_engine = cls.create_inference_engine(algorithm, ckpt_dir, config)
            
            # 创建数据适配器
            data_adapter = cls.create_data_adapter(robot_type)
            
            log.info(f"✅ 学习推理流水线创建完成")
            return inference_engine, data_adapter
        
        @classmethod
        def _validate_config(cls, algorithm: str, config: Dict[str, Any]) -> None:
            """验证算法配置参数.
            
            Args:
                algorithm: 算法名称
                config: 配置参数
                
            Raises:
                ValueError: 当配置参数无效时抛出
            """
            if not isinstance(config, dict):
                raise ValueError("配置参数必须是字典类型")
            
            if algorithm.upper() == "ACT":
                # ACT必需参数验证
                required_keys = ["state_dim", "camera_names"]
                missing_keys = [key for key in required_keys if key not in config]
                
                if missing_keys:
                    raise ValueError(f"ACT配置缺少必需参数: {missing_keys}")
                
                # 参数类型验证
                if not isinstance(config["state_dim"], int) or config["state_dim"] <= 0:
                    raise ValueError("state_dim必须是正整数")
                
                if not isinstance(config["camera_names"], list) or len(config["camera_names"]) == 0:
                    raise ValueError("camera_names必须是非空列表")
            
            # 可扩展其他算法的配置验证
        
        @classmethod
        def get_supported_algorithms(cls) -> list[str]:
            """获取支持的学习算法列表."""
            return list(cls.SUPPORTED_ALGORITHMS.keys())
        
        @classmethod
        def get_supported_robots(cls) -> list[str]:
            """获取支持的机器人类型列表.""" 
            return list(cls.SUPPORTED_ROBOTS.keys())
        
        @classmethod
        def validate_checkpoint_directory(cls, ckpt_dir: str, algorithm: str) -> bool:
            """验证检查点目录的完整性.
            
            Args:
                ckpt_dir: 检查点目录路径
                algorithm: 算法名称
                
            Returns:
                bool: 目录是否有效
            """
            if not os.path.exists(ckpt_dir):
                log.error(f"❌ 检查点目录不存在: {ckpt_dir}")
                return False
            
            if algorithm.upper() == "ACT":
                # ACT必需文件检查
                required_files = ["dataset_stats.pkl"]
                policy_files = ["policy_best.ckpt", "policy_best_bac.ckpt"]
                
                # 检查统计文件
                stats_exists = any(
                    os.path.exists(os.path.join(ckpt_dir, f))
                    for f in ["dataset_stats.pkl", "dataset_stats_bac.pkl"]
                )
                if not stats_exists:
                    log.error(f"❌ 找不到数据集统计文件在: {ckpt_dir}")
                    return False
                
                # 检查策略文件
                policy_exists = any(
                    os.path.exists(os.path.join(ckpt_dir, f))
                    for f in policy_files
                )
                if not policy_exists:
                    log.error(f"❌ 找不到策略模型文件在: {ckpt_dir}")
                    return False
            
            log.info(f"✅ 检查点目录验证通过: {ckpt_dir}")
            return True


    # 便捷的工厂函数
    def create_act_inference(
        ckpt_dir: str,
        state_dim: int,
        camera_names: list[str],
        robot_type: str = "generic",
        **kwargs
    ) -> tuple[ACTInference, RobotDataAdapter]:
        """便捷的ACT推理流水线创建函数.
        
        Args:
            ckpt_dir: ACT模型检查点目录
            state_dim: 状态维度
            camera_names: 相机名称列表
            robot_type: 机器人类型
            **kwargs: 其他ACT参数
            
        Returns:
            tuple: (ACT推理引擎, 数据适配器)
        """
        config = {
            "state_dim": state_dim,
            "camera_names": camera_names,
            **kwargs
        }
        
        return LearningInferenceFactory.create_learning_pipeline(
            algorithm="ACT",
            robot_type=robot_type,
            ckpt_dir=ckpt_dir,
            config=config
        )
        