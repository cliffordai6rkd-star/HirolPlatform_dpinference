import abc
import numpy as np
from hardware.base.utils import RobotJointState
from hardware.base.safety_checker import SafetyChecker, SafetyLevel, SafetyLimits
import threading
import copy
from typing import Dict, Any, Optional, Tuple, Union, List
import glog as log
class ArmBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._dof = config["dof"]
        self._init_joint_positions = config.get("init_joint_positions", None)
        self._joint_states = RobotJointState()
        # self._tcp_pose = np.zeros(7) # [x, y, z, qx, qy, qz, qw]
        self._lock = threading.Lock()
        self._is_initialized = False
        
        # Initialize safety checker
        self._init_safety_checker(config)
        
        # Initialize learning inference components
        self._learning_inference_engine = None
        self._learning_data_adapter = None
        self._learning_enabled = False
        
        self._is_initialized = self.initialize()
    
    def print_state(self):
        print(f"Arm joint states[positions, velocity, torques]: "
              f"{self._joint_states._positions}, {self._joint_states._velocities}, {self._joint_states._torques}")
        # print(f'Arm TCP pose: {self._tcp_pose}')
    
    def get_dof(self):
        if not isinstance(self._dof, list):
            dof = [self._dof]
        else:
            dof = self._dof
        return dof
    
    # def get_tcp_pose(self):
    #     """
    #         return the tcp pose in the format [x,y,z,qx,qy,qz,qw]
    #     """
    #     if self.is_initialized:
    #         self._lock.acquire()
    #         tcp_pose = copy.deepcopy(self._tcp_pose)
    #         self._lock.release()
    #         return tcp_pose
    #     else:
    #         raise RuntimeError("Arm is not initialized, cannot get TCP pose.")

    def get_joint_states(self)-> RobotJointState: 
        if self._is_initialized:
            self._lock.acquire()
            joint_state = copy.deepcopy(self._joint_states)
            self._lock.release()
            # @TODO: hack
            # joint_state._accelerations = np.zeros(len(joint_state._accelerations))
            return joint_state
        else:
            raise RuntimeError("Arm is not initialized, cannot get joint states.")
        
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_arm_states(self):
        """
            This func should not be called from external
            Because this is called in the class thread
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_joint_command(self, mode: Union[str, List[str]], command: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def move_to_start(self):
        raise NotImplementedError
    
    def _init_safety_checker(self, config: Dict[str, Any]) -> None:
        """
        Initialize safety checker from configuration
        
        Args:
            config: Configuration dictionary
        """
        # Get safety level from config
        safety_level = SafetyLevel.NORMAL
        if 'safety_level' in config:
            safety_level = SafetyLevel(config['safety_level'])
        
        # Custom safety limits from config
        custom_limits = None
        if 'safety_limits' in config:
            limits_config = config['safety_limits']
            custom_limits = SafetyLimits(
                max_joint_change=limits_config.get('max_joint_change', 0.5),
                max_position_change=limits_config.get('max_position_change', 0.01),
                max_rotation_change=limits_config.get('max_rotation_change', 0.2),
                max_joint_velocity=limits_config.get('max_joint_velocity', 2.0),
                min_command_interval=limits_config.get('min_command_interval', 0.001)
            )
        
        # Get robot name from config or use default
        robot_name = config.get('robot_name', self.__class__.__name__)
        
        self._safety_checker = SafetyChecker(
            limits=custom_limits,
            safety_level=safety_level,
            robot_name=robot_name
        )
    
    def init_safety_state(self, joint_positions: Optional[np.ndarray] = None) -> None:
        """
        Initialize safety checker state
        
        Args:
            joint_positions: Initial joint positions, if None uses zero positions
        """
        if joint_positions is None:
            # Calculate total DOF
            total_dof = sum(self.get_dof()) if isinstance(self._dof, list) else self._dof
            joint_positions = np.zeros(total_dof)
        
        self._safety_checker.update_state(joint_positions=joint_positions)
        self._safety_checker.commit_valid_state()
        log.info(f"Safety checker initialized with state: {joint_positions}")
    
    def check_joint_command_safety(self, command: np.ndarray) -> Tuple[bool, str]:
        """
        Check if joint command is safe
        
        Args:
            command: Joint command array
            
        Returns:
            Tuple[bool, str]: (is_safe, reason)
        """
        return self._safety_checker.check_joint_command(command)
    
    def update_safety_state(self, joint_positions: np.ndarray) -> None:
        """
        Update safety checker with current robot state
        
        Args:
            joint_positions: Current joint positions
        """
        self._safety_checker.update_state(joint_positions=joint_positions)
    
    def commit_safe_state(self) -> None:
        """Commit current state as safe/valid"""
        self._safety_checker.commit_valid_state()
    
    # ============= Safety Management =============
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety check statistics"""
        return self._safety_checker.get_statistics()
    
    def print_safety_statistics(self) -> None:
        """Print safety check statistics"""
        self._safety_checker.print_statistics()
    
    def reset_safety_tracking(self) -> None:
        """Reset safety state tracking"""
        self._safety_checker.reset_tracking()
        log.info("Safety tracking reset")
    
    def get_valid_joint_positions(self) -> Optional[np.ndarray]:
        """Get last known safe joint positions for emergency rollback"""
        return self._safety_checker.get_valid_joint_positions()
    
    def emergency_rollback(self) -> bool:
        """
        Emergency rollback to last safe joint positions
        
        Returns:
            bool: Whether rollback was successful
        """
        valid_positions = self.get_valid_joint_positions()
        if valid_positions is not None:
            try:
                log.info("Executing emergency rollback to safe position...")
                # Temporarily disable safety check for rollback
                temp_limits = self._safety_checker.limits
                self._safety_checker.limits.max_joint_change = 10.0  # Allow large change
                
                self.set_joint_command(['position'], valid_positions)
                
                # Restore original limits
                self._safety_checker.limits = temp_limits
                log.info("Emergency rollback completed")
                return True
            except Exception as e:
                log.info(f"Emergency rollback failed: {e}")
                return False
        else:
            log.info("No valid rollback position available")
            return False
    
    # ========== 学习推理相关方法 ==========
    
    def set_learning_inference(self, inference_engine, data_adapter) -> None:
        """设置学习推理引擎和数据适配器.
        
        Args:
            inference_engine: 学习推理引擎实例
            data_adapter: 数据适配器实例
        """
        with self._lock:
            self._learning_inference_engine = inference_engine
            self._learning_data_adapter = data_adapter
            self._learning_enabled = True
            
        log.info("✅ 学习推理组件已设置")
        log.info(f"   - 推理引擎: {type(inference_engine).__name__}")
        log.info(f"   - 数据适配器: {type(data_adapter).__name__}")
    
    def get_learning_prediction(self, camera_data: Dict[str, np.ndarray]) -> Optional[List[float]]:
        """获取学习策略预测的动作序列.
        
        Args:
            camera_data: 相机名称到图像数据的映射
            
        Returns:
            Optional[List[float]]: 预测的关节动作序列，失败时返回None
            
        Raises:
            RuntimeError: 当学习推理组件未初始化时抛出
        """
        if not self._learning_enabled:
            raise RuntimeError("学习推理组件未初始化，请先调用set_learning_inference()")
        
        if not self._is_initialized:
            raise RuntimeError("机器人臂未初始化")
        
        try:
            # 获取当前关节状态
            current_joint_state = self.get_joint_states()
            
            # 验证数据有效性
            if not self._learning_data_adapter.validate_robot_state(current_joint_state):
                log.error("❌ 当前关节状态数据无效")
                return None
            
            if not self._learning_data_adapter.validate_camera_data(camera_data):
                log.error("❌ 相机数据无效")
                return None
            
            # 数据格式转换
            state_array = self._learning_data_adapter.robot_state_to_numpy(current_joint_state)
            image_tensor = self._learning_data_adapter.camera_dict_to_tensor(camera_data)
            
            # 学习推理
            with self._lock:
                predicted_actions = self._learning_inference_engine.predict(state_array, image_tensor)
            
            # 转换回机器人动作格式
            action_commands = self._learning_data_adapter.numpy_to_robot_actions(predicted_actions)
            
            log.debug(f"学习推理完成: {len(action_commands)}个动作")
            return action_commands
            
        except Exception as e:
            log.error(f"❌ 学习推理失败: {str(e)}")
            return None
    
    def execute_learned_action_sequence(
        self, 
        actions: List[float],
        validate_safety: bool = True,
        execution_mode: str = "position"
    ) -> bool:
        """执行学习策略输出的动作序列.
        
        Args:
            actions: 动作序列
            validate_safety: 是否进行安全检查
            execution_mode: 执行模式 ("position", "velocity", "torque")
            
        Returns:
            bool: 执行是否成功
        """
        if not self._is_initialized:
            log.error("❌ 机器人臂未初始化")
            return False
        
        if not actions:
            log.error("❌ 动作序列为空")
            return False
        
        try:
            # 验证动作维度
            expected_dof = sum(self.get_dof())
            if len(actions) != expected_dof:
                log.error(f"❌ 动作维度不匹配: 期望{expected_dof}, 实际{len(actions)}")
                return False
            
            # 安全检查
            if validate_safety:
                current_state = self.get_joint_states()
                if not self._safety_checker.check_joint_command_safety(actions, current_state):
                    log.error("❌ 学习动作未通过安全检查")
                    return False
            
            # 执行动作
            success = self.set_joint_command([execution_mode], actions)
            
            if success:
                log.debug(f"✅ 学习动作执行成功: {execution_mode}模式")
            else:
                log.error(f"❌ 学习动作执行失败")
            
            return success
            
        except Exception as e:
            log.error(f"❌ 学习动作执行异常: {str(e)}")
            return False
    
    def run_learning_control_loop(
        self, 
        camera_data: Dict[str, np.ndarray],
        execution_mode: str = "position",
        max_retries: int = 3
    ) -> bool:
        """运行完整的学习控制循环：感知-推理-执行.
        
        Args:
            camera_data: 相机数据
            execution_mode: 执行模式
            max_retries: 最大重试次数
            
        Returns:
            bool: 控制循环是否成功
        """
        for attempt in range(max_retries):
            try:
                # 学习推理
                predicted_actions = self.get_learning_prediction(camera_data)
                if predicted_actions is None:
                    log.warning(f"⚠️ 学习推理失败，尝试 {attempt + 1}/{max_retries}")
                    continue
                
                # 执行动作
                success = self.execute_learned_action_sequence(
                    predicted_actions, 
                    execution_mode=execution_mode
                )
                
                if success:
                    log.info(f"✅ 学习控制循环成功完成")
                    return True
                else:
                    log.warning(f"⚠️ 动作执行失败，尝试 {attempt + 1}/{max_retries}")
                    
            except Exception as e:
                log.error(f"❌ 学习控制循环异常: {str(e)}")
        
        log.error(f"❌ 学习控制循环失败，已重试{max_retries}次")
        return False
    
    def is_learning_enabled(self) -> bool:
        """检查学习推理是否已启用."""
        return self._learning_enabled
    
    def disable_learning_inference(self) -> None:
        """禁用学习推理功能."""
        with self._lock:
            self._learning_inference_engine = None
            self._learning_data_adapter = None
            self._learning_enabled = False
        
        log.info("🛑 学习推理功能已禁用")
    
    