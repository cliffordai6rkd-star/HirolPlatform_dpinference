#!/usr/bin/env python3
"""
ACT模型真机推理演示脚本 - 支持动作时间聚合

该脚本演示如何在真实机器人上运行ACT模型进行实时推理控制。
支持FR3、Monte01等多种机器人平台，具备安全检查和错误处理机制。
新增动作时间聚合功能，通过平滑动作序列提高执行稳定性。

使用方法:
    python run_act_inference.py --robot fr3 --ckpt_dir /path/to/model --config /path/to/config.yaml

作者: HIROLRobotPlatform
日期: 2025-08-20
更新: 2025-08-29 - 添加动作时间聚合功能
"""

import os
import sys
import argparse
import time
import signal
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, Any

# 添加项目根路径到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要模块
import glog as log

# 导入夹爪常量
try:
    from hardware.monte01.defs import CORENETIC_GRIPPER_MAX_POSITION
except ImportError:
    CORENETIC_GRIPPER_MAX_POSITION = 0.074  # 默认值作为备用
from factory.components.robot_factory import RobotFactory
from factory.components.motion_factory import MotionFactory
from factory.components.learning_inference_factory import LearningInferenceFactory
from hardware.base.safety_checker import SafetyChecker

# 导入重构后的模块
try:
    from gripper_controller import SmartGripperController, IncrementalGripperController
    from action_aggregator import ActionAggregator as ExternalActionAggregator
    from joint_position_plotter import JointPositionPlotter
    USE_EXTERNAL_MODULES = True
except ImportError:
    SmartGripperController = None
    IncrementalGripperController = None
    ExternalActionAggregator = None
    JointPositionPlotter = None
    USE_EXTERNAL_MODULES = False

class ACTInferenceRunner:
    """ACT真机推理运行器"""
    
    def __init__(self, config_path: str, ckpt_dir: str, robot_type: str):
        """
        初始化ACT推理运行器
        
        Args:
            config_path: 配置文件路径
            ckpt_dir: 模型检查点目录路径
            robot_type: 机器人类型 (fr3, monte01)
        """
        self.config_path = config_path
        self.ckpt_dir = ckpt_dir
        self.robot_type = robot_type
        self.running = False
        self.emergency_stop = False
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化机器人系统
        self.robot_factory = None
        self.motion_factory = None
        self.robot = None
        self.cameras = {}
        
        # 初始化学习推理组件
        self.inference_engine = None
        self.data_adapter = None
        
        # 安全检查器 - 根据机器人类型初始化
        robot_type = self.config.get('robot', {}).get('type', 'fr3')
        if robot_type == 'monte01':
            # 双臂机器人使用两个独立的安全检查器
            self.left_safety_checker = SafetyChecker(robot_name="Monte01_Left")
            self.right_safety_checker = SafetyChecker(robot_name="Monte01_Right")
            self.safety_checker = None  # 向后兼容
        else:
            # 单臂机器人使用单个安全检查器
            self.safety_checker = SafetyChecker(robot_name="FR3")
            self.left_safety_checker = None
            self.right_safety_checker = None
        
        # 控制参数
        self.control_frequency = self.config.get('control_frequency', 10)  # Hz
        self.max_episode_length = self.config.get('max_episode_length', 1200)  # steps
        self.action_repeat = self.config.get('action_repeat', 1)
        
        # 统计信息
        self.step_count = 0
        self.episode_count = 0
        
        # ACT动作序列管理
        self.action_sequence = None  # 当前的动作序列缓存
        self.action_index = 0        # 当前执行的动作索引（采样步数）
        self.action_chunk_size = 100  # 使用前100个动作
        self.action_sampling_interval = 1  # 动作采样间隔
        self.failed_actions_count = 0  # 连续失败的动作数量
        self.max_failed_actions = 5   # 最大连续失败次数，超过后重新推理
        
        # 滑动窗口推理参数
        self.sliding_window_size = self.config.get('sliding_window_size', 10)  # 每执行N步重新推理
        self.steps_since_last_prediction = 0  # 距离上次推理的步数
        self.force_repredict = False  # 强制重新推理标志（用于事件触发）
        self.last_gripper_state = 'OPEN'  # 记录上一个夹爪状态用于检测变化
        
        # 动作插值参数 - 减少卡顿
        self.enable_action_interpolation = self.config.get('enable_action_interpolation', True)
        self.last_executed_action = None  # 上一次执行的动作
        self.interpolation_steps = self.config.get('interpolation_steps', 3)  # 插值步数
        
        # 频率监控参数
        self.frequency_monitor_enabled = True
        self.frequency_monitor_window = 50  # 每50步统计一次实际频率
        self.last_monitor_time = None
        self.monitor_step_count = 0
        
        # 动作时间聚合器
        action_agg_config = self.config.get('action_aggregation', {})
        self.action_aggregator = None
        if action_agg_config.get('enabled', False):
            if USE_EXTERNAL_MODULES and ExternalActionAggregator is not None:
                self.action_aggregator = ExternalActionAggregator(action_agg_config)
                log.info("✅ 动作时间聚合器已启用（使用独立模块）")
            else:
                log.warning("⚠️ 外部动作聚合器模块不可用，动作聚合功能已禁用")
                self.action_aggregator = None
        else:
            log.info("ℹ️ 动作时间聚合器未启用")
        
        # 初始化夹爪控制器（支持单臂和双臂）
        self._initialize_gripper_controllers()
        
        # 双臂协调控制器（仅Monte01）
        self.dual_arm_coordinator = None
        if robot_type == 'monte01':
            self._initialize_dual_arm_coordinator()
        
        # 图像显示配置
        self.visualization_config = self.config.get('visualization', {})
        self.enable_image_display = self.visualization_config.get('enable_image_display', False)
        self.display_window_size = self.visualization_config.get('display_window_size', [640, 480])
        self.display_refresh_rate = self.visualization_config.get('display_refresh_rate', 15)
        self.display_windows_initialized = False
        
        self.camera_latency_ms = 80.0  # 您的相機延遲，單位：毫秒
        self.action_delay_compensation_steps = 0  # 需要跳過的動作步數
        
        # 关节位置可视化配置
        self._initialize_joint_visualizer()

        # 验证采样间隔参数
        if not isinstance(self.action_sampling_interval, int) or self.action_sampling_interval <= 0:
            raise ValueError(f"action_sampling_interval必须为正整数，当前值: {self.action_sampling_interval}")
        
        if self.action_sampling_interval > self.action_chunk_size:
            log.warning(f"采样间隔({self.action_sampling_interval})大于动作块大小({self.action_chunk_size})，将调整为最大可用值")
            self.action_sampling_interval = self.action_chunk_size
        
        log.info("🚀 ACT推理运行器初始化完成")
        
        # 启动关节位置绘图器
        if self.joint_plotter is not None:
            self.joint_plotter.start_plotting()
        if self.enable_image_display:
            log.info(f"🖥️ 图像显示已启用 - 窗口大小: {self.display_window_size}, 刷新率: {self.display_refresh_rate}Hz")

    def _initialize_gripper_controllers(self):
        """初始化夹爪控制器（单臂或双臂）"""
        self.gripper_postprocess_config = self.config.get('gripper_postprocess', {})
        
        if not USE_EXTERNAL_MODULES or (SmartGripperController is None and IncrementalGripperController is None):
            log.error("❌ 夹爪控制器模块未找到！请确保 gripper_controller.py 存在")
            raise ImportError("GripperController modules are required")
        
        # 检查夹爪控制模式
        control_mode = self.gripper_postprocess_config.get('control_mode', 'binary')  # 默认binary模式(FR3兼容)
        log.info(f"🔧 夹爪控制模式: {control_mode}")
        
        # 选择合适的控制器类
        if control_mode == 'incremental':
            GripperControllerClass = IncrementalGripperController
            controller_name = "增量式"
            log.info("🔧 使用Monte01增量式夹爪控制器")
        else:  # binary mode (FR3)
            GripperControllerClass = SmartGripperController
            controller_name = "智能状态机"
            log.info("🔧 使用FR3智能状态机夹爪控制器")
        
        if self.robot_type == 'monte01':
            # Monte01双臂夹爪控制器
            left_gripper_config = self.gripper_postprocess_config.get('left_gripper', self.gripper_postprocess_config)
            right_gripper_config = self.gripper_postprocess_config.get('right_gripper', self.gripper_postprocess_config)
            
            log.info(f"🔧 初始化Monte01双臂{controller_name}夹爪控制器:")
            log.info(f"   左臂配置: {left_gripper_config}")
            log.info(f"   右臂配置: {right_gripper_config}")
            
            self.left_gripper_controller = GripperControllerClass(left_gripper_config)
            self.right_gripper_controller = GripperControllerClass(right_gripper_config)
            self.gripper_controller = self.left_gripper_controller  # 向后兼容
            
            # 处理状态兼容性
            if hasattr(self.left_gripper_controller, 'state'):
                self.gripper_state = self.left_gripper_controller.state
            else:
                self.gripper_state = None  # 增量模式无状态
                
            log.info(f"✅ Monte01双臂{controller_name}夹爪控制器已配置完成")
        else:
            # FR3单臂夹爪控制器 (强制使用SmartGripperController)
            if control_mode == 'incremental':
                log.warning("⚠️  FR3不支持增量模式，强制使用智能状态机模式")
                GripperControllerClass = SmartGripperController
            
            self.gripper_controller = GripperControllerClass(self.gripper_postprocess_config)
            if hasattr(self.gripper_controller, 'state'):
                self.gripper_state = self.gripper_controller.state
            else:
                self.gripper_state = None
            log.info(f"🔧 FR3单臂{controller_name}夹爪控制器已配置")
    
    def _initialize_dual_arm_coordinator(self):
        """初始化双臂协调控制器（仅Monte01）"""
        try:
            from learning.utils.dual_arm_utils import create_dual_arm_coordinator
            
            dual_arm_config = self.config.get('dual_arm_control', {})
            if dual_arm_config:
                self.dual_arm_coordinator = create_dual_arm_coordinator(dual_arm_config)
                log.info("🤖 Monte01双臂协调控制器已初始化")
            else:
                log.info("ℹ️ 双臂协调配置未找到，跳过初始化")
        except ImportError as e:
            log.warning(f"⚠️ 双臂协调工具导入失败: {e}")
            log.warning("   双臂协调功能将被禁用，仍可进行基本双臂控制")
    
    def _initialize_joint_visualizer(self):
        """初始化关节位置可视化器"""
        self.joint_vis_config = self.config.get('joint_position_visualization', {})
        self.joint_plotter = None
        
        if self.joint_vis_config.get('enabled', False):
            if USE_EXTERNAL_MODULES and JointPositionPlotter is not None:
                # 根据机器人类型定义关节名称
                if self.robot_type == 'monte01':
                    # Monte01双臂16关节名称
                    joint_names = []
                    # 左臂
                    for i in range(1, 8):
                        joint_names.append(f'Left_joint{i}')
                    joint_names.append('Left_Gripper')
                    # 右臂
                    for i in range(1, 8):
                        joint_names.append(f'Right_joint{i}')
                    joint_names.append('Right_Gripper')
                else:
                    # FR3单臂8关节名称
                    joint_names = ['Panda_joint1', 'Panda_joint2', 'Panda_joint3', 'Panda_joint4', 
                                 'Panda_joint5', 'Panda_joint6', 'Panda_joint7', 'Gripper']
                
                self.joint_plotter = JointPositionPlotter(self.joint_vis_config, joint_names)
                log.info(f"📊 {self.robot_type}关节位置绘图器已初始化")
            else:
                log.warning("⚠️ 关节位置绘图器模块不可用，功能已禁用")
        else:
            log.info("ℹ️ 关节位置可视化未启用")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            from hardware.base.utils import dynamic_load_yaml
            config = dynamic_load_yaml(self.config_path)
            log.info(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            log.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """设置信号处理器以支持优雅关闭"""
        def signal_handler(signum, frame):  # pylint: disable=unused-argument
            log.warning("⚠️ 接收到停止信号，正在安全关闭...")
            self.emergency_stop = True
            self.running = False
            
            # 如果多次按Ctrl+C，强制退出（第一次就设置较短超时）
            if hasattr(self, '_signal_count'):
                self._signal_count += 1
                if self._signal_count >= 2:
                    log.warning("🚨 检测到多次停止信号，立即强制退出...")
                    import os
                    os._exit(1)
                else:
                    # 第二次按Ctrl+C，设置5秒超时后强制退出
                    log.warning("⏰ 设置5秒超时，如无法正常退出将强制终止...")
                    import threading
                    def force_exit():
                        time.sleep(5)
                        log.error("💥 超时强制退出！")
                        import os
                        os._exit(1)
                    threading.Thread(target=force_exit, daemon=True).start()
            else:
                self._signal_count = 1
                # 第一次按Ctrl+C，设置3秒超时
                log.warning("⏰ 设置3秒超时，再次按Ctrl+C立即强制退出...")
                import threading
                def force_exit():
                    time.sleep(3)
                    log.error("💥 超时强制退出！")
                    import os
                    os._exit(1)
                threading.Thread(target=force_exit, daemon=True).start()
            
            # 尝试优雅关闭（不阻塞）
            try:
                self.stop()
            except Exception as e:
                log.error(f"❌ 优雅关闭失败: {e}")
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_robot(self):
        """初始化机器人硬件系统"""
        try:
            log.info("🔧 初始化机器人硬件系统...")
            
            # 获取motion_config并创建机器人系统
            motion_config = self.config.get('motion_config', self.config)  # 如果有motion_config则使用，否则回退到整个config
            
            # 使用工厂模式创建机器人系统 - 使用motion_config以确保硬件配置一致
            robot_system = RobotFactory(motion_config)
            
            # 使用MotionFactory包装机器人系统
            log.info(f"📋 Motion配置包含键: {list(motion_config.keys()) if isinstance(motion_config, dict) else type(motion_config)}")
            self.motion_factory = MotionFactory(motion_config, robot_system)
            self.motion_factory.create_motion_components()  # 这里会调用robot_system.create_robot_system()
            self.robot_factory = robot_system  # 保持向后兼容性
            self.robot = robot_system._robot
            
            log.info(f"MotionFactory系统初始化完成")

            # 获取相机传感器  
            camera_names = self.config.get('learning', {}).get('camera_names', [])
            
            for camera_name in camera_names:
                camera_found = False
                # 在相机列表中查找对应名称的相机
                if 'camera' in robot_system._sensors:
                    for cam_info in robot_system._sensors['camera']:
                        if cam_info['name'] == camera_name:
                            self.cameras[camera_name] = cam_info['object']
                            log.info(f"  ✅ 相机 {camera_name} 连接成功")
                            camera_found = True
                            break
                
                if not camera_found:
                    log.warning(f"  ⚠️ 相机 {camera_name} 未找到")
            
            # 等待机器人系统稳定
            time.sleep(1.0)
            
            # 检查机器人状态
            log.info("🔍 开始机器人健康检查...")
            try:
                if not self._check_robot_health():
                    raise RuntimeError("机器人健康检查失败")
            except Exception as health_error:
                log.error(f"❌ 机器人健康检查异常: {health_error}")
                raise health_error
            
            log.info("✅ 机器人硬件系统初始化完成")
            
        except Exception as e:
            log.error(f"❌ 机器人初始化失败: {e}")
            raise
    
    def initialize_learning_system(self):
        """初始化学习推理系统"""
        try:
            log.info("🧠 初始化学习推理系统...")
            
            learning_config = self.config['learning']
            
            # 验证检查点目录
            if not LearningInferenceFactory.validate_checkpoint_directory(self.ckpt_dir, "ACT"):
                raise FileNotFoundError(f"无效的ACT检查点目录: {self.ckpt_dir}")
            
            # 创建推理引擎和数据适配器
            log.info(f"🔍 创建推理引擎参数: robot_type={self.robot_type}, state_dim={learning_config.get('state_dim')}")
            self.inference_engine, self.data_adapter = LearningInferenceFactory.create_learning_pipeline(
                algorithm="ACT",
                robot_type=self.robot_type,
                ckpt_dir=self.ckpt_dir,
                config=learning_config
            )
            
            log.info("✅ 学习推理系统初始化完成")
            log.info(f"   - 算法: ACT")
            log.info(f"   - 机器人类型: {self.robot_type}")
            log.info(f"   - 检查点目录: {self.ckpt_dir}")
            log.info(f"   - 相机: {learning_config.get('camera_names', [])}")
            
        except Exception as e:
            log.error(f"❌ 学习推理系统初始化失败: {e}")
            raise
    
    def _check_robot_health(self) -> bool:
        """检查机器人系统健康状态"""
        try:
            # 获取当前关节状态（对Monte01进行状态组合处理）
            if self.robot_type == 'monte01':
                # 获取处理后的16维状态
                positions = self._get_monte01_dual_arm_state()
                joint_state = self._create_joint_state_from_array(positions)
            else:
                # FR3直接使用原始状态
                joint_state = self.robot.get_joint_states()
                if joint_state is None:
                    log.error("❌ 无法获取机器人关节状态")
                    return False
                positions = np.array(joint_state._positions)
            
            # 基本的关节状态检查
            if joint_state is None or len(positions) == 0:
                log.error("❌ 无法获取机器人关节状态")
                return False
            
            # 基本数值有效性检查
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                log.error("❌ 机器人关节位置包含无效数值")
                return False
            
            # 如果data_adapter已初始化，进行详细验证
            if self.data_adapter is not None:
                if not self.data_adapter.validate_robot_state(joint_state):
                    log.error("❌ 机器人关节状态无效")
                    return False
            else:
                log.warning("⚠️ data_adapter未初始化，跳过详细状态验证")
            
            # 更新安全检查器状态
            robot_type = self.config.get('robot', {}).get('type', 'fr3')
            if robot_type == 'monte01' and self.left_safety_checker and self.right_safety_checker:
                # 双臂模式：分别更新左右臂安全检查器状态
                if len(positions) >= 14:  # 确保有足够的关节数据
                    left_positions = positions[:7]   # 左臂关节
                    right_positions = positions[7:14]  # 右臂关节
                    self.left_safety_checker.update_state(joint_positions=left_positions)
                    self.right_safety_checker.update_state(joint_positions=right_positions)
            elif self.safety_checker:
                # 单臂模式
                self.safety_checker.update_state(joint_positions=positions)
            
            log.info("✅ 机器人健康检查通过")
            return True
            
        except Exception as e:
            log.error(f"❌ 机器人健康检查失败: {e}")
            return False
    
    def _get_camera_observations(self):
        """获取相机观察数据"""
        # 获取推理引擎的设备信息
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(self.inference_engine, 'device'):
            device = self.inference_engine.device
        
        # 获取原始numpy数组用于验证
        raw_observations = {}
        # 转换后的tensor数据用于推理
        tensor_observations = {}
        
        for camera_name, camera in self.cameras.items():
            try:
                # 使用CameraBase的标准接口获取图像数据
                success, image, timestamp = camera.read_image()
                if success and image is not None and isinstance(image, np.ndarray):
                    # 统一图像尺寸到480x640（参考load_and_resize_image_robust）
                    target_size = (480, 640)  # (height, width)
                    
                    # 如果尺寸不匹配，进行resize
                    if image.shape[:2] != target_size:
                        # OpenCV resize需要 (width, height) 参数顺序
                        image = cv2.resize(image, (target_size[1], target_size[0]))
                        log.debug(f"相机 {camera_name} 图像已resize: {image.shape[:2]} -> {target_size}")
                    
                    # 保存处理后的numpy数组用于data_adapter验证
                    raw_observations[camera_name] = image
                    
                    # 转换为PyTorch张量用于ACT推理
                    # BGR -> RGB，使用copy()避免负步长问题
                    image_rgb = image[:, :, ::-1].copy()
                    # (H, W, C) -> (C, H, W)
                    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
                    # 归一化到[0, 1]
                    image_tensor = image_tensor / 255.0
                    # 移动到与模型相同的设备
                    image_tensor = image_tensor.to(device)
                    tensor_observations[camera_name] = image_tensor
                    
                    log.debug(f"相机 {camera_name} 图像张量: {image_tensor.shape}, 设备: {image_tensor.device}")
                else:
                    log.warning(f"⚠️ 相机 {camera_name} 读取失败或返回空图像")
            except Exception as e:
                log.error(f"❌ 获取相机 {camera_name} 数据失败: {e}")
        
        # 显示图像（如果启用）
        self._display_camera_images(raw_observations)
        
        return raw_observations, tensor_observations
    
    
    def _execute_action_safely(self, action: np.ndarray) -> bool:
        """
        安全执行动作，包含对夹爪的后处理逻辑
        支持单臂（FR3: 8 DOF）和双臂（Monte01: 16 DOF）机器人
        """
        try:
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            action = np.asarray(action).flatten()
            log.info(f"🔍 执行动作前: shape={action.shape}, values={action[:8]}...")
            log.info(f"   动作范围: min={np.min(action):.3f}, max={np.max(action):.3f}, mean={np.mean(action):.3f}")
            
            # 根据机器人类型处理动作
            robot_type = self.config.get('robot', {}).get('type', 'fr3')
            if robot_type == 'monte01':
                return self._execute_dual_arm_action_safely(action)
            else:
                # FR3单臂模式
                return self._execute_single_arm_action_safely(action)
            
        except Exception as e:
            log.error(f"❌ 动作执行异常: {e}")
            return False
    
    def _execute_single_arm_action_safely(self, action: np.ndarray) -> bool:
        """
        执行FR3单臂动作（8 DOF：7关节 + 1夹爪）
        """
        joint_action = action[:7]
        gripper_action_raw = action[7] if len(action) > 7 else 0.0
            
        # 验证关节动作
        safety_check_passed = self._validate_single_arm_action(joint_action)
        if not safety_check_passed:
            log.warning("⚠️ 关节动作验证失败，跳过执行")
            # 仍然处理夹爪，传递安全失败信息
            if self.gripper_controller is not None:
                tool_interface = self.robot_factory._tool if hasattr(self.robot_factory, '_tool') else None
                
                # 获取末端位姿
                end_effector_pose = None
                if self.robot:
                    try:
                        end_effector_pose = self.robot.get_ee_pose()
                    except:
                        pass
                
                self.gripper_controller.process(
                    gripper_action_raw,
                    safety_failed=False,  # 夹爪控制不受关节安全检查影响
                    tool_interface=tool_interface,
                    end_effector_pose=end_effector_pose
                )
                self.gripper_state = self.gripper_controller.state
            else:
                self._process_gripper_action(gripper_action_raw, safety_check_failed=True)
            return False
        
        # 执行关节动作
        success = self.robot.set_joint_command('position', joint_action.tolist())
        if not success:
            log.warning("⚠️ 关节动作执行失败")
            return False
        
        # 获取当前关节位置（用于稳定性检测）
        if self.robot:
            joint_state = self.robot.get_joint_states()
            # 如果是 RobotJointState 对象，提取位置数组
            if hasattr(joint_state, '_positions'):
                current_joint_states = joint_state._positions
            elif hasattr(joint_state, 'positions'):
                current_joint_states = joint_state.positions
            else:
                current_joint_states = action
        else:
            current_joint_states = action
        
        # 处理夹爪动作 - 完全依赖外部模块
        tool_interface = self.robot_factory._tool if hasattr(self.robot_factory, '_tool') else None
        
        # 获取末端位姿
        end_effector_pose = None
        if self.robot:
            try:
                end_effector_pose = self.robot.get_ee_pose()  # 获取末端位姿
            except:
                pass
        
        _, _, force_repredict = self.gripper_controller.process(
            gripper_action_raw,
            safety_failed=False,
            tool_interface=tool_interface,
            end_effector_pose=end_effector_pose
        )
        if force_repredict:
            self.force_repredict = True
        # 同步状态
        self.gripper_state = self.gripper_controller.state
        gripper_success = True
        
        # 更新关节位置绘图器（单臂）
        if self.joint_plotter is not None:
            # 获取当前完整状态
            current_state = self._get_current_state()
            joint_positions = current_state[:7]  # 前7维是关节位置
            gripper_position = current_state[7]   # 第8维是夹爪位置
            
            # 将夹爪位置从状态值转换为实际宽度
            actual_gripper_width = self.gripper_controller.get_gripper_width(tool_interface)
            
            self.joint_plotter.update_positions(joint_positions, actual_gripper_width)
        
        # 每100步输出一次状态信息
        if self.step_count % 100 == 0:
            history = list(self.gripper_controller.action_history) if hasattr(self.gripper_controller, 'action_history') else []
            # 获取真实夹爪宽度
            actual_width = self.gripper_controller.get_gripper_width(tool_interface)
            log.info(f"📊 夹爪状态: {self.gripper_state}, 动作值: {gripper_action_raw:.4f}, 实际宽度: {actual_width:.4f}m, 历史: {history}")
            
            # 每1000步输出一次统计信息
            if self.joint_plotter is not None and self.step_count % 1000 == 0:
                self.joint_plotter.print_statistics()
        
        return success and gripper_success

    def _execute_dual_arm_action_safely(self, action: np.ndarray) -> bool:
        """
        执行Monte01双臂动作（16 DOF：左臂8维 + 右臂8维）- 简化版
        """
        if len(action) != 16:
            log.error(f"❌ Monte01双臂动作维度错误: 期望16, 实际{len(action)}")
            return False
        
        # 分离左右臂动作
        left_arm_action = action[:8]   # 左臂: 7关节 + 1夹爪/吸盘
        right_arm_action = action[8:16]  # 右臂: 7关节 + 1夹爪/吸盘
        
        log.info(f"🔧 双臂动作分离:")
        log.info(f"   左臂关节: {left_arm_action[:7].round(3)}")
        log.info(f"   左臂夹爪: {left_arm_action[7]:.4f}")
        log.info(f"   右臂关节: {right_arm_action[:7].round(3)}")
        log.info(f"   右臂夹爪: {right_arm_action[7]:.4f}")
        
        # 简化版：直接使用原始动作，无碰撞检测
        if hasattr(self, 'dual_arm_coordinator') and self.dual_arm_coordinator is not None:
            # 简化处理：不做碰撞检测，只做基本记录
            processed_action, coordination_info = self.dual_arm_coordinator.process_dual_arm_action(action)
            left_arm_action = processed_action[:8]
            right_arm_action = processed_action[8:16]
        
        try:
            # 组合14维关节命令 (不包含夹爪，DuoArm期望14维)
            joint_command_14d = np.concatenate([
                left_arm_action[:7],  # 左臂7个关节
                right_arm_action[:7]  # 右臂7个关节
            ])
            
            # 设置双臂模式
            dual_mode = ['position', 'position']
            
            # 执行双臂动作 (硬件层自动进行安全检查)
            self.robot.set_joint_command(dual_mode, joint_command_14d.tolist())
            success = True
            log.debug(f"✅ 双臂关节命令执行: 14维 = {joint_command_14d.shape}")
            
        except Exception as e:
            log.warning(f"⚠️ 双臂关节动作执行失败: {e}")
            success = False
        
        if not success:
            log.info("💡 硬件层安全检查生效或执行失败")
            # 仍然处理夹爪，因为夹爪动作通常是安全的
            self._process_dual_arm_grippers(left_arm_action[7], right_arm_action[7], safety_failed=True)
            return False
        
        # 处理双臂夹爪/吸盘（使用转换后的夹爪值）
        gripper_success = self._process_dual_arm_grippers(left_arm_action[7], right_arm_action[7])
        
        # 更新关节位置绘图器（双臂）
        if self.joint_plotter is not None:
            # 获取当前双臂状态
            current_state = self._get_current_state()  # 16维状态
            
            # 双臂的绘图器需要更新所有关节和夹爪
            try:
                # Monte01: 16维 = 左臂8维 + 右臂8维
                all_positions = current_state  # 全部位置
                
                # 获取双臂夹爪宽度
                left_gripper_width = 0.04  # 默认值
                right_gripper_width = 0.04
                
                if hasattr(self, 'left_gripper_controller') and self.left_gripper_controller:
                    try:
                        # 通过robot_factory._tool获取工具接口，而不是robot
                        left_tool = getattr(self.robot_factory._tool._tool.get('left'), None) if hasattr(self.robot_factory, '_tool') and self.robot_factory._tool else None
                        left_gripper_width = self.left_gripper_controller.get_gripper_width(left_tool)
                    except Exception as e:
                        log.debug(f"获取左侧夹爪宽度失败: {e}")
                        pass
                        
                if hasattr(self, 'right_gripper_controller') and self.right_gripper_controller:
                    try:
                        # 通过robot_factory._tool获取工具接口，而不是robot
                        right_tool = getattr(self.robot_factory._tool._tool.get('right'), None) if hasattr(self.robot_factory, '_tool') and self.robot_factory._tool else None
                        right_gripper_width = self.right_gripper_controller.get_gripper_width(right_tool)
                    except Exception as e:
                        log.debug(f"获取右侧夹爪宽度失败: {e}")
                        pass
                
                # 传递双臂夹爪宽度列表
                gripper_widths = [left_gripper_width, right_gripper_width]
                self.joint_plotter.update_positions(all_positions, gripper_widths)
                
            except Exception as e:
                log.warning(f"⚠️ 双臂关节绘图器更新失败: {e}")
        
        # 简化的双臂统计输出
        if self.step_count % 200 == 0 and hasattr(self, 'dual_arm_coordinator') and self.dual_arm_coordinator is not None:
            stats = self.dual_arm_coordinator.get_statistics()
            log.info(f"📊 双臂协调统计: {stats}")
        
        # 双臂夹爪状态输出
        if self.step_count % 100 == 0:
            left_state = getattr(self.left_gripper_controller, 'state', 'UNKNOWN') if hasattr(self, 'left_gripper_controller') else 'N/A'
            right_state = getattr(self.right_gripper_controller, 'state', 'UNKNOWN') if hasattr(self, 'right_gripper_controller') else 'N/A'
            log.info(f"📊 双臂夹爪状态: 左臂={left_state}, 右臂={right_state}")
        
        return success and gripper_success
    
    def _process_dual_arm_grippers(self, left_gripper_action: float, right_gripper_action: float, safety_failed: bool = False) -> bool:
        """
        处理双臂夹爪/吸盘动作
        """
        log.info(f"🤏 处理双臂夹爪动作: 左={left_gripper_action:.4f}, 右={right_gripper_action:.4f}, 安全失败={safety_failed}")
        success = True
        
        # 处理左臂夹爪/吸盘
        if hasattr(self.robot_factory, '_tool') and self.robot_factory._tool:
            try:
                # 通过robot_factory._tool获取左侧工具接口
                left_tool_interface = self.robot_factory._tool._tool.get('left') if self.robot_factory._tool._tool else None
                left_ee_pose = self.robot.get_left_ee_pose() if hasattr(self.robot, 'get_left_ee_pose') else None
                
                # 这里需要根据实际的双臂夹爪控制器来实现
                # 暂时使用单臂的夹爪控制器逻辑
                if hasattr(self, 'left_gripper_controller') and self.left_gripper_controller is not None:
                    # 单位转换：ACT输出米制单位 → SmartGripperController期望的归一化值(0-1)
                    left_gripper_normalized = left_gripper_action / CORENETIC_GRIPPER_MAX_POSITION
                    left_gripper_normalized = np.clip(left_gripper_normalized, 0.0, 1.0)
                    
                    log.debug(f"🤏 左臂夹爪控制器处理: action={left_gripper_action:.4f}m → normalized={left_gripper_normalized:.4f}")
                    left_execute_command, left_command_value, left_force_repredict = self.left_gripper_controller.process(
                        left_gripper_normalized,  # 使用归一化值
                        safety_failed=safety_failed,
                        tool_interface=left_tool_interface,
                        end_effector_pose=left_ee_pose
                    )
                    
                    # 执行夹爪命令到硬件
                    if left_execute_command and left_command_value is not None and left_tool_interface:
                        try:
                            left_tool_interface._set_binary_command(left_command_value)
                            log.info(f"🤏 左臂夹爪硬件命令已发送: {left_command_value:.4f}")
                        except Exception as e:
                            log.error(f"❌ 左臂夹爪硬件命令发送失败: {e}")
                    
                    log.debug(f"✅ 左臂夹爪控制器处理完成, execute={left_execute_command}, command={left_command_value}, force_repredict={left_force_repredict}")
                else:
                    log.warning("⚠️ 左臂夹爪控制器不可用")
                    left_force_repredict = False
                
            except Exception as e:
                log.warning(f"⚠️ 左臂夹爪处理失败: {e}")
                success = False
                left_force_repredict = False
                
            # 检查是否需要重新预测
            if left_force_repredict:
                self.force_repredict = True
        
        # 处理右臂夹爪/吸盘
        if hasattr(self.robot_factory, '_tool') and self.robot_factory._tool:
            try:
                # 通过robot_factory._tool获取右侧工具接口
                right_tool_interface = self.robot_factory._tool._tool.get('right') if self.robot_factory._tool._tool else None
                right_ee_pose = self.robot.get_right_ee_pose() if hasattr(self.robot, 'get_right_ee_pose') else None
                
                # 这里需要根据实际的双臂夹爪控制器来实现
                if hasattr(self, 'right_gripper_controller') and self.right_gripper_controller is not None:
                    # 单位转换：ACT输出米制单位 → SmartGripperController期望的归一化值(0-1)
                    right_gripper_normalized = right_gripper_action / CORENETIC_GRIPPER_MAX_POSITION
                    right_gripper_normalized = np.clip(right_gripper_normalized, 0.0, 1.0)
                    
                    log.debug(f"🤏 右臂夹爪控制器处理: action={right_gripper_action:.4f}m → normalized={right_gripper_normalized:.4f}")
                    right_execute_command, right_command_value, right_force_repredict = self.right_gripper_controller.process(
                        right_gripper_normalized,  # 使用归一化值
                        safety_failed=safety_failed,
                        tool_interface=right_tool_interface,
                        end_effector_pose=right_ee_pose
                    )
                    
                    # 执行夹爪命令到硬件
                    if right_execute_command and right_command_value is not None and right_tool_interface:
                        try:
                            right_tool_interface._set_binary_command(right_command_value)
                            log.info(f"🤏 右臂夹爪硬件命令已发送: {right_command_value:.4f}")
                        except Exception as e:
                            log.error(f"❌ 右臂夹爪硬件命令发送失败: {e}")
                    
                    log.debug(f"✅ 右臂夹爪控制器处理完成, execute={right_execute_command}, command={right_command_value}, force_repredict={right_force_repredict}")
                else:
                    log.warning("⚠️ 右臂夹爪控制器不可用")
                    right_force_repredict = False
                
            except Exception as e:
                log.warning(f"⚠️ 右臂夹爪处理失败: {e}")
                success = False
                right_force_repredict = False
                
            # 检查是否需要重新预测
            if right_force_repredict:
                self.force_repredict = True
        
        return success
    

    def _validate_single_arm_action(self, joint_action: np.ndarray, safety_checker=None) -> bool:
        """验证单臂关节动作的安全性"""
        expected_dim = 7
        if len(joint_action) != expected_dim:
            log.error(f"❌ 关节动作维度错误: 期望{expected_dim}, 实际{joint_action.shape}")
            return False
        
        # 检查是否包含NaN或无穷大值
        if np.any(np.isnan(joint_action)) or np.any(np.isinf(joint_action)):
            log.error("❌ 关节动作包含无效数值")
            return False
        
        # 选择合适的安全检查器
        if safety_checker is None:
            safety_checker = self.safety_checker
        
        # 检查关节命令安全性
        is_safe, safety_message = safety_checker.check_joint_command(joint_action)
        if not is_safe:
            log.warning(f"⚠️ 动作安全检查失败: {safety_message}")
            
            # 获取当前关节状态用于安全修正
            current_joint_state = self.robot.get_joint_states()
            if current_joint_state is None:
                log.warning(f"⚠️ 无法获取当前关节状态，跳过执行")
                return False
                
            current_positions = np.array(current_joint_state._positions)
            joint_position_diff = joint_action - current_positions[:7]  # 只取前7个关节
            
            # 处理不同类型的安全问题
            if "High joint velocity" in safety_message:
                # 应用速度限制 - 针对15Hz推理优化
                max_velocity_limit = self.config.get('safety', {}).get('max_velocity_per_step', 1.8)  # 使用配置值
                dt = 1.0 / self.control_frequency
                estimated_velocity = np.abs(joint_position_diff / dt)
                max_velocity = np.max(estimated_velocity)
                
                if max_velocity > max_velocity_limit:
                    scale_factor = max_velocity_limit / max_velocity
                    # 使用安全缩放因子
                    safety_scale = self.config.get('safety', {}).get('safety_scale_factor', 0.85)
                    scale_factor *= safety_scale
                    safe_joint_action = current_positions[:7] + joint_position_diff * scale_factor
                    log.info(f"🛡️ 速度限制应用: {max_velocity:.3f}→{max_velocity_limit:.3f} rad/s, 缩放={scale_factor:.3f}")
                    # 更新关节动作
                    joint_action[:] = safe_joint_action
                    
            elif "Large joint change" in safety_message:
                # 应用位置步长限制 - 使用配置的max_joint_change
                max_step_limit = self.config.get('safety', {}).get('max_joint_change', 0.4)
                max_step = np.max(np.abs(joint_position_diff))
                
                if max_step > max_step_limit:
                    scale_factor = max_step_limit / max_step
                    # 使用安全缩放因子
                    safety_scale = self.config.get('safety', {}).get('safety_scale_factor', 0.85)
                    scale_factor *= safety_scale
                    safe_joint_action = current_positions[:7] + joint_position_diff * scale_factor
                    log.info(f"🛡️ 步长限制应用: {max_step:.3f}→{max_step_limit:.3f} rad, 缩放={scale_factor:.3f}")
                    # 更新关节动作
                    joint_action[:] = safe_joint_action
            else:
                # 其他安全检查失败，直接跳过
                return False
        
        return True
    
    def _reset_action_sequence(self):
        """重置动作序列缓存"""
        self.action_sequence = None
        self.action_index = 0
        self.failed_actions_count = 0
        
        # 重置动作聚合器
        if self.action_aggregator is not None:
            self.action_aggregator.reset()
        
        # 重置夹爪控制器
        if self.gripper_controller is not None:
            self.gripper_controller.reset()
        
        log.info("🔄 动作序列已重置")
    
    def _get_current_state(self) -> np.ndarray:
        """
        获取当前完整状态（关节+夹爪/吸盘）
        
        Returns:
            np.ndarray: 状态向量 
                       - FR3: 8维 [joint_positions(7), gripper_position(1)]
                       - Monte01: 16维 [left_arm(7+1), right_arm(7+1)]
        """
        if self.robot_type == 'monte01':
            # Monte01双臂模式: 16 DOF (左臂8DOF + 右臂8DOF)
            return self._get_monte01_dual_arm_state()
        else:
            # FR3单臂模式: 8 DOF (保持原有逻辑)
            return self._get_fr3_single_arm_state()
    
    def _get_fr3_single_arm_state(self) -> np.ndarray:
        """FR3单臂状态获取 (8 DOF)"""
        joint_state = self.robot.get_joint_states()
        joint_positions = np.array(joint_state._positions)
        
        # 获取当前夹爪状态（关闭=0, 打开=0.08）
        if self.gripper_controller is not None:
            gripper_position = self.gripper_controller.get_position()
        else:
            gripper_position = 0.0 if self.gripper_state in ['CLOSED', 'HOLDING'] else 0.08
        
        # 组合为8维状态向量
        return np.append(joint_positions, gripper_position)
    
    def _get_monte01_dual_arm_state(self) -> np.ndarray:
        """Monte01双臂状态获取 (16 DOF: 左臂8DOF + 右臂8DOF)"""
        # 获取双臂关节状态（duo_arm返回14维：左臂7+右臂7）
        joint_state = self.robot.get_joint_states()
        arm_positions = np.array(joint_state._positions)
        
        if len(arm_positions) == 14:
            # 正常情况：duo_arm返回14维，需要补充夹爪状态
            left_arm = arm_positions[:7]   # 左臂7个关节
            right_arm = arm_positions[7:]  # 右臂7个关节
            
            # 获取真实夹爪状态
            left_gripper_pos = self._get_gripper_position('left')
            right_gripper_pos = self._get_gripper_position('right')
            
            # 组合为16维状态: [左臂7+左夹爪, 右臂7+右夹爪]
            all_positions = np.concatenate([
                left_arm, [left_gripper_pos],
                right_arm, [right_gripper_pos]
            ])
            
            log.debug(f"🔄 Monte01状态组合: 14维arm → 16维(含夹爪)")
            return all_positions
            
        elif len(arm_positions) == 16:
            # 已经包含夹爪状态（可能来自其他源）
            log.debug(f"✅ Monte01直接获取16维状态")
            return arm_positions
            
        else:
            # 异常情况：数据维度不符合预期
            log.error(f"❌ Monte01关节数量异常: 期望14或16, 实际{len(arm_positions)}")
            # 返回零填充的16维数组作为安全回退
            return np.zeros(16)
    
    def _get_gripper_position(self, side: str) -> float:
        """获取指定侧夹爪的真实位置
        
        Args:
            side: 'left' 或 'right'
            
        Returns:
            float: 夹爪位置（0.0=闭合, 0.074=打开）
        """
        try:
            # 修复：使用robot_factory而不是robot来获取工具状态
            # self.robot是Monte01实例，没有get_tool_dict_state方法
            # 应该使用RobotFactory的方法
            if self.robot_factory is None:
                log.debug(f"⚠️ RobotFactory未初始化，{side}侧夹爪状态不可用，使用默认值")
                return 0.0
            
            # 获取工具状态字典
            tool_dict = self.robot_factory.get_tool_dict_state()
            
            if tool_dict is None:
                log.debug(f"⚠️ {side}侧夹爪状态不可用，使用默认值")
                return 0.0  # 默认闭合状态
            
            # 获取指定侧的夹爪状态
            if side in tool_dict and tool_dict[side] is not None:
                gripper_state = tool_dict[side]
                # 获取夹爪位置（ToolState._position）
                position = getattr(gripper_state, '_position', 0.0)
                log.debug(f"📍 {side}侧夹爪位置: {position:.3f}")
                return float(position)
            else:
                # 该侧夹爪未安装或不可用
                log.debug(f"ℹ️ {side}侧夹爪未安装，使用默认值")
                return 0.0
                
        except Exception as e:
            log.warning(f"⚠️ 获取{side}侧夹爪位置失败: {e}")
            return 0.0
    
    def _create_joint_state_from_array(self, state_array: np.ndarray):
        """从numpy数组创建RobotJointState对象
        
        Args:
            state_array: 关节位置数组
            
        Returns:
            RobotJointState: 包含位置信息的关节状态对象
        """
        from hardware.base.utils import RobotJointState
        
        joint_state = RobotJointState()
        joint_state._positions = state_array.copy()
        joint_state._velocities = np.zeros_like(state_array)
        joint_state._accelerations = np.zeros_like(state_array)
        joint_state._torques = np.zeros_like(state_array)
        
        return joint_state
    
    def _initialize_display_windows(self):
        """初始化图像显示窗口"""
        if not self.enable_image_display or self.display_windows_initialized:
            return
            
        try:
            # 为每个相机创建显示窗口
            for camera_name in self.cameras.keys():
                window_name = f"Camera_{camera_name}"
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(window_name, self.display_window_size[0], self.display_window_size[1])
                log.info(f"✅ 创建显示窗口: {window_name}")
            
            self.display_windows_initialized = True
            log.info("🖥️ 图像显示窗口初始化完成")
            
        except Exception as e:
            log.error(f"❌ 显示窗口初始化失败: {e}")
            self.enable_image_display = False
    
    def _display_camera_images(self, raw_observations: dict):
        """显示相机图像"""
        if not self.enable_image_display or not raw_observations:
            return
            
        try:
            for camera_name, image in raw_observations.items():
                if image is not None and isinstance(image, np.ndarray):
                    window_name = f"Camera_{camera_name}"
                    
                    # 调整图像大小以适应显示窗口
                    display_image = cv2.resize(image, tuple(self.display_window_size))
                    
                    # 添加文本信息
                    step_text = f"Step: {self.step_count}"
                    cv2.putText(display_image, step_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 【新增】显示动作聚合信息
                    if self.action_aggregator is not None:
                        agg_stats = self.action_aggregator.get_stats()
                        agg_text = f"Action Agg: {agg_stats['method']} ({agg_stats['history_length']}/{agg_stats['window_size']})"
                        cv2.putText(display_image, agg_text, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # 显示图像
                    cv2.imshow(window_name, display_image)
            
            # 非阻塞式等待，保持窗口响应
            cv2.waitKey(1)
            
        except Exception as e:
            log.error(f"❌ 图像显示失败: {e}")
    
    def _close_display_windows(self):
        """关闭所有显示窗口"""
        if self.enable_image_display and self.display_windows_initialized:
            try:
                cv2.destroyAllWindows()
                log.info("🖥️ 图像显示窗口已关闭")
            except Exception as e:
                log.error(f"❌ 关闭显示窗口失败: {e}")
    
    def _validate_inputs(self, camera_obs: tuple) -> bool:
        """
        验证输入数据的有效性
        
        Args:
            camera_obs: 相机观察数据 (raw_obs, tensor_obs)
            
        Returns:
            bool: 数据是否有效
        """
        if self.data_adapter is None:
            return True
            
        # 验证机器人状态（对Monte01进行状态组合处理）
        if self.robot_type == 'monte01':
            # 获取处理后的16维状态
            combined_state = self._get_monte01_dual_arm_state()
            # 创建包含16维状态的RobotJointState
            joint_state = self._create_joint_state_from_array(combined_state)
        else:
            # FR3直接使用原始状态
            joint_state = self.robot.get_joint_states()
            
        if not self.data_adapter.validate_robot_state(joint_state):
            log.warning("⚠️ 机器人状态无效")
            return False
        
        # 验证相机数据
        raw_camera_observations, _ = camera_obs
        if not self.data_adapter.validate_camera_data(raw_camera_observations):
            log.warning("⚠️ 相机数据无效")
            return False
            
        return True
    
    def _should_predict_new_actions(self) -> bool:
        """
        检查是否需要新的推理
        
        Returns:
            bool: 是否需要新推理
        """
        # 无动作序列缓存
        if self.action_sequence is None:
            return True
        
        # 连续失败太多
        if self.failed_actions_count >= self.max_failed_actions:
            return True
        
        # 修复Bug: 使用当前索引而非下一个索引
        current_real_index = self.action_index * self.action_sampling_interval
        return current_real_index >= len(self.action_sequence)
    
    def _predict_new_action_sequence(self, state: np.ndarray, camera_obs: tuple) -> None:
        """
        預測新的動作序列 (已修改，加入延遲補償)
        
        Args:
            state: 當前機器人狀態
            camera_obs: 相機觀察數據 (raw_obs, tensor_obs)
        """
        _, tensor_camera_observations = camera_obs
        
        # 关键修复：验证推理输入状态坐标系（应为CORENETIC）
        log.info(f"🔍 推理输入验证 - 状态坐标系: CORENETIC, shape: {state.shape}")
        if self.robot_type == 'monte01' and len(state) >= 16:
            log.info(f"   左臂关节状态(CORENETIC): {state[:7].round(3)}")
            log.info(f"   右臂关节状态(CORENETIC): {state[8:15].round(3)}")
            log.info(f"   左右夹爪状态: [{state[7]:.3f}, {state[15]:.3f}]")
        else:
            log.info(f"   关节状态: {state[:7].round(3) if len(state) >= 7 else state}")
        
        # 運行ACT推理
        log.info(f"🤖 执行ACT推理以获取新动作序列...")
        predicted_actions = self.inference_engine.predict(
            state=state,  # 确保为CORENETIC坐标系，与训练数据一致
            images=tensor_camera_observations
        )
        
        log.info(f"🔍 ACT原始輸出: shape={predicted_actions.shape}, type={type(predicted_actions)}, {predicted_actions}")
        
        # 處理ACT動作塊輸出
        if predicted_actions.ndim == 2 and predicted_actions.shape[0] > 1:
            # 獲取完整的動作序列塊
            all_actions = predicted_actions[:self.action_chunk_size]
            
            # 應用延遲補償：從序列中移除前面的N個動作
            if self.action_delay_compensation_steps > 0:
                log.info(f"   => 應用延遲補償，跳過 {self.action_delay_compensation_steps} 個動作...")
                compensated_actions = all_actions[self.action_delay_compensation_steps:]
                
                # 安全檢查：如果補償後序列為空，則至少保留最後一個動作
                if len(compensated_actions) == 0:
                    log.warning(f"⚠️ 延遲補償後無可用動作，將使用原始序列的最後一個動作。")
                    compensated_actions = all_actions[-1:]
                
                self.action_sequence = compensated_actions
                log.info(f"📦 緩存已補償的動作序列: {all_actions.shape} → 使用後 {len(self.action_sequence)} 個動作")
            else:
                # 無需補償，使用原始序列
                self.action_sequence = all_actions
                log.info(f"📦 緩存動作序列: {all_actions.shape} → 使用前 {len(self.action_sequence)} 個動作")

        else:
            # 單步動作輸出（向後兼容）
            if predicted_actions.ndim > 1:
                predicted_actions = predicted_actions.squeeze()
            self.action_sequence = predicted_actions.reshape(1, -1)
            log.info(f"📍 單步動作緩存: {self.action_sequence.shape}")

        # 重置索引和失敗计数
        self.action_index = 0  # 索引總是從0開始，因為我們已經修剪了序列
        self.failed_actions_count = 0
        self.steps_since_last_prediction = 0  # 重置推理步数计数
        
        log.info(f"✅ 新動作序列緩存完成: shape={self.action_sequence.shape}")
    
    def _get_next_action(self) -> np.ndarray:
        """
        获取下一个要执行的动作（支持插值平滑）
        
        Returns:
            np.ndarray: 下一个动作
        """
        # 计算真实索引
        real_index = self.action_index * self.action_sampling_interval
        
        # 获取原始动作
        raw_action = self.action_sequence[real_index]
        
        # 动作插值处理 - 减少卡顿
        if self.enable_action_interpolation and self.last_executed_action is not None:
            # 简单线性插值，减少动作跳跃
            alpha = 0.3  # 插值权重，0表示完全使用上一个动作，1表示完全使用新动作
            interpolated_action = (1 - alpha) * self.last_executed_action + alpha * raw_action
            log.debug(f"🔄 动作插值: alpha={alpha}, 原始→插值")
            action = interpolated_action
        else:
            action = raw_action
        
        log.info(f"🎯 获取动作[{self.action_index}→{real_index}]: shape={action.shape}, 序列总shape={self.action_sequence.shape}")
        
        # 记录当前动作用于下次插值
        self.last_executed_action = action.copy()
        
        # 在获取后递增索引
        self.action_index += 1
        
        return action
    
    def detect_and_fix_discontinuity(self, new_action: np.ndarray, context: str = "unknown") -> np.ndarray:
        """
        检测并修复动作不连续性 - 核心修复函数
        
        Args:
            new_action: 新动作
            context: 上下文信息 ("prediction", "execution", "aggregation")
        """
        if not hasattr(self, '_discontinuity_history'):
            self._discontinuity_history = []
        
        # 获取参考动作
        reference_action = None
        if hasattr(self, '_last_executed_action') and self._last_executed_action is not None:
            reference_action = self._last_executed_action
        elif (hasattr(self, 'action_sequence') and self.action_sequence is not None and 
            self.action_index > 0):
            prev_index = (self.action_index - 1) * self.action_sampling_interval
            if prev_index < len(self.action_sequence):
                reference_action = self.action_sequence[prev_index]
        
        if reference_action is None:
            return new_action  # 没有参考，直接返回
        
        # 确保动作维度匹配
        if len(new_action) != len(reference_action):
            log.warning(f"⚠️ 动作维度不匹配: new={len(new_action)}, ref={len(reference_action)}, 跳过不连续性检查")
            return new_action
        
        # 根据机器人类型确定关节数量
        robot_type = self.config.get('robot', {}).get('type', 'fr3')
        if robot_type == 'monte01':
            # Monte01双臂：14个关节（左臂7 + 右臂7）+ 2个夹爪 = 16维
            joint_count = 14  # 只检查关节，不检查夹爪
        else:
            # FR3单臂：7个关节 + 1个夹爪 = 8维
            joint_count = 7
        
        # 计算关节跳跃幅度（不包括夹爪）
        joint_jump = np.max(np.abs(new_action[:joint_count] - reference_action[:joint_count]))
        
        # 检测不连续性
        discontinuity_threshold = self.config.get('action_aggregation', {}).get('discontinuity_threshold', 0.12)
        
        if joint_jump > discontinuity_threshold:
            # 记录不连续性
            discontinuity_info = {
                'timestamp': time.time(),
                'context': context,
                'jump_magnitude': joint_jump,
                'step': self.step_count
            }
            self._discontinuity_history.append(discontinuity_info)
            
            log.warning(f"🚫 {context}不连续检测: {joint_jump:.3f} rad (阈值: {discontinuity_threshold:.3f})")
            
            # 应用修复
            smooth_factor = self.config.get('action_aggregation', {}).get('discontinuity_smooth_factor', 0.3)
            fixed_action = new_action.copy()
            # 只修复关节部分，保持夹爪不变
            fixed_action[:joint_count] = (smooth_factor * new_action[:joint_count] + 
                                        (1 - smooth_factor) * reference_action[:joint_count])
            
            log.info(f"🔧 不连续性修复: {joint_jump:.3f} → {np.max(np.abs(fixed_action[:joint_count] - reference_action[:joint_count])):.3f}")
            return fixed_action
        
        return new_action

    def enhanced_action_execution(self, raw_action: np.ndarray, action_start_time: float = None) -> bool:
        """
        增强的动作执行流程 - 集成所有修复措施
        """
        log.info(f"🔍 enhanced_action_execution输入: shape={raw_action.shape}")
        
        # 1. 预测层修复
        action_after_pred_fix = self.detect_and_fix_discontinuity(raw_action, "prediction")
        
        # 2. 动作聚合
        if self.action_aggregator is not None:
            action_after_agg = self.action_aggregator.process_action(action_after_pred_fix)
        else:
            action_after_agg = action_after_pred_fix
        
        # 3. 执行层修复
        action_after_exec_fix = self.detect_and_fix_discontinuity(action_after_agg, "execution")
        
        # 4. 最终安全检查和执行
        success = self._execute_action_safely(action_after_exec_fix)
        
        # 记录动作执行耗时
        if action_start_time is not None:
            action_elapsed = time.time() - action_start_time
            log.info(f"⏱️  动作执行耗时: {action_elapsed:.3f}s, 成功: {success}")
        else:
            log.info(f"⏱️  动作执行完成, 成功: {success}")
        
        # 5. 记录成功执行的动作
        if success:
            self._last_executed_action = action_after_exec_fix.copy()
            
            # 可选: 保存动作历史用于分析
            if self.config.get('debug', {}).get('save_action_history', False):
                if not hasattr(self, '_action_execution_history'):
                    self._action_execution_history = []
                self._action_execution_history.append({
                    'step': self.step_count,
                    'raw_action': raw_action.copy(),
                    'final_action': action_after_exec_fix.copy(),
                    'success': success
                })
        
        return success
    
    def _execute_action(self, action: np.ndarray) -> bool:
        """
        执行动作并统一处理失败计数
        
        Args:
            action: 要执行的动作
            
        Returns:
            bool: 执行是否成功
        """
        # 检查停止标志，快速退出
        if not self.running or self.emergency_stop:
            log.info("🛑 检测到停止信号，跳过动作执行")
            return False
            
        log.info(f"🔍 _execute_action输入: shape={action.shape}")
        
        try:
            # 记录动作执行开始时间
            action_start_time = time.time()
            # 设置动作执行超时（避免硬件通信阻塞）
            success = self.enhanced_action_execution(action, action_start_time)
        except Exception as e:
            log.error(f"❌ 动作执行异常: {e}")
            success = False
        
        # 统一失败计数逻辑
        if success:
            self.failed_actions_count = 0
        else:
            self.failed_actions_count += 1
            log.warning(f"⚠️ 连续失败: {self.failed_actions_count}/{self.max_failed_actions}")
        
        return success
    
    def _initialize_episode(self) -> None:
        """初始化episode"""
        log.info("🚀 开始ACT推理控制循环")
        log.info(f"   - 控制频率: {self.control_frequency} Hz")
        log.info(f"   - 最大episode长度: {self.max_episode_length} steps")
        log.info(f"   - ACT动作块大小: {self.action_chunk_size}, 采样间隔: {self.action_sampling_interval}")
        if self.action_aggregator is not None:
            agg_stats = self.action_aggregator.get_stats()
            log.info(f"   - 动作聚合: {agg_stats['method']} (窗口: {agg_stats['window_size']})")
        
        self.running = True
        self.step_count = 0
        
        if self.camera_latency_ms > 0 and self.control_frequency > 0:
            control_period_ms = 1000.0 / self.control_frequency
            self.action_delay_compensation_steps = int(round(self.camera_latency_ms / control_period_ms))
            log.info(f"📷 視覺延遲補償已啟用: {self.camera_latency_ms}ms, 控制週期: {control_period_ms:.1f}ms/步")
            log.info(f"   => 將跳過每個新動作序列的前 {self.action_delay_compensation_steps} 個動作。")
        else:
            self.action_delay_compensation_steps = 0

        # 重置动作序列
        self._reset_action_sequence()
        
        # 初始化图像显示窗口
        self._initialize_display_windows()
        
        # 获取初始状态
        if not self._check_robot_health():
            raise RuntimeError("初始机器人状态检查失败")
    
    def _should_terminate(self) -> bool:
        """检查是否应该终止episode"""
        return self.step_count >= self.max_episode_length
    
    def _control_frequency(self) -> None:
        """控制循环频率 - 带实际频率监控"""
        dt = 1.0 / self.control_frequency
        elapsed = time.time() - self.step_start_time
        sleep_time = max(0, dt - elapsed)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            log.warning(f"⚠️ 控制循环延迟: {elapsed:.3f}s > {dt:.3f}s")
        
        # 频率监控
        if self.frequency_monitor_enabled:
            self.monitor_step_count += 1
            current_time = time.time()
            
            if self.last_monitor_time is None:
                self.last_monitor_time = current_time
            
            # 每N步统计一次实际频率
            if self.monitor_step_count >= self.frequency_monitor_window:
                time_span = current_time - self.last_monitor_time
                actual_frequency = self.frequency_monitor_window / time_span
                efficiency = (actual_frequency / self.control_frequency) * 100
                
                log.info(f"📊 实际控制频率: {actual_frequency:.1f}Hz (目标:{self.control_frequency}Hz, 效率:{efficiency:.1f}%)")
                
                # 重置监控计数器
                self.monitor_step_count = 0
                self.last_monitor_time = current_time
    
    def _finalize_episode(self) -> None:
        """结束Episode"""
        self.episode_count += 1
        log.info(f"🎯 Episode {self.episode_count} 结束")
        
        # 重置动作序列为下一个Episode准备
        self._reset_action_sequence()
    
    def run_inference_loop(self):
        """简化后的推理循环 - 主要逻辑集中在30行内"""
        try:
            self._initialize_episode()
            
            while self.running and not self.emergency_stop:
                self.step_start_time = time.time()
                
                try:
                    # 快速检查停止标志
                    if not self.running or self.emergency_stop:
                        break
                        
                    # 1. 获取状态和观察
                    current_state = self._get_current_state()
                    
                    # 快速检查停止标志 - 在耗时操作前
                    if not self.running or self.emergency_stop:
                        break
                        
                    raw_camera_obs, tensor_camera_obs = self._get_camera_observations()
                    camera_obs_tuple = (raw_camera_obs, tensor_camera_obs)
                    
                    # 2. 验证数据
                    if not self._validate_inputs(camera_obs_tuple):
                        continue
                    
                    # 3. 预测新动作（如需要）
                    if self._should_predict_new_actions():
                        log.info(f"🔍 开始新推理 - 索引:{self.action_index * self.action_sampling_interval}, 失败次数:{self.failed_actions_count}")
                        self._predict_new_action_sequence(current_state, camera_obs_tuple)
                    
                    # 4. 执行下一个动作
                    # 快速检查停止标志 - 在动作执行前
                    if not self.running or self.emergency_stop:
                        break
                        
                    action = self._get_next_action()
                    
                    # 再次检查停止标志 - 在执行前
                    if not self.running or self.emergency_stop:
                        break
                        
                    success = self._execute_action(action)
                    
                    if success:
                        self.step_count += 1
                        self.steps_since_last_prediction += 1
                        
                        if self.step_count % 100 == 0:
                            log.info(f"📊 已执行 {self.step_count} 步")
                            # 打印动作聚合统计
                            if self.action_aggregator is not None:
                                agg_stats = self.action_aggregator.get_stats()
                                log.info(f"   动作聚合统计: {agg_stats}")
                            log.info(f"   滑动窗口: {self.steps_since_last_prediction}/{self.sliding_window_size}")
                    
                    # 5. 检查终止条件
                    if self._should_terminate():
                        log.info(f"✅ Episode完成 ({self.step_count} 步)")
                        break
                    
                    # 6. 控制频率
                    self._control_frequency()
                    
                except KeyboardInterrupt:
                    log.info("🛑 检测到键盘中断，正在停止...")
                    break
                except Exception as e:
                    log.error(f"❌ 步骤执行失败: {e}")
                    time.sleep(1.0 / self.control_frequency)
                    continue
            
            log.info(f"🎯 推理循环结束 - 总步数: {self.step_count}")
            
        except Exception as e:
            log.error(f"❌ 推理控制循环发生致命错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._finalize_episode()
            self.running = False
    
    def stop(self):
        """停止推理运行器"""
        log.info("🛑 正在停止ACT推理运行器...")
        self.running = False
        
        # 停止关节位置绘图器
        if hasattr(self, 'joint_plotter') and self.joint_plotter is not None:
            self.joint_plotter.stop_plotting()
            # 保存最终数据
            if self.joint_plotter.save_data:
                self.joint_plotter.save_data(f"final_joint_positions_{int(time.time())}.csv")
            self.joint_plotter.print_statistics()
        
        # 安全停止机器人
        if self.robot:
            try:
                # 停止机器人运动
                self.robot.close()
                log.info("✅ 机器人已安全停止")
            except Exception as e:
                log.error(f"❌ 机器人停止失败: {e}")
        
        # 关闭相机
        for camera_name, camera in self.cameras.items():
            try:
                camera.close()
                log.info(f"✅ 相机 {camera_name} 已断开")
            except Exception as e:
                log.error(f"❌ 相机 {camera_name} 断开失败: {e}")
        
        # 关闭显示窗口
        self._close_display_windows()
        
        log.info("✅ ACT推理运行器已停止")
        exit(0)
    
    def print_statistics(self):
        """打印运行统计信息"""
        log.info("📊 运行统计:")
        log.info(f"   - Episodes: {self.episode_count}")
        log.info(f"   - 总步数: {self.step_count}")
        log.info(f"   - 平均episode长度: {self.step_count / max(1, self.episode_count):.1f}")
        
        # 【新增】打印动作聚合统计
        if self.action_aggregator is not None:
            agg_stats = self.action_aggregator.get_stats()
            log.info(f"   - 动作聚合统计: {agg_stats}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ACT模型真机推理演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # FR3机器人推理
  python run_act_inference.py --robot fr3 --ckpt_dir ./models/act_fr3 --config ./configs/fr3_inference.yaml
  
  # Monte01机器人推理
  python run_act_inference.py --robot monte01 --ckpt_dir ./models/act_monte01 --config ./configs/monte01_inference.yaml
  
  # 自定义控制频率
  python run_act_inference.py --robot fr3 --ckpt_dir ./models/act_fr3 --config ./configs/fr3_inference.yaml --frequency 20
        """
    )
    
    parser.add_argument(
        '--robot', 
        type=str, 
        required=True, 
        choices=['fr3', 'monte01', 'unitree_g1'],
        help='机器人类型'
    )
    
    parser.add_argument(
        '--ckpt_dir', 
        type=str, 
        required=True,
        help='ACT模型检查点目录路径'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='推理配置文件路径'
    )
    
    parser.add_argument(
        '--frequency', 
        type=float, 
        default=10.0,
        help='控制频率 (Hz), 默认: 10.0'
    )
    
    parser.add_argument(
        '--max_steps', 
        type=int, 
        default=1000,
        help='最大episode步数, 默认: 1000'
    )
    
    parser.add_argument(
        '--log_level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别, 默认: INFO'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 配置日志
    log.setLevel(getattr(log, args.log_level))
    
    # 验证输入参数
    if not os.path.exists(args.ckpt_dir):
        log.error(f"❌ 检查点目录不存在: {args.ckpt_dir}")
        return 1
    
    if not os.path.exists(args.config):
        log.error(f"❌ 配置文件不存在: {args.config}")
        return 1
    
    # 创建推理运行器
    runner = None
    try:
        log.info("🚀 启动ACT真机推理演示")
        log.info(f"   - 机器人类型: {args.robot}")
        log.info(f"   - 检查点目录: {args.ckpt_dir}")
        log.info(f"   - 配置文件: {args.config}")
        log.info(f"   - 控制频率: {args.frequency} Hz")
        
        # 初始化运行器
        runner = ACTInferenceRunner(
            config_path=args.config,
            ckpt_dir=args.ckpt_dir,
            robot_type=args.robot
        )
        
        # 更新配置参数
        runner.control_frequency = args.frequency
        runner.max_episode_length = args.max_steps
        
        # 设置信号处理器
        runner._setup_signal_handlers()
        
        # 初始化系统 - 先初始化学习系统，再初始化机器人
        runner.initialize_learning_system()
        runner.initialize_robot()
        
        # 运行推理循环
        runner.run_inference_loop()
        
        # 打印统计信息
        runner.print_statistics()
        
        return 0
        
    except KeyboardInterrupt:
        log.info("⚠️ 用户中断")
        return 0
    except Exception as e:
        log.error(f"❌ 程序运行失败: {e}")
        return 1
    finally:
        if runner:
            runner.stop()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)