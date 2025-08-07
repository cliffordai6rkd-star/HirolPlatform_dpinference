#!/usr/bin/env python3
"""
机器人API适配器 - 简化版
为test/robot_arm_api_test.py提供统一的机器人接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import time
import glog as log


class RobotTestInterface(ABC):
    """
    简化的机器人测试接口 - 只读模式
    """
    
    @abstractmethod
    def get_joint_angles(self) -> List[float]:
        """
        获取关节角度
        
        Returns:
            List[float]: 7个关节角度，弧度制
        """
        pass
    
    @abstractmethod
    def get_robot_info(self) -> Dict[str, Any]:
        """获取机器人基本信息"""
        pass
    
    def close(self) -> None:
        """关闭连接（可选实现）"""
        pass


class Xarm7Adapter(RobotTestInterface):
    """xArm7只读适配器"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._arm = None
        self._is_connected = False
        
        # 尝试初始化硬件
        try:
            from hardware.monte01.xarm7_arm import Xarm7Arm
            self._arm = Xarm7Arm(config)
            
            # 等待初始化
            timeout = 5.0
            start_time = time.time()
            while not self._arm._is_initialized and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if self._arm._is_initialized:
                self._is_connected = True
                log.info(f"xArm7 connected (IP: {config['ip']})")
            else:
                raise TimeoutError("Initialization timeout")
                
        except Exception as e:
            log.warning(f"xArm7 hardware unavailable: {e}")
            self._is_connected = False
            raise RuntimeError(f"Cannot initialize xArm7: {e}")
    
    def get_joint_angles(self) -> List[float]:
        """获取关节角度"""
        if not self._is_connected:
            raise RuntimeError("xArm7 not connected")
            
        try:
            joint_states = self._arm.get_joint_states()
            return joint_states._positions.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to read xArm7 angles: {e}")
    
    def set_teaching_mode(self) -> bool:
        """启用teaching mode"""
        if not self._is_connected:
            return False
        try:
            return self._arm.set_teaching_mode()
        except Exception as e:
            log.warning(f"Failed to set teaching mode: {e}")
            return False
    
    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "type": "xarm7",
            "ip": self._config["ip"],
            "connected": self._is_connected
        }
    
    def close(self) -> None:
        if self._arm:
            self._arm.close()


class Monte01Adapter(RobotTestInterface):
    """Monte01只读适配器"""
    
    def __init__(self, config: Dict[str, Any], arm_side: str = 'left'):
        self._config = config
        self._arm_side = arm_side
        self._robot = None
        self._is_connected = False
        
        # 验证参数
        if arm_side not in ['left', 'right']:
            raise ValueError(f"arm_side must be 'left' or 'right', got: {arm_side}")
        
        # 尝试初始化硬件
        try:
            from hardware.monte01.monte01 import Monte01
            self._robot = Monte01(config)
            self._is_connected = True
            log.info(f"Monte01 connected (IP: {config['ip']}, arm: {arm_side})")
        except Exception as e:
            log.warning(f"Monte01 hardware unavailable: {e}")
            self._is_connected = False
            raise RuntimeError(f"Cannot initialize Monte01: {e}")
    
    def get_joint_angles(self) -> List[float]:
        """获取指定手臂的关节角度"""
        if not self._is_connected:
            raise RuntimeError("Monte01 not connected")
            
        try:
            joint_states = self._robot.get_joint_states()
            
            # 提取指定手臂的7个关节
            if self._arm_side == 'left':
                return joint_states._positions[:7].tolist()
            else:
                return joint_states._positions[7:14].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to read Monte01 {self._arm_side} arm angles: {e}")
    
    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "type": "monte01",
            "arm_side": self._arm_side,
            "ip": self._config["ip"],
            "connected": self._is_connected
        }
    
    def close(self) -> None:
        if self._robot:
            self._robot.close()


class MockAdapter(RobotTestInterface):
    """
    Mock适配器：兼容原有MockAPIWithOffset
    """
    
    def __init__(self, robot_type: str = 'mock'):
        """
        初始化Mock适配器
        
        Args:
            robot_type: 被模拟的机器人类型，影响模拟参数
        """
        self._robot_type = robot_type
        self._current_position = np.array([0.0] * 7)
        
        # 根据机器人类型设置不同的模拟参数
        if robot_type == 'xarm7':
            self._zero_offsets = np.array([0, np.pi/4, -np.pi/6, np.pi/2, 0, -np.pi/3, np.pi/8])
            self._sign_flips = np.array([1, -1, 1, 1, -1, 1, 1])
        elif robot_type.startswith('monte01'):
            self._zero_offsets = np.array([0.1, -0.2, 0.15, -0.3, 0.05, 0.25, -0.1])
            self._sign_flips = np.array([1, 1, -1, 1, -1, 1, 1])
        else:
            # 默认Mock参数（兼容原有测试）
            self._zero_offsets = np.array([0, np.pi/4, -np.pi/6, np.pi/2, 0, -np.pi/3, np.pi/8])
            self._sign_flips = np.array([1, -1, 1, 1, -1, 1, 1])
        
        # 添加随机噪声模拟真实硬件
        self._noise_level = 0.001  # 1毫弧度噪声
        
        log.info(f"Mock adapter created for {robot_type}")
    
    def get_joint_angles(self) -> List[float]:
        """获取模拟的关节角度"""
        
        # 基础位置 + 变换 + 噪声
        transformed_angles = (self._sign_flips * self._current_position + 
                            self._zero_offsets + 
                            np.random.normal(0, self._noise_level, 7))
        
        # 限制在[-π, π]范围内
        wrapped_angles = np.arctan2(np.sin(transformed_angles), 
                                  np.cos(transformed_angles))
        
        angles = wrapped_angles.tolist()
        log.debug(f"Mock {self._robot_type} angles: {[f'{a:.3f}' for a in angles]}")
        
        return angles
    
    def move_to(self, angles: np.ndarray) -> bool:
        """模拟移动到指定位置（用于数据生成）"""
        if len(angles) != 7:
            log.warning(f"Invalid angles length: {len(angles)}")
            return False
        
        # 模拟移动时间
        time.sleep(0.1)
        
        # 更新当前位置（添加一些误差模拟）
        self._current_position = angles + np.random.normal(0, 0.001, 7)
        
        log.debug(f"Mock {self._robot_type} moved to: {[f'{a:.3f}' for a in angles]}")
        return True
    
    def get_robot_info(self) -> Dict[str, Any]:
        """获取Mock机器人信息"""
        return {
            "type": f"mock_{self._robot_type}",
            "connected": True,
            "hardware_available": True,
            "is_simulation": True,
            "current_position": self._current_position.tolist()
        }
    
    def close(self) -> None:
        """关闭Mock适配器"""
        log.info(f"Mock adapter for {self._robot_type} closed")


class TeachModeGuide:
    """
    Teach Mode引导助手
    
    职责：
    - 引导用户手动移动机械臂
    - 实时显示关节角度
    - 收集测试数据
    """
    
    def __init__(self, adapter: RobotTestInterface):
        self.adapter = adapter
        self.collected_data = []
    
    def start_teach_session(self, target_samples: int = 50) -> List[Dict]:
        """
        启动teach mode数据收集会话
        
        Args:
            target_samples: 目标采样数量
            
        Returns:
            List[Dict]: 收集的数据点
        """
        print(f"\n{'='*60}")
        print("🤖 TEACH MODE 数据收集会话")
        print(f"{'='*60}")
        print(f"目标样本数: {target_samples}")
        print(f"机器人类型: {self.adapter.get_robot_info()['type']}")
        
        # 检查硬件连接
        try:
            current_angles = self.adapter.get_joint_angles()
            print(f"✅ 硬件连接正常")
            print(f"当前关节角度: {[f'{a:.3f}' for a in current_angles]}")
        except Exception as e:
            print(f"❌ 硬件连接失败: {e}")
            return []
        
        # 启用teach mode（仅xArm7支持）
        self._enable_teach_mode()
        
        print(f"\n📋 数据收集指南:")
        print(f"1. 手动移动机械臂到不同位置")
        print(f"2. 按 [ENTER] 记录当前位置")
        print(f"3. 按 [q] + [ENTER] 结束收集")
        print(f"4. 尽量覆盖各关节的运动范围")
        
        sample_count = 0
        
        while sample_count < target_samples:
            try:
                # 显示当前状态
                current_angles = self.adapter.get_joint_angles()
                print(f"\n[{sample_count+1}/{target_samples}] 当前角度: {[f'{a:.3f}' for a in current_angles]}")
                
                # 等待用户输入
                user_input = input("按 [ENTER] 记录位置, [q] 退出: ").strip().lower()
                
                if user_input == 'q':
                    print("用户主动结束数据收集")
                    break
                elif user_input == '':
                    # 记录当前位置
                    data_point = {
                        'sample_id': sample_count,
                        'angles': current_angles,
                        'timestamp': time.time()
                    }
                    self.collected_data.append(data_point)
                    sample_count += 1
                    print(f"✅ 已记录样本 {sample_count}")
                else:
                    print("无效输入，请按 [ENTER] 记录或 [q] 退出")
                    
            except KeyboardInterrupt:
                print("\n用户中断，结束数据收集")
                break
            except Exception as e:
                print(f"❌ 读取角度失败: {e}")
                break
        
        print(f"\n📊 数据收集完成!")
        print(f"总样本数: {len(self.collected_data)}")
        
        return self.collected_data
    
    def _enable_teach_mode(self):
        """启用teach mode（如果支持）"""
        try:
            robot_info = self.adapter.get_robot_info()
            if robot_info['type'] == 'xarm7':
                # xArm7支持teach mode
                if hasattr(self.adapter, 'set_teaching_mode'):
                    success = self.adapter.set_teaching_mode()
                    if success:
                        print("✅ xArm7 Teach Mode 已启用")
                    else:
                        print("⚠️  xArm7 Teach Mode 启用失败")
            else:
                print("ℹ️  当前机器人不支持自动teach mode，请手动设置")
        except Exception as e:
            print(f"⚠️  Teach Mode设置失败: {e}")
    
    def export_data(self, filename: str = "teach_mode_data.json"):
        """导出收集的数据"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
        print(f"📁 数据已导出到: {filename}")