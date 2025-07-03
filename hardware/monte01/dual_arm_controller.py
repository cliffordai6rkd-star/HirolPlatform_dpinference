import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Callable
import glog as log
from hardware.monte01.arm import Arm
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

class DualArmController:
    """
    双臂控制器，提供并发控制和协调功能
    """
    def __init__(self, agent):
        self.agent = agent
        self.arm_left:Arm = None
        self.arm_right:Arm = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.arm_states = {"left": "idle", "right": "idle"}
        self.state_lock = threading.Lock()
        
    def initialize(self):
        """初始化双臂，并发移动到起始位置"""
        log.info("Initializing dual arm controller...")
        self.agent.wait_for_ready()
        self.arm_left = self.agent.arm_left()
        self.arm_right = self.agent.arm_right()
        
        # 并发移动到起始位置
        self.concurrent_move_to_start()
        log.info("Dual arm controller initialized successfully")
        
    def concurrent_move_to_start(self):
        """并发移动双臂到起始位置"""
        def move_arm_to_start(arm, arm_name):
            with self.state_lock:
                self.arm_states[arm_name] = "moving_to_start"
            
            log.info(f"Moving {arm_name} arm to start position...")
            arm.move_to_start()
            arm.hold_position_for_duration(0.2)
            
            with self.state_lock:
                self.arm_states[arm_name] = "holding"
            
            log.info(f"{arm_name} arm moved to start position")
        
        # 同时启动两个手臂的移动
        futures = [
            self.executor.submit(move_arm_to_start, self.arm_left, "left"),
            self.executor.submit(move_arm_to_start, self.arm_right, "right")
        ]
        
        # 等待两个手臂都完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"Error in concurrent_move_to_start: {e}")
        
        log.info("Both arms are ready!")
    
    def execute_arm_sequence(self, arm_name: str, actions: List[Callable], hold_other_arm: bool = True):
        """
        执行单个手臂的动作序列
        
        Args:
            arm_name: "left" 或 "right"
            actions: 动作函数列表
            hold_other_arm: 是否保持另一个手臂位置
        """
        log.info(f"Starting execute_arm_sequence for {arm_name} arm with {len(actions)} actions")
        arm = self.arm_left if arm_name == "left" else self.arm_right
        other_arm = self.arm_right if arm_name == "left" else self.arm_left
        other_name = "right" if arm_name == "left" else "left"
        
        def execute_actions():
            with self.state_lock:
                self.arm_states[arm_name] = "executing_sequence"
            
            try:
                for i, action in enumerate(actions):
                    log.info(f"Executing {arm_name} arm action {i+1}/{len(actions)}")
                    action(arm)
            except Exception as e:
                log.error(f"Error executing {arm_name} arm actions: {e}")
            finally:
                with self.state_lock:
                    self.arm_states[arm_name] = "holding"
        
        def hold_other_arm_position():
            if not hold_other_arm:
                return
                
            with self.state_lock:
                self.arm_states[other_name] = "holding_position"
            
            try:
                # 持续保持另一个手臂的位置
                while True:
                    with self.state_lock:
                        if self.arm_states[arm_name] != "executing_sequence":
                            break
                    
                    # 每100ms保持一次位置
                    other_arm.hold_position_for_duration(0.1)
            except Exception as e:
                log.error(f"Error holding {other_name} arm position: {e}")
            finally:
                with self.state_lock:
                    self.arm_states[other_name] = "holding"
        
        # 并发执行
        futures = [self.executor.submit(execute_actions)]
        if hold_other_arm:
            futures.append(self.executor.submit(hold_other_arm_position))
        
        # 等待执行完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"Error in execute_arm_sequence: {e}")
        
        log.info(f"Completed execute_arm_sequence for {arm_name} arm")
    
    def execute_synchronized_actions(self, left_actions: List[Callable], 
                                   right_actions: List[Callable]):
        """同步执行双臂动作"""
        def execute_arm_actions(arm, actions, arm_name):
            with self.state_lock:
                self.arm_states[arm_name] = "synchronized_execution"
            
            try:
                for i, action in enumerate(actions):
                    log.info(f"Executing {arm_name} arm synchronized action {i+1}")
                    action(arm)
            except Exception as e:
                log.error(f"Error in synchronized execution for {arm_name}: {e}")
        
        # 确保两个动作列表长度相同
        max_len = max(len(left_actions), len(right_actions))
        left_actions = left_actions + [lambda arm: arm.hold_position_for_duration(0.1)] * (max_len - len(left_actions))
        right_actions = right_actions + [lambda arm: arm.hold_position_for_duration(0.1)] * (max_len - len(right_actions))
        
        # 并发执行同步动作
        futures = [
            self.executor.submit(execute_arm_actions, self.arm_left, left_actions, "left"),
            self.executor.submit(execute_arm_actions, self.arm_right, right_actions, "right")
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"Error in synchronized execution: {e}")
        
        # 重置状态
        with self.state_lock:
            self.arm_states["left"] = "holding"
            self.arm_states["right"] = "holding"
    
    def get_arm_states(self) -> Dict[str, str]:
        """获取双臂状态"""
        with self.state_lock:
            return self.arm_states.copy()
    
    def emergency_stop(self):
        """紧急停止所有动作"""
        log.warning("Emergency stop activated!")
        self.executor.shutdown(wait=False)
        with self.state_lock:
            self.arm_states["left"] = "stopped"
            self.arm_states["right"] = "stopped"
    
    def shutdown(self):
        """正常关闭控制器"""
        log.info("Shutting down dual arm controller...")
        self.executor.shutdown(wait=True)


def create_demo_left_arm_actions():
    """创建左臂演示动作序列"""
    def move_sideways(arm):
        log.info("Left arm: Moving sideways")
        pose = arm.get_tcp_pose()
        pose_before = pose.copy()
        pose[AXIS_Y, 3] -= 0.1  # Y轴负向移动10cm
        arm.move_to_pose(pose)
        
        pose_after = arm.get_tcp_pose()
        delta = pose_after[:3, 3] - pose_before[:3, 3]
        log.info(f"Left arm moved: {delta}")
    
    def operate_gripper(arm):
        log.info("Left arm: Operating gripper")
        gripper = arm.get_gripper()
        if gripper is not None:
            gripper.gripper_move(0.5)
            arm.hold_position_for_duration(1.0)
            
            gripper.gripper_close()
            arm.hold_position_for_duration(1.0)
            
            gripper.gripper_open()
            arm.hold_position_for_duration(0.5)
    
    def move_up(arm):
        log.info("Left arm: Moving up")
        pose = arm.get_tcp_pose()
        pose[AXIS_Z, 3] += 0.15  # Z轴正向移动15cm
        arm.move_to_pose(pose)
    
    return [move_sideways, operate_gripper, move_up]


def create_demo_right_arm_actions():
    """创建右臂演示动作序列"""
    def move_forward(arm):
        log.info("Right arm: Moving forward")
        pose = arm.get_tcp_pose()
        pose[AXIS_X, 3] += 0.1  # X轴正向移动10cm
        arm.move_to_pose(pose)
    
    def wave_motion(arm):
        log.info("Right arm: Wave motion")
        base_pose = arm.get_tcp_pose()
        
        # 左右摆动
        for i in range(3):
            pose = base_pose.copy()
            pose[AXIS_Y, 3] += 0.05 * (1 if i % 2 == 0 else -1)
            arm.move_to_pose(pose)
            arm.hold_position_for_duration(0.5)
    
    def move_back(arm):
        log.info("Right arm: Moving back to position")
        pose = arm.get_tcp_pose()
        pose[AXIS_X, 3] -= 0.1  # X轴负向移动10cm
        arm.move_to_pose(pose)

    def operate_gripper(arm):
        log.info("Left arm: Operating gripper")
        gripper = arm.get_gripper()
        if gripper is not None:
            gripper.gripper_move(0.5)
            arm.hold_position_for_duration(1.0)
            
            gripper.gripper_close()
            arm.hold_position_for_duration(1.0)
            
            gripper.gripper_open()
            arm.hold_position_for_duration(0.5)
    
    return [operate_gripper, wave_motion, move_back]