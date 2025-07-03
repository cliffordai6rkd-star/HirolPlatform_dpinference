from typing import Text, Mapping, Any, Optional, Dict, Sequence, Union, Tuple
from threading import Thread
import threading
import numpy as np

from hardware.base.robot import Robot

from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco
import importlib.util
import os
from .defs import ROBOTLIB_SO_PATH
spec = importlib.util.spec_from_file_location(
    "RobotLib", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), ROBOTLIB_SO_PATH))
)
RobotLib_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RobotLib_module)
RobotLib = RobotLib_module.Robot

from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco

from .camera import Camera
import glog as log
from hardware.monte01.trunk import Trunk
from data_types.se3 import Transform

class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any], use_real_robot=False, start_viewer_main_loop=True):
        
        self.robot = None
        self._arms_ready = threading.Event() # 一個事件旗標，用來表示手臂是否已載入完成
        self._arm_left_instance = None
        self._arm_right_instance = None
        self.sim = None
        self.config = config
        self.use_real_robot = use_real_robot
        self.start_viewer_main_loop = start_viewer_main_loop
        self._disable_motion = False
        self._name = config.get('name', 'Agent')
        
        # VR teleoperation parameters from config
        vr_config = config.get('vr_teleop', {})
        self.movement_speed = vr_config.get("movement_speed", 0.001)
        self.rotation_speed = vr_config.get("rotation_speed", 0.002)
        self.gripper_speed = vr_config.get("gripper_speed", 0.01)
        self.max_movement_speed = vr_config.get("max_movement_speed", 0.008)
        self.min_movement_speed = vr_config.get("min_movement_speed", 0.0005)
        self.max_gripper_speed = vr_config.get("max_gripper_speed", 0.05)
        self.min_gripper_speed = vr_config.get("min_gripper_speed", 0.001)
        self.max_gripper_position = vr_config.get("max_gripper_position", 1.0)
        
        self._load_thread = threading.Thread(
            target=self._load_all_in_background, 
            args=(config, use_real_robot, start_viewer_main_loop),
            daemon=True
        )
        self._load_thread.start()
        log.info("Agent 初始化已發起，正在背景載入模型...")

        self.camera = Camera()

    def _load_all_in_background(self, config: Mapping[Text, Any], use_real_robot: bool, start_viewer_main_loop: bool):
        """
        這個函式在背景執行緒中運行，包含了所有耗時的初始化操作。
        """
        print("背景載入執行緒已啟動...")
        
        # Import Arm class and setup hardware interface
        from hardware.monte01.arm import Arm
        print("使用 RobotLib 實現")
        
        if use_real_robot:
            self.robot = RobotLib("192.168.11.3:50051", "", "")
            print("Robot connection established.")
        else:
            self.robot = None

        # 模擬器的初始化通常很快，但也可以放在這裡
        self.sim = Monte01Mujoco()
        if start_viewer_main_loop:
            # Traditional mode: sim_thread runs the full viewer loop
            self.sim_thread = threading.Thread(target=self.sim.start, daemon=True)
            self.sim_thread.start()
        else:
            # VR mode: only start viewer and simulation thread, main loop handled externally
            self.sim_thread = self.sim.start_viewer_only()

        print("正在初始化躯体组件...")
        self.trunk = Trunk(config, self.robot, self.sim)
        
        print("正在预加载URDF模型...")
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['arm']['urdf_path']))
        Arm.preload_urdf(urdf_path)
        #TODO: load urdf once, and construct reduced model separately
        # --- 這裡是最耗時的部分 ---
        print("正在載入左臂...")
        # Use RobotLib instance as hardware interface
        self._arm_left_instance = Arm(config=config['arm'], hardware_interface=self.robot, simulator=self.sim, isLeft=True, trunk=self.trunk)
        self._arm_left_instance._agent_ref = self  # Set agent reference for sync
        
        print("正在載入右臂...")
        # Use RobotLib instance as hardware interface
        self._arm_right_instance = Arm(config=config['arm'], hardware_interface=self.robot, simulator=self.sim, isLeft=False, trunk=self.trunk)
        self._arm_right_instance._agent_ref = self  # Set agent reference for sync
        # --- 耗時部分結束 ---

        print("所有手臂模型已載入完成！")
        # 設定事件，通知其他執行緒，手臂已經準備好了
        self._arms_ready.set()

    def arm_left(self):
        self._arms_ready.wait() # 等待事件被設定
        return self._arm_left_instance
    
    def arm_right(self):
        self._arms_ready.wait() # 等待事件被設定
        return self._arm_right_instance
    
    def wait_for_ready(self, timeout: float = None):
        """提供一個方法來等待所有資源載入完成。"""
        print("主程式正在等待所有資源載入完成...")
        ready = self._arms_ready.wait(timeout=timeout)
        if ready:
            print("資源已就緒！")
        else:
            print(f"等待超時 ({timeout}秒)！")
        return ready
    
    def head_front_camera(self) -> Camera:
        return self.camera
    
    def sync_dual_arms_to_simulator(self):
        """Sync both arms' real robot joint states to simulator for visualization"""
        if self.robot is not None and self.sim is not None:
            try:
                # Wait for arms to be ready
                if not self._arms_ready.is_set():
                    return
                    
                # Get body joint positions via trunk component
                body_positions = self.trunk.get_body_joint_positions()
                all_joint_targets = {}
                
                # Add body joints
                BODY_JOINT_IDS = [1, 2, 3]
                for i, joint_id in enumerate(BODY_JOINT_IDS):
                    all_joint_targets[joint_id] = body_positions[i]
                
                # Add left arm joints
                left_positions = self._arm_left_instance.get_joint_positions()
                for i, joint_id in enumerate(self._arm_left_instance.get_joint_ids()):
                    all_joint_targets[joint_id] = left_positions[i]
                
                # Add right arm joints  
                right_positions = self._arm_right_instance.get_joint_positions()
                for i, joint_id in enumerate(self._arm_right_instance.get_joint_ids()):
                    all_joint_targets[joint_id] = right_positions[i]
                
                # Set all joint positions in simulator at once
                self.sim.set_joint_positions(all_joint_targets)
                
            except Exception as e:
                import glog as log
                log.warning(f"Failed to sync dual arms to simulator: {e}")
    
    # VR Teleoperation Interface Methods
    @property
    def robot_name(self):
        """Returns the name of the robot."""
        return self._name
    
    @property
    def name(self):
        """Returns the name of the robot."""
        return self._name
    
    @property
    def disable_motion(self):
        """Returns the disable motion flag."""
        return self._disable_motion

    @disable_motion.setter
    def disable_motion(self, value: bool):
        """Sets the disable motion flag."""
        if not isinstance(value, bool):
            raise ValueError("disable_motion must be a boolean")
        self._disable_motion = value

    def get_tcp_pose(self, arm: Text) -> Optional[Dict[Text, Transform]]:
        """Gets the flange pose of the arm.
        
        Args:
            arm: side of arm, 'left' or 'right' or 'both'
        Returns:
            flange pose of the arm as Transform objects
        """
        try:
            if arm == "left":
                pose_matrix = self.arm_left().get_tcp_pose()
                transform = Transform(matrix=pose_matrix)
                return {"left": transform}
            elif arm == "right": 
                pose_matrix = self.arm_right().get_tcp_pose()
                transform = Transform(matrix=pose_matrix)
                return {"right": transform}
            elif arm == "both":
                left_pose = self.arm_left().get_tcp_pose()
                right_pose = self.arm_right().get_tcp_pose()
                return {
                    "left": Transform(matrix=left_pose),
                    "right": Transform(matrix=right_pose)
                }
            else:
                raise ValueError(f"Invalid arm name: {arm}, must be 'left', 'right', or 'both'")
        except Exception as e:
            log.error(f"Failed to get flange pose for {arm}: {e}")
            return None

    def set_arm_servo_flange_pose(self, flange_pose: Dict[str, Transform | Sequence[float]]):
        """Sets the servo flange pose of the arm.
        
        Args:
            flange_pose: Dictionary with 'left' and/or 'right' flange poses
        """
        if self._disable_motion:
            return
            
        try:
            if "left" in flange_pose:
                left_transform = flange_pose["left"]
                if isinstance(left_transform, Sequence):
                    left_transform = Transform(xyz=left_transform[0:3], rot=left_transform[3:7])
                
                pose_matrix = left_transform.matrix
                
                # Transform target pose from world frame to chest_link frame 
                # This is needed for both real robot and simulation modes
                try:
                    if self.trunk is not None:
                        world_to_chest = self.trunk.get_world_to_chest_transform()
                        body_positions = self.trunk.get_body_joint_positions()
                        log.debug(f"Body joints: {body_positions}")
                    else:
                        world_to_chest = np.eye(4)
                        log.warning("No trunk component available, using identity transform")
                    
                    # Transform target from world frame to chest_link frame
                    # target_world -> target_chest = inv(world_to_chest) * target_world
                    pose_matrix = np.linalg.inv(world_to_chest) @ pose_matrix
                    
                except Exception as e:
                    log.error(f"Failed to transform target pose: {e}")
                    log.warning("Using target pose as-is (assuming it's already in chest frame)")
                
                self.arm_left().move_to_pose(pose_matrix)
                
            if "right" in flange_pose:
                right_transform = flange_pose["right"]
                if isinstance(right_transform, Sequence):
                    right_transform = Transform(xyz=right_transform[0:3], rot=right_transform[3:7])
                
                pose_matrix = right_transform.matrix
                
                # Transform target pose from world frame to chest_link frame 
                # This is needed for both real robot and simulation modes
                try:
                    if self.trunk is not None:
                        world_to_chest = self.trunk.get_world_to_chest_transform()
                        body_positions = self.trunk.get_body_joint_positions()
                        log.debug(f"Body joints: {body_positions}")
                    else:
                        world_to_chest = np.eye(4)
                        log.warning("No trunk component available, using identity transform")
                    
                    # Transform target from world frame to chest_link frame
                    # target_world -> target_chest = inv(world_to_chest) * target_world
                    pose_matrix = np.linalg.inv(world_to_chest) @ pose_matrix
                    
                except Exception as e:
                    log.error(f"Failed to transform target pose: {e}")
                    log.warning("Using target pose as-is (assuming it's already in chest frame)")
                
                self.arm_right().move_to_pose(pose_matrix)
                    
        except Exception as e:
            log.error(f"Failed to set servo flange pose: {e}")

    def get_joint_positions(self, arm_side: str) -> Optional[Sequence[float]]:
        """Gets the joint positions of the robot.
        
        Args:
            arm_side: The side of the arm ('left' or 'right').
        Returns:
            Sequence[float]: The joint positions of the robot.
        """
        try:
            if arm_side == "left":
                return self.arm_left().get_joint_positions().tolist()
            elif arm_side == "right":
                return self.arm_right().get_joint_positions().tolist()
            else:
                raise ValueError(f"Invalid arm side: {arm_side}, must be 'left' or 'right'")
        except Exception as e:
            log.error(f"Failed to get joint positions for {arm_side}: {e}")
            return None

    def set_joint_positions(self, arm_side: str, joint_positions: Sequence[float]):
        """Sets the joint positions of the robot.
        
        Args:
            arm_side: The side of the arm ('left' or 'right').
            joint_positions: The joint positions to set.
        """
        if self._disable_motion:
            return
            
        try:
            jp_array = np.array(joint_positions)
            if arm_side == "left":
                self.arm_left().move_to_joint_target(jp_array, blocking=False)
            elif arm_side == "right":
                self.arm_right().move_to_joint_target(jp_array, blocking=False)
            else:
                raise ValueError(f"Invalid arm side: {arm_side}, must be 'left' or 'right'")
        except Exception as e:
            log.error(f"Failed to set joint positions for {arm_side}: {e}")

    def get_gripper_state(self, gripper_side: str) -> Optional[Sequence[float]]:
        """Gets the gripper state of the robot.
        
        Args:
            gripper_side: The side of the gripper ('left' or 'right').
        Returns:
            Sequence[float]: The state of the gripper.
        """
        try:
            if gripper_side == "left":
                gripper = self.arm_left().get_gripper()
                if gripper is None:
                    return [0.0]
                return [gripper.get_position()]
            elif gripper_side == "right":
                gripper = self.arm_right().get_gripper()
                if gripper is None:
                    return [0.0] 
                return [gripper.get_position()]
            else:
                raise ValueError(f"Invalid gripper side: {gripper_side}, must be 'left' or 'right'")
        except Exception as e:
            log.error(f"Failed to get gripper state for {gripper_side}: {e}")
            return [0.0]

    def set_gripper_state(self, gripper_side: str, gripper_state: float):
        """Sets the gripper state of the robot.
        
        Args:
            gripper_side: The side of the gripper ('left' or 'right').
            gripper_state: The state of the gripper.
        """
        if self._disable_motion:
            return
            
        try:
            if gripper_side == "left":
                gripper = self.arm_left().get_gripper()
                if gripper is not None:
                    gripper.gripper_move(gripper_state)
            elif gripper_side == "right":
                gripper = self.arm_right().get_gripper()
                if gripper is not None:
                    gripper.gripper_move(gripper_state)
            else:
                raise ValueError(f"Invalid gripper side: {gripper_side}, must be 'left' or 'right'")
        except Exception as e:
            log.error(f"Failed to set gripper state for {gripper_side}: {e}")

    def stop_all_motion(self):
        """Stops all motion of the robot."""
        self.disable_motion = True
        stop_thread = threading.Thread(target=self.stop_arm_motion, args=("left",), daemon=True)
        stop_thread.start()
        self.stop_arm_motion("right")

    def resume_all_motion(self):
        """Resumes all motion of the robot."""
        self.disable_motion = False

    def stop_arm_motion(self, arm: str):
        """Stops motion for a specific arm.
        
        Args:
            arm: The arm to stop ('left' or 'right').
        """
        try:
            if arm == "left":
                current_pos = self.arm_left().get_joint_positions()
                self.arm_left().set_joint_positions(current_pos)
            elif arm == "right":
                current_pos = self.arm_right().get_joint_positions()
                self.arm_right().set_joint_positions(current_pos)
            else:
                raise ValueError(f"Invalid arm: {arm}, must be 'left' or 'right'")
        except Exception as e:
            log.error(f"Failed to stop motion for {arm}: {e}")

    def enable_interactive_control(self, arm: Text = "left") -> bool:
        """Enables interactive control for the arm.
        
        Args:
            arm: side of arm, 'left' or 'right'
        Returns:
            True if successful, False otherwise
        """
        log.info(f"Interactive control enabled for {arm} arm")
        return True

    def enable_position_control(self, arm: Text = "left") -> bool:
        """Enables position control for the arm.
        
        Args:
            arm: side of arm, 'left' or 'right'
        Returns:
            True if successful, False otherwise
        """
        log.info(f"Position control enabled for {arm} arm")
        return True

    def enable_servo_control(self, arm: Text = "left") -> bool:
        """Enables servo control for the arm.
        
        Args:
            arm: side of arm, 'left' or 'right'
        Returns:
            True if successful, False otherwise
        """
        log.info(f"Servo control enabled for {arm} arm")
        return True

    def move_to_start_position(self, arm: str = "both"):
        """Move arms to start position.
        
        Args:
            arm: Which arm to move ('left', 'right', or 'both')
        """
        try:
            if arm in ["left", "both"]:
                self.arm_left().move_to_start()
            if arm in ["right", "both"]:
                self.arm_right().move_to_start()
            log.info(f"Moved {arm} arm(s) to start position")
        except Exception as e:
            log.error(f"Failed to move {arm} to start position: {e}")

    def sync_to_simulator(self):
        """Sync robot state to simulator for visualization."""
        try:
            self.arm_left().sync_robot_state_to_simulator()
            self.arm_right().sync_robot_state_to_simulator()
        except Exception as e:
            log.warning(f"Failed to sync to simulator: {e}")

    # Backward compatibility methods that map to the Arm API
    def get_arm_jpos(self, arm: Text = "left") -> Optional[Tuple]:
        """Gets the joint positions of the arm (compatibility method).
        
        Args:
            arm: side of arm, 'left' or 'right'
        Returns:
            Tuple of (positions, velocities, efforts) - velocities and efforts are None
        """
        positions = self.get_joint_positions(arm)
        if positions is not None:
            return (np.array(positions), None, None)
        return None

    def set_arm_jpos(self, jpos: Union[Sequence, np.ndarray], speed: float, arm: Text, wait: bool = True):
        """Sets the joint positions of the arm (compatibility method).
        
        Args:
            jpos: joint positions of the arm
            speed: joint speed (ignored in Arm)
            arm: side of arm, 'left' or 'right'
            wait: wait for the motion to complete (ignored in Arm)
        """
        self.set_joint_positions(arm, jpos)

    def set_arm_jpos_j(self, jpos: Union[Sequence, np.ndarray], arm: Text):
        """Sets the joint positions of the arm with servo control (compatibility method).
        
        Args:
            jpos: joint positions of the arm
            arm: side of arm, 'left' or 'right'
        """
        self.set_joint_positions(arm, jpos)

    def get_gripper_position(self, arm: Text = "left") -> Tuple[float, float]:
        """Gets position and effort of the gripper (compatibility method).
        
        Args:
            arm: side of arm, 'left' or 'right'
        Returns:
            Tuple of (position, effort)
        """
        state = self.get_gripper_state(arm)
        if state is not None and len(state) > 0:
            return (state[0], 0.0)  # Return position and zero effort
        return (0.0, 0.0)

    def set_gripper_position(self, arm: Text, position: float):
        """Sets the gripper position (compatibility method).
        
        Args:
            arm: side of arm, 'left' or 'right'
            position: position of the gripper
        """
        self.set_gripper_state(arm, position)
