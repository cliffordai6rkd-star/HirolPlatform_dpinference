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

from typing import Text, Mapping, Any, Sequence

from hardware.base.arm import ArmBase

from motion.kinematics import PinocchioKinematicsModel as KinematicsModel, UrdfModelManager

from motion import trajectory_planner, trajectory_executor

import glog as log
import time,os
import numpy as np

#=================================== Switch gripper type!===================================
from hardware.monte01.gripper_corenetic import Gripper
# from hardware.monte01.gripper_xarm import Gripper
#=================================== =================== ===================================

from hardware.monte01.trunk import Trunk

from .defs import *

BODY_JOINT_IDS = [1,2,3]
HEAD_JOINT_IDS = [4,5]
LEFT_ARM_JOINT_IDS = [6,7,8,9,10,11,12,]
LEFT_GRIPPER_JOINT_IDS = [13, 14, 15, 16, 17, 18, 19]  # left_drive_gear + all follower joints
RIGHT_ARM_JOINT_IDS = [20,21,22,23,24,25,26]  # Updated for new joint structure  
RIGHT_GRIPPER_JOINT_IDS = [27, 28, 29, 30, 31, 32, 33]  # right_drive_gear + all follower joints

DEFAULT_JP_LEFT = np.array([0.6265732, 1.64933479, -1.2618717, -1.65806019, -6.30063725, 1.58824956, 0.32637656])  # Default joint positions for left arm
DEFAULT_JP_RIGHT = np.array([-1.484266996383667, 2.322235584259033, -1.3403406143188477, -2.110537052154541, -3.470266342163086, 1.7113285064697266, 0.9290168285369873])
class Arm(ArmBase):
    # 类级别的URDF预加载函数
    @classmethod 
    def preload_urdf(cls, urdf_path: str):
        """
        预加载URDF模型到共享缓存中。
        这个函数可以在创建任何Arm实例之前调用，以加速后续的初始化。
        
        Args:
            urdf_path: URDF文件的路径
        """
        model_manager = UrdfModelManager()
        model_manager.get_full_model(urdf_path)
        log.info(f"URDF模型已预加载: {urdf_path}")
    def __init__(self, config: Mapping[Text, Any], hardware_interface: RobotLib, simulator: Monte01Mujoco, isLeft: bool = True):
        super().__init__()
        component_type = COM_TYPE_LEFT if isLeft else COM_TYPE_RIGHT
        self.component_type = COM_TYPE_LEFT if isLeft else COM_TYPE_RIGHT
        self.config = config
        self.joint_ids = LEFT_ARM_JOINT_IDS if isLeft else RIGHT_ARM_JOINT_IDS
        self.gripper_joint_ids = LEFT_GRIPPER_JOINT_IDS if isLeft else RIGHT_GRIPPER_JOINT_IDS
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
        base_link = 'chest_link'
        end_link = 'left_arm_link_7' if isLeft else 'right_arm_link_7'
        try:
            self.kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)
            
            # Initialize body kinematics for coordinate transformations (both real robot and simulation)
            try:
                body_joint_names = ['body_joint_1', 'body_joint_2', 'body_joint_3']
                self.body_joint_names = body_joint_names
                self.body_kinematics = KinematicsModel(urdf_path=urdf_path, base_link='base_link', end_effector_link='chest_link')
                log.info("Body kinematics initialized for coordinate transformations")
            except Exception as e:
                log.error(f"Failed to initialize body kinematics: {e}")
                self.body_kinematics = None
        except Exception as e:
            log.error(f"Failed to load URDF: {e}")

        self.ctrl_dt = 1.0 / config['control_rate']

        self.flange_t_tcp = np.eye(4)

        self.tcp_t_flange = self.flange_t_tcp.T

        self.joint_names = config['joint_names_left'] if isLeft else config['joint_names_right']
        self.robot = hardware_interface
        self.simulator = simulator

        trajectory_args = {'rate': config['control_rate'],
                       'time_func': self.get_time,
                       'logging': config.get('log_trajectory', False)}
        if config['control_mode'] == 'position':
            trajectory_args['pos_setter'] = self.set_joint_positions
        elif config['control_mode'] == 'velocity':
            trajectory_args['vel_setter'] = self.set_joint_velocities
        else:
            raise ValueError('control_mode should be one of position or velocity.')
        self.trajectory_executor = trajectory_executor.OpenTrajectoryExecutor(
        **trajectory_args)

        time.sleep(0.1) # 給予一點緩衝時間
        if hardware_interface is not None:
            hardware_interface.set_arm_enable(component_type, ARM_ENABLE)
            hardware_interface.set_arm_mode(component_type, ARM_MODE_SERVO_MOTION)
            hardware_interface.set_arm_state(component_type, ARM_STATE_SPORT)
            self.component_type = component_type

            gripper_ip = "192.168.11.11" if isLeft else "192.168.11.12"
            try:
                self.gripper = Gripper(config=config['gripper'], ip=gripper_ip, simulator=simulator)
                # Test gripper communication using hardware interface
                # success, _, _ = hardware_interface.get_gripper_position(component_type)
                # if not success:
                #     log.warning(f"Gripper communication test failed, running arm without gripper")
                #     self.gripper_available = False
                #     self.gripper = None
                # else:
                log.info("Gripper initialized and communication verified")
            except Exception as e:
                log.error(f"Failed to initialize gripper: {e}, running arm without gripper")
                self.gripper_available = False
                self.gripper = None

            self.trunk = Trunk(config, hardware_interface, simulator)
        else:
            try:
                self.gripper = Gripper(config=config['gripper'], ip=None, simulator=simulator, is_left=isLeft)
                log.info("Gripper initialized in simulation mode")
            except Exception as e:
                log.error(f"Failed to initialize gripper in simulation: {e}")
                self.gripper_available = False
                self.gripper = None

        self.qs_prev = self.get_joint_positions()
        self.qs_last_valid = self.qs_prev.copy()  # Store last valid joint positions for IK safety check
        self.gripper_available = True  # Track gripper availability

        self.time_log = []
        self.jp_log = []

    def get_gripper(self):
        if not self.gripper_available:
            log.warning("Gripper not available, returning None")
            return None
        return self.gripper
    
    def hold_position_for_duration(self, duration: float):
        """
        在指定的持續時間內，主動保持在當前關節位置。
        """
        log.info(f"Holding current position for {duration} seconds...++++")
        q_hold = self.get_joint_positions()
        log.info(f"q_hold == {q_hold}")
        start_time = self.get_time()
        while self.get_time() - start_time < duration:
            self.set_joint_positions(q_hold)
            time.sleep(self.ctrl_dt)
        log.info("Holding finished.--------------------------------------")
        
    def get_time(self) -> float:
        """獲取當前時間，優先從模擬器獲取，否則使用系統時間。"""
        if self.robot is not None:
            return time.time()
        else:
            return self.simulator.get_time()
    
    def get_joint_positions(self, joint_names: Any = None) -> np.ndarray:
        names = joint_names if joint_names is not None else self.joint_names
        if self.robot is None:
            return self.simulator.get_joint_positions(names)
        success, angles = self.robot.get_arm_servo_angle(self.component_type)
        log.debug(f"get_joint_positions: success={success}, angles={angles}")
        if not success:
            raise RuntimeError("Failed to get joint positions from the robot.")
        return np.array(angles)
    
    def hold_joint_positions(self):
        self.simulator.hold_joint_positions(self.joint_ids)
        if self.gripper_available and self.gripper:
            self.simulator.hold_joint_positions(self.gripper_joint_ids)

    def set_joint_velocities(self, velocities: np.ndarray):
        raise NotImplementedError("set_joint_velocities is not implemented for this arm.")
    
    def set_joint_positions(self, positions: np.ndarray):
        target_positions = {}
        for i, joint_name in enumerate(self.joint_names):
            target_positions[self.joint_ids[i]] = positions[i]

        if self.robot:
            self.robot.set_arm_mode(self.component_type, ARM_MODE_SERVO_MOTION)
            self.robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
            self.robot.set_arm_servo_angle_j(self.component_type, list(positions), 1, 0, 1)
        if self.simulator:
            self.simulator.set_joint_positions(target_positions)

    def get_flange_pose(self) -> np.matrix:
        """Gets the pose of the flange.
        """
        return self.kinematics.fk(self.get_joint_positions())
    def ik(self, target: np.ndarray, seed: np.ndarray = None) -> np.ndarray:
        """Gets joint configuration via IK.
        
        Args:
            target: The target pose of the flange in base frame.
            seed: Optional seed for the IK solver.
        
        Returns:
            Joint positions that achieve the target pose, or None if no solution is found.
        """
        return self.kinematics.ik(target, seed=seed, max_iter=5000)
    def get_tcp_pose_chest(self) -> np.matrix:
        """Get TCP pose in chest_link frame (legacy behavior)"""
        return self.get_flange_pose() @ self.flange_t_tcp
    
    def get_tcp_pose(self) -> np.matrix:
        """Get TCP pose in world/base frame"""
        # Sync robot state to simulator for consistent calculations
        self.sync_robot_state_to_simulator()
        
        try:
            # Get TCP pose in chest_link frame
            tcp_in_chest = self.get_tcp_pose_chest()
            
            # Get transformation from base_link to chest_link
            if self.robot is not None and self.body_kinematics is not None:
                # For real robot: get actual body joint positions
                body_positions = self.get_body_joint_positions()
                world_to_chest = self.body_kinematics.fk(body_positions)
            else:
                # For simulation: get body joint positions from simulator  
                if hasattr(self, 'body_joint_names') and self.simulator is not None:
                    body_positions = self.simulator.get_joint_positions(self.body_joint_names)
                    if self.body_kinematics is not None:
                        world_to_chest = self.body_kinematics.fk(body_positions)
                    else:
                        world_to_chest = np.eye(4)
                else:
                    world_to_chest = np.eye(4)
            # Transform TCP pose from chest_link frame to world frame
            tcp_in_world = world_to_chest @ tcp_in_chest
            return tcp_in_world
            
        except Exception as e:
            log.error(f"Failed to get TCP pose in world frame: {e}")
            # Fallback to chest frame pose
            return self.get_tcp_pose_chest()
    
    def move_thru_joint_targets(self, targets: Sequence[np.ndarray],blocking: bool = True) -> bool:
        cur_jp = self.get_joint_positions()
        log.info('before constructing trajectory from ' +
                    ','.join(map(str, cur_jp)) + ' to ' +
                    ','.join(map(str, targets[-1])))
        trajectory = trajectory_planner.TimeOptimalTrajectoryWrapper(
        [cur_jp] + targets, self.config['max_deviation'],
        self.config['jvel_limit'], self.config['jacc_limit'])
        log.info('after constructing trajectory')
        return self.execute_trajectory(
        trajectory, timeout=trajectory.get_duration() + 0.5, blocking=blocking)
    
    def execute_trajectory(
        self, trajectory: trajectory_planner.TimeOptimalTrajectoryWrapper,
        timeout: float = None, blocking: bool = True) -> bool:
        self.trajectory_executor.follow_trajectory(trajectory)
        if blocking:
            return self.wait_for_trajectory_done(timeout)
        else:
            return True
        
    def log_trajectory(self, timestamp: float):
        # self.time_log.append(timestamp)
        # self.jp_log.append(self.get_joint_positions())
        pass

    def wait_for_trajectory_done(self, timeout: float = None) -> bool:
        result = self.trajectory_executor.wait(
        timeout=timeout, callback=self.log_trajectory)
        return result
    
    def move_to_joint_target(self, target: np.ndarray, blocking: bool = True) -> bool:
        """Gets current jp and directly moves to target.

        Using the two waypoint trajectory.
        """
        success = self.move_thru_joint_targets([target], blocking)
        if success and blocking:
            # Update last valid joint positions after successful movement
            self.qs_last_valid = target.copy()
        return success

    def get_joint_target_from_pose(self, target: np.matrix,
                                start: np.ndarray = None, 
                                max_joint_diff: float = 1.0):
        """Gets joint configuration via IK with safety checks.
        
        Args:
            target: Target TCP pose in base frame
            start: Seed joint positions for IK solver (if None, uses current positions)
            max_joint_diff: Maximum allowed difference per joint from last valid position (radians)
            
        Returns:
            (success, joint_positions): Tuple of success flag and joint positions
        """
        if start is None:
            start = self.get_joint_positions()
            
        flange_target = target @ self.tcp_t_flange
        ik_result = self.ik(flange_target, seed=start)
        
        # Handle different IK return types
        if isinstance(ik_result, tuple):
            success, jp = ik_result
        else:
            # Some IK implementations return None on failure or joint positions on success
            if ik_result is None:
                success, jp = False, None
            else:
                success, jp = True, ik_result
        
        if not success or jp is None:
            log.warning("IK solver failed to find solution, returning last valid joint positions")
            return False, self.qs_last_valid.copy()
        
        # Safety check: compare with last valid joint positions
        joint_diff = np.abs(jp - self.qs_last_valid)
        max_diff = np.max(joint_diff)
        
        if max_diff > max_joint_diff:
            log.error(f"IK solution has large joint change (max diff: {max_diff:.3f} rad > {max_joint_diff:.3f} rad)")
            log.error(f"Previous valid JP: {self.qs_last_valid}")
            log.error(f"New IK solution JP: {jp}")
            log.error(f"Joint differences: {joint_diff}")
            log.error("Rejecting IK solution, returning last valid joint positions")
            return False, self.qs_last_valid.copy()
        
        # Update last valid joint positions if solution is acceptable
        self.qs_last_valid = jp.copy()
        return True, jp
    
    def get_body_joint_positions(self) -> np.ndarray:
        if self.robot is not None:
            success,positions,_,_ = self.robot.get_joint_state(BODY_JOINT_IDS)
            log.debug(f"get_body_joint_positions from robot: success={success}, positions={positions}")
            if not success:
                log.error(f"Failed to get body joint positions: {success}")
                return np.zeros(len(BODY_JOINT_IDS))
            return np.array(positions)
        else:
            positions = self.simulator.get_joint_positions(self.body_joint_names)
            log.debug(f"get_body_joint_positions from simulator: positions={positions}")
            return positions

    def sync_robot_state_to_simulator(self):
        """Sync current real robot joint states to simulator for visualization"""
        if self.robot is not None and self.simulator is not None:
            try:
                # Use Agent's dual arm sync method to avoid conflicts
                # Try to get agent reference from the robot controller
                if hasattr(self, '_agent_ref') and self._agent_ref is not None:
                    self._agent_ref.sync_dual_arms_to_simulator()
                else:
                    # Fallback to individual arm sync (original behavior)
                    # Sync body joint positions
                    body_positions = self.get_body_joint_positions()
                    body_joint_targets = {}
                    for i, joint_id in enumerate(BODY_JOINT_IDS):
                        body_joint_targets[joint_id] = body_positions[i]
                    
                    # Sync current arm joint positions
                    arm_positions = self.get_joint_positions()
                    arm_joint_targets = {}
                    for i, joint_id in enumerate(self.joint_ids):
                        arm_joint_targets[joint_id] = arm_positions[i]
                    
                    # Combine and set all joint positions in simulator
                    all_joint_targets = {**body_joint_targets, **arm_joint_targets}
                    self.simulator.set_joint_positions(all_joint_targets)
                    log.debug(f"Synced real robot state to simulator")
                
            except Exception as e:
                log.warning(f"Failed to sync robot state to simulator: {e}")

    def move_to_pose(self, target: np.matrix, blocking: bool = True) -> bool:
        """Moves to the target that specifies TCP pose in world/base frame.
        Converts the target from world frame to chest_link frame before IK.
        """
        # For real robot mode: sync all joint states to simulator for correct visualization
        self.sync_robot_state_to_simulator()
        
        # Convert target pose from world frame to chest_link frame
        try:
            # Get current transformation from world to chest_link
            if self.robot is not None and self.body_kinematics is not None:
                # For real robot: get actual body joint positions
                body_positions = self.get_body_joint_positions()
                world_to_chest = self.body_kinematics.fk(body_positions)
                log.info(f"Real robot body joints: {body_positions}")
            else:
                # For simulation: get body joint positions from simulator
                if hasattr(self, 'body_joint_names') and self.simulator is not None:
                    body_positions = self.simulator.get_joint_positions(self.body_joint_names)
                    if self.body_kinematics is not None:
                        world_to_chest = self.body_kinematics.fk(body_positions)
                    else:
                        world_to_chest = np.eye(4)
                    log.info(f"Simulation body joints: {body_positions}")
                else:
                    world_to_chest = np.eye(4)
                    log.warning("Using identity transform (no body kinematics)")
            
            # Transform target from world frame to chest_link frame
            # target_world -> target_chest = inv(world_to_chest) * target_world
            chest_to_world = np.linalg.inv(world_to_chest)
            target_in_chest_frame = chest_to_world @ target
            
            # log.info(f"Target in world frame:\n{target}")
            # log.info(f"World to chest transform:\n{world_to_chest}")
            # log.info(f"Target in chest frame:\n{target_in_chest_frame}")
            
        except Exception as e:
            log.error(f"Failed to transform target pose: {e}")
            log.warning("Using target pose as-is (assuming it's already in chest frame)")
            target_in_chest_frame = target
        
        success, jp = self.get_joint_target_from_pose(target_in_chest_frame, self.get_joint_positions())
        if success:
            self.move_to_joint_target(jp, blocking=blocking)
        else:
            log.error("Failed to get joint target from pose, will NOT move!!")
        return success
    
    def is_trajectory_done(self) -> bool:
        """Check if current trajectory execution is complete.
        """
        return self.trajectory_executor.finished == True

    def print_state(self):
        if self.robot is not None:
            success, pos, vel, effort = self.robot.get_joint_state(self.joint_ids)
            log.info(f"Arm state: \nsuccess: {success}\npos: {pos}\nvel: {vel}\neffort: {effort}\n")

    def move_to_start(self):
        """Move arm to start position, supporting both real robot and simulation modes"""
        try:
            if self.robot is not None:
                # Real robot mode
                if self.component_type == COM_TYPE_LEFT:
                    # Left arm real robot
                    robot = self.robot
                    robot.set_arm_enable(self.component_type, ARM_ENABLE)
                    robot.set_arm_mode(self.component_type, ARM_MODE_POSITION_CTRL)
                    robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
                    success = robot.set_arm_servo_angle(self.component_type, DEFAULT_JP_LEFT, 1, 0, 1)
                    log.info(f"Left arm move to start position success: {success}")
                    time.sleep(1)
                elif self.component_type == COM_TYPE_RIGHT:
                    # Right arm real robot 
                    robot = self.robot
                    robot.set_arm_enable(self.component_type, ARM_ENABLE)
                    robot.set_arm_mode(self.component_type, ARM_MODE_POSITION_CTRL)
                    robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
                    success = robot.set_arm_servo_angle(self.component_type, DEFAULT_JP_RIGHT, 1, 0, 1)
                    log.info(f"Right arm move to start position success: {success}")
                    time.sleep(1)
            else:
                # Simulation mode
                if self.component_type == COM_TYPE_LEFT:
                    # Left arm simulation
                    start_positions = DEFAULT_JP_LEFT
                    log.info("Moving left arm to start position in simulation")
                else:
                    # Right arm simulation
                    start_positions = DEFAULT_JP_RIGHT
                    log.info("Moving right arm to start position in simulation")
                
                # Use the trajectory-based movement for simulation
                success = self.move_to_joint_target(start_positions, blocking=True)
                if success:
                    log.info(f"Arm moved to start position successfully in simulation")
                else:
                    log.warning(f"Failed to move arm to start position in simulation")
                    
        except Exception as e:
            log.error(f"Error in move_to_start: {e}")
            # Fallback: try using trajectory-based movement
            try:
                start_positions = DEFAULT_JP_LEFT if self.component_type == COM_TYPE_LEFT else np.array([-0.6265732, 1.64933479, 1.2618717, -1.65806019, 6.30063725, 1.58824956, -0.32637656])
                self.move_to_joint_target(start_positions, blocking=True)
                log.info("Fallback move_to_start completed")
            except Exception as fallback_error:
                log.error(f"Fallback move_to_start also failed: {fallback_error}")
    
    def get_tf_transform(self, parent_frame_id, child_frame_id, start_position):
        ret, transform = self.robot.get_tf_transform(parent_frame_id, child_frame_id, start_position)
        if not ret:
            log.error(f"get_tf_transform Failed!")
            return np.eye(4)
        log.info(f"get_tf_transform\n{transform}")
        return transform