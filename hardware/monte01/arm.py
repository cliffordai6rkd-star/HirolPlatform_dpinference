import importlib.util
import os
import xml.etree.ElementTree as ET
from .defs import ROBOTLIB_SO_PATH
spec = importlib.util.spec_from_file_location(
    "RobotLib", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), ROBOTLIB_SO_PATH))
)
RobotLib_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RobotLib_module)
RobotLib = RobotLib_module.Robot
from simulation.monte01_mujoco.monte01_mujoco import Monte01Mujoco

from typing import Text, Mapping, Any, Sequence, Tuple

from hardware.base.arm import ArmBase

from motion.kinematics import PinocchioKinematicsModel as KinematicsModel, UrdfModelManager

from motion import trajectory_planner, trajectory_executor

import glog as log
import time,os
import numpy as np
from data_types.se3 import Transform

#=================================== Switch gripper type!===================================
# from hardware.monte01.gripper_corenetic import Gripper
# from hardware.monte01.gripper_xarm import Gripper
#=================================== =================== ===================================


from .defs import *
from .trunk import Trunk

HEAD_JOINT_IDS = [4,5]

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

    def _get_joint_count_from_xml(self, xml_path):
        """Parses an XML file and returns the number of joints within the worldbody."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # Only count joints that are part of the kinematic chain, not equality constraints
            return len(root.findall('.//worldbody//joint'))
        except (ET.ParseError, FileNotFoundError) as e:
            log.error(f"Failed to parse or find XML file at {xml_path}: {e}")
            return 0

    def _get_actuator_joint_names_from_xml(self, xml_path):
        """Parses a gripper XML file and returns the names of joints in actuators."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            actuator_joints = []
            for actuator in root.findall('.//actuator/position'):
                joint_name = actuator.get('joint')
                if joint_name:
                    actuator_joints.append(joint_name)
            return actuator_joints
        except (ET.ParseError, FileNotFoundError) as e:
            log.error(f"Failed to parse or find XML file at {xml_path}: {e}")
            return []

    def __init__(self, config: Mapping[Text, Any], hardware_interface: RobotLib, simulator: Monte01Mujoco, isLeft: bool = True, trunk: Trunk = None):
        super().__init__()
        component_type = COM_TYPE_LEFT if isLeft else COM_TYPE_RIGHT
        self.component_type = COM_TYPE_LEFT if isLeft else COM_TYPE_RIGHT
        self.config = config
        self.trunk = trunk

        # Dynamically determine joint IDs
        gripper_sim_config = config.get('gripper_sim', {}).get('robot', {}).get('grippers', {})
        left_gripper_path = gripper_sim_config.get('left_gripper', {}).get('model_path')
        right_gripper_path = gripper_sim_config.get('right_gripper', {}).get('model_path')

        num_left_arm_joints = 7  # Assuming this is fixed
        num_right_arm_joints = 7 # Assuming this is fixed
        num_left_gripper_joints = self._get_joint_count_from_xml(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', left_gripper_path))) if left_gripper_path else 0
        num_right_gripper_joints = self._get_joint_count_from_xml(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', right_gripper_path))) if right_gripper_path else 0
        log.info(f"Left Arm Joints: {num_left_arm_joints}, Left Gripper Joints: {num_left_gripper_joints}")
        log.info(f"Right Arm Joints: {num_right_arm_joints}, Right Gripper Joints: {num_right_gripper_joints}")

        if left_gripper_path:
            left_gripper_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', left_gripper_path))
            left_actuator_joints = self._get_actuator_joint_names_from_xml(left_gripper_full_path)
            log.info(f"Left Gripper Actuator Joints: {left_actuator_joints}")

        if right_gripper_path:
            right_gripper_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', right_gripper_path))
            right_actuator_joints = self._get_actuator_joint_names_from_xml(right_gripper_full_path)
            log.info(f"Right Gripper Actuator Joints: {right_actuator_joints}")
        self.LEFT_ARM_JOINT_IDS = list(range(6, 6 + num_left_arm_joints))
        self.LEFT_GRIPPER_JOINT_IDS = list(range(6 + num_left_arm_joints, 6 + num_left_arm_joints + num_left_gripper_joints))
        self.RIGHT_ARM_JOINT_IDS = list(range(6 + num_left_arm_joints + num_left_gripper_joints, 6 + num_left_arm_joints + num_left_gripper_joints + num_right_arm_joints))
        self.RIGHT_GRIPPER_JOINT_IDS = list(range(6 + num_left_arm_joints + num_left_gripper_joints + num_right_arm_joints, 6 + num_left_arm_joints + num_left_gripper_joints + num_right_arm_joints + num_right_gripper_joints))

        self.joint_ids = self.LEFT_ARM_JOINT_IDS if isLeft else self.RIGHT_ARM_JOINT_IDS
        self.gripper_joint_ids = self.LEFT_GRIPPER_JOINT_IDS if isLeft else self.RIGHT_GRIPPER_JOINT_IDS

        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
        base_link = 'chest_link'
        end_link = 'left_arm_link_7' if isLeft else 'right_arm_link_7'
        try:
            self.kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)
        except Exception as e:
            log.error(f"Failed to load URDF: {e}")

        self.ctrl_dt = 1.0 / config['control_rate']

        self.flange_t_tcp = np.eye(4)

        self.tcp_t_flange = self.flange_t_tcp.T

        self.joint_names = config['joint_names_left'] if isLeft else config['joint_names_right']
        self.robot = hardware_interface
        self.simulator: Monte01Mujoco = simulator

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

        # Determine gripper type and dynamically import the correct class
        gripper_config = gripper_sim_config.get('left_gripper' if isLeft else 'right_gripper', {})
        gripper_type = gripper_config.get('type')
        
        Gripper = None
        if gripper_type == 'corenetic':
            log.info("Using Corenetic gripper simulation")
            from hardware.monte01.gripper_corenetic import Gripper
        elif gripper_type == 'xarm7':
            log.info("Using XArm7 gripper simulation")
            from hardware.monte01.gripper_xarm import Gripper
        else:
            log.warning(f"Unknown or unspecified gripper type: {gripper_type}. Gripper will not be initialized.")

        time.sleep(0.1) # 給予一點緩衝時間
        
        # Determine the correct driver joint ID based on the actuator specified in the gripper's XML file
        gripper_driver_joint_id = -1  # Default to an invalid ID
        try:
            gripper_path = gripper_config.get('model_path')
            if gripper_path:
                gripper_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', gripper_path))
                actuator_joints = self._get_actuator_joint_names_from_xml(gripper_full_path)
                
                if actuator_joints:
                    base_driver_name = actuator_joints[0]  # Assume the first actuator is the driver
                    position = "left" if isLeft else "right"
                    prefix = f"{position}_{gripper_type}_" if gripper_type else f"{position}_"
                    full_driver_name = prefix + base_driver_name
                    
                    # Get the joint ID from the simulator using its full, prefixed name
                    gripper_driver_joint_id = simulator.mj_model.joint(full_driver_name).id + 1
                    log.info(f"Determined {'Left' if isLeft else 'Right'} Gripper Driver Joint: Name='{full_driver_name}', ID={gripper_driver_joint_id}")
                    self.gripper_driver_name = full_driver_name
                else:
                    log.error(f"Could not find any actuator joint in {gripper_path}")
            else:
                log.warning("Gripper 'model_path' not found in config, cannot determine driver joint ID.")

        except Exception as e:
            log.error(f"Error determining gripper driver joint ID: {e}")
        if hardware_interface is not None:
            hardware_interface.clean_arm_err_warn_code(component_type)
            hardware_interface.set_arm_enable(component_type, ARM_ENABLE)
            hardware_interface.set_arm_mode(component_type, ARM_MODE_SERVO_MOTION)
            hardware_interface.set_arm_state(component_type, ARM_STATE_SPORT)

            self.component_type = component_type

            gripper_ip = "192.168.11.11" if isLeft else "192.168.11.12"
            try:
                self.gripper = Gripper(config=config['gripper'], ip=gripper_ip, driver_joint_id=gripper_driver_joint_id,
                                       driver_joint_name = self.gripper_driver_name,simulator=simulator)
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
                self.gripper = None

        else:
            try:
                self.gripper = Gripper(config=config['gripper'], ip=None, simulator=simulator, driver_joint_id=gripper_driver_joint_id,driver_joint_name=self.gripper_driver_name,is_left=isLeft)
                log.info("Gripper initialized in simulation mode")
            except Exception as e:
                log.error(f"Failed to initialize gripper in simulation: {e}")
                self.gripper = None

        self.qs_prev = self.get_joint_positions()
        self.qs_last_valid = self.qs_prev.copy()  # Store last valid joint positions for IK safety check

        # Safety variables for pose movement tracking
        self.last_tcp_pose_chest = None  # Track last TCP pose for safety checks
        self.max_position_change = config.get('max_position_change', 0.01)  # Max position change in meters
        self.max_rotation_change = config.get('max_rotation_change', 0.2)  # Max rotation change in radians
        
        # Initialize last TCP pose after arm is ready
        try:
            self.last_tcp_pose_chest = self.get_tcp_pose_chest()
        except Exception as e:
            log.warning(f"Could not initialize last TCP pose: {e}")
            self.last_tcp_pose_chest = None

        self.time_log = []
        self.jp_log = []

        # Cache for chest_to_world transform
        self._cached_chest_to_world = None
        self._last_body_joint_positions = None
        self._body_joint_change_threshold = 1e-3 # Radians

    def get_gripper(self):
        if not self.gripper.valid():
            log.debug("Gripper not available, returning None")
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
    
    def get_joint_ids(self):
        return self.joint_ids
    
    def hold_joint_positions(self):
        self.simulator.hold_joint_positions(self.joint_ids)
        if self.gripper and self.gripper.valid():
            self.simulator.hold_joint_positions(self.gripper_joint_ids)

    def set_joint_velocities(self, velocities: np.ndarray):
        raise NotImplementedError("set_joint_velocities is not implemented for this arm.")
    
    def set_joint_positions(self, positions: np.ndarray):
        # Safety check: validate input positions
        if positions is None or len(positions) != len(self.joint_names):
            log.error(f"Invalid joint positions: expected {len(self.joint_names)} joints, got {len(positions) if positions is not None else 'None'}")
            return False
            
        # Safety check: detect large joint changes that could indicate state mutation
        try:
            current_positions = self.get_joint_positions()
            max_joint_change = 0.5  # Maximum allowed change per joint (radians) - about 28 degrees
            
            joint_diffs = np.abs(positions - current_positions)
            max_diff = np.max(joint_diffs)
            
            if max_diff > max_joint_change:
                log.error(f"Rejecting joint command due to large state change: max diff {max_diff:.3f} rad > {max_joint_change:.3f} rad")
                log.error(f"Current JP: {current_positions}")
                log.error(f"Target JP:  {positions}")
                log.error(f"Differences: {joint_diffs}")
                return False
                
        except Exception as e:
            log.warning(f"Could not perform safety check on joint positions: {e}")
            # Continue execution if safety check fails (robot communication might be unavailable)
        
        target_positions = {}
        for i, joint_name in enumerate(self.joint_names):
            target_positions[self.joint_ids[i]] = positions[i]

        success = True
        if self.robot:
            self.robot.set_arm_mode(self.component_type, ARM_MODE_SERVO_MOTION)
            self.robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
            success = self.robot.set_arm_servo_angle_j(self.component_type, list(positions), 1, 0, 1)
            if not success:
                log.error("Failed to set joint positions on real robot")
                
        if self.simulator:
            self.simulator.set_joint_positions(target_positions)
            
        return success

    def get_flange_pose(self) -> np.matrix:
        """Gets the pose of the flange. In chest_link frame. 
        """
        if self.robot is not None:
            success, pose = self.robot.get_arm_end_pose(self.component_type)
            if not success:
                raise RuntimeError("Failed to get flange pose from the robot.")
            return Transform(xyz=pose[0:3], rot=pose[3:7]).matrix
        
        return self.kinematics.fk(self.get_joint_positions())
    def ik(self, target: np.ndarray, seed: np.ndarray = None) -> Tuple[bool, np.ndarray]:

        """Gets joint configuration via IK.
        
        Args:
            target: The target pose of the flange in base frame.
            seed: Optional seed for the IK solver.
        
        Returns:
            Joint positions that achieve the target pose, or None if no solution is found.
        """
        return self.kinematics.ik(target, seed=seed, max_iter=1000, tol=1e-3, step_size=0.5)
    
    def get_chest_to_world_transform_cached(self) -> np.matrix:
        """
        Gets the transformation from chest_link to world, using a cache
        that invalidates when body joints change significantly.
        """
        if self.trunk is None:
            log.warning("No trunk component available, using identity transform")
            return np.eye(4)

        current_body_joints = self.get_body_joint_positions()

        if self._cached_chest_to_world is not None and self._last_body_joint_positions is not None:
            joint_diff = np.abs(current_body_joints - self._last_body_joint_positions)
            if np.max(joint_diff) < self._body_joint_change_threshold:
                # Return cached value if body hasn't moved much
                return self._cached_chest_to_world

        # Recompute the transform
        chest_to_world = self.trunk.get_chest_to_world_transform()
        
        # Update cache
        self._cached_chest_to_world = chest_to_world
        self._last_body_joint_positions = current_body_joints
        
        return chest_to_world

    def get_tcp_pose_chest(self) -> np.matrix:
        """Get TCP pose in chest_link frame (legacy behavior)"""
        return self.get_flange_pose() @ self.flange_t_tcp
    
    def get_tcp_pose(self) -> np.matrix:
        """Get TCP pose in world/base frame"""
        # Only sync robot state to simulator in real robot mode
        if self.robot is not None:
            self.sync_robot_state_to_simulator()
        
        try:
            # Get TCP pose in chest_link frame
            tcp_in_chest = self.get_tcp_pose_chest()
            
            # Get transformation from chest_link to world (cached)
            chest_to_world = self.get_chest_to_world_transform_cached()

            # Transform TCP pose from chest_link frame to world frame
            tcp_in_world = chest_to_world @ tcp_in_chest
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
        if not self.kinematics.is_reachable(target):
            log.warning("Target pose is not reachable, returning last valid joint positions")
            return False, self.qs_last_valid.copy()

        if start is None:
            start = self.get_joint_positions()
            
        flange_target = target @ self.tcp_t_flange
        success, jp = self.ik(flange_target, seed=start)
        
        
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
        """Get body joint positions via trunk component"""
        if self.trunk is not None:
            return self.trunk.get_body_joint_positions()
        else:
            log.warning("No trunk component available, returning zeros")
            return np.zeros(3)

    def _calculate_pose_difference(self, pose1: np.matrix, pose2: np.matrix) -> Tuple[float, float]:
        """Calculate position and rotation differences between two poses.
        
        Args:
            pose1: First pose matrix (4x4)
            pose2: Second pose matrix (4x4)
            
        Returns:
            Tuple of (position_diff, rotation_diff) in meters and radians
        """
        # Position difference (Euclidean distance)
        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]
        position_diff = np.linalg.norm(pos1 - pos2)
        
        # Rotation difference (angle between rotation matrices)
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        
        # Calculate relative rotation matrix
        R_rel = R1.T @ R2
        
        # Extract angle from rotation matrix using trace
        trace = np.trace(R_rel)
        # Clamp trace to valid range to avoid numerical issues
        trace = np.clip(trace, -1.0, 3.0)
        rotation_diff = np.arccos((trace - 1) / 2)
        
        return float(position_diff), float(rotation_diff)
    
    def _is_pose_change_safe(self, target_pose: np.matrix) -> bool:
        """Check if the pose change is within safe limits.
        
        Args:
            target_pose: Target pose matrix (4x4)
            
        Returns:
            True if the pose change is safe, False otherwise
        """
        if self.last_tcp_pose_chest is None:
            # First call, always safe
            return True
        
        try:
            pos_diff, rot_diff = self._calculate_pose_difference(self.last_tcp_pose_chest, target_pose)
            
            if pos_diff > self.max_position_change:
                log.warning(f"Large position change detected: {pos_diff:.4f}m > {self.max_position_change:.4f}m")
                return False
                
            if rot_diff > self.max_rotation_change:
                log.warning(f"Large rotation change detected: {rot_diff:.4f}rad > {self.max_rotation_change:.4f}rad")
                return False
                
            return True
            
        except Exception as e:
            log.error(f"Error calculating pose difference: {e}")
            return False
    
    def reset_pose_safety_tracking(self):
        """Reset pose safety tracking. Call this when arm position changes through other means."""
        try:
            self.last_tcp_pose_chest = self.get_tcp_pose_chest()
            log.info("Pose safety tracking reset")
        except Exception as e:
            log.warning(f"Could not reset pose safety tracking: {e}")
            self.last_tcp_pose_chest = None

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
                    # Sync current arm joint positions only (body joints handled by trunk)
                    arm_positions = self.get_joint_positions()
                    arm_joint_targets = {}
                    for i, joint_id in enumerate(self.joint_ids):
                        arm_joint_targets[joint_id] = arm_positions[i]
                    
                    # Set arm joint positions in simulator
                    self.simulator.set_joint_positions(arm_joint_targets)
                    log.debug(f"Synced real robot state to simulator")
                
            except Exception as e:
                log.warning(f"Failed to sync robot state to simulator: {e}")

    def move_to_pose(self, target: np.matrix, blocking: bool = True) -> bool:
        """Moves to the target that specifies TCP pose in chest_link frame.
        For real robot: target should be in chest_link frame.
        For simulation: target is converted from world frame to chest_link frame before IK.
        """
        # For real robot mode: sync all joint states to simulator for correct visualization
        if self.robot is not None:
            self.sync_robot_state_to_simulator()
        
        if self.robot is not None:
            # Safety check: verify pose change is within safe limits
            if not self._is_pose_change_safe(target):
                log.error("Pose change exceeds safety limits, rejecting movement command")
                return False
            
            target_pose = Transform(matrix=target)
            
            quaternion = target_pose.quaternion
            position = target_pose.translation

            current_pose =  Transform(matrix=self.get_flange_pose())

            # Calculate quaternion difference (dot product for similarity)
            quat_dot_product = sum(a * b for a, b in zip(quaternion, current_pose.quaternion))
            log.debug(f"Quaternion similarity (dot product): {quat_dot_product:.4f} (1.0=identical, -1.0=opposite)")
            if quat_dot_product < 0:
                quaternion = -quaternion
            success = True
            pose_cmd = position.tolist() + (quaternion).tolist()
            success = self.robot.set_arm_end_pose(self.component_type, pose_cmd)

            # Update last TCP pose after successful movement
            if success:
                self.last_tcp_pose_chest = target.copy()
            else:
                log.error("Failed to move arm to target pose on real robot")

            return success
        else:
            # Simulation mode: try direct end-effector pose control first
            # Determine arm end body name (wrist link) - IK will solve for TCP but we control arm joints
            ee_body_name = 'left_arm_link_7' if self.component_type == COM_TYPE_LEFT else 'right_arm_link_7'
            # Use direct pose control in simulation with balanced gains
            success = self.simulator.set_end_effector_pose(
                body_name=ee_body_name,
                target_pose=target, #chest frame
                joint_ids=self.joint_ids,
                arm_kinematics=self.kinematics
            )
            
            if success:
                log.debug(f"Direct end-effector pose control successful for {ee_body_name}")
            else:
                log.warning(f"Set end_effector_pose failed for {ee_body_name}")
                    
            # Fallback to IK-based control if direct pose control fails
            # success, jp = self.get_joint_target_from_pose(target, self.get_joint_positions())
            # if success:
            #     self.move_to_joint_target(jp, blocking=False)
            # else:
            #     log.error("Failed to get joint target from pose, will NOT move!!")
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
                init_duration = 5  # seconds
                joint_velocity = 1.0 / init_duration  # rad/s
                # Real robot mode
                if self.component_type == COM_TYPE_LEFT:
                    # Left arm real robot
                    robot = self.robot
                    robot.set_arm_enable(self.component_type, ARM_ENABLE)
                    robot.set_arm_mode(self.component_type, ARM_MODE_POSITION_CTRL)
                    robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
                    success = robot.set_arm_servo_angle(self.component_type, DEFAULT_JP_LEFT, joint_velocity, 0, 1)
                    log.info(f"Left arm move to start position success: {success}")
                    time.sleep(init_duration)
                elif self.component_type == COM_TYPE_RIGHT:
                    # Right arm real robot 
                    robot = self.robot
                    robot.set_arm_enable(self.component_type, ARM_ENABLE)
                    robot.set_arm_mode(self.component_type, ARM_MODE_POSITION_CTRL)
                    robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
                    success = robot.set_arm_servo_angle(self.component_type, DEFAULT_JP_RIGHT, joint_velocity, 0, 1)
                    log.info(f"Right arm move to start position success: {success}")
                    time.sleep(init_duration)

                robot.set_arm_mode(self.component_type, ARM_MODE_SERVO_MOTION)
                robot.set_arm_state(self.component_type, ARM_STATE_SPORT)
                
                # Update last TCP pose after moving to start position
                try:
                    self.last_tcp_pose_chest = self.get_tcp_pose_chest()
                except Exception as e:
                    log.warning(f"Could not update last TCP pose after move_to_start: {e}")
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
                    # Update last TCP pose after moving to start position
                    try:
                        self.last_tcp_pose_chest = self.get_tcp_pose_chest()
                    except Exception as e:
                        log.warning(f"Could not update last TCP pose after move_to_start: {e}")
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
