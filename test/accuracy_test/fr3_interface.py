"""
FR3 Robot Interface for Accuracy Testing
"""
import sys
import os
import numpy as np
import yaml
import logging
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R
import time
import threading

# Add the HIROLRobotPlatform to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
platform_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, platform_dir)

from hardware.fr3.fr3_arm import Fr3Arm
from controller.controller_base import IKController
from controller.cartesian_impedance_controller import CartesianImpedanceController
from motion.pin_model import RobotModel
from hardware.base.utils import RobotJointState, Buffer, TrajectoryState
from trajectory.cartesian_trajectory import CartessianTrajectory
from trajectory.joint_trajectory import JointTrajectory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FR3Interface:
    """FR3 robot interface for pose accuracy testing"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize FR3 robot interface
        
        Args:
            config_path: Path to FR3 configuration file, uses default if None
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(platform_dir, 'hardware/fr3/config/fr3_cfg.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize FR3 arm
        self._config = config['fr3']
        self._ip = self._config['ip']
        self._fr3_arm = Fr3Arm(self._config)
        
        # Get panda robot object for teaching mode
        self._fr3_robot = self._fr3_arm._fr3_robot
        
        # Define start position (same as in arm.py)
        self._start_joint_positions = np.array([ 
                                                1.62230305e-03,
                                                -7.84216639e-01, 
                                                -9.56568750e-04,
                                                -2.35610871e+00,
                                                -2.86549018e-03, 
                                                1.56793816e+00,
                                                7.78829413e-01
                                                ])
        
        # Initialize robot model for IK
        robot_model_config = {
            "urdf_path": os.path.join(platform_dir, "assets/franka_fr3/fr3_franka_hand.urdf"),
            "mesh_offset": "",
            "frames": ["base", "fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", 
                        "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8", "fr3_hand_tcp",
                        "fr3_hand", "fr3_leftfinger", "fr3_rightfinger"],
            "base_link": "base",
            "ee_link": "fr3_hand_tcp",
            "fixed_base": True,
            "lock_joints": [{"base": "fr3_hand","end": "fr3_leftfinger"}, {"base": "fr3_hand", "end": "fr3_rightfinger"}]
        }
        self._robot_model = RobotModel(robot_model_config)
        
        # Initialize IK controller with LM method
        ik_config = {
            "damping_weight": 0.3,
            "ik_type": "lm",  # Use Levenberg-Marquardt method
            "tolerance": 0.0001,
            "max_iteration": 1000
        }
        self._ik_controller = IKController(ik_config, self._robot_model)
        
        # Initialize trajectory components
        traj_config_path = os.path.join(platform_dir, 'trajectory/config/cartesian_polynomial_traj_cfg.yaml')
        with open(traj_config_path, 'r') as f:
            traj_config = yaml.safe_load(f)
        
        # Create buffer and lock for trajectory execution
        self._traj_buffer = Buffer(traj_config["buffer"]["size"], traj_config["buffer"]["dim"])
        self._traj_buffer_lock = threading.Lock()
        
        # Create trajectory generator
        self._cartesian_traj = CartessianTrajectory(traj_config["cart_polynomial"], 
                                                    self._traj_buffer, 
                                                    self._traj_buffer_lock)
        
        # Initialize joint trajectory components
        joint_traj_config_path = os.path.join(platform_dir, 'trajectory/config/joint_polynomial_traj_cfg.yaml')
        with open(joint_traj_config_path, 'r') as f:
            joint_traj_config = yaml.safe_load(f)
        
        # Create separate buffer for joint trajectory
        self._joint_traj_buffer = Buffer(joint_traj_config["buffer"]["size"], 
                                       joint_traj_config["buffer"]["dim"])
        self._joint_traj_buffer_lock = threading.Lock()
        
        # Create joint trajectory generator
        self._joint_traj = JointTrajectory(joint_traj_config["joint_polynomial"],
                                         self._joint_traj_buffer,
                                         self._joint_traj_buffer_lock)
        
        # Initialize Cartesian Impedance Controller
        impedance_config_path = os.path.join(platform_dir, 'controller/config/cartesian_impedance_fr3_cfg.yaml')
        with open(impedance_config_path, 'r') as f:
            impedance_config = yaml.safe_load(f)
        
        self._impedance_controller = CartesianImpedanceController(
            impedance_config["cartesian_impedance"], 
            self._robot_model
        )
                # Initialize trajectory components
        impedance_traj_config_path = os.path.join(platform_dir, 'trajectory/config/cartesian_polynomial_traj_cfg.yaml')
        with open(traj_config_path, 'r') as f:
            self.impedance_traj_config = yaml.safe_load(f)
        
        # Create separate buffer for impedance trajectory
        self._impedance_traj_buffer = Buffer(traj_config["buffer"]["size"], 
                                           traj_config["buffer"]["dim"])
        self._impedance_traj_buffer_lock = threading.Lock()
        
        # Thread control
        self._traj_thread = None
        self._joint_traj_thread = None
        self._impedance_traj_thread = None
        self._stop_execution = threading.Event()
        
        # Control mode tracking
        self._current_control_mode = None
        self._impedance_control_active = False
        
        logger.info(f"FR3 interface initialized with IP: {self._ip}")
        self.counter = 0
    
    def enter_teach_mode(self) -> None:
        """Enter teaching mode"""
        try:
            self._fr3_robot.teaching_mode(True)
            logger.info("Entered teaching mode")
        except Exception as e:
            logger.error(f"Failed to enter teaching mode: {e}")
            raise
    
    def exit_teach_mode(self) -> None:
        """Exit teaching mode"""
        try:
            self._fr3_robot.teaching_mode(False)
            logger.info("Exited teaching mode")
        except Exception as e:
            logger.error(f"Failed to exit teaching mode: {e}")
            raise
    
    def get_current_pose(self) -> Dict[str, float]:
        """
        Get current end-effector pose
        
        Returns:
            Dictionary containing x, y, z, qx, qy, qz, qw
        """
        try:
            # Get current state
            state = self._fr3_arm._fr3_state
            if state is None:
                raise RuntimeError("Robot state not available")
            
            # Get end-effector transformation matrix
            O_T_EE = np.array(state.O_T_EE).reshape(4, 4).T
            
            # Extract position (in meters)
            position = O_T_EE[:3, 3] 
            
            # Extract rotation and convert to quaternion
            rotation_matrix = O_T_EE[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()  # Returns [x, y, z, w]
            
            # joint_state = self._fr3_arm.get_joint_states()._positions
            # pose = self._robot_model.get_frame_pose('fr3_hand_tcp', joint_state, True)
            # position = pose[:3, 3] 
            # rotation = R.from_matrix(pose[:3, :3])
            # quaternion = rotation.as_quat()  # Returns [x, y, z, w]
            
            pose = {
                'x': float(position[0]),
                'y': float(position[1]),
                'z': float(position[2]),
                'qx': float(quaternion[0]),
                'qy': float(quaternion[1]),
                'qz': float(quaternion[2]),
                'qw': float(quaternion[3])
            }
            
            return pose
            
        except Exception as e:
            logger.error(f"Failed to get current pose: {e}")
            raise
    
    def move_to_pose_servo(self, target_pose: Dict[str, float], mode: str = 'position') -> None:
        """
        Move to target pose using IK controller
        
        Args:
            target_pose: Dictionary containing x, y, z, qx, qy, qz, qw
            mode: Control mode (default 'position')
        
        Raises:
            RuntimeError: If IK solution fails
        """
        try:
            # Get current joint state
            current_joint_state = self._fr3_arm.get_joint_states()
            
            # Prepare target in the format expected by IK controller
            # Convert pose to [x, y, z, qx, qy, qz, qw] format
            pose_7d = np.array([
                target_pose['x'],
                target_pose['y'],
                target_pose['z'],
                target_pose['qx'],
                target_pose['qy'],
                target_pose['qz'],
                target_pose['qw']
            ])
            
            # IK controller expects dict with frame name as key
            ik_target = {"fr3_hand_tcp": pose_7d}
            
            # Compute IK solution
            success, joint_target, control_mode = self._ik_controller.compute_controller(
                ik_target, current_joint_state
            )
        
            if not success:
                logger.error("IK solution failed to converge")
                raise RuntimeError("IK solution failed to converge")
            
            # Send joint command
            nums = 100; i = 0
            while i < nums:
                self._fr3_arm.set_joint_command(control_mode, joint_target)
                time.sleep(0.1)
                i+=1
            logger.debug(f"Sent joint command: {joint_target}")
            
        except Exception as e:
            logger.error(f"Failed to move to pose: {e}")
            raise
    
    def move_to_pose_traj(self, target_pose: Dict[str, float], 
                         finish_time: Optional[float] = None,
                         mode: str = 'position') -> None:
        """
        Move to target pose using trajectory planning
        
        Args:
            target_pose: Dictionary containing x, y, z, qx, qy, qz, qw
            finish_time: Time to complete trajectory in seconds (None for auto)
            mode: Control mode (default 'position')
        """
        try:
            # Get current pose
            current_pose = self.get_current_pose()
            
            # Prepare trajectory target
            start = np.array([
                current_pose['x'], current_pose['y'], current_pose['z'],
                current_pose['qx'], current_pose['qy'], current_pose['qz'], current_pose['qw']
            ])
            
            end = np.array([
                target_pose['x'], target_pose['y'], target_pose['z'],
                target_pose['qx'], target_pose['qy'], target_pose['qz'], target_pose['qw']
            ])
            
            # Create TrajectoryState
            traj_target = TrajectoryState()
            traj_target._zero_order_values = np.vstack((start, end))
            traj_target._first_order_values = np.zeros((2, 7))
            traj_target._second_order_values = np.zeros((2, 7))
            
            # Clear buffer before starting
            self._traj_buffer_lock.acquire()
            while self._traj_buffer.size() > 0:
                self._traj_buffer.pop_data()
            self._traj_buffer_lock.release()
            
            # Reset stop flag
            self._stop_execution.clear()
            
            # Start trajectory generation thread
            self._traj_thread = threading.Thread(
                target=self._trajectory_thread,
                args=(traj_target, finish_time)
            )
            self._traj_thread.start()
            
            # Execute trajectory from buffer
            self._trajectory_execution_loop()
            
            # Wait for trajectory thread to complete
            if self._traj_thread.is_alive():
                self._traj_thread.join(timeout=1.0)
            
            logger.info("Trajectory execution completed")
            
        except Exception as e:
            logger.error(f"Failed to move to pose via trajectory: {e}")
            self._stop_execution.set()
            if self._traj_thread and self._traj_thread.is_alive():
                self._traj_thread.join(timeout=1.0)
            raise
    
    def _reinitialize_controller(self) -> None:
        """Reinitialize servo controller after controller stop"""
        try:
            logger.info("Reinitializing FR3 controller...")
            
            # Wait for robot to stabilize
            time.sleep(0.5)
            
            # Reinitialize the FR3 arm controller
            self._fr3_arm.initialize()
            
            # Verify state update is working
            timeout = 5.0
            start_time = time.time()
            while not self._fr3_arm._fr3_state_update_flag:
                if time.time() - start_time > timeout:
                    raise RuntimeError("Controller reinitialization timeout")
                time.sleep(0.1)
            
            logger.info("Controller reinitialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to reinitialize controller: {e}")
            raise
    
    def move_to_start(self) -> None:
        """Move to start position and reinitialize controller for servo control"""
        try:
            # Use panda_py's move_to_start
            self._fr3_robot.move_to_start()
            logger.info("Moved to start position")
            
            # Reinitialize controller for subsequent servo control
            self._reinitialize_controller()
            
        except Exception as e:
            logger.error(f"Failed to move to start: {e}")
            raise
    
    def _trajectory_thread(self, target: TrajectoryState, finish_time: Optional[float]) -> None:
        """
        Thread function for trajectory generation
        
        Args:
            target: Trajectory state with start and end poses
            finish_time: Time to complete trajectory
        """
        try:
            logger.debug("Starting trajectory generation")
            self._cartesian_traj.plan_trajectory(target, finish_time)
            logger.debug("Trajectory generation completed")
        except Exception as e:
            logger.error(f"Trajectory generation failed: {e}")
            self._stop_execution.set()
    
    def _trajectory_execution_loop(self) -> None:
        """
        Execute trajectory points from buffer using IK controller
        """
        control_mode = None
        points_executed = 0
        
        while not self._stop_execution.is_set():
            # Get point from buffer
            self._traj_buffer_lock.acquire()
            success, traj_point = self._traj_buffer.pop_data()
            buffer_size = self._traj_buffer.size()
            self._traj_buffer_lock.release()
            
            if not success:
                # Buffer empty, check if trajectory generation is done
                if not self._cartesian_traj.trajectory_idle:
                    # Still generating, wait a bit
                    time.sleep(0.001)
                    continue
                elif buffer_size == 0 and self._cartesian_traj.trajectory_idle:
                    # Generation done and buffer empty, we're finished
                    logger.debug(f"Trajectory execution finished, executed {points_executed} points")
                    break
            else:
                # Execute trajectory point
                try:
                    # Get current joint state
                    current_joint_state = self._fr3_arm.get_joint_states()
                    
                    # Prepare IK target
                    ik_target = {"fr3_hand_tcp": traj_point}
                    
                    # Compute IK solution
                    success, joint_target, control_mode = self._ik_controller.compute_controller(
                        ik_target, current_joint_state
                    )
                    
                    if not success:
                        logger.warning(f"IK failed for trajectory point {points_executed}")
                        continue
                    
                    # Send joint command
                    self._fr3_arm.set_joint_command(control_mode, joint_target)
                    points_executed += 1
                    
                    # Small delay to match trajectory dt (slightly less to account for computation time)
                    time.sleep(self._cartesian_traj.dt * 0.8)
                    
                except Exception as e:
                    logger.error(f"Error executing trajectory point: {e}")
                    self._stop_execution.set()
                    break
    
    def _joint_trajectory_thread(self, target: TrajectoryState, finish_time: Optional[float]) -> None:
        """
        Thread function for joint trajectory generation
        
        Args:
            target: Trajectory state with start and end joint positions
            finish_time: Time to complete trajectory
        """
        try:
            logger.debug("Starting joint trajectory generation")
            self._joint_traj.plan_trajectory(target, finish_time)
            logger.debug("Joint trajectory generation completed")
        except Exception as e:
            logger.error(f"Joint trajectory generation failed: {e}")
            self._stop_execution.set()
    
    def _joint_trajectory_execution_loop(self) -> None:
        """
        Execute joint trajectory points from buffer
        """
        control_mode = "position"
        points_executed = 0
        
        while not self._stop_execution.is_set():
            # Get point from buffer
            self._joint_traj_buffer_lock.acquire()
            success, joint_point = self._joint_traj_buffer.pop_data()
            buffer_size = self._joint_traj_buffer.size()
            self._joint_traj_buffer_lock.release()
            
            if not success:
                # Buffer empty, check if trajectory generation is done
                if not self._joint_traj.trajectory_idle:
                    # Still generating, wait a bit
                    time.sleep(0.001)
                    continue
                elif buffer_size == 0 and self._joint_traj.trajectory_idle:
                    # Generation done and buffer empty, we're finished
                    logger.debug(f"Joint trajectory execution finished, executed {points_executed} points")
                    break
            else:
                # Execute joint trajectory point directly
                try:
                    # Send joint command directly
                    self._fr3_arm.set_joint_command(control_mode, joint_point)
                    points_executed += 1
                    
                    # Small delay to match trajectory dt (slightly less to account for computation time)
                    time.sleep(self._joint_traj.dt * 0.8)
                    
                except Exception as e:
                    logger.error(f"Error executing joint trajectory point: {e}")
                    self._stop_execution.set()
                    break
    
    def _impedance_trajectory_thread(self, target: TrajectoryState, finish_time: Optional[float]) -> None:
        """
        Thread function for impedance trajectory generation
        
        Args:
            target: Trajectory state with start and end poses
            finish_time: Time to complete trajectory
        """
        try:
            logger.debug("Starting impedance trajectory generation")
            # Create a new CartesianTrajectory instance for impedance control
            impedance_cartesian_traj = CartessianTrajectory(
                self.impedance_traj_config["cart_polynomial"],
                self._impedance_traj_buffer,
                self._impedance_traj_buffer_lock
            )
            impedance_cartesian_traj.plan_trajectory(target, finish_time)
            logger.debug("Impedance trajectory generation completed")
        except Exception as e:
            logger.error(f"Impedance trajectory generation failed: {e}")
            self._stop_execution.set()
    
    def _impedance_execution_loop(self) -> None:
        """
        Execute impedance control trajectory from buffer
        High-frequency control loop for torque commands
        """
        points_executed = 0
        control_frequency = 1000.0  # Target 1kHz for impedance control
        dt = 1.0 / control_frequency
        
        # Reset impedance controller for fresh start
        self._impedance_controller.initialized = False
        
        while not self._stop_execution.is_set():
            loop_start = time.time()
            
            # Get point from buffer
            self._impedance_traj_buffer_lock.acquire()
            success, traj_point = self._impedance_traj_buffer.pop_data()
            buffer_size = self._impedance_traj_buffer.size()
            self._impedance_traj_buffer_lock.release()
            
            if not success:
                # Buffer empty, check if trajectory generation is done
                if buffer_size == 0 and not self._impedance_traj_thread.is_alive():
                    # Generation done and buffer empty, we're finished
                    logger.debug(f"Impedance trajectory execution finished, executed {points_executed} points")
                    break
                else:
                    # Still generating or buffer temporarily empty, wait a bit
                    time.sleep(0.0001)
                    continue
            else:
                # Execute impedance control
                try:
                    # Get current joint state
                    current_joint_state = self._fr3_arm.get_joint_states()
                    
                    # Prepare target for impedance controller
                    impedance_target = {self._robot_model.ee_link: traj_point}
                    
                    # Compute torque command
                    success, torque_command, control_mode = self._impedance_controller.compute_controller(
                        impedance_target, current_joint_state
                    )
                    
                    if not success:
                        logger.warning(f"Impedance controller failed at point {points_executed}")
                        continue
                    
                    if control_mode != 'torque':
                        logger.error(f"Impedance controller returned unexpected mode: {control_mode}")
                        self._stop_execution.set()
                        break
                    
                    # Send torque command
                    self._fr3_arm.set_joint_command(control_mode, torque_command)
                    points_executed += 1
                    
                    # Maintain control frequency
                    loop_time = time.time() - loop_start
                    if loop_time < dt:
                        time.sleep(dt - loop_time)
                    elif points_executed % 100 == 0:
                        actual_freq = 1.0 / loop_time
                        if actual_freq < control_frequency * 0.8:
                            logger.warning(f"Impedance control frequency dropped to {actual_freq:.1f} Hz")
                    
                except Exception as e:
                    logger.error(f"Error executing impedance control: {e}")
                    self._stop_execution.set()
                    break
    
    def close(self) -> None:
        """Close robot connection"""
        try:
            # Stop any ongoing trajectory execution
            self._stop_execution.set()
            
            # Wait for trajectory thread if running
            if self._traj_thread and self._traj_thread.is_alive():
                self._traj_thread.join(timeout=2.0)
            
            if self._joint_traj_thread and self._joint_traj_thread.is_alive():
                self._joint_traj_thread.join(timeout=2.0)
                
            if self._impedance_traj_thread and self._impedance_traj_thread.is_alive():
                self._impedance_traj_thread.join(timeout=2.0)
            
            self._fr3_arm.close()
            logger.info("FR3 interface closed")
        except Exception as e:
            logger.error(f"Error closing FR3 interface: {e}")
            raise
        
    def move_to_joint(self, target_joints: np.ndarray, 
                     finish_time: Optional[float] = None) -> None:
        """
        Move to target joint positions using joint trajectory planning
        
        Args:
            target_joints: Target joint positions array (7 values in radians)
            finish_time: Time to complete trajectory in seconds (None for auto)
        
        Raises:
            ValueError: If target_joints is not 7-dimensional
            RuntimeError: If trajectory execution fails
        """
        try:
            # Validate input
            if not isinstance(target_joints, np.ndarray):
                target_joints = np.array(target_joints)
            
            if target_joints.shape != (7,):
                raise ValueError(f"Expected 7 joint values, got {target_joints.shape}")
            
            # Get current joint positions
            current_joints = self._fr3_arm.get_joint_states()._positions
            
            # Create TrajectoryState for joint trajectory
            traj_target = TrajectoryState()
            traj_target._zero_order_values = np.vstack((current_joints, target_joints))
            traj_target._first_order_values = np.zeros((2, 7))
            traj_target._second_order_values = np.zeros((2, 7))
            
            # Clear joint buffer before starting
            self._joint_traj_buffer_lock.acquire()
            while self._joint_traj_buffer.size() > 0:
                self._joint_traj_buffer.pop_data()
            self._joint_traj_buffer_lock.release()
            
            # Reset stop flag
            self._stop_execution.clear()
            
            # Start joint trajectory generation thread
            self._joint_traj_thread = threading.Thread(
                target=self._joint_trajectory_thread,
                args=(traj_target, finish_time)
            )
            self._joint_traj_thread.start()
            
            # Execute joint trajectory from buffer
            self._joint_trajectory_execution_loop()
            
            # Wait for trajectory thread to complete
            if self._joint_traj_thread.is_alive():
                self._joint_traj_thread.join(timeout=1.0)
            
            logger.info("Joint trajectory execution completed")
            
        except Exception as e:
            logger.error(f"Failed to move to joint positions: {e}")
            self._stop_execution.set()
            if self._joint_traj_thread and self._joint_traj_thread.is_alive():
                self._joint_traj_thread.join(timeout=1.0)
            raise
    
    def move_cartesian_impedance(self, target_pose: Dict[str, float], 
                                finish_time: Optional[float] = None,
                                stiffness: Optional[Dict[str, float]] = None,
                                damping: Optional[Dict[str, float]] = None) -> None:
        """
        Move to target pose using Cartesian impedance control with trajectory planning
        
        Args:
            target_pose: Dictionary containing x, y, z, qx, qy, qz, qw
            finish_time: Time to complete trajectory in seconds (None for auto)
            stiffness: Optional stiffness parameters {"translational": float, "rotational": float}
            damping: Optional damping parameters {"translational": float, "rotational": float}
        
        Raises:
            RuntimeError: If controller initialization or execution fails
        """
        try:
            # Stop any ongoing control
            self._stop_execution.set()
            if self._traj_thread and self._traj_thread.is_alive():
                self._traj_thread.join(timeout=1.0)
            if self._joint_traj_thread and self._joint_traj_thread.is_alive():
                self._joint_traj_thread.join(timeout=1.0)
            
            # Apply optional impedance parameters
            if stiffness:
                self._impedance_controller.set_stiffness(
                    translational_stiffness=stiffness.get("translational"),
                    rotational_stiffness=stiffness.get("rotational")
                )
            if damping:
                self._impedance_controller.set_damping(
                    translational_damping=damping.get("translational"),
                    rotational_damping=damping.get("rotational")
                )
            
            # Get current pose
            current_pose = self.get_current_pose()
            
            # Prepare trajectory target
            start = np.array([
                current_pose['x'], current_pose['y'], current_pose['z'],
                current_pose['qx'], current_pose['qy'], current_pose['qz'], current_pose['qw']
            ])
            
            end = np.array([
                target_pose['x'], target_pose['y'], target_pose['z'],
                target_pose['qx'], target_pose['qy'], target_pose['qz'], target_pose['qw']
            ])
            
            # Create TrajectoryState
            traj_target = TrajectoryState()
            traj_target._zero_order_values = np.vstack((start, end))
            traj_target._first_order_values = np.zeros((2, 7))
            traj_target._second_order_values = np.zeros((2, 7))
            
            # Clear impedance buffer
            self._impedance_traj_buffer_lock.acquire()
            while self._impedance_traj_buffer.size() > 0:
                self._impedance_traj_buffer.pop_data()
            self._impedance_traj_buffer_lock.release()
            
            # Reset stop flag and set impedance control active
            self._stop_execution.clear()
            self._impedance_control_active = True
            
            # Switch to torque control mode if needed
            if self._fr3_arm._control_mode != 'torque':
                logger.info("Switching to torque control mode for impedance control")
                self._fr3_robot.stop_controller()
                time.sleep(0.5)
                self._reinitialize_controller()
            
            # Start impedance trajectory generation thread
            self._impedance_traj_thread = threading.Thread(
                target=self._impedance_trajectory_thread,
                args=(traj_target, finish_time)
            )
            self._impedance_traj_thread.start()
            
            # Execute impedance control from buffer
            self._impedance_execution_loop()
            
            # Wait for trajectory thread to complete
            if self._impedance_traj_thread.is_alive():
                self._impedance_traj_thread.join(timeout=1.0)
            
            self._impedance_control_active = False
            logger.info("Cartesian impedance control completed")
            
        except Exception as e:
            logger.error(f"Failed to execute cartesian impedance control: {e}")
            self._stop_execution.set()
            self._impedance_control_active = False
            if self._impedance_traj_thread and self._impedance_traj_thread.is_alive():
                self._impedance_traj_thread.join(timeout=1.0)
            raise
    
    def set_impedance_params(self,
                           translational_stiffness: Optional[float] = None,
                           rotational_stiffness: Optional[float] = None,
                           translational_damping: Optional[float] = None,
                           rotational_damping: Optional[float] = None) -> None:
        """
        Dynamically adjust impedance control parameters
        
        Args:
            translational_stiffness: Translational stiffness in N/m
            rotational_stiffness: Rotational stiffness in Nm/rad
            translational_damping: Translational damping in Ns/m
            rotational_damping: Rotational damping in Nms/rad
        """
        self._impedance_controller.set_stiffness(
            translational_stiffness=translational_stiffness,
            rotational_stiffness=rotational_stiffness
        )
        self._impedance_controller.set_damping(
            translational_damping=translational_damping,
            rotational_damping=rotational_damping
        )
        logger.debug(f"Updated impedance params - Stiffness: trans={translational_stiffness}, "
                    f"rot={rotational_stiffness}; Damping: trans={translational_damping}, "
                    f"rot={rotational_damping}")
    
    def stop_controller(self) -> None:
        self._fr3_robot.stop_controller()
        
        
        
    


# Example usage
if __name__ == "__main__":
    # Initialize interface
    interface = FR3Interface()
    
    try:
        # Get current pose
        current_pose = interface.get_current_pose()
        print(f"Current pose: {current_pose}")
        
        # Move to start
        interface.move_to_start()
        
        # Test selection menu
        print("\n=== FR3 Interface Test Menu ===")
        print("1. Test Joint Trajectory (movej)")
        print("2. Test Cartesian Impedance Control")
        print("3. Test Teaching Mode")
        test_choice = input("Select test (1-3): ")
        
        if test_choice == "1":
            # Test joint trajectory (movej)
            print("\n=== Testing Joint Trajectory (movej) ===")
            
            # Get current joint positions
            current_joints = interface._fr3_arm.get_joint_states()._positions
            print(f"Current joint positions: {current_joints}")
            
            # Define target joint positions (small movement from current)
            target_joints = current_joints.copy()
            target_joints[0] += 0.3  # Rotate base joint by 0.3 rad
            target_joints[1] -= 0.2  # Adjust shoulder joint
            target_joints[3] -= 0.2  # Adjust elbow joint
            
            print(f"Target joint positions: {target_joints}")
            
            # Execute joint trajectory
            print("Executing joint trajectory...")
            interface.move_to_joint(target_joints, finish_time=3.0)
            
            # Verify final position
            final_joints = interface._fr3_arm.get_joint_states()._positions
            print(f"Final joint positions: {final_joints}")
            print(f"Joint position error: {np.linalg.norm(final_joints - target_joints):.6f} rad")
            
            input("Press Enter to return to start position...")
            interface.move_to_start()
            
        elif test_choice == "2":
            # Test Cartesian impedance control
            print("\n=== Testing Cartesian Impedance Control ===")
            
            # Get current pose
            start_pose = interface.get_current_pose()
            print(f"Start pose: {start_pose}")
            
            # Define target pose (move 10cm in x direction)
            target_pose = start_pose.copy()
            target_pose['x'] += 0.1
            
            print(f"Target pose: {target_pose}")
            
            # Set custom impedance parameters (optional)
            stiffness = {"translational": 1500.0, "rotational": 250.0}
            damping = {"translational": 89.0, "rotational": 9.0}
            
            print("\nImpedance parameters:")
            print(f"  Stiffness: {stiffness}")
            print(f"  Damping: {damping}")
            
            # Execute impedance control
            print("\nExecuting cartesian impedance control...")
            interface.move_cartesian_impedance(
                target_pose, 
                finish_time=None,
                stiffness=stiffness,
                damping=damping
            )
            
            # Verify final pose
            final_pose = interface.get_current_pose()
            print(f"\nFinal pose: {final_pose}")
            
            position_error = np.linalg.norm(np.array([
                final_pose['x'] - target_pose['x'],
                final_pose['y'] - target_pose['y'],
                final_pose['z'] - target_pose['z']
            ]))
            print(f"Position error: {position_error*1000:.2f} mm")
            
            # Test dynamic parameter adjustment
            input("\nPress Enter to test dynamic parameter adjustment...")
            
            # Return to start with different parameters
            print("\nReturning to start with softer impedance...")
            interface.set_impedance_params(
                translational_stiffness=800.0,
                rotational_stiffness=80.0
            )
            interface.move_cartesian_impedance(start_pose, finish_time=None)
            
            input("Press Enter to return to home position...")
            interface.move_to_start()
            
        elif test_choice == "3":
            # Test teaching mode
            print("\n=== Testing Teaching Mode ===")
            interface.enter_teach_mode()
            input("Move robot manually and press Enter to continue...")
            interface.exit_teach_mode()
            
            # Get new pose
            new_pose = interface.get_current_pose()
            print(f"New pose after teaching: {new_pose}")
            
            input("Press Enter to return to start...")
            interface.move_to_start()
        
        else:
            print("Invalid selection")
        
    finally:
        interface.close()