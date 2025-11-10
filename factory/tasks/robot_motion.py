"""
RobotMotion - Motion Planning Task with Data Collection

Provides a high-level interface for motion planning and sensor data collection,
integrating RobotFactory and MotionFactory components.

Features:
- Motion control (Cartesian/Joint space)
- Complete data collection (compatible with LeRobot format)
- Dual-thread architecture (control + data collection)
- Rerun + OpenCV visualization
- Keyboard interaction (hardware enable, recording, reset, quit)

Author: Haotian Liang
Date: 2025-09-30
"""

import os
import sys
import numpy as np
import threading
import time
import copy
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
import glog as log
import cv2

# Add HIROLRobotPlatform to path if needed
platform_path = Path(__file__).parent.parent.parent
if str(platform_path) not in sys.path:
    sys.path.insert(0, str(platform_path))

from factory.components.robot_factory import RobotFactory
from factory.components.motion_factory import MotionFactory, Robot_Space
from dataset.lerobot.data_process import EpisodeWriter
from hardware.base.utils import dynamic_load_yaml, convert_homo_2_7D_pose, ToolControlMode
from hardware.base.img_utils import combine_image
from sshkeyboard import listen_keyboard, stop_listening


class RobotMotion:
    """
    Motion Planning Task with Data Collection

    Integrates RobotFactory + MotionFactory for high-level motion control and
    complete sensor data collection (colors/depths/joint_states/ee_states/
    tools/tactiles/imus/actions).

    Example:
        >>> config_path = "factory/tasks/config/robot_motion_fr3_cfg.yaml"
        >>> robot_motion = RobotMotion(config_path)
        >>>
        >>> # Motion control
        >>> robot_motion.send_pose_command(target_pose)
        >>> robot_motion.send_gripper_command(0.5)
        >>>
        >>> # Data collection
        >>> robot_motion.start_recording()
        >>> # ... execute motions ...
        >>> robot_motion.stop_recording()
        >>>
        >>> robot_motion.close()
    """

    def __init__(self, config_path: str, auto_initialize: bool = True):
        """
        Initialize RobotMotion task

        Args:
            config_path: Path to configuration file (YAML format)
            auto_initialize: Whether to auto-initialize hardware connection

        Configuration file must contain:
            - motion_config: MotionFactory configuration (includes RobotFactory config)
            - data_collection: Data collection settings
            - motion_control: Motion control parameters
        """
        # Load configuration
        self._config = dynamic_load_yaml(config_path)
        log.info(f"Loaded configuration from {config_path}")

        # Extract configurations
        self._motion_config = self._config["motion_config"]
        self._data_config = self._config.get("data_collection", {})
        self._control_config = self._config.get("motion_control", {})

        # Initialize robot system using factory components
        self._robot_system = RobotFactory(self._motion_config)
        self._motion_factory = MotionFactory(self._motion_config, self._robot_system)

        # Data collection configuration
        self._data_record_frequency = self._data_config.get("data_record_frequency", 30)
        self._img_visualization = self._data_config.get("image_visualization", True)
        self._rerun_visualization = self._data_config.get("rerun_visualization", True)
        self._task_description = self._data_config.get("task_description", None)
        self._task_description_goal = self._data_config.get("task_description_goal", None)
        self._task_description_steps = self._data_config.get("task_description_steps", None)

        # Data save path
        self._save_path_prefix = self._data_config.get("save_path_prefix", None)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if self._save_path_prefix is not None:
            self._save_path_dir = os.path.join(cur_path, "../../dataset/data",
                                               self._save_path_prefix)
        else:
            self._save_path_dir = os.path.join(cur_path, "../../dataset/data/robot_motion")

        # Motion control configuration
        self._control_loop_time = self._control_config.get("control_loop_time", 0.02)  # 50Hz default
        self._enable_keyboard_listener = self._control_config.get("enable_keyboard_listener", True)  # Default: enabled
        self._reset_arm_command = self._control_config.get("reset_arm_command", None)
        self._reset_space = self._control_config.get("reset_space", None)
        self._reset_space = Robot_Space(self._reset_space) if self._reset_space is not None else None
        self._reset_tool_command = self._control_config.get("reset_tool_command", None)

        # State flags
        self._is_initialized = False
        self._enable_hardware = False
        self._enable_recording = False
        self._update_motion_state = True  # Pause motion updates during reset
        self._main_thread_running = False

        # Data recorder
        self.data_recorder: Optional[EpisodeWriter] = None

        # Action synchronization (thread-safe)
        self._latest_tool_action = {}
        self._tool_action_lock = threading.Lock()

        # Data recording thread
        self._data_recording_thread: Optional[threading.Thread] = None

        # Robot metadata (populated after initialization)
        self._ee_links = []
        self._robot_index = []
        self._ee_index = []

        # Initialize if requested
        if auto_initialize:
            self.initialize()

        log.info("RobotMotion initialized successfully")

    def initialize(self) -> None:
        """
        Initialize robot system and motion components

        Steps:
            1. Create MotionFactory components (auto-creates RobotFactory)
            2. Enable hardware execution
            3. Get end effector link list
            4. Start data collection thread
            5. Start keyboard listener thread

        Raises:
            RuntimeError: If hardware initialization fails
        """
        log.info("Initializing RobotMotion system...")

        try:
            # Create motion components (includes robot system creation)
            self._motion_factory.create_motion_components()
            self._robot_system = self._motion_factory._robot_system

            # Get robot metadata
            self._ee_links = self._motion_factory.get_model_end_effector_link_list()
            self._robot_index = self._motion_factory.get_model_types()
            self._ee_index = ['left', 'right'] if len(self._ee_links) > 1 else ['single']
            log.info(f"End effector links: {self._ee_links}")
            log.info(f"Robot indices: {self._ee_index}")

            # Start data recording thread
            self._main_thread_running = True
            self._data_recording_thread = threading.Thread(target=self._data_collection_loop)
            self._data_recording_thread.start()
            log.info("Data collection thread started")

            # Start keyboard listener (optional, controlled by config)
            if self._enable_keyboard_listener:
                self._keyboard_thread = threading.Thread(
                    target=listen_keyboard,
                    kwargs={
                        "on_press": self._keyboard_callback,
                        "until": None,
                        "sequential": False,
                    },
                    daemon=True
                )
                self._keyboard_thread.start()
                log.info("Keyboard listener started (h=hardware, r=record, o=reset, q=quit)")
            else:
                self._keyboard_thread = None
                log.info("Keyboard listener DISABLED (use API methods for control)")

            self._is_initialized = True
            log.info("RobotMotion system initialized successfully")

        except Exception as e:
            log.error(f"Failed to initialize RobotMotion: {e}")
            raise RuntimeError(f"Initialization failed: {e}")

    def get_state(self) -> Dict[str, Any]:
        """
        Get current robot state

        Returns:
            Dictionary containing:
                - pose: np.ndarray, shape=(7,) - TCP pose [x,y,z, qx,qy,qz,qw]
                - vel: np.ndarray, shape=(6,) - TCP velocity [vx,vy,vz, wx,wy,wz]
                - q: np.ndarray, shape=(7,) - Joint positions (rad)
                - dq: np.ndarray, shape=(7,) - Joint velocities (rad/s)
                - torque: np.ndarray, shape=(7,) - Joint torques (Nm)
                - gripper_pos: float - Gripper position [0-1]
                - time_stamp: float - Timestamp

        Note:
            This interface is for external scripts to query state in real-time
        """
        # Get joint states
        joint_states = self._robot_system.get_joint_states()

        # Get end effector pose
        ee_name = self._ee_links[0] if self._ee_links else None
        model_type = self._robot_index[0] if self._robot_index else "model"

        if ee_name:
            tcp_pose = self._motion_factory.get_frame_pose(ee_name, model_type)

            # Get TCP velocity using Jacobian
            jacobian = self._motion_factory._robot_model.get_jacobian(
                ee_name, joint_states._positions, model_type=model_type
            )
            if jacobian is not None:
                tcp_velocity = jacobian @ joint_states._velocities
            else:
                tcp_velocity = np.zeros(6)
        else:
            tcp_pose = np.zeros(7)
            tcp_velocity = np.zeros(6)

        # Get gripper state
        gripper_pos = 0.0
        tool_state = self._robot_system.get_tool_dict_state()
        if tool_state and "single" in tool_state:
            tool_state_obj = tool_state["single"]
            if hasattr(tool_state_obj, "_position"):
                raw_pos = np.asarray(tool_state_obj._position, dtype=np.float64).reshape(-1)
                if raw_pos.size > 0:
                    gripper_pos = float(raw_pos[0])
                    if gripper_pos <= 0.1:
                        gripper_pos = min(1.0, gripper_pos / 0.08)

        return {
            "pose": tcp_pose,
            "vel": tcp_velocity,
            "q": joint_states._positions,
            "dq": joint_states._velocities,
            "torque": joint_states._torques,
            "gripper_pos": gripper_pos,
            "time_stamp": joint_states._time_stamp
        }

    def send_pose_command(self, pose: np.ndarray) -> None:
        """
        Send Cartesian space pose command

        Args:
            pose: Target pose(s)
                - Single arm: shape=(7,), [x, y, z, qx, qy, qz, qw]
                - Dual arm: shape=(14,), [left_pose(7), right_pose(7)]

        Note:
            Whether the command is smoothed depends on the configuration file's
            use_smoother parameter:
            - use_smoother: true (default) - Command smoothed to 800Hz by Ruckig/CriticalDamped
            - use_smoother: false - Command sent directly without smoothing

            User code remains the same regardless of smoother configuration.

        Raises:
            ValueError: If pose dimension doesn't match robot configuration
        """
        # Dynamic validation
        num_ee = len(self._ee_links)
        expected_dim = 7 * num_ee

        if pose.shape[0] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}D pose for {num_ee} end effector(s), "
                f"got {pose.shape[0]}D"
            )

        self._motion_factory.update_high_level_command(pose)

    def send_joint_command(self, joints: np.ndarray) -> None:
        """
        Send joint space position command

        Args:
            joints: Joint positions (rad)
                - Single arm (FR3): shape=(7,)
                - Dual arm (Duo XArm): shape=(14,)
                - Whole body (UnitreeG1): shape=(29,)

        Note:
            Smoother behavior is controlled by configuration file (see send_pose_command).

        Raises:
            ValueError: If joints dimension doesn't match robot DOF
        """
        # Get total DOF dynamically
        dofs = self._robot_system.get_robot_dofs()
        total_dof = sum(dofs)

        if joints.shape[0] != total_dof:
            raise ValueError(
                f"Expected {total_dof}D joints for robot with DOFs {dofs}, "
                f"got {joints.shape[0]}D"
            )

        # Set joint command directly (MotionFactory will handle conversion)
        self._motion_factory.set_joint_positions(joints, is_continous_joint_command=True)

    def send_gripper_command(self, gripper_commands: Dict[str, float]) -> None:
        """
        Send gripper command(s)

        Args:
            gripper_commands: Dictionary of gripper commands
                - Single arm: {"single": 0.5}
                - Dual arm: {"left": 0.3, "right": 0.8}

        Note:
            This also updates _latest_tool_action for data recording

        Example:
            >>> # Single arm
            >>> robot_motion.send_gripper_command({"single": 0.5})
            >>>
            >>> # Dual arm
            >>> robot_motion.send_gripper_command({"left": 0.0, "right": 1.0})

        Raises:
            ValueError: If invalid gripper keys provided
        """
        # Validate keys
        valid_keys = set(self._ee_index)
        provided_keys = set(gripper_commands.keys())

        if not provided_keys.issubset(valid_keys):
            raise ValueError(
                f"Invalid gripper keys: {provided_keys - valid_keys}. "
                f"Valid keys: {valid_keys}"
            )

        # Send commands
        for key, width in gripper_commands.items():
            if not 0.0 <= width <= 1.0:
                log.warning(f"Gripper width {width} for {key} out of range [0, 1], clamping")
                width = np.clip(width, 0.0, 1.0)

            # Send to robot
            self._robot_system.set_tool_command({key: np.array([width])})

            # Update action cache for data recording (thread-safe)
            if self._enable_recording:
                # Apply binary threshold if tool is in BINARY mode
                tool = self._robot_system._tool
                if tool is not None and hasattr(tool, '_control_mode') and tool._control_mode == ToolControlMode.BINARY:
                    threshold = getattr(tool, '_binary_threshold', 0.5)
                    width_recorded = 1.0 if width >= threshold else 0.0
                else:
                    # INCREMENTAL or continuous mode: record original value
                    width_recorded = width

                with self._tool_action_lock:
                    if key not in self._latest_tool_action:
                        self._latest_tool_action[key] = {}
                    self._latest_tool_action[key]["tool"] = {
                        "position": width_recorded,
                        "time_stamp": time.perf_counter()
                    }

    def send_gripper_command_simple(self, width: float) -> None:
        """
        Send gripper command (simplified for single arm)

        Args:
            width: Gripper opening width, range [0-1] (0=closed, 1=open)

        Note:
            For single arm robots only. For dual arm, use send_gripper_command()

        Raises:
            RuntimeError: If robot has multiple arms
        """
        if len(self._ee_index) > 1:
            raise RuntimeError(
                f"Robot has multiple arms: {self._ee_index}. "
                f"Use send_gripper_command() with dict input"
            )

        key = self._ee_index[0]
        self.send_gripper_command({key: width})

    def execute_trajectory(self, waypoints: List[np.ndarray],
                          timing: Optional[List[float]] = None) -> None:
        """
        Execute trajectory sequence (blocking)

        Args:
            waypoints: List of poses, each element shape=(7,) [x,y,z,qx,qy,qz,qw]
            timing: Optional list of arrival times (seconds) for each waypoint.
                   If None, uses default interval (2.0 seconds per waypoint)

        Workflow:
            1. Send each waypoint sequentially
            2. Wait for arrival (checked by position error)
            3. Continue to next waypoint

        Raises:
            ValueError: If waypoints is empty or dimensions don't match

        Example:
            >>> waypoints = [
            ...     np.array([0.3, 0.0, 0.5, 1, 0, 0, 0]),
            ...     np.array([0.4, 0.0, 0.5, 1, 0, 0, 0]),
            ... ]
            >>> robot_motion.execute_trajectory(waypoints, timing=[2.0, 2.0])
        """
        if not waypoints:
            raise ValueError("Waypoints list is empty")

        if timing is None:
            timing = [2.0] * len(waypoints)  # Default 2 seconds per waypoint

        if len(timing) != len(waypoints):
            raise ValueError(f"Timing length {len(timing)} doesn't match waypoints {len(waypoints)}")

        log.info(f"Executing trajectory with {len(waypoints)} waypoints")

        for i, (waypoint, wait_time) in enumerate(zip(waypoints, timing)):
            if waypoint.shape[0] != 7:
                raise ValueError(f"Waypoint {i} has wrong dimension: {waypoint.shape[0]} (expected 7)")

            log.info(f"Moving to waypoint {i+1}/{len(waypoints)}")
            self.send_pose_command(waypoint)
            time.sleep(wait_time)

        log.info("Trajectory execution completed")

    def move_line(self,
                  target_pose: np.ndarray,
                  duration: Optional[float] = None,
                  buffer_size: int = 200) -> None:
        """
        Execute linear motion from current pose to target pose using CartesianTrajectory planner

        Args:
            target_pose: Target end-effector pose
                - Single arm: shape=(7,) [x,y,z,qx,qy,qz,qw]
                - Dual arm: shape=(14,) [left_pose(7), right_pose(7)]
            duration: Trajectory duration in seconds (None=auto-calculated based on distance)
            buffer_size: Trajectory buffer size (default: 200 points)

        Features:
            - Linear interpolation in position space
            - Slerp (Spherical Linear Interpolation) for orientation
            - Quintic polynomial for smooth velocity/acceleration profiles
            - Does NOT require use_trajectory_planner=true in config

        Note:
            This method is blocking and will return after trajectory completion

        Example:
            >>> # Move to target position in straight line
            >>> target = np.array([0.5, 0.0, 0.3, 0, 0, 0, 1])
            >>> robot_motion.move_line(target, duration=3.0)

            >>> # Auto-calculate duration based on distance
            >>> robot_motion.move_line(target)
        """
        from trajectory.cartesian_trajectory import CartessianTrajectory
        from hardware.base.utils import Buffer, TrajectoryState
        import threading

        # Validate target pose dimension
        num_ee = len(self._ee_links)
        expected_dim = 7 * num_ee

        if target_pose.shape[0] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}D pose for {num_ee} end effector(s), "
                f"got {target_pose.shape[0]}D"
            )

        # 1. Get current pose
        current_state = self.get_state()
        start_pose = current_state['pose']

        # For dual arm, need to get both arm poses
        if num_ee > 1:
            # Dual arm: get poses for each arm
            poses = []
            all_joint_states = self._robot_system.get_joint_states()
            for i, ee_link in enumerate(self._ee_links):
                key = self._ee_index[i]
                ee_pose = self._motion_factory.get_frame_pose_with_joint_state(
                    all_joint_states, ee_link, key, need_vel=False
                )
                poses.append(ee_pose[:7])
            start_pose = np.concatenate(poses)

        # 2. Auto-calculate duration if not specified
        if duration is None:
            # Calculate Cartesian distance
            distance = np.linalg.norm(target_pose[:3] - start_pose[:3])
            # Use conservative speed: 0.1 m/s
            duration = max(distance / 0.1, 1.0)  # Minimum 1 second
            log.info(f"Auto-calculated duration: {duration:.2f}s for {distance:.3f}m distance")

        # 3. Create trajectory planner instance
        dim = 7 if num_ee == 1 else 14
        buffer = Buffer(size=buffer_size, dim=dim)
        buffer_lock = threading.Lock()

        traj_config = {
            "dt": 0.01,  # 100Hz planning frequency
            "interpolation_type": "quintic",  # Smooth quintic polynomial
            "max_velocity": 1.0,  # Max Cartesian velocity (m/s)
            "enable_motion": False,
            "enable_online_planning": False
        }

        cart_traj = CartessianTrajectory(traj_config, buffer, buffer_lock)

        # 4. Build trajectory target
        traj_target = TrajectoryState()
        traj_target._zero_order_values = np.vstack([start_pose, target_pose])
        traj_target._first_order_values = np.zeros((2, dim))
        traj_target._second_order_values = np.zeros((2, dim))

        # 5. Start trajectory planning in background thread
        planning_thread = threading.Thread(
            target=cart_traj.plan_trajectory,
            args=(traj_target, duration)
        )
        planning_thread.daemon = True
        planning_thread.start()

        # 6. Execute trajectory by reading from buffer
        log.info(f"Executing linear trajectory to position {target_pose[:3]} over {duration:.2f}s")
        start_time = time.perf_counter()
        waypoint_count = 0

        try:
            # Wait for initial buffer fill
            time.sleep(0.05)

            while planning_thread.is_alive() or buffer.size() > 0:
                buffer_lock.acquire()
                success, waypoint, timestamp = buffer.pop_data()
                buffer_lock.release()

                if success:
                    # Send waypoint as high-level command
                    self._motion_factory.update_high_level_command(waypoint)
                    waypoint_count += 1

                    # Wait for control loop timing
                    time.sleep(self._control_loop_time)
                else:
                    time.sleep(0.001)  # Avoid busy waiting

            # Wait for planning thread to complete
            planning_thread.join(timeout=duration + 2.0)

            elapsed = time.perf_counter() - start_time
            log.info(f"Linear trajectory completed: {waypoint_count} waypoints in {elapsed:.2f}s")

        except Exception as e:
            log.error(f"Error during linear trajectory execution: {e}")
            raise
        finally:
            # Ensure thread cleanup
            if planning_thread.is_alive():
                log.warning("Planning thread still alive after trajectory completion")

    def enable_hardware(self) -> None:
        """
        Enable hardware execution (robot will move)

        This allows commands sent via send_pose_command(), send_joint_command(), etc.
        to actually control the physical robot hardware.

        Note:
            - Equivalent to pressing 'h' key when hardware is disabled
            - Safe to call multiple times (idempotent)

        Example:
            >>> robot_motion.enable_hardware()
            >>> robot_motion.send_pose_command(target_pose)  # Robot will move
        """
        if not self._enable_hardware:
            self._enable_hardware = True
            self._motion_factory.update_execute_hardware(True)
            log.info("=" * 15 + " Hardware execution: ENABLED " + "=" * 15)
        else:
            log.info("Hardware already enabled")

    def disable_hardware(self) -> None:
        """
        Disable hardware execution (dry-run mode for testing)

        Commands will be processed but not sent to physical robot hardware.
        Useful for testing motion logic without moving the robot.

        Note:
            - Equivalent to pressing 'h' key when hardware is enabled
            - Safe to call multiple times (idempotent)

        Example:
            >>> robot_motion.disable_hardware()
            >>> robot_motion.send_pose_command(target_pose)  # No physical motion
        """
        if self._enable_hardware:
            self._enable_hardware = False
            self._motion_factory.update_execute_hardware(False)
            log.info("=" * 15 + " Hardware execution: DISABLED " + "=" * 15)
        else:
            log.info("Hardware already disabled")

    def reset_to_home(self, home_pose: Optional[np.ndarray] = None,
                     space: Robot_Space = None) -> None:
        """
        Reset robot to home position

        Args:
            home_pose: Target pose/joint positions. If None, uses configuration default
            space: Motion space (CARTESIAN_SPACE or JOINT_SPACE)

        Workflow:
            1. Pause motion updates (_update_motion_state = False)
            2. Call motion_factory.reset_robot_system()
            3. Wait for arrival
            4. Resume motion updates (_update_motion_state = True)
        """
        log.info(f"Resetting to home position (space: {space})")

        # Pause motion updates
        self._update_motion_state = False

        try:
            # Ensure controller thread can accept new targets
            self._motion_factory.release_blocking_motion()
            self._motion_factory.clear_high_level_command()

            # Use configured defaults if not provided
            if home_pose is None:
                home_pose = self._reset_arm_command
            if space is None:
                space = self._reset_space or Robot_Space.CARTESIAN_SPACE

            # Reset robot
            self._motion_factory.reset_robot_system(
                arm_command=home_pose,
                space=space,
                tool_command=self._reset_tool_command
            )

            log.info("Robot reset completed")

        finally:
            # Resume motion updates
            time.sleep(0.1)
            self._update_motion_state = True

            # Clear previous high-level command by setting current pose as target
            # This prevents robot from returning to previous command after reset
            current_state = self.get_state()
            self._motion_factory.update_high_level_command(current_state['pose'])

    def start_recording(self) -> None:
        """
        Start data recording

        Workflow:
            1. If data_recorder is None, create EpisodeWriter instance
            2. Call data_recorder.create_episode()
            3. Enable motion_factory action recording
            4. Set _enable_recording = True

        Raises:
            RuntimeError: If previous episode is not saved yet
        """
        if self.data_recorder is None:
            os.makedirs(self._save_path_dir, exist_ok=True)
            log.info(f"Creating data recorder at {self._save_path_dir}")
            self.data_recorder = EpisodeWriter(
                task_dir=self._save_path_dir,
                frequency=self._data_record_frequency,
                rerun_log=self._rerun_visualization,
                task_description=self._task_description,
                task_description_goal=self._task_description_goal,
                task_description_steps=self._task_description_steps
            )

        # Create new episode
        if not self.data_recorder.create_episode():
            raise RuntimeError("Failed to create episode - previous episode may not be saved")

        # Enable action recording in motion factory
        self._motion_factory.change_update_action_status(True)

        self._enable_recording = True
        log.info("=" * 15 + " Data recording started " + "=" * 15)

    def stop_recording(self) -> None:
        """
        Stop data recording

        Workflow:
            1. Set _enable_recording = False
            2. Call data_recorder.save_episode()
            3. Disable motion_factory action recording
        """
        if not self._enable_recording:
            log.warning("Recording is not active")
            return

        self._enable_recording = False

        # Save episode
        if self.data_recorder is not None:
            self.data_recorder.save_episode()

        # Disable action recording
        self._motion_factory.change_update_action_status(False)

        time.sleep(0.5)
        log.info("=" * 15 + " Data recording stopped " + "=" * 15)

    def start(self) -> None:
        """
        Start main loop (non-blocking for scripted control)

        Note:
            This method is for script-based interaction. After calling start(),
            the system runs in background, and users can send commands via API.

            To exit, press 'q' or call close()

        Example:
            >>> robot_motion = RobotMotion(config_path)
            >>> robot_motion.start()
            >>>
            >>> # User code here
            >>> robot_motion.send_pose_command(target_pose)
            >>> robot_motion.start_recording()
            >>> # ... more commands ...
            >>> robot_motion.stop_recording()
            >>>
            >>> robot_motion.close()
        """
        log.info("RobotMotion started - press 'q' to quit")
        log.info("Keyboard controls: h=hardware, r=record, o=reset, q=quit")

        # System is already running (data collection thread + keyboard listener)
        # Just keep main thread alive
        try:
            while self._main_thread_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
            self.close()

    def close(self) -> None:
        """
        Close system and release resources

        Workflow:
            1. Stop main thread (_main_thread_running = False)
            2. Stop data collection thread
            3. If recording, save episode
            4. Close data_recorder
            5. Close motion_factory
            6. Close robot_system
            7. Close OpenCV windows
        """
        log.info("=" * 15 + " Closing RobotMotion " + "=" * 15)

        try:
            # Stop threads
            self._main_thread_running = False
            self._update_motion_state = False

            # Stop keyboard listener
            stop_listening()

            # Wait for data thread to finish
            if self._data_recording_thread is not None:
                self._data_recording_thread.join(timeout=2.0)

            # Save ongoing recording
            if self._enable_recording and self.data_recorder is not None:
                log.info("Saving ongoing recording...")
                self.data_recorder.save_episode()

            # Close data recorder
            if self.data_recorder is not None:
                self.data_recorder.close()

            # Close motion factory
            self._motion_factory.close()

            # Close robot system
            self._robot_system.close()

            # Close OpenCV windows
            if self._img_visualization:
                cv2.destroyAllWindows()

            log.info("RobotMotion closed successfully")

        except Exception as e:
            log.error(f"Error during close: {e}")

    # ==================== Internal Methods ====================

    def _data_collection_loop(self) -> None:
        """
        Data collection loop (runs in separate thread at configured frequency)

        Collects sensor data and robot states, saves to EpisodeWriter if recording enabled
        """
        log.info("Data collection loop started")

        target_period = 1.0 / self._data_record_frequency
        next_run_time = time.time()

        while self._main_thread_running:
            loop_start = time.time()

            # Collect and save data
            try:
                self._collect_and_save_data()
            except Exception as e:
                log.error(f"Error in data collection: {e}")

            # Frequency control (precise timing)
            next_run_time += target_period
            sleep_time = next_run_time - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Handle overrun
                log.warning(f"Data collection loop overrun: {-sleep_time:.3f}s")
                next_run_time = time.time()

        log.info("Data collection loop stopped")

    def _collect_and_save_data(self) -> None:
        """
        Collect sensor data and save to recorder (called by data collection loop)

        Collects:
            - colors: Camera RGB images
            - depths: Depth images
            - imus: IMU data
            - joint_states: Joint positions/velocities/accelerations/torques
            - ee_states: End effector pose/twist
            - tools: Gripper states
            - tactiles: Tactile sensor data
            - actions: Motion commands (joint/ee/tool)
        """
        # Get camera data
        cameras_data = self._robot_system.get_cameras_infos()
        image_list = []
        cur_colors = {}
        cur_depths = {}
        cur_imus = {}

        if cameras_data is not None:
            for cam_data in cameras_data:
                name = cam_data['name']
                if 'color' in name:
                    cur_colors[name] = {
                        "data": cam_data['img'],
                        "time_stamp": cam_data['time_stamp']
                    }
                    image_list.append(cam_data['img'])
                if 'depth' in name:
                    cur_depths[name] = {
                        "data": cam_data['img'],
                        "time_stamp": cam_data['time_stamp']
                    }
                if 'imu' in name:
                    cur_imus[name] = {
                        "data": cam_data['imu'],
                        "time_stamp": cam_data['time_stamp']
                    }

        # Image visualization
        if len(image_list) > 0 and self._img_visualization:
            # Debug: print image shapes on first call
            if not hasattr(self, '_debug_shapes_printed'):
                log.info("=" * 60)
                log.info("Camera image shapes (H×W×C):")
                for i, img in enumerate(image_list):
                    log.info(f"  Camera {i}: {img.shape}")
                log.info("=" * 60)
                self._debug_shapes_printed = True

            combined_imgs = image_list[0]
            for img in image_list[1:]:
                combined_imgs = combine_image(combined_imgs, img)

            # Debug: print combined image size on first call
            if not hasattr(self, '_debug_combined_printed'):
                log.info(f"Combined image size: {combined_imgs.shape} (will be displayed in OpenCV window)")
                log.info(f"Note: If your screen is smaller than {combined_imgs.shape[1]}px wide, window will be auto-scaled")
                self._debug_combined_printed = True

            cv2.imshow('Robot Cameras', combined_imgs)
            cv2.waitKey(1)

        # Recording data
        if self._enable_recording and self.data_recorder is not None:
            # Get joint states
            joint_states = {}
            ee_states = {}
            gripper_state = {}

            all_joint_states = self._robot_system.get_joint_states()
            tool_state_dict = self._robot_system.get_tool_dict_state()

            # Process each robot
            for i, cur_ee_link in enumerate(self._ee_links):
                key = self._ee_index[i]

                # Joint states
                sliced_joint_states = self._motion_factory.get_type_joint_state(
                    all_joint_states, key
                )
                joint_states[key] = {
                    "position": sliced_joint_states._positions.tolist(),
                    "velocity": sliced_joint_states._velocities.tolist(),
                    "acceleration": sliced_joint_states._accelerations.tolist(),
                    "torque": sliced_joint_states._torques.tolist(),
                    "time_stamp": sliced_joint_states._time_stamp
                }

                # End effector states
                cur_ee_pose = self._motion_factory.get_frame_pose_with_joint_state(
                    all_joint_states, cur_ee_link, key, need_vel=True
                )
                ee_states[key] = {
                    "pose": cur_ee_pose[:7].tolist(),
                    "twist": cur_ee_pose[7:13].tolist(),
                    "time_stamp": sliced_joint_states._time_stamp
                }

                # Tool states
                if tool_state_dict is not None and key in tool_state_dict:
                    gripper_state[key] = {
                        'position': tool_state_dict[key]._position,
                        'time_stamp': tool_state_dict[key]._time_stamp
                    }

            # Get tactile data
            tactiles = self._robot_system.get_tactile_data()

            # Get actions
            motion_action = self._motion_factory.get_latest_action()
            actions = {}

            # Build actions for each arm
            for i, cur_ee_link in enumerate(self._ee_links):
                key = self._ee_index[i]
                actions[key] = {}

                # Joint action: use motion_action if available, otherwise use current joint state
                if motion_action is not None and key in motion_action:
                    joint_action_pos = motion_action[key]["joint"]["position"]
                    joint_action_time = motion_action[key]["joint"]["time_stamp"]
                else:
                    # No command yet - action equals current state (at rest)
                    joint_action_pos = joint_states[key]["position"]
                    joint_action_time = joint_states[key]["time_stamp"]

                actions[key]["joint"] = {
                    "position": joint_action_pos,
                    "time_stamp": joint_action_time
                }

                # EE action: compute from joint action using FK
                # This provides continuous trajectory instead of discrete target poses
                ee_action_pose_homo = self._motion_factory._robot_model.get_frame_pose(
                    cur_ee_link,
                    np.array(joint_action_pos),
                    need_update=True,
                    model_type=key
                )
                # Convert SE3 (4x4 homogeneous matrix) to 7D pose (x,y,z,qx,qy,qz,qw)
                ee_action_pose_7d = convert_homo_2_7D_pose(ee_action_pose_homo)
                actions[key]["ee"] = {
                    "pose": ee_action_pose_7d.tolist(),
                    "time_stamp": joint_action_time
                }

                # Tool action: use recorded action if available, otherwise use current gripper state
                # Note: action["tool"] (singular) is normalized (0-1), state["tools"] (plural) keeps hardware units
                with self._tool_action_lock:
                    if key in self._latest_tool_action and "tool" in self._latest_tool_action[key]:
                        # Use cached action (already normalized 0-1)
                        actions[key]["tool"] = self._latest_tool_action[key]["tool"]
                    elif key in gripper_state:
                        # Fallback to current state - need to normalize
                        tool_pos_raw = gripper_state[key]["position"]
                        # Normalize if in meters (Franka hand: 0-0.08m → 0-1)
                        tool_pos_normalized = tool_pos_raw
                        if tool_pos_raw <= 0.1:
                            tool_pos_normalized = min(1.0, tool_pos_raw / 0.08)

                        # Apply binary threshold if tool is in BINARY mode
                        tool = self._robot_system._tool
                        if tool is not None and hasattr(tool, '_control_mode') and tool._control_mode == ToolControlMode.BINARY:
                            threshold = getattr(tool, '_binary_threshold', 0.5)
                            tool_pos_normalized = 1.0 if tool_pos_normalized >= threshold else 0.0

                        actions[key]["tool"] = {
                            "position": tool_pos_normalized,
                            "time_stamp": gripper_state[key]["time_stamp"]
                        }

            # Save data
            self.data_recorder.add_item(
                colors=cur_colors if len(cur_colors) > 0 else None,
                depths=cur_depths if len(cur_depths) > 0 else None,
                joint_states=joint_states,
                ee_states=ee_states,
                tools=gripper_state,
                tactiles=tactiles,
                imus=cur_imus if len(cur_imus) > 0 else None,
                actions=actions
            )

    def _keyboard_callback(self, key: str) -> None:
        """
        Keyboard event callback

        Args:
            key: Pressed key

        Controls:
            - 'h': Toggle hardware execution
            - 'r': Toggle recording
            - 'o': Reset to home
            - 'q': Quit
        """
        if key == 'h':
            self._toggle_hardware()
        elif key == 'r':
            self._toggle_recording()
        elif key == 'o':
            self._reset_robot()
        elif key == 'q':
            self.close()

    def _toggle_hardware(self) -> None:
        """Toggle hardware execution on/off"""
        self._enable_hardware = not self._enable_hardware
        self._motion_factory.update_execute_hardware(self._enable_hardware)
        log.info(f"=" * 15 + f" Hardware execution: {self._enable_hardware} " + "=" * 15)

    def _toggle_recording(self) -> None:
        """Toggle data recording on/off"""
        if self._enable_recording:
            self.stop_recording()
        else:
            try:
                self.start_recording()
            except RuntimeError as e:
                log.error(f"Failed to start recording: {e}")

    def _reset_robot(self) -> None:
        """Reset robot to home position (triggered by keyboard)"""
        try:
            self.reset_to_home()
        except Exception as e:
            log.error(f"Failed to reset robot: {e}")
