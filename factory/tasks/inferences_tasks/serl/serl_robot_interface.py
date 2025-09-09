"""
SERL Robot Interface for HIROLRobotPlatform
Provides a unified interface for HIL-SERL to interact with robots
"""

import sys
import numpy as np
import yaml
import time
from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import glog as log

# Add HIROLRobotPlatform to path
platform_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(platform_path))

from factory.components.robot_factory import RobotFactory
from factory.components.motion_factory import MotionFactory, Robot_Space
from hardware.base.utils import dynamic_load_yaml


@dataclass
class ComplianceParams:
    """Compliance parameters for impedance control"""
    translational_stiffness: float = 1500.0
    translational_damping: float = 80.0
    rotational_stiffness: float = 100.0
    rotational_damping: float = 10.0


class SerlRobotInterface:
    """
    SERL Robot Interface using HIROLRobotPlatform components
    Provides a unified interface compatible with HIL-SERL
    """
    
    def __init__(self, config_path: Optional[str] = None, auto_initialize: bool = True):
        """
        Initialize SERL Robot Interface
        
        Args:
            config_path: Path to configuration file
            auto_initialize: Whether to auto-initialize robot connection
        """
        # Load configuration
        if config_path is None:
            config_path = str(Path(__file__).parent / "config" / "serl_fr3_config.yaml")
        
        # Use dynamic_load_yaml to handle !include tags
        self._config = dynamic_load_yaml(config_path)
        
        # Initialize robot system using factory components
        self._robot_system = RobotFactory(self._config["motion_config"])
        self._motion_factory = MotionFactory(self._config["motion_config"], self._robot_system)
        
        # State tracking
        self._current_pose = None
        self._current_gripper_state = 1.0  # Default to open (1.0 = open, 0.0 = closed)
        self._compliance_params = ComplianceParams()
        
        # Initialize if requested
        if auto_initialize:
            self.initialize()
            
        log.info("SerlRobotInterface initialized successfully")
    
    def initialize(self) -> None:
        """Initialize robot system and motion components"""
        # Create motion components (includes robot system creation)
        self._motion_factory.create_motion_components()
        self._motion_factory.update_execute_hardware(True)
        time.sleep(0.5)
        
        # Get initial state
        self._update_state()
        
        log.info("Robot system initialized and ready")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current robot state
        
        Returns:
            Dictionary containing:
                - pose: [x, y, z, qx, qy, qz, qw]
                - vel: TCP velocity (6D)
                - force: TCP force (3D)
                - torque: TCP torque (3D)
                - gripper_pos: Gripper position [0-1]
                - gripper_is_grasped: Whether object is grasped
                - q: Joint positions
                - dq: Joint velocities
                - jacobian: Jacobian matrix
        """
        # Update internal state
        self._update_state()
        
        # Get joint states
        joint_states = self._robot_system.get_joint_states()
        
        # Get end effector pose
        ee_links = self._motion_factory.get_model_end_effector_link_list()
        ee_name = ee_links[0] if ee_links else "fr3_hand_tcp"
        model_types = self._motion_factory.get_model_types()
        model_type = model_types[0] if model_types else "model"
        
        # Get TCP pose (returns 7D pose: [x, y, z, qx, qy, qz, qw])
        tcp_pose_7d = self._motion_factory.get_frame_pose(ee_name, model_type)
        
        # Get Jacobian
        jacobian = self._motion_factory._robot_model.get_jacobian(
            ee_name, 
            joint_states._positions,
            model_type=model_type
        )
        if jacobian is None:
            jacobian = np.zeros((6, 7))
        
        # Calculate TCP velocity using Jacobian
        tcp_velocity = jacobian @ joint_states._velocities if joint_states._velocities is not None else np.zeros(6)
        
        # Get force/torque 
        # TODO: Decouple this when RobotFactory implements get_tcp_force_torque
        # Currently assumes FR3 robot with panda-py interface
        force = np.zeros(3)
        torque = np.zeros(3)
        
        # Try to get force/torque from FR3 robot directly
        if hasattr(self._robot_system, '_robot') and hasattr(self._robot_system._robot, '_fr3_state'):
            try:
                fr3_state = self._robot_system._robot._fr3_state
                if fr3_state is not None:
                    # Use external forces in base frame (O_F_ext_hat_K)
                    force_torque = np.array(fr3_state.O_F_ext_hat_K)
                    force = force_torque[:3]  # First 3 values are forces (N)
                    torque = force_torque[3:]  # Last 3 values are torques (Nm)
            except Exception as e:
                # Fallback to zeros if not available
                pass
        
        # Alternative: Try external force sensor if available
        if np.allclose(force, 0) and np.allclose(torque, 0):
            if hasattr(self._robot_system, '_sensors') and 'FT_sensor' in self._robot_system._sensors:
                try:
                    ft_sensors = self._robot_system._sensors['FT_sensor']
                    if ft_sensors and len(ft_sensors) > 0:
                        ft_data = ft_sensors[0]['object'].get_ft_data()
                        if ft_data is not None:
                            force = ft_data[:3]
                            torque = ft_data[3:]
                except:
                    pass
        
        # Get gripper state
        tool_state = self._robot_system.get_tool_dict_state()
        if tool_state and 'single' in tool_state:
            # tool_state['single'] is a ToolState object
            tool_state_obj = tool_state['single']
            if hasattr(tool_state_obj, '_position'):
                gripper_pos = float(tool_state_obj._position)
                # The gripper position from Franka hand is in meters (0-0.08m)
                # We need to normalize it to 0-1 range for consistency
                # But tool_state_obj._position is already normalized by franka_hand.py
                # It stores the raw value in meters, so we normalize here
                if gripper_pos <= 0.1:  # Likely in meters (max 0.08m + margin)
                    gripper_pos = gripper_pos / 0.08  # Normalize to 0-1
                    gripper_pos = min(1.0, gripper_pos)  # Clamp to 1.0 max
            else:
                # Fallback for array-like state
                gripper_pos = float(tool_state_obj[0]) if hasattr(tool_state_obj, '__getitem__') else self._current_gripper_state
        else:
            # Use internal state if tool state not available
            gripper_pos = self._current_gripper_state
        
        # Check if object is grasped (simplified logic)
        gripper_is_grasped = gripper_pos < 0.5
        
        return {
            "pose": tcp_pose_7d,
            "vel": tcp_velocity,
            "force": force,
            "torque": torque,
            "gripper_pos": np.array([gripper_pos]),
            "gripper_is_grasped": gripper_is_grasped,
            "q": joint_states._positions,
            "dq": joint_states._velocities,
            "jacobian": jacobian
        }
    
    def send_pos_command(self, pose: np.ndarray) -> None:
        """
        Send position control command
        
        Args:
            pose: Target TCP pose [x, y, z, qx, qy, qz, qw]
        """
        # Ensure pose is 7D
        if pose.shape[0] != 7:
            raise ValueError(f"Expected 7D pose, got {pose.shape[0]}D")
        
        # Update high-level command for motion factory
        self._motion_factory.update_high_level_command(pose)
        self._current_pose = pose.copy()
    
    # def send_pos_trajectory_command(self, pose: np.ndarray, finish_time: float) -> None:
    #     """
    #     Send trajectory command with specified finish time
        
    #     Args:
    #         pose: Target TCP pose [x, y, z, qx, qy, qz, qw]
    #         finish_time: Time to reach target (seconds)
    #     """
    #     # Send trajectory command
    #     self.send_pos_command(pose)
        
    #     # Wait for motion to complete
    #     time.sleep(finish_time)
    
    def open_gripper(self) -> None:
        """Open the gripper (1.0 = fully open)"""
        log.info("[DEBUG] Opening gripper to 1.0")
        self._robot_system.set_tool_command({"single": np.array([1.0])})
        # time.sleep(0.5)  # Wait for gripper to open
        # Update state after gripper moves
        self._update_state()
        log.info(f"[DEBUG] Gripper opened, state={self._current_gripper_state:.2f}")
    
    def close_gripper(self) -> None:
        """Close the gripper (0.0 = fully closed)"""
        self._robot_system.set_tool_command({"single": np.array([0.])})
        # Update state after gripper moves
        self._update_state()
        log.info(f"[DEBUG] Gripper closed, state={self._current_gripper_state:.2f}")
    
    def send_gripper_command(self, width: float, mode: str = "binary") -> None:
        """
        Send gripper command
        
        Args:
            width: Gripper width (0=closed, 1=open for binary mode)
            mode: Control mode ("binary" or "continuous")
        """
        if mode == "binary":
            if width < 0.5:
                self.close_gripper()
            else:
                self.open_gripper()
        else:
            # Continuous mode
            self._robot_system.set_tool_command({"single": np.array([width])})
            time.sleep(0.1)
            self._update_state()
    
    def clear_errors(self) -> None:
        """Clear any robot errors"""
        # Reset robot system if needed
        if hasattr(self._robot_system._robot, 'clear_errors'):
            self._robot_system._robot.clear_errors()
        # log.info("Errors cleared")
    
    def recover_gripper(self) -> bool:
        """
        Recover gripper from error state
        
        Returns:
            True if recovery successful
        """
        try:
            # Try to reinitialize gripper
            if hasattr(self._robot_system, '_tool') and self._robot_system._tool:
                # self._robot_system._tool.initialize()
                self._robot_system._tool.recover()
            
            # Open and close to verify
            self.open_gripper()
            time.sleep(0.5)
            return True
        except Exception as e:
            log.error(f"Gripper recovery failed: {e}")
            return False
    
    def joint_reset(self, home_joints: Optional[np.ndarray] = None) -> None:
        """
        Reset robot to home joint position
        
        Args:
            home_joints: Optional home joint positions. If None, uses robot's default home position
        """
        # Reset using motion factory (it handles None gracefully with defaults)
        self._motion_factory.reset_robot_system(
            arm_command=home_joints,
            space=Robot_Space.JOINT_SPACE,
            tool_command={"single": np.array([1.0])}  # Open gripper
        )
        
        log.info("Joint reset completed")
    
    def cartesian_reset(self, home_pose: Optional[np.ndarray] = None) -> None:
        """
        Reset robot to home Cartesian position
        
        Args:
            home_pose: Optional home pose [x, y, z, qx, qy, qz, qw]. If None, uses robot's current position
        """
        # Reset using motion factory in Cartesian space
        self._motion_factory.reset_robot_system(
            arm_command=home_pose,
            space=Robot_Space.CARTESIAN_SPACE,
            tool_command={"single": np.array([1.0])}  # Open gripper
        )
        
        log.info("Cartesian reset completed")
    
    def update_params(self, params: ComplianceParams) -> None:
        """
        Update compliance parameters
        
        Args:
            params: New compliance parameters
        """
        self._compliance_params = params
        
        # Update controller parameters based on available methods
        controller = self._motion_factory._controller
        
        # For CartesianImpedanceController
        if hasattr(controller, 'set_stiffness'):
            controller.set_stiffness(
                translational_stiffness=params.translational_stiffness,
                rotational_stiffness=params.rotational_stiffness
            )
        
        if hasattr(controller, 'set_damping'):
            controller.set_damping(
                translational_damping=params.translational_damping,
                rotational_damping=params.rotational_damping
            )
        
        log.debug(f"Updated compliance params: trans_stiff={params.translational_stiffness}, "
                 f"rot_stiff={params.rotational_stiffness}")
    
    def _update_state(self) -> None:
        """Update internal state from robot"""
        # Get current gripper state and update internal tracking
        tool_state = self._robot_system.get_tool_dict_state()
        if tool_state and 'single' in tool_state:
            tool_state_obj = tool_state['single']
            if hasattr(tool_state_obj, '_position'):
                gripper_pos = float(tool_state_obj._position)
                # Normalize if in meters
                if gripper_pos <= 0.1:  # Likely in meters
                    gripper_pos = gripper_pos / 0.08
                    gripper_pos = min(1.0, gripper_pos)
                self._current_gripper_state = gripper_pos
    
    def close(self) -> None:
        """Clean up resources"""
        try:
            # Stop motion factory threads
            if hasattr(self._motion_factory, '_controller_thread_running'):
                self._motion_factory._controller_thread_running = False
            if hasattr(self._motion_factory, '_traj_thread_running'):
                self._motion_factory._traj_thread_running = False
            
            # Close robot system
            self._robot_system.close()
            
            log.info("SerlRobotInterface closed")
        except Exception as e:
            log.error(f"Error closing interface: {e}")