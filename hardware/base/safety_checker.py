"""
Robot Safety Checker Module

Provides common safety checking functionality for robot motion control.
Can be used across different robot implementations.
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety check severity levels"""
    PERMISSIVE = "permissive"    # Loose limits for fast motion
    NORMAL = "normal"            # Standard safety limits
    STRICT = "strict"            # Tight limits for precise tasks


@dataclass
class SafetyLimits:
    """Safety limit configuration"""
    # Joint motion limits  
    max_joint_change: float = 1.5          # Max joint change per command (rad)
    max_joint_velocity: float = 25.0       # Max joint velocity (rad/s)
    max_joint_acceleration: float = 25.0    # Max joint acceleration (rad/s²)
    
    # Cartesian motion limits
    max_position_change: float = 0.01      # Max position change per command (m)
    max_rotation_change: float = 0.2       # Max rotation change per command (rad)
    max_linear_velocity: float = 0.1       # Max linear velocity (m/s)
    max_angular_velocity: float = 1.0      # Max angular velocity (rad/s)
    
    # Timing limits
    min_command_interval: float = 0.001    # Min time between commands (s)
    command_timeout: float = 1.0           # Command timeout (s)


class SafetyChecker:
    """
    Universal robot safety checker
    
    Provides safety validation for joint and Cartesian motions across different robot types.
    """
    
    # Predefined safety profiles
    SAFETY_PROFILES = {
        SafetyLevel.PERMISSIVE: SafetyLimits(
            max_joint_change=1.0,
            max_position_change=0.05,
            max_rotation_change=0.5
        ),
        SafetyLevel.NORMAL: SafetyLimits(),  # Default values
        SafetyLevel.STRICT: SafetyLimits(
            max_joint_change=0.1,
            max_position_change=0.005,
            max_rotation_change=0.05
        )
    }
    
    def __init__(self, 
                 limits: Optional[SafetyLimits] = None,
                 safety_level: SafetyLevel = SafetyLevel.NORMAL,
                 robot_name: str = "Robot"):
        """
        Initialize safety checker
        
        Args:
            limits: Custom safety limits, or None to use predefined profile
            safety_level: Safety level if using predefined profile
            robot_name: Robot name for logging
        """
        self.robot_name = robot_name
        self.limits = limits or self.SAFETY_PROFILES[safety_level]
        
        # State tracking
        self._last_joint_positions: Optional[np.ndarray] = None
        self._last_joint_velocities: Optional[np.ndarray] = None
        self._last_cartesian_pose: Optional[np.ndarray] = None
        self._last_command_time: float = 0.0
        
        # Valid state storage for rollback
        self._valid_joint_positions: Optional[np.ndarray] = None
        self._valid_cartesian_pose: Optional[np.ndarray] = None
        
        # Statistics
        self._checks_performed = 0
        self._checks_failed = 0
    
    def update_state(self, 
                    joint_positions: Optional[np.ndarray] = None,
                    joint_velocities: Optional[np.ndarray] = None,
                    cartesian_pose: Optional[np.ndarray] = None):
        """
        Update current robot state for safety tracking
        
        Args:
            joint_positions: Current joint positions (rad)
            joint_velocities: Current joint velocities (rad/s)
            cartesian_pose: Current TCP pose [x,y,z,qx,qy,qz,qw] or 4x4 matrix
        """
        if joint_positions is not None:
            self._last_joint_positions = np.array(joint_positions)
            # Store as valid state if no current valid state exists
            if self._valid_joint_positions is None:
                self._valid_joint_positions = self._last_joint_positions.copy()
        
        if joint_velocities is not None:
            self._last_joint_velocities = np.array(joint_velocities)
        
        if cartesian_pose is not None:
            self._last_cartesian_pose = np.array(cartesian_pose)
            if self._valid_cartesian_pose is None:
                self._valid_cartesian_pose = self._last_cartesian_pose.copy()
    
    def check_joint_command(self, 
                           target_positions: np.ndarray,
                           dt: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if joint command is safe
        
        Args:
            target_positions: Target joint positions (rad)
            dt: Time since last command (s), or None for auto-calculation
            
        Returns:
            (is_safe, reason): Safety check result and failure reason
        """
        self._checks_performed += 1
        
        try:
            target_positions = np.array(target_positions)
            
            # Check command timing
            current_time = time.time()
            if dt is None:
                dt = current_time - self._last_command_time
            
            if dt > 0 and dt < self.limits.min_command_interval:
                self._checks_failed += 1
                return False, f"Command too frequent: {dt:.6f}s < {self.limits.min_command_interval}s"
            
            # Check against last known position
            if self._last_joint_positions is not None:
                joint_diff = np.abs(target_positions - self._last_joint_positions)
                max_diff = np.max(joint_diff)
                
                if max_diff > self.limits.max_joint_change:
                    self._checks_failed += 1
                    return False, f"Large joint change: max={max_diff:.3f}rad > {self.limits.max_joint_change:.3f}rad"
                
                # Check velocity if dt is available
                if dt > 0:
                    velocities = joint_diff / dt
                    max_vel = np.max(velocities)
                    if max_vel > self.limits.max_joint_velocity:
                        self._checks_failed += 1
                        return False, f"High joint velocity: max={max_vel:.3f}rad/s > {self.limits.max_joint_velocity:.3f}rad/s"
            
            # Check against valid reference position
            if self._valid_joint_positions is not None:
                valid_diff = np.abs(target_positions - self._valid_joint_positions)
                max_valid_diff = np.max(valid_diff)
                
                # Use a larger threshold for valid position check
                valid_threshold = self.limits.max_joint_change * 3.0
                if max_valid_diff > valid_threshold:
                    self._checks_failed += 1
                    return False, f"Large deviation from valid state: max={max_valid_diff:.3f}rad > {valid_threshold:.3f}rad"
            
            # Update timing
            self._last_command_time = current_time
            
            return True, "OK"
            
        except Exception as e:
            self._checks_failed += 1
            return False, f"Safety check error: {e}"
    
    def check_cartesian_command(self, 
                               target_pose: np.ndarray) -> Tuple[bool, str]:
        """
        Check if Cartesian command is safe
        
        Args:
            target_pose: Target TCP pose [x,y,z,qx,qy,qz,qw] or 4x4 matrix
            
        Returns:
            (is_safe, reason): Safety check result and failure reason
        """
        self._checks_performed += 1
        
        try:
            target_pose = np.array(target_pose)
            
            if self._last_cartesian_pose is None:
                return True, "OK (no reference pose)"
            
            # Calculate pose difference
            pos_diff, rot_diff = self._calculate_pose_difference(
                self._last_cartesian_pose, target_pose)
            
            # Check position change
            if pos_diff > self.limits.max_position_change:
                self._checks_failed += 1
                return False, f"Large position change: {pos_diff:.4f}m > {self.limits.max_position_change:.4f}m"
            
            # Check rotation change
            if rot_diff > self.limits.max_rotation_change:
                self._checks_failed += 1
                return False, f"Large rotation change: {rot_diff:.4f}rad > {self.limits.max_rotation_change:.4f}rad"
            
            return True, "OK"
            
        except Exception as e:
            self._checks_failed += 1
            return False, f"Cartesian safety check error: {e}"
    
    def _calculate_pose_difference(self, 
                                  pose1: np.ndarray, 
                                  pose2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate position and rotation differences between poses
        
        Supports both [x,y,z,qx,qy,qz,qw] and 4x4 matrix formats
        
        Returns:
            (position_diff, rotation_diff): Differences in meters and radians
        """
        # Handle different pose formats
        if pose1.shape == (7,) and pose2.shape == (7,):
            # [x,y,z,qx,qy,qz,qw] format
            pos1, pos2 = pose1[:3], pose2[:3]
            quat1, quat2 = pose1[3:], pose2[3:]
            
            # Position difference
            position_diff = np.linalg.norm(pos2 - pos1)
            
            # Rotation difference via quaternion dot product
            dot_product = np.abs(np.dot(quat1, quat2))
            dot_product = np.clip(dot_product, 0.0, 1.0)
            rotation_diff = 2 * np.arccos(dot_product)
            
        elif pose1.shape == (4,4) and pose2.shape == (4,4):
            # 4x4 transformation matrix format
            pos1, pos2 = pose1[:3, 3], pose2[:3, 3]
            R1, R2 = pose1[:3, :3], pose2[:3, :3]
            
            # Position difference
            position_diff = np.linalg.norm(pos2 - pos1)
            
            # Rotation difference via trace of relative rotation
            R_rel = R1.T @ R2
            trace = np.trace(R_rel)
            trace = np.clip(trace, -1.0, 3.0)
            rotation_diff = np.arccos((trace - 1) / 2)
            
        else:
            raise ValueError(f"Unsupported pose formats: {pose1.shape}, {pose2.shape}")
        
        return float(position_diff), float(rotation_diff)
    
    def commit_valid_state(self):
        """Mark current state as valid (safe rollback point)"""
        if self._last_joint_positions is not None:
            self._valid_joint_positions = self._last_joint_positions.copy()
        if self._last_cartesian_pose is not None:
            self._valid_cartesian_pose = self._last_cartesian_pose.copy()
    
    def get_valid_joint_positions(self) -> Optional[np.ndarray]:
        """Get last valid joint positions for rollback"""
        return self._valid_joint_positions.copy() if self._valid_joint_positions is not None else None
    
    def get_valid_cartesian_pose(self) -> Optional[np.ndarray]:
        """Get last valid Cartesian pose for rollback"""
        return self._valid_cartesian_pose.copy() if self._valid_cartesian_pose is not None else None
    
    def reset_tracking(self):
        """Reset all state tracking"""
        self._last_joint_positions = None
        self._last_joint_velocities = None
        self._last_cartesian_pose = None
        self._last_command_time = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety check statistics"""
        success_rate = 1.0 - (self._checks_failed / max(1, self._checks_performed))
        return {
            'robot_name': self.robot_name,
            'checks_performed': self._checks_performed,
            'checks_failed': self._checks_failed,
            'success_rate': success_rate,
            'safety_level': self.limits.__dict__
        }
    
    def print_statistics(self):
        """Print safety check statistics"""
        stats = self.get_statistics()
        print(f"=== Safety Statistics ({stats['robot_name']}) ===")
        print(f"Checks performed: {stats['checks_performed']}")
        print(f"Checks failed: {stats['checks_failed']}")
        print(f"Success rate: {stats['success_rate']:.2%}")