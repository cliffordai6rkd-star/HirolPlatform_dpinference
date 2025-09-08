"""
Accuracy metric evaluation for IK algorithms.

Evaluates position and orientation accuracy of IK solutions
compared to target poses.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R
import glog as log


@dataclass
class AccuracyResult:
    """Results from accuracy evaluation."""
    mean_position_error: float
    std_position_error: float  
    max_position_error: float
    mean_rotation_error: float
    std_rotation_error: float
    max_rotation_error: float
    converged_tests: int
    total_tests: int
    convergence_rate: float


class AccuracyMetric:
    """Evaluates accuracy of IK solutions against target poses."""
    
    def __init__(self, position_tolerance: float = 1e-6, 
                 rotation_tolerance: float = 1e-6):
        """
        Initialize accuracy metric evaluator.
        
        Args:
            position_tolerance: Tolerance for position accuracy (meters)
            rotation_tolerance: Tolerance for rotation accuracy (radians)
        """
        self._pos_tol = position_tolerance
        self._rot_tol = rotation_tolerance
    
    def evaluate(self, solutions: List[np.ndarray], 
                target_poses: List[np.ndarray],
                achieved_poses: List[np.ndarray],
                converged_flags: List[bool]) -> AccuracyResult:
        """
        Evaluate accuracy of IK solutions.
        
        Args:
            solutions: List of joint angle solutions
            target_poses: List of target 4x4 transformation matrices
            achieved_poses: List of achieved 4x4 transformation matrices from FK
            converged_flags: List of convergence flags for each solution
            
        Returns:
            AccuracyResult containing statistical accuracy metrics
        """
        assert len(target_poses) == len(achieved_poses) == len(converged_flags), \
            "Input lists must have same length"
        
        position_errors = []
        rotation_errors = []
        converged_count = 0
        
        for i, (target, achieved, converged) in enumerate(
            zip(target_poses, achieved_poses, converged_flags)):
            
            if converged:
                pos_error, rot_error = self._compute_pose_error(achieved, target)
                position_errors.append(pos_error)
                rotation_errors.append(rot_error)
                converged_count += 1
            else:
                # For non-converged solutions, use large error values
                position_errors.append(float('inf'))
                rotation_errors.append(float('inf'))
        
        # Filter out infinite errors for statistical calculations
        finite_pos_errors = [e for e in position_errors if np.isfinite(e)]
        finite_rot_errors = [e for e in rotation_errors if np.isfinite(e)]
        
        if len(finite_pos_errors) == 0:
            log.warning("No converged solutions found for accuracy evaluation")
            return AccuracyResult(
                mean_position_error=float('inf'),
                std_position_error=0.0,
                max_position_error=float('inf'),
                mean_rotation_error=float('inf'),
                std_rotation_error=0.0,
                max_rotation_error=float('inf'),
                converged_tests=0,
                total_tests=len(target_poses),
                convergence_rate=0.0
            )
        
        return AccuracyResult(
            mean_position_error=np.mean(finite_pos_errors),
            std_position_error=np.std(finite_pos_errors),
            max_position_error=np.max(finite_pos_errors),
            mean_rotation_error=np.mean(finite_rot_errors),
            std_rotation_error=np.std(finite_rot_errors),
            max_rotation_error=np.max(finite_rot_errors),
            converged_tests=converged_count,
            total_tests=len(target_poses),
            convergence_rate=converged_count / len(target_poses)
        )
    
    def _compute_pose_error(self, achieved: np.ndarray, 
                           target: np.ndarray) -> Tuple[float, float]:
        """
        Compute position and rotation error between achieved and target poses.
        
        Args:
            achieved: 4x4 achieved transformation matrix
            target: 4x4 target transformation matrix
            
        Returns:
            Tuple of (position_error, rotation_error)
        """
        # Position error (Euclidean distance)
        pos_error = np.linalg.norm(achieved[:3, 3] - target[:3, 3])
        
        # Rotation error (angle between rotation matrices)
        rot_error = self._compute_rotation_error(achieved[:3, :3], target[:3, :3])
        
        return pos_error, rot_error
    
    def _compute_rotation_error(self, R_achieved: np.ndarray, 
                               R_target: np.ndarray) -> float:
        """
        Compute rotation error as angle between two rotation matrices.
        
        Args:
            R_achieved: 3x3 achieved rotation matrix
            R_target: 3x3 target rotation matrix
            
        Returns:
            Rotation error in radians
        """
        # Compute relative rotation
        R_relative = R_achieved @ R_target.T
        
        # Extract angle from rotation matrix
        trace = np.trace(R_relative)
        # Clamp trace to valid range to avoid numerical issues
        trace = np.clip(trace, -1, 3)
        angle = np.arccos((trace - 1) / 2)
        
        return abs(angle)