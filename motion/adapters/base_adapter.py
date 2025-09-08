"""
Base Adapter for IK Solvers - Abstract interface for IK solver integration.

This module defines the abstract base class that all IK solver adapters must implement
to ensure consistent integration with the HIROL benchmark system.
"""

import abc
import numpy as np
from typing import Tuple, Dict, Any, Optional
import glog as log


class IKAdapterBase(abc.ABC):
    """
    Abstract base class for IK solver adapters.
    
    This class defines the interface that all IK solver adapters must implement
    to integrate with the HIROL platform's benchmark system.
    """
    
    def __init__(self, urdf_path: str, end_effector_link: str, **kwargs):
        """
        Initialize the IK adapter.
        
        Args:
            urdf_path: Path to robot URDF file
            end_effector_link: Name of end effector link
            **kwargs: Additional adapter-specific parameters
        """
        self._urdf_path = urdf_path
        self._ee_link = end_effector_link
        self._config = kwargs
        self._initialized = False
        self._robot_model = None
        
        log.info(f"Created {self.__class__.__name__} adapter for {end_effector_link}")
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialize the adapter and underlying solver.
        
        This method should:
        1. Load robot model from URDF
        2. Set up solver-specific configurations
        3. Prepare any necessary computational graphs/optimizations
        """
        pass
    
    @abc.abstractmethod
    def solve_single(self, 
                    target_pose: np.ndarray, 
                    initial_guess: np.ndarray,
                    tolerance: float = 1e-6,
                    max_iterations: int = 1000) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Solve single IK problem.
        
        Args:
            target_pose: Target 4x4 homogeneous transformation matrix
            initial_guess: Initial joint configuration (n_dof,)
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (converged, solution_joints, solve_time_seconds)
        """
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """
        Check if the IK solver is available and properly configured.
        
        Returns:
            True if solver can be used, False otherwise
        """
        pass
    
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics (optional, for verification).
        
        Args:
            joint_positions: Joint configuration (n_dof,)
            
        Returns:
            4x4 homogeneous transformation matrix
            
        Note:
            Default implementation returns None. Subclasses can override
            this method to provide FK capabilities for solution verification.
        """
        return None
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get robot joint limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits) arrays
            
        Note:
            Default implementation returns None. Subclasses should override
            this method to provide joint limit information.
        """
        return None, None
    
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get information about the solver.
        
        Returns:
            Dictionary with solver information (name, version, capabilities, etc.)
        """
        return {
            'adapter_name': self.__class__.__name__,
            'urdf_path': self._urdf_path,
            'end_effector_link': self._ee_link,
            'initialized': self._initialized,
            'available': self.is_available()
        }
    
    def cleanup(self) -> None:
        """
        Cleanup resources (GPU memory, etc.).
        
        Default implementation does nothing. Subclasses can override
        to release solver-specific resources.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Utility functions for common operations
def homogeneous_to_quaternion_position(pose_4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 homogeneous transformation to quaternion + position.
    
    Args:
        pose_4x4: 4x4 transformation matrix
        
    Returns:
        Tuple of (quaternion_wxyz, position_xyz)
    """
    from scipy.spatial.transform import Rotation
    
    rotation_matrix = pose_4x4[:3, :3]
    position = pose_4x4[:3, 3]
    
    # Convert rotation matrix to quaternion (w, x, y, z format)
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()  # Returns [x, y, z, w]
    
    # Reorder to [w, x, y, z] format
    quaternion_wxyz = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    
    return quaternion_wxyz, position


def quaternion_position_to_homogeneous(quaternion_wxyz: np.ndarray, 
                                     position_xyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion + position to 4x4 homogeneous transformation.
    
    Args:
        quaternion_wxyz: Quaternion in [w, x, y, z] format
        position_xyz: Position vector [x, y, z]
        
    Returns:
        4x4 transformation matrix
    """
    from scipy.spatial.transform import Rotation
    
    # Convert from [w, x, y, z] to [x, y, z, w] for scipy
    quaternion_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], 
                               quaternion_wxyz[3], quaternion_wxyz[0]])
    
    r = Rotation.from_quat(quaternion_xyzw)
    rotation_matrix = r.as_matrix()
    
    # Create homogeneous transformation matrix
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = rotation_matrix
    pose_4x4[:3, 3] = position_xyz
    
    return pose_4x4


def validate_pose_matrix(pose: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate that a matrix is a proper SE(3) transformation.
    
    Args:
        pose: 4x4 transformation matrix
        tolerance: Numerical tolerance
        
    Returns:
        True if pose is valid, False otherwise
    """
    if pose.shape != (4, 4):
        return False
    
    # Check if bottom row is [0, 0, 0, 1]
    expected_bottom = np.array([0, 0, 0, 1])
    if not np.allclose(pose[3, :], expected_bottom, atol=tolerance):
        return False
    
    # Check if rotation part is orthogonal
    R = pose[:3, :3]
    should_be_identity = R.T @ R
    identity = np.eye(3)
    if not np.allclose(should_be_identity, identity, atol=tolerance):
        return False
    
    # Check if determinant is 1 (proper rotation)
    if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance):
        return False
    
    return True