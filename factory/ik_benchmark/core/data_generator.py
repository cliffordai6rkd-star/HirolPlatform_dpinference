"""
Data generator for IK benchmark testing.

Generates various types of test data including random poses, trajectories,
and singularity-near poses for comprehensive IK algorithm evaluation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial.transform import Rotation as R, Slerp
import glog as log
# from motion.pin_model import RobotModel


class TrajectoryGenerator:
    """Static methods for generating different trajectory types."""
    
    @staticmethod
    def generate_line_trajectory(start_pose: np.ndarray, end_pose: np.ndarray, 
                               waypoints: int) -> List[np.ndarray]:
        """
        Generate linear trajectory between two poses.
        
        Args:
            start_pose: 4x4 start transformation matrix
            end_pose: 4x4 end transformation matrix
            waypoints: Number of waypoints along trajectory
            
        Returns:
            List of 4x4 transformation matrices
        """
        trajectory = []
        
        # Extract positions and rotations
        start_pos = start_pose[:3, 3]
        end_pos = end_pose[:3, 3]
        start_rot = R.from_matrix(start_pose[:3, :3])
        end_rot = R.from_matrix(end_pose[:3, :3])
        
        for i in range(waypoints):
            t = i / (waypoints - 1) if waypoints > 1 else 0.0
            
            # Interpolate position
            pos = start_pos + t * (end_pos - start_pos)
            
            # Interpolate rotation using slerp
            if t == 0.0:
                rot = start_rot
            elif t == 1.0:
                rot = end_rot
            else:
                # Use Slerp for scipy compatibility
                key_rots = R.from_matrix(np.stack([start_rot.as_matrix(), end_rot.as_matrix()]))
                key_times = [0, 1]
                slerp = Slerp(key_times, key_rots)
                rot = slerp([t])[0]
            
            # Construct pose matrix
            pose = np.eye(4)
            pose[:3, 3] = pos
            pose[:3, :3] = rot.as_matrix()
            trajectory.append(pose)
            
        return trajectory
    
    @staticmethod
    def generate_circular_trajectory(center: np.ndarray, radius: float,
                                   normal: np.ndarray, waypoints: int,
                                   start_angle: float = 0.0) -> List[np.ndarray]:
        """
        Generate circular trajectory around a center point.
        
        Args:
            center: 3D center point
            radius: Circle radius
            normal: Normal vector to circle plane
            waypoints: Number of waypoints
            start_angle: Starting angle in radians
            
        Returns:
            List of 4x4 transformation matrices
        """
        trajectory = []
        
        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Create two perpendicular vectors in the circle plane
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        
        for i in range(waypoints):
            angle = start_angle + 2 * np.pi * i / waypoints
            
            # Point on circle
            pos = center + radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
            
            # Orientation pointing toward center
            direction = center - pos
            direction = direction / np.linalg.norm(direction)
            
            # Create rotation matrix (Z-axis pointing toward center)
            z_axis = direction
            x_axis = v1  # Keep X consistent
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            pose = np.eye(4)
            pose[:3, 3] = pos
            pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
            trajectory.append(pose)
            
        return trajectory
    
    @staticmethod
    def generate_helix_trajectory(center: np.ndarray, radius: float, 
                                pitch: float, turns: float, waypoints: int) -> List[np.ndarray]:
        """
        Generate helical trajectory.
        
        Args:
            center: 3D center point
            radius: Helix radius
            pitch: Vertical distance per turn
            turns: Number of turns
            waypoints: Number of waypoints
            
        Returns:
            List of 4x4 transformation matrices
        """
        trajectory = []
        
        for i in range(waypoints):
            t = i / (waypoints - 1) if waypoints > 1 else 0.0
            angle = 2 * np.pi * turns * t
            
            # Helix position
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)  
            z = center[2] + pitch * turns * t
            pos = np.array([x, y, z])
            
            # Tangent direction for orientation
            dx = -radius * np.sin(angle)
            dy = radius * np.cos(angle)
            dz = pitch
            tangent = np.array([dx, dy, dz])
            tangent = tangent / np.linalg.norm(tangent)
            
            # Create rotation matrix
            z_axis = tangent
            x_axis = np.array([1, 0, 0])  # Default
            if abs(np.dot(z_axis, x_axis)) > 0.9:
                x_axis = np.array([0, 1, 0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            
            pose = np.eye(4)
            pose[:3, 3] = pos
            pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
            trajectory.append(pose)
            
        return trajectory


class DataGenerator:
    """Main data generator for IK benchmark testing."""
    
    def __init__(self, robot_model, config: Dict[str, Any]):
        """
        Initialize data generator.
        
        Args:
            robot_model: Robot model for FK and joint limits (RobotModel instance)
            config: Configuration dictionary
        """
        self._robot_model = robot_model
        self._config = config
        
        # 动态获取末端执行器链接名称
        self._ee_link = self._get_ee_link_name()
        
        self._workspace_bounds = self._compute_workspace_bounds()
        
        log.info(f"DataGenerator initialized for robot with {robot_model.nq} joints, ee_link: {self._ee_link}")
        log.info(f"Workspace bounds: {self._workspace_bounds}")
    
    def generate_random_poses(self, count: int, 
                            seed: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate reachable poses using random joint configurations + FK.
        
        Args:
            count: Number of poses to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of 4x4 transformation matrices
        """
        if seed is not None:
            np.random.seed(seed)
            
        poses = []
        joint_limits_low = self._robot_model.model.lowerPositionLimit
        joint_limits_high = self._robot_model.model.upperPositionLimit
        
        log.info(f"Generating {count} random poses...")
        
        for i in range(count):
            if count >= 10 and i % (count // 10) == 0 and i > 0:
                log.info(f"Generated {i}/{count} poses")
                
            # Generate random joint configuration
            q_random = np.random.uniform(joint_limits_low, joint_limits_high)
            
            # Compute forward kinematics
            try:
                pose = self._robot_model.get_frame_pose(self._ee_link, q_random, need_update=True)
                poses.append(pose)
            except Exception as e:
                log.warning(f"FK failed for joint config {i}: {e}")
                # Skip this configuration and continue
                continue
        
        log.info(f"Successfully generated {len(poses)} random poses")
        return poses
    
    def _get_ee_link_name(self) -> str:
        """
        Get end effector link name from robot model.
        
        Returns:
            str: End effector link name (e.g., "fr3_ee", "fr3_hand_tcp")
            
        Raises:
            AttributeError: If robot model doesn't have ee_link attribute
            ValueError: If ee_link is list (multi-ee scenario not supported yet)
        """
        if not hasattr(self._robot_model, 'ee_link'):
            raise AttributeError("Robot model must have 'ee_link' attribute")
            
        ee_link = self._robot_model.ee_link
        
        if isinstance(ee_link, list):
            raise ValueError(
                f"Multi-end-effector scenario not supported yet. "
                f"Found ee_link={ee_link}. Please use single end-effector configuration."
            )
            
        if not isinstance(ee_link, str):
            raise ValueError(f"ee_link must be string, got {type(ee_link)}: {ee_link}")
            
        return ee_link
    
    def generate_trajectory_poses(self, trajectory_type: str, 
                                waypoints: int = 50,
                                **kwargs) -> List[np.ndarray]:
        """
        Generate poses along specified trajectory type.
        
        Args:
            trajectory_type: 'line', 'circle', 'helix'
            waypoints: Number of waypoints along trajectory
            **kwargs: Additional parameters for trajectory generation
            
        Returns:
            List of 4x4 transformation matrices
        """
        log.info(f"Generating {waypoints} waypoints for {trajectory_type} trajectory")
        
        if trajectory_type == 'line':
            return self._generate_line_trajectory_poses(waypoints, **kwargs)
        elif trajectory_type == 'circle':
            return self._generate_circular_trajectory_poses(waypoints, **kwargs)
        elif trajectory_type == 'helix':
            return self._generate_helix_trajectory_poses(waypoints, **kwargs)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    def generate_singular_poses(self, count: int,
                              margin: float = 0.01) -> List[np.ndarray]:
        """
        Generate poses near singularities (simplified approach).
        
        This implementation generates poses near workspace boundaries
        as a proxy for singularities. A more sophisticated approach
        would analyze Jacobian condition numbers.
        
        Args:
            count: Number of poses to generate
            margin: Distance margin from workspace boundary
            
        Returns:
            List of 4x4 transformation matrices
        """
        log.info(f"Generating {count} poses near singularities")
        
        poses = []
        joint_limits_low = self._robot_model.model.lowerPositionLimit
        joint_limits_high = self._robot_model.model.upperPositionLimit
        
        # Generate poses near joint limits (common source of singularities)
        for i in range(count):
            # Choose random joint to be near limit
            joint_idx = np.random.randint(0, len(joint_limits_low))
            
            # Random joint configuration
            q = np.random.uniform(joint_limits_low, joint_limits_high)
            
            # Set chosen joint near its limit
            if np.random.random() < 0.5:
                # Near lower limit
                q[joint_idx] = joint_limits_low[joint_idx] + margin * np.random.random()
            else:
                # Near upper limit  
                q[joint_idx] = joint_limits_high[joint_idx] - margin * np.random.random()
            
            # Ensure within limits
            q = np.clip(q, joint_limits_low, joint_limits_high)
            
            try:
                pose = self._robot_model.get_frame_pose(self._ee_link, q, need_update=True)
                poses.append(pose)
            except Exception as e:
                log.warning(f"FK failed for singular config {i}: {e}")
                continue
        
        log.info(f"Successfully generated {len(poses)} singular poses")
        return poses
    
    def generate_workspace_grid(self, resolution: Tuple[int, int, int],
                              orientation_samples: int = 8) -> List[np.ndarray]:
        """
        Generate poses on regular grid within workspace.
        
        Args:
            resolution: Grid resolution in (x, y, z)
            orientation_samples: Number of orientation samples per position
            
        Returns:
            List of 4x4 transformation matrices
        """
        min_bounds, max_bounds = self._workspace_bounds
        
        log.info(f"Generating workspace grid with resolution {resolution}, "
                f"{orientation_samples} orientations per point")
        
        # Create position grid
        x = np.linspace(min_bounds[0], max_bounds[0], resolution[0])
        y = np.linspace(min_bounds[1], max_bounds[1], resolution[1])
        z = np.linspace(min_bounds[2], max_bounds[2], resolution[2])
        
        poses = []
        
        # Set fixed seed for consistent orientation sampling across runs
        np.random.seed(123)  # Different from workspace bounds seed
        
        for xi in x:
            for yi in y:
                for zi in z:
                    position = np.array([xi, yi, zi])
                    
                    # Generate different orientations for this position
                    for ori_idx in range(orientation_samples):
                        # Generate random orientation (now reproducible)
                        rot = R.random()
                        
                        pose = np.eye(4)
                        pose[:3, 3] = position
                        pose[:3, :3] = rot.as_matrix()
                        poses.append(pose)
        
        log.info(f"Generated {len(poses)} grid poses")
        return poses
    
    def _compute_workspace_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate workspace bounds using random sampling.
        
        Returns:
            Tuple of (min_bounds, max_bounds)
        """
        log.info("Computing workspace bounds...")
        
        # Set fixed seed for consistent workspace bounds across runs
        np.random.seed(42)
        
        sample_count = 1000
        positions = []
        
        joint_limits_low = self._robot_model.model.lowerPositionLimit
        joint_limits_high = self._robot_model.model.upperPositionLimit
        
        for _ in range(sample_count):
            q = np.random.uniform(joint_limits_low, joint_limits_high)
            try:
                pose = self._robot_model.get_frame_pose(self._ee_link, q, need_update=True)
                positions.append(pose[:3, 3])
            except Exception:
                continue
        
        if not positions:
            log.warning("Failed to compute workspace bounds, using default")
            return np.array([-1, -1, -1]), np.array([1, 1, 1])
        
        positions = np.array(positions)
        min_bounds = np.min(positions, axis=0) - 0.05  # Add small margin
        max_bounds = np.max(positions, axis=0) + 0.05
        
        return min_bounds, max_bounds
    
    def _generate_line_trajectory_poses(self, waypoints: int, **kwargs) -> List[np.ndarray]:
        """Generate linear trajectory poses."""
        # Use workspace bounds to create default trajectory
        min_bounds, max_bounds = self._workspace_bounds
        
        start_pos = kwargs.get('start_pos', min_bounds + 0.2 * (max_bounds - min_bounds))
        end_pos = kwargs.get('end_pos', max_bounds - 0.2 * (max_bounds - min_bounds))
        
        start_pose = np.eye(4)
        start_pose[:3, 3] = start_pos
        end_pose = np.eye(4) 
        end_pose[:3, 3] = end_pos
        
        return TrajectoryGenerator.generate_line_trajectory(start_pose, end_pose, waypoints)
    
    def _generate_circular_trajectory_poses(self, waypoints: int, **kwargs) -> List[np.ndarray]:
        """Generate circular trajectory poses."""
        min_bounds, max_bounds = self._workspace_bounds
        center = kwargs.get('center', (min_bounds + max_bounds) / 2)
        radius = kwargs.get('radius', 0.2)
        normal = kwargs.get('normal', np.array([0, 0, 1]))
        
        return TrajectoryGenerator.generate_circular_trajectory(center, radius, normal, waypoints)
    
    def _generate_helix_trajectory_poses(self, waypoints: int, **kwargs) -> List[np.ndarray]:
        """Generate helical trajectory poses."""
        min_bounds, max_bounds = self._workspace_bounds
        center = kwargs.get('center', (min_bounds + max_bounds) / 2)
        radius = kwargs.get('radius', 0.15)
        pitch = kwargs.get('pitch', 0.1)
        turns = kwargs.get('turns', 2.0)
        
        return TrajectoryGenerator.generate_helix_trajectory(center, radius, pitch, turns, waypoints)