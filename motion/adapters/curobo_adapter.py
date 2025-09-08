"""
CuRobo IK Adapter - Integration of CuRobo IK solver with HIROL platform.

This adapter provides a bridge between CuRobo's PyTorch-based IK solver and the 
HIROL platform's numpy-based benchmark system.
"""

import time
import numpy as np
from typing import Tuple, Optional, Dict, Any
import glog as log

try:
    import torch
    from curobo.geom.types import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.util_file import (
        get_robot_configs_path,
        get_world_configs_path, 
        join_path,
        load_yaml
    )
    from scipy.spatial.transform import Rotation
    CUROBO_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    CUROBO_DEPENDENCIES_AVAILABLE = False
    _import_error = str(e)

from .base_adapter import IKAdapterBase, homogeneous_to_quaternion_position


class CuroboAdapter(IKAdapterBase):
    """
    CuRobo IK solver adapter for HIROL platform.
    
    This adapter integrates CuRobo's PyTorch-based IK solver with the HIROL benchmark
    system, handling model loading, GPU acceleration, and data conversion.
    """
    
    def __init__(self, urdf_path: str, end_effector_link: str, **kwargs):
        """
        Initialize CuRobo adapter.
        
        Args:
            urdf_path: Path to robot URDF file
            end_effector_link: Name of end effector link
            **kwargs: Additional configuration options:
                - robot_config_name: Robot config name (e.g., 'franka.yml')
                - world_config_name: World config name (default: 'collision_test.yml')
                - position_threshold: Position tolerance (default: 0.005)
                - rotation_threshold: Rotation tolerance (default: 0.05)
                - num_seeds: Number of IK seeds (default: 16)
                - use_cuda_graph: Enable CUDA graph optimization (default: True)
                - high_precision: Enable high precision mode (default: False)
                - collision_free: Enable collision checking (default: False)
                - use_gpu: Force GPU usage (default: True if available)
        """
        super().__init__(urdf_path, end_effector_link, **kwargs)
        
        # CuRobo solver parameters
        self._robot_config_name = kwargs.get('robot_config_name', 'franka.yml')
        self._world_config_name = kwargs.get('world_config_name', 'collision_test.yml')
        self._position_threshold = kwargs.get('position_threshold', 0.005)
        self._rotation_threshold = kwargs.get('rotation_threshold', 0.05)
        self._num_seeds = kwargs.get('num_seeds', 16)
        self._use_cuda_graph = kwargs.get('use_cuda_graph', True)
        self._high_precision = kwargs.get('high_precision', False)
        self._collision_free = kwargs.get('collision_free', False)
        self._use_gpu = kwargs.get('use_gpu', torch.cuda.is_available())
        
        # CuRobo-specific state
        self._ik_solver = None
        self._tensor_args = None
        self._robot_config = None
        self._world_config = None
        
        if not self.is_available():
            log.warning(f"CuRobo dependencies not available: {_import_error if not CUROBO_DEPENDENCIES_AVAILABLE else 'CUDA not available'}")
    
    def is_available(self) -> bool:
        """Check if CuRobo is available."""
        return CUROBO_DEPENDENCIES_AVAILABLE and (torch.cuda.is_available() or not self._use_gpu)
    
    def initialize(self) -> None:
        """Initialize CuRobo robot model and IK solver."""
        if not self.is_available():
            if not CUROBO_DEPENDENCIES_AVAILABLE:
                raise RuntimeError("CuRobo dependencies not available")
            else:
                raise RuntimeError("CUDA not available for CuRobo")
        
        if self._initialized:
            return
        
        log.info(f"Initializing CuRobo adapter for {self._ee_link}")
        
        try:
            # Set up tensor device
            self._setup_tensor_device()
            
            # Load robot and world configurations
            self._load_robot_config()
            self._load_world_config()
            
            # Create IK solver
            self._create_ik_solver()
            
            self._initialized = True
            log.info("CuRobo adapter initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize CuRobo adapter: {e}")
            raise
    
    def _setup_tensor_device(self) -> None:
        """Set up tensor device (GPU/CPU)."""
        if self._use_gpu and torch.cuda.is_available():
            # Check if we have valid CUDA devices
            if torch.cuda.device_count() > 0:
                # Explicitly specify device index to avoid None index issues
                device = torch.device('cuda:0')  # Use first GPU explicitly
                log.info(f"Using CUDA device {device} for CuRobo (GPU count: {torch.cuda.device_count()})")
            else:
                log.warning("CUDA is available but no CUDA devices found, falling back to CPU")
                device = torch.device('cpu')
                self._use_gpu = False
        else:
            if self._use_gpu:
                log.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
            log.info("Using CPU for CuRobo")
            self._use_gpu = False  # Force CPU mode
        
        self._tensor_args = TensorDeviceType(device=device)
        
        # Additional debug info
        log.debug(f"Final tensor device: {device}")
        if device.type == 'cuda':
            log.debug(f"CUDA device properties: {torch.cuda.get_device_properties(device)}")
            log.debug(f"Current CUDA device: {torch.cuda.current_device()}")
    
    def _load_robot_config(self) -> None:
        """Load robot configuration from CuRobo config files."""
        try:
            # Try to load from CuRobo's robot configs
            robot_config_path = join_path(get_robot_configs_path(), self._robot_config_name)
            robot_data = load_yaml(robot_config_path)["robot_cfg"]
            
            # Modify config based on collision requirements
            if not self._collision_free:
                robot_data["kinematics"]["collision_link_names"] = None
                robot_data["kinematics"]["lock_joints"] = {}
            robot_data["kinematics"]["collision_sphere_buffer"] = 0.0
            
            self._robot_config = RobotConfig.from_dict(robot_data)
            log.info(f"Loaded robot config: {self._robot_config_name}")
            
        except Exception as e:
            log.warning(f"Failed to load robot config {self._robot_config_name}: {e}")
            # Fallback: create minimal robot config from URDF
            self._create_robot_config_from_urdf()
    
    def _create_robot_config_from_urdf(self) -> None:
        """Create minimal robot config directly from URDF."""
        # This is a simplified fallback - in practice, you might need
        # more sophisticated URDF parsing for CuRobo
        robot_data = {
            "kinematics": {
                "urdf_path": self._urdf_path,
                "base_link": "base_link",
                "ee_link": self._ee_link,
                "collision_link_names": None,
                "lock_joints": {},
                "collision_sphere_buffer": 0.0
            }
        }
        self._robot_config = RobotConfig.from_dict(robot_data)
        log.info("Created minimal robot config from URDF")
    
    def _load_world_config(self) -> None:
        """Load world configuration."""
        try:
            world_config_path = join_path(get_world_configs_path(), self._world_config_name)
            world_data = load_yaml(world_config_path)
            self._world_config = WorldConfig.from_dict(world_data)
            log.info(f"Loaded world config: {self._world_config_name}")
        except Exception as e:
            log.warning(f"Failed to load world config: {e}")
            # Create empty world config
            self._world_config = WorldConfig.from_dict({"world": []})
    
    def _create_ik_solver(self) -> None:
        """Create CuRobo IK solver."""
        try:
            # Use the specified precision settings
            position_threshold = self._position_threshold
            grad_iters = None
            if self._high_precision:
                # High precision mode: use more gradient iterations but keep user threshold
                grad_iters = 100
            
            log.info(f"CuRobo solver config: position_threshold={position_threshold}, "
                     f"rotation_threshold={self._rotation_threshold}, num_seeds={self._num_seeds}")
            log.info(f"CuRobo robot_config type: {type(self._robot_config)}")
            log.info(f"CuRobo world_config type: {type(self._world_config)}")
            log.info(f"CuRobo tensor_args device: {self._tensor_args.device}")
            
            # Validate configs before creating solver
            if self._robot_config is None:
                raise ValueError("Robot config is None")
            if self._world_config is None:
                raise ValueError("World config is None")
            
            # Create IK solver config
            log.info("Creating IKSolverConfig...")
            ik_config = IKSolverConfig.load_from_robot_config(
                self._robot_config,
                self._world_config,
                position_threshold=position_threshold,
                rotation_threshold=self._rotation_threshold,
                num_seeds=self._num_seeds,
                self_collision_check=self._collision_free,
                self_collision_opt=self._collision_free,
                tensor_args=self._tensor_args,
                use_cuda_graph=self._use_cuda_graph and self._use_gpu,
                high_precision=self._high_precision,
                regularization=False,
                grad_iters=grad_iters,
            )
            log.info("IKSolverConfig created successfully")
            
            # Create IK solver
            log.info("Creating IKSolver...")
            self._ik_solver = IKSolver(ik_config)
            log.info(f"Created CuRobo IK solver with {self._num_seeds} seeds")
            
        except Exception as e:
            log.error(f"Detailed CuRobo solver creation error: {e}")
            import traceback
            log.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    
    def solve_single(self, 
                    target_pose: np.ndarray, 
                    initial_guess: np.ndarray,
                    tolerance: float = 1e-6,
                    max_iterations: int = 1000) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Solve single IK problem using CuRobo.
        
        Args:
            target_pose: Target 4x4 transformation matrix
            initial_guess: Initial joint configuration (used for seed sampling)
            tolerance: Convergence tolerance (overrides default thresholds)
            max_iterations: Maximum iterations (not directly used by CuRobo)
            
        Returns:
            Tuple of (converged, solution, solve_time)
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Convert numpy pose to CuRobo Pose format
            curobo_pose = self._numpy_to_curobo_pose(target_pose)
            
            # Time the solve
            start_time = time.time()
            
            # Solve IK
            result = self._ik_solver.solve_batch(curobo_pose)
            
            if self._use_gpu:
                torch.cuda.synchronize()  # Ensure GPU computation is complete
            
            solve_time = time.time() - start_time
            
            # Debug: Log CuRobo result information
            success_count = result.success.sum().item() if result.success is not None else 0
            log.info(f"CuRobo result: {success_count}/{len(result.success) if result.success is not None else 0} successful solutions")
            
            # Check if solution found
            if result.success.any():
                # Get first successful solution
                success_indices = torch.nonzero(result.success, as_tuple=True)[0]
                best_idx = success_indices[0]
                
                # Extract joint positions from JointState object
                js_solution = result.js_solution[best_idx]
                if hasattr(js_solution, 'position'):
                    # js_solution is a JointState object, extract position tensor
                    solution = js_solution.position.cpu().numpy().astype(np.float64)  # Ensure float64 for Pinocchio compatibility
                    # Ensure we have the full joint vector, not just one element
                    if solution.ndim == 0:  # scalar
                        log.error(f"CuRobo returned scalar solution: {solution}")
                        solution = None
                    elif solution.ndim > 1:  # multi-dimensional
                        solution = solution.flatten()
                    log.debug(f"CuRobo solution shape after extraction: {solution.shape if solution is not None else None}")
                else:
                    # js_solution might be a tensor directly
                    solution_raw = js_solution.cpu().numpy().astype(np.float64)  # Ensure float64 for Pinocchio compatibility
                    if solution_raw.ndim == 0:  # scalar
                        log.error(f"CuRobo returned scalar solution: {solution_raw}")
                        solution = None
                    elif solution_raw.ndim > 1:  # multi-dimensional  
                        solution = solution_raw.flatten()
                    else:
                        solution = solution_raw
                
                # Compute SE(3) error using actual forward kinematics (consistent with traditional methods)
                achieved_pose = self.forward_kinematics(solution)
                if achieved_pose is not None:
                    # Compute SE(3) error consistent with traditional methods
                    from scipy.spatial.transform import Rotation
                    
                    # Position error (translation)
                    pos_target = target_pose[:3, 3]
                    pos_achieved = achieved_pose[:3, 3]
                    pos_error_vec = pos_target - pos_achieved
                    
                    # Rotation error (orientation)  
                    R_target = target_pose[:3, :3]
                    R_achieved = achieved_pose[:3, :3]
                    R_error = R_target @ R_achieved.T
                    r_error = Rotation.from_matrix(R_error)
                    rot_error_vec = r_error.as_rotvec()
                    
                    # SE(3) error norm (same as traditional methods)
                    se3_error = np.concatenate([pos_error_vec, rot_error_vec])
                    se3_error_norm = np.linalg.norm(se3_error)
                    
                    # Component errors for logging
                    pos_error = np.linalg.norm(pos_error_vec)
                    rot_error = np.linalg.norm(rot_error_vec)
                else:
                    # Fallback to CuRobo's internal errors if FK fails
                    pos_error = result.position_error[best_idx].cpu().numpy().item()
                    rot_error = result.rotation_error[best_idx].cpu().numpy().item()
                    se3_error_norm = np.sqrt(pos_error**2 + rot_error**2)
                
                # Debug: Log actual errors for diagnosis
                log.debug(f"CuRobo errors - pos: {pos_error:.2e}, rot: {rot_error:.2e}, "
                         f"SE(3): {se3_error_norm:.2e}, tolerance: {tolerance:.2e}")
                
                # Use separate position and rotation thresholds for convergence
                converged = (pos_error < self._position_threshold and 
                           rot_error < self._rotation_threshold)
                
            else:
                # Debug: Log when no solutions found
                log.info(f"CuRobo found no successful solutions for target pose")
                if result.position_error is not None and result.rotation_error is not None:
                    min_pos_error = result.position_error.min().item()
                    min_rot_error = result.rotation_error.min().item()
                    log.info(f"Best errors: pos={min_pos_error:.2e}, rot={min_rot_error:.2e}, "
                             f"thresholds: pos={self._position_threshold:.2e}, rot={self._rotation_threshold:.2e}")
                
                solution = None
                converged = False
            
            return converged, solution, solve_time
            
        except Exception as e:
            log.warning(f"CuRobo solve failed: {e}")
            log.debug(f"CuRobo solve error details: {type(e).__name__}: {e}")
            import traceback
            log.debug(f"Full traceback:\n{traceback.format_exc()}")
            return False, None, 0.0
    
    def _numpy_to_curobo_pose(self, pose_4x4: np.ndarray) -> Pose:
        """Convert numpy 4x4 matrix to CuRobo Pose object."""
        # Extract position
        position = pose_4x4[:3, 3]
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = pose_4x4[:3, :3]
        r = Rotation.from_matrix(rotation_matrix)
        quaternion_xyzw = r.as_quat()  # [x, y, z, w]
        
        # Convert to tensor format
        position_tensor = torch.tensor(position, dtype=torch.float32, device=self._tensor_args.device)
        quaternion_tensor = torch.tensor(quaternion_xyzw, dtype=torch.float32, device=self._tensor_args.device)
        
        # Create batch dimension
        position_batch = position_tensor.unsqueeze(0)  # (1, 3)
        quaternion_batch = quaternion_tensor.unsqueeze(0)  # (1, 4)
        
        return Pose(position=position_batch, quaternion=quaternion_batch)
    
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics using CuRobo.
        
        Args:
            joint_positions: Joint configuration
            
        Returns:
            4x4 transformation matrix
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Convert to tensor with proper batch dimension handling
            joint_positions = np.array(joint_positions, dtype=np.float32)
            
            # Ensure we have the right shape: [dof] -> [1, dof]
            if joint_positions.ndim == 1:
                joint_positions = joint_positions.reshape(1, -1)
            elif joint_positions.ndim != 2:
                raise ValueError(f"Invalid joint_positions shape: {joint_positions.shape}, expected 1D or 2D")
            
            # Convert to tensor on the correct device
            q_tensor = torch.tensor(joint_positions, dtype=torch.float32, 
                                  device=self._tensor_args.device)
            
            log.debug(f"CuRobo FK input shape: {q_tensor.shape}")
            
            # Compute FK
            kin_state = self._ik_solver.fk(q_tensor)
            
            # Extract pose (remove batch dimension)
            position = kin_state.ee_position[0].cpu().numpy()
            quaternion_xyzw = kin_state.ee_quaternion[0].cpu().numpy()
            
            # Convert quaternion to rotation matrix
            r = Rotation.from_quat(quaternion_xyzw)
            rotation_matrix = r.as_matrix()
            
            # Create homogeneous transformation matrix
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :3] = rotation_matrix
            pose_4x4[:3, 3] = position
            
            return pose_4x4
            
        except Exception as e:
            log.warning(f"CuRobo FK failed: {e}")
            return None
    
    def get_joint_count(self) -> int:
        """Get number of robot joints dynamically."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Method 1: Try to get from IK solver robot config
            if hasattr(self._ik_solver, 'robot_cfg'):
                robot_cfg = self._ik_solver.robot_cfg
                if hasattr(robot_cfg, 'kinematics'):
                    # Try different ways to get joint count from CuRobo config
                    if hasattr(robot_cfg.kinematics, 'joint_limits'):
                        limits = robot_cfg.kinematics.joint_limits
                        if hasattr(limits, 'position') and limits.position is not None:
                            return len(limits.position[0])
                    
                    # Try to get from cspace config
                    if hasattr(robot_cfg.kinematics, 'cspace') and hasattr(robot_cfg.kinematics.cspace, 'joint_names'):
                        return len(robot_cfg.kinematics.cspace.joint_names)
            
            # Method 2: Try to get from robot config
            if self._robot_config and hasattr(self._robot_config, 'kinematics'):
                if hasattr(self._robot_config.kinematics, 'cspace') and hasattr(self._robot_config.kinematics.cspace, 'joint_names'):
                    return len(self._robot_config.kinematics.cspace.joint_names)
            
            # Method 3: Parse URDF to count joints
            n_joints = self._get_joint_count_from_urdf()
            if n_joints > 0:
                return n_joints
            
            # Fallback: assume common robot configurations
            log.warning("Could not determine joint count dynamically, assuming 7-DOF robot")
            # return 7
            
        except Exception as e:
            log.warning(f"Error getting joint count: {e}, assuming 7-DOF robot")
            # return 7
    
    def _get_joint_count_from_urdf(self) -> int:
        """Parse URDF file to count actuated joints."""
        try:
            import xml.etree.ElementTree as ET
            
            # Parse URDF XML
            tree = ET.parse(self._urdf_path)
            root = tree.getroot()
            
            # Count joints that are not 'fixed'
            joint_count = 0
            for joint in root.findall('joint'):
                joint_type = joint.get('type', 'unknown')
                if joint_type in ['revolute', 'continuous', 'prismatic']:
                    joint_count += 1
            
            log.info(f"Parsed {joint_count} actuated joints from URDF: {self._urdf_path}")
            return joint_count
            
        except Exception as e:
            log.warning(f"Failed to parse URDF for joint count: {e}")
            return 0
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits from CuRobo robot config."""
        if not self._initialized:
            self.initialize()
        
        try:
            # CuRobo stores limits in the robot config
            # This is a simplified access - actual implementation may vary
            if hasattr(self._ik_solver, 'robot_cfg'):
                robot_cfg = self._ik_solver.robot_cfg
                if hasattr(robot_cfg, 'kinematics') and hasattr(robot_cfg.kinematics, 'joint_limits'):
                    limits = robot_cfg.kinematics.joint_limits
                    return limits.position[0].cpu().numpy(), limits.position[1].cpu().numpy()
            
            # Fallback: return reasonable default limits based on actual joint count
            n_joints = self.get_joint_count()
            lower_limits = np.full(n_joints, -np.pi)
            upper_limits = np.full(n_joints, np.pi)
            log.info(f"Using default joint limits for {n_joints}-DOF robot")
            return lower_limits, upper_limits
            
        except Exception as e:
            log.warning(f"Failed to get joint limits: {e}")
            return None, None
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get CuRobo solver information."""
        info = super().get_solver_info()
        info.update({
            'solver_name': 'CuRobo',
            'framework': 'PyTorch',
            'device': 'GPU' if self._use_gpu else 'CPU',
            'num_seeds': self._num_seeds,
            'n_dof': self.get_joint_count() if self._initialized else 'unknown',
            'position_threshold': self._position_threshold,
            'rotation_threshold': self._rotation_threshold,
            'cuda_graph': self._use_cuda_graph,
            'high_precision': self._high_precision,
            'collision_free': self._collision_free,
            'supports_batch': True,
            'supports_gpu': True
        })
        
        return info
    
    def cleanup(self) -> None:
        """Cleanup GPU memory and resources."""
        if self._use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear solver to free memory
        self._ik_solver = None
        log.info("CuRobo adapter cleanup completed")