"""
Pyroki IK Adapter - Integration of Pyroki IK solver with HIROL platform.

This adapter provides a bridge between Pyroki's JAX-based IK solver and the 
HIROL platform's numpy-based benchmark system.
"""

import time
import numpy as np
from typing import Tuple, Optional, Dict, Any
import glog as log

try:
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk
    import yourdfpy
    from io import StringIO
    import xml.etree.ElementTree as ET
    PYROKI_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    PYROKI_DEPENDENCIES_AVAILABLE = False
    _import_error = str(e)

from .base_adapter import IKAdapterBase, homogeneous_to_quaternion_position


class PyrokiAdapter(IKAdapterBase):
    """
    Pyroki IK solver adapter for HIROL platform.
    
    This adapter integrates Pyroki's JAX-based IK solver with the HIROL benchmark
    system, handling model loading, data conversion, and solver optimization.
    """
    
    def __init__(self, urdf_path: str, end_effector_link: str, **kwargs):
        """
        Initialize Pyroki adapter.
        
        Args:
            urdf_path: Path to robot URDF file
            end_effector_link: Name of end effector link
            **kwargs: Additional configuration options:
                - num_seeds_init: Number of initial seeds (default: 64)
                - num_seeds_final: Number of final seeds (default: 4) 
                - total_steps: Total optimization steps (default: 16)
                - init_steps: Initial optimization steps (default: 6)
                - pos_weight: Position error weight (default: 10.0)
                - ori_weight: Orientation error weight (default: 5.0)
                - limit_weight: Joint limit weight (default: 50.0)
                - lambda_initial: Initial damping parameter (default: 10.0)
        """
        super().__init__(urdf_path, end_effector_link, **kwargs)
        
        # Pyroki solver parameters
        self._num_seeds_init = kwargs.get('num_seeds_init', 64)
        self._num_seeds_final = kwargs.get('num_seeds_final', 4)
        self._total_steps = kwargs.get('total_steps', 16)
        self._init_steps = kwargs.get('init_steps', 6)
        self._pos_weight = kwargs.get('pos_weight', 10.0)
        self._ori_weight = kwargs.get('ori_weight', 5.0)
        self._limit_weight = kwargs.get('limit_weight', 50.0)
        self._lambda_initial = kwargs.get('lambda_initial', 10.0)
        
        # Smoothness cost parameters
        self._smoothness_weight = kwargs.get('smoothness_weight', 10.0)
        self._enable_smoothness = kwargs.get('enable_smoothness', False)
        
        # Pyroki-specific state
        self._pyroki_robot = None
        self._target_link_index = None
        self._joint_limits_low = None
        self._joint_limits_high = None
        self._root = None  # For quasi-random seed generation
        
        # JIT compiled functions (initialized later)
        self._jit_solve_single = None
        self._jit_fk = None
        
        if not self.is_available():
            log.warning(f"Pyroki dependencies not available: {_import_error if not PYROKI_DEPENDENCIES_AVAILABLE else 'Unknown'}")
    
    def is_available(self) -> bool:
        """Check if Pyroki is available."""
        return PYROKI_DEPENDENCIES_AVAILABLE
    
    def initialize(self) -> None:
        """Initialize Pyroki robot model and JIT compile functions."""
        if not self.is_available():
            raise RuntimeError("Pyroki dependencies not available")
        
        if self._initialized:
            return
        
        log.info(f"Initializing Pyroki adapter for {self._ee_link}")
        
        try:
            # Load URDF and create Pyroki robot
            self._load_pyroki_robot()
            
            # Set up quasi-random seed generation
            self._setup_seed_generation()
            
            # JIT compile functions for performance
            self._setup_jit_functions()
            
            self._initialized = True
            log.info("Pyroki adapter initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize Pyroki adapter: {e}")
            raise
    
    def _load_pyroki_robot(self) -> None:
        """Load robot from URDF and create Pyroki model."""
        # Load URDF
        urdf = yourdfpy.URDF.load(self._urdf_path)
        
        # Fix prismatic joints to match typical IK setup
        xml_tree = urdf.write_xml()
        for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics"):
                child = joint.find(tag)
                if child is not None:
                    joint.remove(child)
        
        # Reload modified URDF
        xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
        buf = StringIO(xml_str)
        urdf = yourdfpy.URDF.load(buf)
        
        if not urdf.validate():
            raise ValueError("Invalid URDF after modifications")
        
        # Create Pyroki robot
        self._pyroki_robot = pk.Robot.from_urdf(urdf)
        
        # Find target link index
        try:
            self._target_link_index = jnp.array(
                self._pyroki_robot.links.names.index(self._ee_link)
            )
        except ValueError:
            raise ValueError(f"End effector link '{self._ee_link}' not found in robot")
        
        # Get joint limits
        self._joint_limits_low = self._pyroki_robot.joints.lower_limits
        self._joint_limits_high = self._pyroki_robot.joints.upper_limits
        
        log.info(f"Loaded Pyroki robot with {self._pyroki_robot.joints.num_actuated_joints} DOF")
    
    def _setup_seed_generation(self) -> None:
        """Set up quasi-random seed generation using Roberts sequence."""
        def newton_raphson(f, x, iters):
            """Newton-Raphson method for root finding."""
            def update(x, _):
                y = x - f(x) / jax.grad(f)(x)
                return y, None
            x, _ = jax.lax.scan(update, x, length=iters)
            return x
        
        # Calculate root for Roberts sequence
        exp = self._pyroki_robot.joints.num_actuated_joints
        self._root = newton_raphson(
            lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000
        )
    
    def _setup_jit_functions(self) -> None:
        """Set up JIT compiled functions for performance."""
        # Create single solve function
        def solve_single_ik(target_wxyz, target_position, initial_seeds, previous_config):
            """Single IK solve with multiple seeds."""
            robot = self._pyroki_robot
            
            def solve_one(initial_q, lambda_initial, max_iters):
                joint_var = robot.joint_var_cls(0)
                factors = [
                    pk.costs.pose_cost_analytic_jac(
                        robot,
                        joint_var,
                        jaxlie.SE3.from_rotation_and_translation(
                            jaxlie.SO3(target_wxyz), target_position
                        ),
                        self._target_link_index,
                        pos_weight=self._pos_weight,
                        ori_weight=self._ori_weight,
                    ),
                    pk.costs.limit_cost(
                        robot,
                        joint_var,
                        weight=self._limit_weight,
                    ),
                ]
                
                # Add smoothness cost if enabled and previous config is provided
                problem_vars = [joint_var]
                initial_vals = [joint_var.with_value(initial_q)]
                
                if self._enable_smoothness and previous_config is not None:
                    previous_joint_var = robot.joint_var_cls(1)
                    factors.append(
                        pk.costs.smoothness_cost(
                            joint_var,
                            previous_joint_var, 
                            weight=self._smoothness_weight,
                        )
                    )
                    problem_vars.append(previous_joint_var)
                    initial_vals.append(previous_joint_var.with_value(previous_config))
                
                problem = jaxls.LeastSquaresProblem(factors, problem_vars)
                sol, summary = problem.analyze().solve(
                    initial_vals=jaxls.VarValues.make(initial_vals),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
                return sol[joint_var], summary
            
            vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))
            
            # Initial optimization with many seeds
            initial_sols, summary1 = vmapped_solve(
                initial_seeds, 
                jnp.full(initial_seeds.shape[:1], self._lambda_initial),
                self._init_steps
            )
            
            # Get best seeds for final optimization
            best_indices = jnp.argsort(
                summary1.cost_history[jnp.arange(self._num_seeds_init), -1]
            )[:self._num_seeds_final]
            
            # Final optimization with fewer seeds
            final_sols, summary2 = vmapped_solve(
                initial_sols[best_indices],
                summary1.lambda_history[jnp.arange(self._num_seeds_init), -1][best_indices],
                self._total_steps - self._init_steps
            )
            
            # Return best solution
            best_final_idx = jnp.argmin(
                summary2.cost_history[jnp.arange(self._num_seeds_final), summary2.iterations]
            )
            
            return final_sols[best_final_idx], summary2.cost_history[best_final_idx, summary2.iterations[best_final_idx]]
        
        # JIT compile the solve function
        self._jit_solve_single = jax.jit(solve_single_ik)
        
        # Create and JIT compile forward kinematics function
        def forward_kinematics_jax(q):
            return self._pyroki_robot.forward_kinematics(q)[self._target_link_index]
        
        self._jit_fk = jax.jit(forward_kinematics_jax)
        
        log.info("JIT compiled Pyroki functions")
    
    def _generate_roberts_sequence(self, num_points: int, dim: int, root: float) -> jnp.ndarray:
        """Generate quasi-random Roberts sequence for initial seeds."""
        basis = 1 - (1 / root ** (1 + jnp.arange(dim)))
        n = jnp.arange(num_points)
        x = n[:, None] * basis[None, :]
        x, _ = jnp.modf(x)
        return x
    
    def _generate_smooth_seeds(self, initial_guess: np.ndarray) -> jnp.ndarray:
        """Generate seeds with smoothness consideration using initial guess."""
        n_joints = self._pyroki_robot.joints.num_actuated_joints
        
        # Normalize initial_guess to [0,1] range
        initial_normalized = ((initial_guess - self._joint_limits_low) / 
                            (self._joint_limits_high - self._joint_limits_low))
        initial_normalized = jnp.clip(initial_normalized, 0.0, 1.0)
        
        # Strategy 1: Smart seeding - first seed is initial_guess, others around it
        seeds = []
        
        # First seed: exact initial guess
        seeds.append(initial_normalized)
        
        # Generate remaining seeds around initial guess with decreasing locality
        remaining_seeds = self._num_seeds_init - 1
        if remaining_seeds > 0:
            # 50% local around initial_guess, 50% global Roberts sequence
            n_local = remaining_seeds // 2
            n_global = remaining_seeds - n_local
            
            if n_local > 0:
                # Local seeds: Gaussian noise around initial guess
                noise_scale = 0.1  # 10% of joint range
                local_seeds = []
                for _ in range(n_local):
                    noise = jnp.array(np.random.normal(0, noise_scale, n_joints))
                    perturbed = initial_normalized + noise
                    perturbed = jnp.clip(perturbed, 0.0, 1.0)
                    local_seeds.append(perturbed)
                seeds.extend(local_seeds)
            
            if n_global > 0:
                # Global seeds: Roberts sequence for diversity
                global_seeds_norm = self._generate_roberts_sequence(n_global, n_joints, self._root)
                for i in range(n_global):
                    seeds.append(global_seeds_norm[i])
        
        # Convert to JAX array and scale to joint limits
        seeds_array = jnp.array(seeds)
        seeds_scaled = (self._joint_limits_low + seeds_array * 
                       (self._joint_limits_high - self._joint_limits_low))
        
        return seeds_scaled
    
    def solve_single(self, 
                    target_pose: np.ndarray, 
                    initial_guess: np.ndarray,
                    tolerance: float = 1e-6,
                    max_iterations: int = 1000) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Solve single IK problem using Pyroki.
        
        Args:
            target_pose: Target 4x4 transformation matrix
            initial_guess: Initial joint configuration used for seed generation and smoothness cost
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations (not used directly by Pyroki)
            
        Returns:
            Tuple of (converged, solution, solve_time)
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Convert pose to JAX format
            quat_wxyz, position = homogeneous_to_quaternion_position(target_pose)
            target_wxyz_jax = jnp.array(quat_wxyz)
            target_position_jax = jnp.array(position)
            
            # === Original Roberts sequence seed generation (saved for later iteration) ===
            # initial_seeds_normalized = self._generate_roberts_sequence(
            #     self._num_seeds_init, 
            #     self._pyroki_robot.joints.num_actuated_joints,
            #     self._root
            # )
            # initial_seeds = (self._joint_limits_low + initial_seeds_normalized * 
            #                (self._joint_limits_high - self._joint_limits_low))
            
            # === New smoothness-aware seed generation ===
            initial_seeds = self._generate_smooth_seeds(initial_guess)
            
            # Time the solve
            start_time = time.time()
            solution_jax, final_cost = self._jit_solve_single(
                target_wxyz_jax, target_position_jax, initial_seeds, jnp.array(initial_guess)
            )
            jax.block_until_ready(solution_jax)  # Ensure computation is complete
            solve_time = time.time() - start_time
            
            # Convert solution back to numpy
            solution = np.array(solution_jax)
            
            # Check convergence based on actual SE(3) error (consistent with traditional methods)
            if solution is not None:
                # Compute forward kinematics to get achieved pose
                achieved_pose = self.forward_kinematics(solution)
                if achieved_pose is not None:
                    # Compute SE(3) error consistent with traditional methods
                    from scipy.spatial.transform import Rotation
                    
                    # Position error (translation)
                    pos_target = target_pose[:3, 3]
                    pos_achieved = achieved_pose[:3, 3]
                    pos_error = pos_target - pos_achieved
                    
                    # Rotation error (orientation)
                    R_target = target_pose[:3, :3]
                    R_achieved = achieved_pose[:3, :3]
                    R_error = R_target @ R_achieved.T
                    r_error = Rotation.from_matrix(R_error)
                    rot_error = r_error.as_rotvec()
                    
                    # SE(3) error norm (same as traditional methods)
                    se3_error = np.concatenate([pos_error, rot_error])
                    error_norm = np.linalg.norm(se3_error)
                    
                    converged = error_norm < tolerance
                else:
                    # If FK fails
                    converged = False
            else:
                converged = False
            
            return converged, solution, solve_time
            
        except Exception as e:
            log.warning(f"Pyroki solve failed: {e}")
            return False, None, 0.0
    
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics using Pyroki.
        
        Args:
            joint_positions: Joint configuration
            
        Returns:
            4x4 transformation matrix
        """
        if not self._initialized:
            self.initialize()
        
        try:
            q_jax = jnp.array(joint_positions)
            fk_result = self._jit_fk(q_jax)
            
            # Extract quaternion and position
            quat_wxyz = np.array(fk_result[:4])
            position = np.array(fk_result[4:7])
            
            # Convert back to homogeneous matrix
            from .base_adapter import quaternion_position_to_homogeneous
            return quaternion_position_to_homogeneous(quat_wxyz, position)
            
        except Exception as e:
            log.warning(f"Pyroki FK failed: {e}")
            return None
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits from Pyroki robot."""
        if not self._initialized:
            self.initialize()
        
        return (np.array(self._joint_limits_low), 
                np.array(self._joint_limits_high))
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get Pyroki solver information."""
        info = super().get_solver_info()
        info.update({
            'solver_name': 'Pyroki',
            'framework': 'JAX',
            'num_seeds_init': self._num_seeds_init,
            'num_seeds_final': self._num_seeds_final,
            'optimization_steps': self._total_steps,
            'supports_batch': True,
            'supports_gpu': True
        })
        
        if self._initialized:
            info['n_dof'] = self._pyroki_robot.joints.num_actuated_joints
        
        return info
    
    def cleanup(self) -> None:
        """Cleanup JAX resources."""
        # JAX handles memory management automatically
        # But we can clear cached JIT functions if needed
        self._jit_solve_single = None
        self._jit_fk = None
        log.info("Pyroki adapter cleanup completed")