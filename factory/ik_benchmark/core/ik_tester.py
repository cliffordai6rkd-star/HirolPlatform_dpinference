"""
IK Tester - Unified interface for testing IK algorithms.

Provides consistent testing interface for different IK methods and
integrates with the benchmark evaluation metrics.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import glog as log

from factory.components.robot_factory import RobotFactory
# from motion.pin_model import RobotModel
from hardware.base.utils import convert_homo_2_7D_pose, convert_7D_2_homo, RobotJointState
from controller.controller_base import IKController

from ..metrics.accuracy_metric import AccuracyMetric, AccuracyResult
from ..metrics.solvability_metric import SolvabilityMetric, SolvabilityResult
from ..metrics.efficiency_metric import EfficiencyMetric, EfficiencyResult
from ..metrics.robustness_metric import RobustnessMetric, RobustnessResult


@dataclass
class IKTestResult:
    """Results from a single IK test."""
    target_pose: np.ndarray
    converged: bool
    solution: Optional[np.ndarray]
    achieved_pose: Optional[np.ndarray]
    solve_time: float
    iterations: Optional[int]
    error_message: Optional[str]


class IKTester:
    """Unified tester for IK algorithms with comprehensive evaluation."""
    
    def __init__(self, robot_factory: RobotFactory, config: Dict[str, Any]):
        """
        Initialize IK tester.
        
        Args:
            robot_factory: Robot factory instance
            config: Configuration dictionary
        """
        self._robot_factory = robot_factory
        self._config = config
        # Create robot model if needed
        if hasattr(robot_factory, '_robot_model'):
            self._robot_model = robot_factory._robot_model
        else:
            # Create robot model using factory config (same as MotionFactory approach)
            from motion.pin_model import RobotModel
            factory_config = robot_factory._config
            model_config = factory_config["model_config"]
            model_name = model_config["name"]
            model_cfg = model_config["cfg"]
            self._robot_model = RobotModel(model_cfg[model_name])
        
        # 动态获取末端执行器链接名称
        factory_config = robot_factory._config
        model_config = factory_config["model_config"]
        model_name = model_config["name"]
        model_cfg = model_config["cfg"]
        self._ee_link = model_cfg[model_name]["ee_link"]
        
        # Initialize evaluation metrics
        self._accuracy_metric = AccuracyMetric(
            position_tolerance=config.get('position_tolerance', 1e-6),
            rotation_tolerance=config.get('rotation_tolerance', 1e-6)
        )
        self._solvability_metric = SolvabilityMetric()
        self._efficiency_metric = EfficiencyMetric(
            timeout_threshold=config.get('timeout_threshold', 1.0)
        )
        self._robustness_metric = RobustnessMetric(
            dt=config.get('trajectory_dt', 0.01)
        )
        
        # Method-specific parameters for optimal performance
        self._method_specific_params = config.get('method_specific_params', {})
        
        # Cache for adapter-based methods to avoid repeated initialization
        self._adapter_cache: Dict[str, Any] = {}
        
        log.info(f"IKTester initialized for robot: {config.get('robot_type', 'unknown')}")
        if self._method_specific_params:
            log.info(f"Method-specific parameters configured for: {list(self._method_specific_params.keys())}")
    
    def test_accuracy(self, ik_method: str, poses: List[np.ndarray], 
                     tolerance: float = 1e-6, **kwargs) -> AccuracyResult:
        """
        Test accuracy of IK method on given poses.
        
        Args:
            ik_method: IK method name ('gaussian_newton', 'dls', 'lm')
            poses: List of target 4x4 transformation matrices
            tolerance: Convergence tolerance
            **kwargs: Additional IK parameters
            
        Returns:
            AccuracyResult with accuracy statistics
        """
        log.info(f"Testing accuracy of {ik_method} on {len(poses)} poses")
        
        test_results = self._run_ik_tests(ik_method, poses, tolerance, **kwargs)
        
        # Extract results for accuracy evaluation
        solutions = [r.solution for r in test_results]
        achieved_poses = [r.achieved_pose for r in test_results]
        converged_flags = [r.converged for r in test_results]
        
        # Filter out None values
        valid_indices = [i for i, (sol, ach) in enumerate(zip(solutions, achieved_poses)) 
                        if sol is not None and ach is not None]
        
        filtered_solutions = [solutions[i] for i in valid_indices]
        filtered_achieved = [achieved_poses[i] for i in valid_indices] 
        filtered_poses = [poses[i] for i in valid_indices]
        filtered_converged = [converged_flags[i] for i in valid_indices]
        
        return self._accuracy_metric.evaluate(
            filtered_solutions, filtered_poses, filtered_achieved, filtered_converged
        )
    
    def test_solvability(self, ik_method: str, poses: List[np.ndarray],
                        tolerance: float = 1e-6, **kwargs) -> SolvabilityResult:
        """
        Test solvability of IK method.
        
        Args:
            ik_method: IK method name
            poses: List of target poses
            tolerance: Convergence tolerance
            **kwargs: Additional IK parameters
            
        Returns:
            SolvabilityResult with solvability metrics
        """
        log.info(f"Testing solvability of {ik_method} on {len(poses)} poses")
        
        test_results = self._run_ik_tests(ik_method, poses, tolerance, **kwargs)
        
        # Extract results for solvability evaluation
        ik_results = [(r.converged, r.solution) for r in test_results]
        
        return self._solvability_metric.evaluate(ik_results, poses)
    
    def test_efficiency(self, ik_method: str, poses: List[np.ndarray],
                       tolerance: float = 1e-6, **kwargs) -> EfficiencyResult:
        """
        Test efficiency of IK method.
        
        Args:
            ik_method: IK method name
            poses: List of target poses
            tolerance: Convergence tolerance
            **kwargs: Additional IK parameters
            
        Returns:
            EfficiencyResult with timing and iteration statistics
        """
        log.info(f"Testing efficiency of {ik_method} on {len(poses)} poses")
        
        test_results = self._run_ik_tests(ik_method, poses, tolerance, **kwargs)
        
        # Extract timing and iteration data
        solve_times = [r.solve_time for r in test_results]
        iteration_counts = [r.iterations for r in test_results if r.iterations is not None]
        converged_flags = [r.converged for r in test_results]
        
        return self._efficiency_metric.evaluate(solve_times, iteration_counts, converged_flags)
    
    def test_robustness(self, ik_method: str, poses: List[np.ndarray], 
                       noise_levels: List[float] = [0.001, 0.005, 0.01, 0.05],
                       tolerance: float = 1e-6, **kwargs) -> RobustnessResult:
        """
        Test robustness of IK method.
        
        Args:
            ik_method: IK method name
            poses: List of target poses
            noise_levels: Noise levels for testing
            tolerance: Convergence tolerance
            **kwargs: Additional IK parameters
            
        Returns:
            RobustnessResult with robustness metrics
        """
        log.info(f"Testing robustness of {ik_method} on {len(poses)} poses")
        
        # Create IK solver function for robustness testing
        def ik_solver_func(pose: np.ndarray, seed: np.ndarray) -> Tuple[bool, np.ndarray]:
            # Remove 'initial_guess' from kwargs to avoid duplicate parameter error
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'initial_guess'}
            converged, solution, _ = self._solve_single_ik(ik_method, pose, tolerance, seed, **filtered_kwargs)
            return converged, solution
        
        return self._robustness_metric.evaluate(
            ik_solver_func, poses, self._robot_model, noise_levels
        )
    
    def _run_ik_tests(self, ik_method: str, poses: List[np.ndarray],
                     tolerance: float, **kwargs) -> List[IKTestResult]:
        """
        Run IK tests on list of poses.
        
        Args:
            ik_method: IK method name
            poses: List of target poses
            tolerance: Convergence tolerance
            **kwargs: Additional parameters
            
        Returns:
            List of IKTestResult objects
        """
        results = []
        
        # Default seed (middle of joint range)
        joint_limits_low = self._robot_model.model.lowerPositionLimit
        joint_limits_high = self._robot_model.model.upperPositionLimit
        default_seed = (joint_limits_low + joint_limits_high) / 2
        
        for i, pose in enumerate(poses):
            if len(poses) >= 10 and i % (len(poses) // 10) == 0 and i > 0:
                log.info(f"Processed {i}/{len(poses)} poses")
            
            initial_guess = kwargs.get('initial_guess', default_seed)
            
            # Remove 'initial_guess' from kwargs to avoid duplicate parameter error
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'initial_guess'}
            
            try:
                converged, solution, solve_time = self._solve_single_ik(ik_method, pose, tolerance, initial_guess, **filtered_kwargs)
                
                # Compute achieved pose if converged
                achieved_pose = None
                if converged and solution is not None:
                    try:
                        achieved_pose = self._robot_model.get_frame_pose(self._ee_link, solution, need_update=True)
                    except Exception as e:
                        log.warning(f"FK failed for solution {i}: {e}")
                
                result = IKTestResult(
                    target_pose=pose,
                    converged=converged,
                    solution=solution,
                    achieved_pose=achieved_pose,
                    solve_time=solve_time,
                    iterations=kwargs.get('max_iterations'),  # Could be tracked if available
                    error_message=None
                )
                
            except Exception as e:
                log.warning(f"IK test {i} failed: {e}")
                result = IKTestResult(
                    target_pose=pose,
                    converged=False,
                    solution=None,
                    achieved_pose=None,
                    solve_time=0.0,  # Exception in main loop, no solve_time available
                    iterations=None,
                    error_message=str(e)
                )
            
            results.append(result)
        
        log.info(f"Completed {len(results)} IK tests")
        return results
    
    def _solve_single_ik(self, ik_method: str, target_pose: np.ndarray,
                        tolerance: float, seed: np.ndarray, 
                        **kwargs) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Solve single IK problem.
        
        Args:
            ik_method: IK method name
            target_pose: 4x4 target transformation matrix
            tolerance: Convergence tolerance
            seed: Initial joint configuration
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (converged, solution, solve_time)
        """
        # Handle adapter-based methods (pyroki, curobo)
        if ik_method in ['pyroki', 'curobo']:
            return self._solve_with_adapter(ik_method, target_pose, tolerance, seed, **kwargs)
        
        # Handle traditional methods using IKController
        return self._solve_with_controller(ik_method, target_pose, tolerance, seed, **kwargs)
    
    def _solve_with_adapter(self, ik_method: str, target_pose: np.ndarray,
                           tolerance: float, seed: np.ndarray, 
                           **kwargs) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Solve IK using adapter-based methods (pyroki, curobo).
        
        Args:
            ik_method: IK method name ('pyroki' or 'curobo')
            target_pose: 4x4 target transformation matrix
            tolerance: Convergence tolerance
            seed: Initial joint configuration
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (converged, solution, solve_time)
        """
        try:
            # Get or create cached adapter
            adapter = self._get_cached_adapter(ik_method, **kwargs)
            
            # Get method-specific parameters for max_iterations
            method_params = self._method_specific_params.get(ik_method, {})
            
            # Solve using adapter
            max_iterations = method_params.get('max_iterations', kwargs.get('max_iterations', 1000))
            converged, solution, solve_time = adapter.solve_single(
                target_pose, 
                seed,
                tolerance=tolerance,
                max_iterations=max_iterations
            )
            
            return converged, solution, solve_time
            
        except ImportError as e:
            log.error(f"Adapter for {ik_method} not available: {e}")
            return False, None, 0.0
        except Exception as e:
            log.warning(f"Adapter-based IK solver ({ik_method}) failed: {e}")
            return False, None, 0.0
    
    def _get_cached_adapter(self, ik_method: str, **kwargs):
        """
        Get cached adapter or create new one if not exists.
        
        Args:
            ik_method: IK method name ('pyroki' or 'curobo')
            **kwargs: Additional parameters (ignored for caching)
            
        Returns:
            Cached adapter instance
        """
        # Check if adapter already cached
        if ik_method in self._adapter_cache:
            log.debug(f"Using cached {ik_method} adapter")
            return self._adapter_cache[ik_method]
        
        # Create new adapter
        log.info(f"Creating new {ik_method} adapter (first time)")
        
        from motion.adapters import create_adapter
        
        # Get method-specific parameters
        method_params = self._method_specific_params.get(ik_method, {})
        
        # Get robot model information
        factory_config = self._robot_factory._config
        model_config = factory_config["model_config"]
        model_name = model_config["name"]
        model_cfg = model_config["cfg"]
        
        # Debug: Log configuration structure
        log.debug(f"Getting URDF path for {ik_method}")
        log.debug(f"model_name: {model_name}")
        log.debug(f"model_cfg keys: {list(model_cfg.keys()) if model_cfg else 'None'}")
        
        # Safe URDF path extraction
        if model_name not in model_cfg:
            raise ValueError(f"Model '{model_name}' not found in model config. Available: {list(model_cfg.keys())}")
        
        model_data = model_cfg[model_name]
        if not isinstance(model_data, dict):
            raise ValueError(f"Model config for '{model_name}' is not a dict: {type(model_data)}")
        
        urdf_path = model_data.get("urdf_path", None)
        if not urdf_path:
            raise ValueError(f"No urdf_path found in model config for '{model_name}'. Available keys: {list(model_data.keys())}")
        
        log.debug(f"Found URDF path: {urdf_path}")
        
        # Create adapter with configuration
        adapter_config = {
            **method_params,
            # Note: Don't include runtime kwargs in config to allow caching
        }
        
        # Create adapter with additional debug logging
        log.debug(f"Creating {ik_method} adapter with urdf_path: {urdf_path}, ee_link: {self._ee_link}")
        log.debug(f"Adapter config keys: {list(adapter_config.keys())}")
        
        try:
            adapter = create_adapter(ik_method, urdf_path, self._ee_link, **adapter_config)
            log.debug(f"{ik_method} adapter created successfully")
            
            # Cache the adapter for reuse
            self._adapter_cache[ik_method] = adapter
            log.info(f"Cached {ik_method} adapter for future use")
            
        except Exception as e:
            log.error(f"Failed to create {ik_method} adapter: {e}")
            import traceback
            log.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        
        return adapter
    
    def _solve_with_controller(self, ik_method: str, target_pose: np.ndarray,
                              tolerance: float, seed: np.ndarray, 
                              **kwargs) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Solve IK using traditional IKController methods.
        
        Args:
            ik_method: IK method name
            target_pose: 4x4 target transformation matrix
            tolerance: Convergence tolerance
            seed: Initial joint configuration
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (converged, solution, solve_time)
        """
        # Get method-specific parameters if available
        method_params = self._method_specific_params.get(ik_method, {})
        
        # Create IK controller configuration with method-specific overrides
        ik_config = {
            'ik_type': ik_method,
            'tolerance': tolerance,
            'max_iteration': method_params.get('max_iterations', kwargs.get('max_iterations', 1000)),
            'damping_weight': method_params.get('damping_weight', kwargs.get('damping_weight', 0.3))
        }
        
        # Log method-specific parameter usage for debugging
        if method_params:
            log.debug(f"Using method-specific params for {ik_method}: {method_params}")
        
        # Create IK controller
        ik_controller = IKController(ik_config, self._robot_model)
        
        # Convert pose to required format
        pose_7d = convert_homo_2_7D_pose(target_pose)
        
        # Create target format for controller
        # Use pre-initialized ee_link for consistency
        target = [{self._ee_link: pose_7d}]
        
        # Create robot state with seed
        robot_state = RobotJointState()
        robot_state._positions = seed.copy()
        robot_state._velocities = np.zeros_like(seed)
        robot_state._accelerations = np.zeros_like(seed)
        
        try:
            # Time only the actual IK computation
            start_time = time.time()
            success, joint_target, mode = ik_controller.compute_controller(target, robot_state)
            solve_time = time.time() - start_time
            
            return success, joint_target if success else None, solve_time
            
        except Exception as e:
            log.warning(f"IK solver failed: {e}")
            return False, None, 0.0
    
    def get_available_ik_methods(self) -> List[str]:
        """
        Get list of available IK methods.
        
        Returns:
            List of IK method names
        """
        available_methods = ['gaussian_newton', 'dls', 'lm', 'pink']
        
        # Check for optional adapters
        try:
            from motion.adapters import PYROKI_AVAILABLE, CUROBO_AVAILABLE
            if PYROKI_AVAILABLE:
                available_methods.append('pyroki')
            if CUROBO_AVAILABLE:
                available_methods.append('curobo')
        except ImportError:
            # Adapters not available, continue with standard methods
            pass
        
        return available_methods
    
    def cleanup_adapters(self):
        """
        Clean up cached adapters to free memory.
        Call this when benchmark testing is complete.
        """
        if not self._adapter_cache:
            return
        
        log.info(f"Cleaning up {len(self._adapter_cache)} cached adapters")
        
        for method_name, adapter in self._adapter_cache.items():
            try:
                # Call adapter cleanup if available
                if hasattr(adapter, 'cleanup'):
                    adapter.cleanup()
                    log.debug(f"Cleaned up {method_name} adapter")
            except Exception as e:
                log.warning(f"Error cleaning up {method_name} adapter: {e}")
        
        # Clear the cache
        self._adapter_cache.clear()
        log.info("Adapter cache cleared")
    
    def validate_ik_method(self, ik_method: str) -> bool:
        """
        Validate if IK method is available.
        
        Args:
            ik_method: IK method name
            
        Returns:
            True if method is available
        """
        return ik_method in self.get_available_ik_methods()