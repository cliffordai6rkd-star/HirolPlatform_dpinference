"""
Robustness metric evaluation for IK algorithms.

Evaluates initial value sensitivity, noise tolerance, and trajectory continuity
of IK algorithms.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional
import glog as log


@dataclass
class RobustnessResult:
    """Results from robustness evaluation."""
    noise_levels: List[float]
    success_rates_by_noise: List[float]
    error_increases_by_noise: List[float]
    consistency_scores: List[float]
    overall_robustness_score: float
    noise_sensitivity: float
    stability_metrics: Dict[str, float]


class RobustnessMetric:
    """Evaluates robustness characteristics of IK algorithms."""
    
    def __init__(self, dt: float = 0.01):
        """
        Initialize robustness metric evaluator.
        
        Args:
            dt: Time step for trajectory continuity analysis
        """
        self._dt = dt
    
    def evaluate(self, ik_solver_func: Callable, 
                test_poses: List[np.ndarray],
                robot_model: 'RobotModel',
                noise_levels: List[float] = [0.001, 0.005, 0.01, 0.05],
                num_seeds_per_pose: int = 10) -> RobustnessResult:
        """
        Evaluate robustness of IK algorithm.
        
        Args:
            ik_solver_func: Function that takes (pose, seed) and returns (converged, solution)
            test_poses: List of target poses for testing
            robot_model: Robot model for FK and joint limits
            noise_levels: List of noise levels to test
            num_seeds_per_pose: Number of different seeds per pose
            
        Returns:
            RobustnessResult containing robustness metrics
        """
        log.info(f"Evaluating robustness with {len(test_poses)} poses, "
                f"{len(noise_levels)} noise levels, {num_seeds_per_pose} seeds per pose")
        
        # Test initial value sensitivity
        initial_sensitivity = self._test_initial_sensitivity(
            ik_solver_func, test_poses, robot_model, num_seeds_per_pose
        )
        
        # Test noise tolerance  
        noise_tolerance = self._test_noise_tolerance(
            ik_solver_func, test_poses, robot_model, noise_levels
        )
        
        # Test trajectory continuity (use subset of poses as trajectory)
        trajectory_poses = test_poses[:min(50, len(test_poses))]  # Limit for performance
        continuity_score, mean_vel, max_vel = self._test_trajectory_continuity(
            ik_solver_func, trajectory_poses, robot_model
        )
        
        # Convert results to expected format
        success_rates = [noise_tolerance.get(level, 0.0) for level in noise_levels]
        error_increases = [max(0.0, 1.0 - rate) for rate in success_rates]
        consistency_scores = [initial_sensitivity.get('random', 0.0) for _ in noise_levels]
        
        overall_score = np.mean(success_rates) if success_rates else 0.0
        sensitivity = np.std(success_rates) if len(success_rates) > 1 else 0.0
        
        stability_metrics = {
            'trajectory_continuity': continuity_score,
            'mean_joint_velocity': mean_vel,
            'max_joint_velocity': max_vel,
            **initial_sensitivity
        }
        
        return RobustnessResult(
            noise_levels=noise_levels,
            success_rates_by_noise=success_rates,
            error_increases_by_noise=error_increases,
            consistency_scores=consistency_scores,
            overall_robustness_score=overall_score,
            noise_sensitivity=sensitivity,
            stability_metrics=stability_metrics
        )
    
    def _test_initial_sensitivity(self, ik_solver_func: Callable,
                                poses: List[np.ndarray],
                                robot_model: 'RobotModel',
                                num_seeds: int) -> Dict[str, float]:
        """
        Test sensitivity to different initial joint configurations.
        
        Args:
            ik_solver_func: IK solver function
            poses: Test poses
            robot_model: Robot model
            num_seeds: Number of seeds to test per pose
            
        Returns:
            Dictionary with convergence rates for different seed strategies
        """
        joint_limits_low = robot_model.model.lowerPositionLimit
        joint_limits_high = robot_model.model.upperPositionLimit
        
        seed_strategies = {
            'random': lambda: np.random.uniform(joint_limits_low, joint_limits_high),
            'zero': lambda: np.zeros(len(joint_limits_low)),
            'middle': lambda: (joint_limits_low + joint_limits_high) / 2,
            'random_middle': lambda: (joint_limits_low + joint_limits_high) / 2 + \
                                   0.1 * np.random.randn(len(joint_limits_low))
        }
        
        sensitivity_results = {}
        
        for strategy_name, seed_generator in seed_strategies.items():
            total_tests = 0
            successful_tests = 0
            
            for pose in poses[:min(20, len(poses))]:  # Limit poses for performance
                for _ in range(num_seeds):
                    seed = seed_generator()
                    # Ensure seed is within joint limits
                    seed = np.clip(seed, joint_limits_low, joint_limits_high)
                    
                    try:
                        converged, _ = ik_solver_func(pose, seed)
                        total_tests += 1
                        if converged:
                            successful_tests += 1
                    except Exception as e:
                        log.warning(f"IK solver failed with seed {strategy_name}: {e}")
                        total_tests += 1
            
            convergence_rate = successful_tests / total_tests if total_tests > 0 else 0.0
            sensitivity_results[strategy_name] = convergence_rate
            
            log.debug(f"Initial sensitivity {strategy_name}: "
                     f"{successful_tests}/{total_tests} ({convergence_rate:.2%})")
        
        return sensitivity_results
    
    def _test_noise_tolerance(self, ik_solver_func: Callable,
                            poses: List[np.ndarray],
                            robot_model: 'RobotModel',
                            noise_levels: List[float]) -> Dict[float, float]:
        """
        Test tolerance to noise in target poses.
        
        Args:
            ik_solver_func: IK solver function
            poses: Test poses
            robot_model: Robot model
            noise_levels: Noise levels to test
            
        Returns:
            Dictionary mapping noise levels to performance degradation
        """
        joint_limits_low = robot_model.model.lowerPositionLimit
        joint_limits_high = robot_model.model.upperPositionLimit
        nominal_seed = (joint_limits_low + joint_limits_high) / 2
        
        # Test baseline performance (no noise)
        baseline_success = 0
        baseline_total = 0
        test_poses_subset = poses[:min(30, len(poses))]  # Limit for performance
        
        for pose in test_poses_subset:
            try:
                converged, _ = ik_solver_func(pose, nominal_seed)
                baseline_total += 1
                if converged:
                    baseline_success += 1
            except Exception:
                baseline_total += 1
        
        baseline_rate = baseline_success / baseline_total if baseline_total > 0 else 0.0
        
        # Test with different noise levels
        noise_tolerance = {}
        
        for noise_level in noise_levels:
            noisy_success = 0
            noisy_total = 0
            
            for pose in test_poses_subset:
                # Add noise to pose
                noisy_pose = self._add_pose_noise(pose, noise_level)
                
                try:
                    converged, _ = ik_solver_func(noisy_pose, nominal_seed)
                    noisy_total += 1
                    if converged:
                        noisy_success += 1
                except Exception:
                    noisy_total += 1
            
            noisy_rate = noisy_success / noisy_total if noisy_total > 0 else 0.0
            
            # Performance degradation relative to baseline
            degradation = (baseline_rate - noisy_rate) / baseline_rate if baseline_rate > 0 else 1.0
            noise_tolerance[noise_level] = max(0.0, 1.0 - degradation)  # Convert to tolerance score
            
            log.debug(f"Noise tolerance {noise_level}: "
                     f"{noisy_success}/{noisy_total} ({noisy_rate:.2%}), "
                     f"degradation: {degradation:.2%}")
        
        return noise_tolerance
    
    def _test_trajectory_continuity(self, ik_solver_func: Callable,
                                  trajectory_poses: List[np.ndarray],
                                  robot_model: 'RobotModel') -> Tuple[float, float, float]:
        """
        Test continuity of joint solutions along a trajectory.
        
        Args:
            ik_solver_func: IK solver function
            trajectory_poses: Ordered list of poses forming trajectory
            robot_model: Robot model
            
        Returns:
            Tuple of (continuity_score, mean_joint_velocity, max_joint_velocity)
        """
        if len(trajectory_poses) < 2:
            return 1.0, 0.0, 0.0
        
        joint_limits_low = robot_model.model.lowerPositionLimit
        joint_limits_high = robot_model.model.upperPositionLimit
        
        joint_solutions = []
        
        # Solve IK for trajectory, using previous solution as seed
        seed = (joint_limits_low + joint_limits_high) / 2  # Initial seed
        
        for pose in trajectory_poses:
            try:
                converged, solution = ik_solver_func(pose, seed)
                if converged:
                    joint_solutions.append(solution)
                    seed = solution  # Use current solution as next seed
                else:
                    # If failed, use current seed for next attempt
                    joint_solutions.append(seed)
            except Exception:
                joint_solutions.append(seed)
        
        if len(joint_solutions) < 2:
            return 0.0, 0.0, 0.0
        
        # Compute joint velocities
        joint_solutions_array = np.array(joint_solutions)
        joint_velocities = np.diff(joint_solutions_array, axis=0) / self._dt
        
        # Compute continuity metrics
        velocity_magnitudes = np.linalg.norm(joint_velocities, axis=1)
        mean_velocity = np.mean(velocity_magnitudes)
        max_velocity = np.max(velocity_magnitudes)
        
        # Continuity score based on velocity smoothness
        # Lower velocities indicate better continuity
        velocity_threshold = 10.0  # rad/s, adjust based on robot capabilities
        continuity_score = np.exp(-mean_velocity / velocity_threshold)
        
        return continuity_score, mean_velocity, max_velocity
    
    def _add_pose_noise(self, pose: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add noise to a pose matrix.
        
        Args:
            pose: 4x4 transformation matrix
            noise_level: Noise level (standard deviation)
            
        Returns:
            Noisy pose matrix
        """
        noisy_pose = pose.copy()
        
        # Add position noise
        position_noise = np.random.normal(0, noise_level, 3)
        noisy_pose[:3, 3] += position_noise
        
        # Add rotation noise (small angle approximation)
        rotation_noise = np.random.normal(0, noise_level, 3)
        
        # Create small rotation matrix
        from scipy.spatial.transform import Rotation as R
        small_rotation = R.from_rotvec(rotation_noise).as_matrix()
        
        # Apply rotation noise
        noisy_pose[:3, :3] = small_rotation @ noisy_pose[:3, :3]
        
        return noisy_pose