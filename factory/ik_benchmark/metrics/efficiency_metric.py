"""
Efficiency metric evaluation for IK algorithms.

Evaluates computational performance including solve times and iteration counts.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import glog as log


@dataclass
class EfficiencyResult:
    """Results from efficiency evaluation."""
    mean_solve_time: float
    std_solve_time: float
    max_solve_time: float
    median_solve_time: float
    mean_iterations: float
    std_iterations: float
    max_iterations: int
    timeout_count: int
    total_tests: int
    # Separated timing statistics
    converged_mean_time: float = 0.0
    converged_std_time: float = 0.0
    failed_mean_time: float = 0.0
    failed_std_time: float = 0.0
    converged_count: int = 0
    failed_count: int = 0
    convergence_rate: float = 0.0


class EfficiencyMetric:
    """Evaluates computational efficiency of IK algorithms."""
    
    def __init__(self, timeout_threshold: float = 1.0):
        """
        Initialize efficiency metric evaluator.
        
        Args:
            timeout_threshold: Time threshold in seconds to consider as timeout
        """
        self._timeout_threshold = timeout_threshold
    
    def evaluate(self, solve_times: List[float], 
                iteration_counts: Optional[List[int]] = None,
                converged_flags: Optional[List[bool]] = None) -> EfficiencyResult:
        """
        Evaluate efficiency of IK algorithm.
        
        Args:
            solve_times: List of solve times in seconds
            iteration_counts: Optional list of iteration counts for each solve
            converged_flags: Optional list of convergence flags
            
        Returns:
            EfficiencyResult containing efficiency metrics
        """
        if not solve_times:
            log.warning("No solve times provided for efficiency evaluation")
            return EfficiencyResult(
                mean_solve_time=0.0,
                std_solve_time=0.0,
                max_solve_time=0.0,
                median_solve_time=0.0,
                mean_iterations=0.0,
                std_iterations=0.0,
                max_iterations=0,
                timeout_count=0,
                total_tests=0,
                converged_mean_time=0.0,
                converged_std_time=0.0,
                failed_mean_time=0.0,
                failed_std_time=0.0,
                converged_count=0,
                failed_count=0,
                convergence_rate=0.0
            )
        
        solve_times_array = np.array(solve_times)
        
        # Filter out infinite or invalid times
        valid_times = solve_times_array[np.isfinite(solve_times_array)]
        
        # Count timeouts
        timeout_count = np.sum(solve_times_array >= self._timeout_threshold)
        
        # Compute time statistics
        if len(valid_times) > 0:
            mean_time = np.mean(valid_times)
            std_time = np.std(valid_times)
            max_time = np.max(valid_times)
            median_time = np.median(valid_times)
        else:
            mean_time = std_time = max_time = median_time = 0.0
        
        # Compute iteration statistics if provided
        mean_iter = std_iter = 0.0
        max_iter = 0
        
        if iteration_counts:
            iter_array = np.array(iteration_counts)
            valid_iters = iter_array[np.isfinite(iter_array)]
            
            if len(valid_iters) > 0:
                mean_iter = np.mean(valid_iters)
                std_iter = np.std(valid_iters)
                max_iter = int(np.max(valid_iters))
        
        # Compute separated time statistics if convergence flags provided
        converged_mean_time = 0.0
        converged_std_time = 0.0
        failed_mean_time = 0.0
        failed_std_time = 0.0
        converged_count = 0
        failed_count = 0
        convergence_rate = 0.0
        
        if converged_flags is not None:
            analysis = self.analyze_convergence_time_relationship(solve_times, converged_flags)
            converged_mean_time = analysis.get('converged_mean_time', 0.0)
            converged_std_time = analysis.get('converged_std_time', 0.0)
            failed_mean_time = analysis.get('failed_mean_time', 0.0)
            failed_std_time = analysis.get('failed_std_time', 0.0)
            converged_count = analysis.get('converged_count', 0)
            failed_count = analysis.get('failed_count', 0)
            convergence_rate = analysis.get('convergence_rate', 0.0)

        result = EfficiencyResult(
            mean_solve_time=mean_time,
            std_solve_time=std_time,
            max_solve_time=max_time,
            median_solve_time=median_time,
            mean_iterations=mean_iter,
            std_iterations=std_iter,
            max_iterations=max_iter,
            timeout_count=timeout_count,
            total_tests=len(solve_times),
            converged_mean_time=converged_mean_time,
            converged_std_time=converged_std_time,
            failed_mean_time=failed_mean_time,
            failed_std_time=failed_std_time,
            converged_count=converged_count,
            failed_count=failed_count,
            convergence_rate=convergence_rate
        )
        
        log.info(f"Efficiency evaluation: Overall={mean_time:.4f}s, "
                f"Success={converged_mean_time:.4f}s, Failed={failed_mean_time:.4f}s, "
                f"Timeouts={timeout_count}/{len(solve_times)}")
        
        return result
    
    def analyze_convergence_time_relationship(self, solve_times: List[float],
                                            converged_flags: List[bool]) -> dict:
        """
        Analyze relationship between solve times and convergence.
        
        Args:
            solve_times: List of solve times
            converged_flags: List of convergence flags
            
        Returns:
            Dictionary with convergence-time analysis
        """
        assert len(solve_times) == len(converged_flags), \
            "Times and flags must have same length"
        
        times_array = np.array(solve_times)
        flags_array = np.array(converged_flags)
        
        # Separate converged and non-converged times
        converged_times = times_array[flags_array]
        failed_times = times_array[~flags_array]
        
        analysis = {
            'converged_count': len(converged_times),
            'failed_count': len(failed_times),
            'total_count': len(solve_times),
            'convergence_rate': np.mean(flags_array),
        }
        
        if len(converged_times) > 0:
            analysis['converged_mean_time'] = np.mean(converged_times)
            analysis['converged_std_time'] = np.std(converged_times)
        else:
            analysis['converged_mean_time'] = 0.0
            analysis['converged_std_time'] = 0.0
            
        if len(failed_times) > 0:
            analysis['failed_mean_time'] = np.mean(failed_times)
            analysis['failed_std_time'] = np.std(failed_times)
        else:
            analysis['failed_mean_time'] = 0.0
            analysis['failed_std_time'] = 0.0
        
        return analysis
    
    def compute_percentiles(self, solve_times: List[float]) -> dict:
        """
        Compute percentile statistics for solve times.
        
        Args:
            solve_times: List of solve times
            
        Returns:
            Dictionary with percentile statistics
        """
        if not solve_times:
            return {}
        
        times_array = np.array(solve_times)
        valid_times = times_array[np.isfinite(times_array)]
        
        if len(valid_times) == 0:
            return {}
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(valid_times, percentiles)
        
        return {
            f'p{p}': v for p, v in zip(percentiles, percentile_values)
        }