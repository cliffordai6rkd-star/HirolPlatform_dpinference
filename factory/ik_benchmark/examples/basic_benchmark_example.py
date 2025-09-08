#!/usr/bin/env python3
"""
Basic IK Benchmark Example

This example demonstrates how to use the IK benchmark platform to evaluate
different IK methods on a robot configuration.
"""

import sys
from pathlib import Path
import glog as log

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from factory.components.robot_factory import RobotFactory
from factory.ik_benchmark.benchmark_factory import IKBenchmarkFactory
from hardware.base.utils import dynamic_load_yaml


def run_basic_benchmark():
    """Run a basic IK benchmark comparison."""
    log.info("Starting basic IK benchmark example...")
    
    # Load robot configuration
    config_path = str(project_root / "factory" / "ik_benchmark" / "config" / "test_robot_config.yaml")
    robot_config = dynamic_load_yaml(config_path)
    
    # Create robot factory
    robot_factory = RobotFactory(robot_config)
    robot_factory.create_robot_system()
    log.info("Robot system created successfully")
    
    # Create benchmark configuration
    benchmark_config = {
        'robot_config': robot_config,
        'ik_methods': ['gaussian_newton', 'dls', 'pink'],  # Compare three methods
        'test_scenarios': {
            'random_sampling': {
                'count': 100,  # Test on 100 random poses
                'seed': 42     # For reproducible results
            },
            'trajectory_tests': [
                {
                    'type': 'line',
                    'waypoints': 20,
                    'start_pos': [0.4, -0.2, 0.3],
                    'end_pos': [0.5, 0.2, 0.4]
                }
            ]
        },
        'evaluation': {
            'tolerance': 1e-6,
            'max_iterations': 1000,
            'noise_levels': [0.001, 0.01, 0.05]  # For robustness testing
        },
        'method_specific_params': {
            'gaussian_newton': {
                'damping_weight': 1e-6,
                'max_iterations': 1000
            },
            'dls': {
                'damping_weight': 0.2,
                'max_iterations': 1000
            },
            'pink': {
                'damping_weight': 1e-6,
                'max_iterations': 1000
            }
        },
        'simulation': {
            'validation_enabled': False  # Disable simulation for faster testing
        },
        'output': {
            'directory': 'benchmark_results',
            'formats': ['html', 'json'],
            'plots': ['accuracy', 'timing', 'convergence']
        }
    }
    
    # Create benchmark factory
    benchmark_factory = IKBenchmarkFactory(robot_factory, benchmark_config)
    
    # Show available IK methods
    available_methods = benchmark_factory.get_available_ik_methods()
    log.info(f"Available IK methods: {available_methods}")
    
    # Run single test first (quick validation)
    log.info("Running single test for validation...")
    single_result = benchmark_factory.run_single_test('gaussian_newton', 'random_sampling')
    log.info(f"Single test completed - Method: {single_result.method_name}")
    if single_result.accuracy_result:
        log.info(f"Accuracy: {single_result.accuracy_result.convergence_rate:.2%} success rate")
    
    # Run full benchmark
    log.info("Running full benchmark comparison...")
    ik_methods = ['gaussian_newton', 'dls']
    test_scenarios = ['random_sampling', 'trajectory_tests']
    
    report = benchmark_factory.run_full_benchmark(ik_methods, test_scenarios)
    
    # Display summary results
    log.info("="*50)
    log.info("BENCHMARK RESULTS SUMMARY")
    log.info("="*50)
    
    for method_name, method_result in report.ik_methods_results.items():
        log.info(f"\n{method_name.upper()}:")
        
        if method_result.accuracy_result:
            acc = method_result.accuracy_result
            log.info(f"  Accuracy: {acc.convergence_rate:.2%} success, "
                    f"pos_error={acc.mean_position_error:.2e}m, "
                    f"rot_error={acc.mean_rotation_error:.2e}rad")
        
        if method_result.solvability_result:
            sol = method_result.solvability_result
            log.info(f"  Solvability: {sol.convergence_rate:.2%} solved")
        
        if method_result.efficiency_result:
            eff = method_result.efficiency_result
            log.info(f"  Efficiency: {eff.mean_solve_time:.4f}s avg, "
                    f"{eff.mean_iterations:.1f} iters avg")
    
    # Results are saved to benchmark_results/ directory
    log.info(f"\nDetailed results saved to: {benchmark_config['output']['directory']}/")
    log.info("- benchmark_report.html (interactive report)")
    log.info("- benchmark_report.json (raw data)")
    log.info("- statistical_tables.html (comparison tables)")
    log.info("- comparative_plots/ (accuracy, timing plots)")
    
    log.info("Basic benchmark example completed successfully!")


if __name__ == "__main__":
    # Configure logging
    log.setLevel(log.INFO)
    
    try:
        run_basic_benchmark()
    except Exception as e:
        log.error(f"Benchmark failed: {e}")
        sys.exit(1)