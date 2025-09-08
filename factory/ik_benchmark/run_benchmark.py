#!/usr/bin/env python3
"""
IK Benchmark Runner - Simple command-line interface for running IK benchmarks.

Usage:
    python factory/ik_benchmark/run_benchmark.py [options]

Examples:
    # Basic benchmark with default settings
    python factory/ik_benchmark/run_benchmark.py
    
    # Custom configuration file
    python factory/ik_benchmark/run_benchmark.py --config my_config.yaml
    
    # Quick test with limited scenarios
    python factory/ik_benchmark/run_benchmark.py --quick
    
    # Specific methods and scenarios
    python factory/ik_benchmark/run_benchmark.py --methods gaussian_newton dls lm --scenarios random_sampling
"""

import sys
import argparse
from pathlib import Path
import glog as log

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from factory.components.robot_factory import RobotFactory
from factory.ik_benchmark.benchmark_factory import IKBenchmarkFactory
from hardware.base.utils import dynamic_load_yaml


def create_default_config():
    """Create default benchmark configuration."""
    return {
        'ik_methods': ['gaussian_newton', 'dls', 'lm', 'pink'],
        'test_scenarios': {
            'random_sampling': {
                'count': 200,
                'seed': 42
            },
            'trajectory_tests': [
                {
                    'type': 'line',
                    'waypoints': 15,
                    'start_pos': [0.4, -0.2, 0.3],
                    'end_pos': [0.5, 0.2, 0.4]
                }
            ]
        },
        'evaluation': {
            'tolerance': 1e-6,
            'max_iterations': 200,
            'noise_levels': [0.001, 0.01]
        },
        'method_specific_params': {
            'gaussian_newton': {
                'damping_weight': 1e-6,
                'max_iterations': 200
            },
            'dls': {
                'damping_weight': 0.2,
                'max_iterations': 200
            },
            'lm': {
                'damping_weight': 0.05,
                'max_iterations': 200
            },
            'pink': {
                'damping_weight': 1e-6,
                'max_iterations': 200
            }
        },
        'simulation': {
            'validation_enabled': False
        },
        'output': {
            'directory': 'benchmark_results',
            'formats': ['html', 'json'],
            'plots': ['accuracy', 'timing']
        }
    }


def create_quick_config():
    """Create quick test configuration with minimal scenarios."""
    return {
        'ik_methods': ['gaussian_newton'],
        'test_scenarios': {
            'random_sampling': {
                'count': 20,
                'seed': 42
            }
        },
        'evaluation': {
            'tolerance': 1e-6,
            'max_iterations': 100
        },
        'method_specific_params': {
            'gaussian_newton': {
                'damping_weight': 1e-6,
                'max_iterations': 100
            },
            'dls': {
                'damping_weight': 0.2,
                'max_iterations': 100
            },
            'lm': {
                'damping_weight': 0.05,
                'max_iterations': 100
            },
            'pink': {
                'damping_weight': 1e-6,
                'max_iterations': 100
            }
        },
        'simulation': {
            'validation_enabled': False
        },
        'output': {
            'directory': 'quick_test_results',
            'formats': ['json'],
            'plots': ['accuracy']
        }
    }


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description='IK Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default benchmark
  %(prog)s --config my_config.yaml           # Custom configuration
  %(prog)s --quick                           # Quick test
  %(prog)s --methods gaussian_newton dls    # Specific methods
  %(prog)s --scenarios random_sampling       # Specific scenarios
  %(prog)s --output my_results               # Custom output directory
        """
    )
    
    parser.add_argument('--config', '-c', type=str,
                       help='Path to benchmark configuration YAML file')
    parser.add_argument('--robot-config', '-r', type=str,
                       help='Path to robot configuration YAML file (default: test_robot_config.yaml)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick test with minimal scenarios')
    parser.add_argument('--methods', '-m', nargs='+', 
                       choices=['gaussian_newton', 'dls', 'lm', 'pink'],
                       help='IK methods to test')
    parser.add_argument('--scenarios', '-s', nargs='+',
                       choices=['random_sampling', 'trajectory_tests', 'singular_tests', 'workspace_grid'],
                       help='Test scenarios to run')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory name')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log.setLevel(log.INFO if args.verbose else log.WARNING)
    
    log.info("IK Benchmark Platform")
    log.info("=" * 50)
    
    try:
        # Load robot configuration
        if args.robot_config:
            robot_config_path = args.robot_config
        else:
            robot_config_path = str(project_root / "factory" / "ik_benchmark" / "config" / "test_robot_config.yaml")
        
        log.info(f"Loading robot configuration: {robot_config_path}")
        robot_config = dynamic_load_yaml(robot_config_path)
        
        # Create robot factory
        robot_factory = RobotFactory(robot_config)
        robot_factory.create_robot_system()
        log.info("Robot system initialized successfully")
        
        # Load or create benchmark configuration
        if args.config:
            # Load from file
            import yaml
            with open(args.config, 'r') as f:
                benchmark_config = yaml.safe_load(f)
            log.info(f"Loaded benchmark configuration from: {args.config}")
        elif args.quick:
            # Quick test configuration
            benchmark_config = create_quick_config()
            log.info("Using quick test configuration")
        else:
            # Default configuration
            benchmark_config = create_default_config()
            log.info("Using default configuration")
        
        # Override configuration with command line arguments
        if args.methods:
            benchmark_config['ik_methods'] = args.methods
            log.info(f"Testing methods: {args.methods}")
        
        if args.scenarios:
            # Filter scenarios based on command line
            filtered_scenarios = {}
            for scenario in args.scenarios:
                if scenario in benchmark_config['test_scenarios']:
                    filtered_scenarios[scenario] = benchmark_config['test_scenarios'][scenario]
            benchmark_config['test_scenarios'] = filtered_scenarios
            log.info(f"Testing scenarios: {args.scenarios}")
        
        if args.output:
            benchmark_config['output']['directory'] = args.output
            log.info(f"Output directory: {args.output}")
        
        # Set robot config
        benchmark_config['robot_config'] = robot_config
        
        # Create benchmark factory
        benchmark_factory = IKBenchmarkFactory(robot_factory, benchmark_config)
        
        # Show available methods
        available_methods = benchmark_factory.get_available_ik_methods()
        log.info(f"Available IK methods: {available_methods}")
        
        # Validate requested methods
        requested_methods = benchmark_config['ik_methods']
        for method in requested_methods:
            if method not in available_methods:
                log.error(f"IK method '{method}' not available")
                sys.exit(1)
        
        # Run benchmark
        log.info("Starting benchmark execution...")
        
        test_scenarios = list(benchmark_config['test_scenarios'].keys())
        report = benchmark_factory.run_full_benchmark(requested_methods, test_scenarios)
        
        # Display results summary
        log.info("\n" + "=" * 50)
        log.info("BENCHMARK RESULTS SUMMARY")
        log.info("=" * 50)
        
        for method_name, method_result in report.ik_methods_results.items():
            log.info(f"\n{method_name.upper()}:")
            
            if method_result.accuracy_result:
                acc = method_result.accuracy_result
                log.info(f"  Accuracy: {acc.convergence_rate:.2%} success rate")
                log.info(f"  Position error: {acc.mean_position_error:.2e} m")
                log.info(f"  Rotation error: {acc.mean_rotation_error:.2e} rad")
            
            if method_result.solvability_result:
                sol = method_result.solvability_result
                log.info(f"  Solvability: {sol.convergence_rate:.2%} solved")
            
            if method_result.efficiency_result:
                eff = method_result.efficiency_result
                log.info(f"  Overall Time: {eff.mean_solve_time:.4f}s average")
                log.info(f"  Success Time: {eff.converged_mean_time:.4f}s average")
                log.info(f"  Failed Time: {eff.failed_mean_time:.4f}s average")
                log.info(f"  Convergence: {eff.convergence_rate:.2%} ({eff.converged_count}/{eff.converged_count + eff.failed_count})")
                log.info(f"  Iterations: {eff.mean_iterations:.1f} average")
        
        # Output information
        output_dir = benchmark_config['output']['directory']
        log.info(f"\nResults saved to: {output_dir}/")
        
        formats = benchmark_config['output']['formats']
        for fmt in formats:
            if fmt == 'html':
                log.info(f"  - benchmark_report.html (interactive report)")
            elif fmt == 'json':
                log.info(f"  - benchmark_report.json (raw data)")
            elif fmt == 'csv':
                log.info(f"  - results.csv (tabular data)")
        
        log.info("Benchmark completed successfully! 🎉")
        
    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()