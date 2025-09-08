#!/usr/bin/env python3
"""
Advanced IK Benchmark Example

This example demonstrates advanced features of the IK benchmark platform including:
- Testing all 6 available IK solvers (traditional + JAX/PyTorch-based)
- Custom test scenarios (trajectories, singularities, workspace coverage)
- Robustness evaluation with noise injection
- Simulation validation
- Configuration file loading
- Statistical analysis and performance comparison
- GPU-accelerated solvers (Pyroki JAX, CuRobo PyTorch)
"""

import sys
from pathlib import Path
import glog as log
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from factory.components.robot_factory import RobotFactory
from factory.ik_benchmark.benchmark_factory import IKBenchmarkFactory
from factory.ik_benchmark.visualizer.plot_utils import PlotUtils
from hardware.base.utils import dynamic_load_yaml
import matplotlib.pyplot as plt


def create_advanced_config():
    """Create advanced benchmark configuration with all features enabled."""
    return {
        'robot_config': {
            'use_hardware': False,
            'use_simulation': True,
            'robot': "fr3",
            'model_type': "model",
            'model_config': {
                'name': "fr3_only",
                'cfg': "!include motion/config/robot_model_fr3_cfg.yaml"
            },
            'simulation': "mujoco",
            'simulation_config': "!include simulation/config/mujoco_fr3_cfg.yaml"
        },
        'ik_methods': [
            'gaussian_newton', 'dls', 'lm', 
            # 'pink',
            'pyroki',
            'curobo'
                       ],  # Test all methods including new solvers
        'test_scenarios': {
            'random_sampling': {
                'count': 200,
                'seed': 12345  # This seed is for pose generation, different from initial_guess
            },
            'trajectory_tests': [
                {
                    'type': 'line',
                    'waypoints': 30,
                    'start_pos': [0.3, -0.3, 0.2],
                    'end_pos': [0.6, 0.3, 0.5]
                },
                {
                    'type': 'circle',
                    'waypoints': 24,
                    'center': [0.45, 0.0, 0.35],
                    'radius': 0.15,
                    'normal': [0, 0, 1]
                },
                {
                    'type': 'helix',
                    'waypoints': 40,
                    'center': [0.45, 0.0, 0.30],
                    'radius': 0.12,
                    'pitch': 0.08,
                    'turns': 2.0
                }
            ],
            'singular_tests': {
                'count': 80,
                'margin': 0.02
            }
        },
        'evaluation': {
            'tolerance': 1e-6,
            'max_iterations': 200,
            'noise_levels': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            'timeout_threshold': 2.0
        },
        'method_specific_params': {
            'gaussian_newton': {
                'damping_weight': 0,
                'max_iterations': 200
            },
            'dls': {
                'damping_weight': 0.2,
                'max_iterations': 200
            },
            'lm': {
                'damping_weight': 1e-3-1e-6,
                'max_iterations': 200
            },
            'pink': {
                'damping_weight': 1e-6,
                'max_iterations': 200
            },
            'pyroki': {
                # Basic Pyroki parameters for fast performance (close to 2000Hz)
                # 'num_seeds_init': 1,          # Minimal seeds for speed
                # 'num_seeds_final': 1,         # Fewer final candidates
                # 'total_steps': 100,             # Reduced optimization steps
                # 'init_steps': 3,              # Minimal initial steps
                # 'pos_weight': 50.0,           # Basic pyroki position weight
                # 'ori_weight': 10.0,           # Basic pyroki orientation weight
                # 'limit_weight': 100.0,        # Basic pyroki limit weight
                # 'lambda_initial': 1.0,        # Basic pyroki initial damping
                # 'max_iterations': 100           # Match total_steps
                
                # # Original advanced parameters (commented out for performance)
                'num_seeds_init': 64,        # More seeds for thorough testing
                'num_seeds_final': 4,         # More final candidates  
                'total_steps': 22,            # More optimization steps
                'init_steps': 8,             # More initial steps
                'pos_weight': 50.0,           # Higher position weight
                'ori_weight': 10.0,            # Higher orientation weight
                'limit_weight': 100.0,        # Strong joint limit enforcement
                'lambda_initial': 1.0,        # Moderate initial damping
                'max_iterations': 22          # Equivalent to total_steps
                
                
                # test parameters (commented out for performance)
                # 'num_seeds_init': 16,        # More seeds for thorough testing
                # 'num_seeds_final': 4,         # More final candidates  
                # 'total_steps': 20,            # More optimization steps
                # 'init_steps': 12,             # More initial steps
                # 'pos_weight': 50.0,           # Higher position weight
                # 'ori_weight': 10.0,            # Higher orientation weight
                # 'limit_weight': 100.0,        # Strong joint limit enforcement
                # 'lambda_initial': 0.1,        # Moderate initial damping
                # 'max_iterations': 20          # Equivalent to total_steps
                
                
            },
            'curobo': {
                # CuRobo-specific parameters - optimized for performance and reliability
                'robot_config_name': "franka.yml",
                'world_config_name': "collision_test.yml",
                'position_threshold': 0.0001,      # Relaxed but reasonable precision (5mm)
                'rotation_threshold': 0.001,       # Relaxed rotation tolerance (~3 degrees)
                'num_seeds': 64,                  # Balanced seeds for good coverage vs speed
                'use_cuda_graph': True,           # Enable CUDA optimization
                'high_precision': True,          # Disable for better performance
                'collision_free': False,          # Disable collision for pure IK
                'use_gpu': True,                  # Use GPU acceleration
                'max_iterations': 100             # Reasonable iteration limit
            }
        },
        'simulation': {
            'validation_enabled': True,
            'validation_tolerance': 1e-3,
            'physics_timestep': 0.001,
            'validation_steps': 100
        },
        'output': {
            'directory': 'factory/ik_benchmark/results/advanced_benchmark_results',
            'formats': ['html', 'json', 'csv'],
            'plots': [
                'accuracy', 'timing', 'convergence', 'robustness', 
                'workspace_coverage', 'singularity_performance'
            ],
            'statistical_analysis': True,
            'comparative_analysis': True,
            'collect_detailed_results': True,
            'generate_3d_plots': True
        }
    }


def get_custom_initial_guess_strategies():
    """Define custom initial guess strategies for IK solving."""
    return {
        'home_position': np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),  # FR3 home position
        # 'stretched_up': np.array([0.0, -1.0, 0.0, -1.5, 0.0, 0.5, 0.785]),        # More vertical reach
        # 'side_reach': np.array([1.2, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785]),          # Side reaching pose
        # 'folded': np.array([0.0, 0.2, 0.0, -2.8, 0.0, 3.0, 0.785])                # Compact folded pose
    }


def run_comprehensive_analysis():
    """Run comprehensive IK analysis with detailed reporting."""
    log.info("Starting advanced IK benchmark analysis...")
    
    # Load robot configuration 
    config_path = str(project_root / "factory" / "ik_benchmark" / "config" / "test_robot_config.yaml")
    robot_config = dynamic_load_yaml(config_path)
    
    # Create robot factory
    robot_factory = RobotFactory(robot_config)
    robot_factory.create_robot_system()
    
    # Create advanced benchmark configuration
    benchmark_config = create_advanced_config()
    benchmark_config['robot_config'] = robot_config  # Use loaded config
    
    # Create benchmark factory
    benchmark_factory = IKBenchmarkFactory(robot_factory, benchmark_config)
    
    # Get custom initial guess strategies
    initial_guess_strategies = get_custom_initial_guess_strategies()
    custom_seed = initial_guess_strategies['home_position']  # Use home position as default
    log.info(f"Using custom initial guess: {custom_seed[:3]}... (home_position strategy)")
    
    # Run comprehensive IK benchmark with all methods and scenarios
    log.info("Running comprehensive IK benchmark...")
    
    methods = [
        'gaussian_newton', 'dls', 'lm', 
            #    'pink', 
               'pyroki',
               'curobo'
               ]
    test_scenarios = ['random_sampling', 'trajectory_tests']
    
    try:
        report = benchmark_factory.run_full_benchmark(methods, test_scenarios, initial_guess=custom_seed)
        
        # Display comprehensive analysis
        display_comprehensive_results(report)
        
        # Generate 3D visualizations if detailed results are available
        if (report.detailed_results and 
            benchmark_config.get('output', {}).get('generate_3d_plots', False)):
            log.info("Generating 3D visualizations from collected detailed results...")
            generate_3d_from_detailed_results(report.detailed_results, methods)
        
    except Exception as e:
        log.error(f"Full benchmark failed: {e}")
    
    log.info("Advanced benchmark analysis completed!")


def display_comprehensive_results(report):
    """Display comprehensive benchmark results with detailed analysis."""
    log.info("\n" + "="*60)
    log.info("COMPREHENSIVE IK BENCHMARK ANALYSIS")
    log.info("="*60)
    
    # Overall summary
    log.info(f"\nReport generated at: {report.timestamp}")
    log.info(f"Methods tested: {len(report.ik_methods_results)}")
    
    # Method comparison table
    log.info("\nMETHOD PERFORMANCE COMPARISON:")
    log.info("-" * 70)
    log.info(f"{'Method':<15} {'Success Rate':<12} {'Avg Time':<10} {'Avg Error':<12} {'Robustness':<10}")
    log.info("-" * 70)
    
    for method, method_result in report.ik_methods_results.items():
        convergence_rate = method_result.accuracy_result.convergence_rate if method_result.accuracy_result else 0.0
        avg_time = method_result.efficiency_result.mean_solve_time if method_result.efficiency_result else 0.0
        avg_error = method_result.accuracy_result.mean_position_error if method_result.accuracy_result else float('inf')
        robustness_score = method_result.robustness_result.overall_robustness_score if method_result.robustness_result else 0.0
        
        log.info(f"{method:<15} {convergence_rate:<12.2%} {avg_time:<10.4f} {avg_error:<12.2e} {robustness_score:<10.3f}")
    
    # Recommendations
    log.info("\nRECOMMENDATIONS:")
    log.info("-" * 20)
    
    # Find best method for different criteria
    best_accuracy_item = max(report.ik_methods_results.items(), 
                            key=lambda x: x[1].accuracy_result.convergence_rate if x[1].accuracy_result else 0)
    best_speed_item = min(report.ik_methods_results.items(),
                         key=lambda x: x[1].efficiency_result.mean_solve_time if x[1].efficiency_result else float('inf'))
    best_robust_item = max(report.ik_methods_results.items(),
                          key=lambda x: x[1].robustness_result.overall_robustness_score if x[1].robustness_result else 0.0)
    
    log.info(f"• Best accuracy: {best_accuracy_item[0]} "
            f"({best_accuracy_item[1].accuracy_result.convergence_rate:.2%} success)")
    log.info(f"• Fastest method: {best_speed_item[0]} "
            f"({best_speed_item[1].efficiency_result.mean_solve_time:.4f}s avg)")
    log.info(f"• Most robust: {best_robust_item[0]} "
            f"({best_robust_item[1].robustness_result.overall_robustness_score:.3f} score)")


def generate_3d_visualizations(benchmark_factory, methods, initial_guess=None):
    """
    Generate 3D visualizations of IK solver performance.
    
    Runs a focused test to collect pose data for 3D scatter plot visualization,
    showing success/failure patterns in workspace.
    """
    log.info("Generating 3D visualizations...")
    
    try:
        # Generate test poses for 3D visualization
        log.info("Generating test poses for 3D visualization...")
        test_poses = []
        
        # Create a focused set of test poses in the workspace
        # Sample poses in a 3D grid within reachable workspace
        for x in np.linspace(0.3, 0.7, 8):  # 8 points along x
            for y in np.linspace(-0.3, 0.3, 6):  # 6 points along y  
                for z in np.linspace(0.2, 0.6, 5):  # 5 points along z
                    # Create homogeneous transformation matrix
                    pose = np.eye(4)
                    pose[0, 3] = x
                    pose[1, 3] = y 
                    pose[2, 3] = z
                    
                    # Add some random rotation variations
                    import random
                    random.seed(42)  # For reproducible results
                    angle = random.uniform(-0.2, 0.2)  # Small rotation range
                    axis = random.choice([[1,0,0], [0,1,0], [0,0,1]])  # Random axis
                    from scipy.spatial.transform import Rotation
                    rotation = Rotation.from_rotvec(angle * np.array(axis))
                    pose[:3, :3] = rotation.as_matrix()
                    
                    test_poses.append(pose)
        
        log.info(f"Generated {len(test_poses)} test poses for 3D visualization")
        
        # Collect pose results for each method
        pose_data = {}
        workspace_data = {}
        
        for method in methods:
            log.info(f"Testing {method} for 3D visualization...")
            pose_results = []
            positions = []
            success_mask = []
            
            # Use the IK tester to get convergence results
            if hasattr(benchmark_factory, '_benchmark_suite') and benchmark_factory._benchmark_suite:
                ik_tester = benchmark_factory._benchmark_suite._ik_tester
                
                if ik_tester.validate_ik_method(method):
                    for i, target_pose in enumerate(test_poses):
                        try:
                            # Test single pose
                            converged, solution, solve_time = ik_tester._test_single_ik(
                                method, target_pose, tolerance=1e-4, max_iterations=100,
                                initial_guess=initial_guess
                            )
                            
                            pose_results.append((target_pose, converged))
                            positions.append(target_pose[:3, 3])
                            success_mask.append(converged)
                            
                            if i % 50 == 0:  # Progress update
                                log.info(f"  Progress: {i+1}/{len(test_poses)}")
                                
                        except Exception as e:
                            log.warning(f"Failed to test pose {i} for {method}: {e}")
                            pose_results.append((target_pose, False))
                            positions.append(target_pose[:3, 3])
                            success_mask.append(False)
                    
                    pose_data[method] = pose_results
                    workspace_data[method] = {
                        'positions': np.array(positions),
                        'success_mask': np.array(success_mask)
                    }
                    
                    success_rate = np.mean(success_mask)
                    log.info(f"  {method}: {success_rate:.1%} success rate ({np.sum(success_mask)}/{len(success_mask)})")
                    
                else:
                    log.warning(f"Method {method} not available for 3D visualization")
        
        # Generate 3D visualizations
        if pose_data:
            log.info("Generating 3D pose scatter plot...")
            
            # Create output directory
            output_dir = Path('factory/ik_benchmark/results/advanced_benchmark_results')
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate 3D pose scatter plot
            fig_scatter = PlotUtils.plot_3d_pose_scatter(
                pose_data, 
                title="3D IK Success/Failure Distribution - All Methods",
                save_path=str(plots_dir / "3d_pose_scatter.png")
            )
            log.info(f"3D pose scatter plot saved to {plots_dir / '3d_pose_scatter.png'}")
            
            # Generate individual 3D workspace coverage plots
            for method, data in workspace_data.items():
                if len(data['positions']) > 0:
                    fig_workspace = PlotUtils.plot_workspace_3d_coverage(
                        data['positions'],
                        data['success_mask'], 
                        method,
                        save_path=str(plots_dir / f"3d_workspace_{method}.png")
                    )
                    log.info(f"3D workspace coverage for {method} saved to {plots_dir / f'3d_workspace_{method}.png'}")
            
            # Show plots if running interactively
            try:
                plt.show()
            except:
                log.info("Plots saved to files (display not available)")
                
            log.info("3D visualizations completed successfully!")
            
        else:
            log.warning("No pose data collected for 3D visualization")
            
    except Exception as e:
        log.error(f"3D visualization generation failed: {e}")
        import traceback
        log.error(f"Full traceback:\n{traceback.format_exc()}")


def generate_3d_from_detailed_results(detailed_results, methods):
    """
    Generate 3D visualizations from collected detailed IK test results.
    
    Args:
        detailed_results: Dict mapping method names to List[IKTestResult]
        methods: List of method names
    """
    log.info("Converting detailed results to 3D visualization data...")
    
    try:
        # Convert detailed results to pose data format
        pose_data = {}
        
        for method, test_results in detailed_results.items():
            if method in methods:
                pose_results = []
                
                for test_result in test_results:
                    if test_result.target_pose is not None:
                        pose_results.append((test_result.target_pose, test_result.converged))
                
                pose_data[method] = pose_results
                log.info(f"Converted {len(pose_results)} detailed results for {method}")
        
        if pose_data:
            # Create output directory
            from pathlib import Path
            output_dir = Path('factory/ik_benchmark/results/advanced_benchmark_results')
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate 3D pose scatter plot
            fig_scatter = PlotUtils.plot_3d_pose_scatter(
                pose_data, 
                title="3D IK Success/Failure Distribution - All Methods",
                save_path=str(plots_dir / "3d_pose_scatter.png")
            )
            log.info(f"3D pose scatter plot saved to {plots_dir / '3d_pose_scatter.png'}")
            
            # Generate individual 3D workspace coverage plots
            for method, pose_results in pose_data.items():
                if pose_results:
                    positions = np.array([pose[0][:3, 3] for pose in pose_results])
                    success_mask = np.array([pose[1] for pose in pose_results])
                    
                    fig_workspace = PlotUtils.plot_workspace_3d_coverage(
                        positions,
                        success_mask, 
                        method,
                        save_path=str(plots_dir / f"3d_workspace_{method}.png")
                    )
                    log.info(f"3D workspace coverage for {method} saved to {plots_dir / f'3d_workspace_{method}.png'}")
            
            # Show plots if running interactively
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except:
                log.info("Plots saved to files (display not available)")
                
            log.info("3D visualizations generated successfully from detailed results!")
            
        else:
            log.warning("No valid pose data found in detailed results")
            
    except Exception as e:
        log.error(f"Failed to generate 3D visualizations from detailed results: {e}")
        import traceback
        log.error(f"Full traceback:\n{traceback.format_exc()}")





if __name__ == "__main__":
    # Configure logging
    log.setLevel(log.INFO)
    
    try:
        # Run single-step comprehensive analysis (includes 3D visualization if configured)
        run_comprehensive_analysis()
        
        # Demonstrate config file usage (optional)
        # run_configuration_file_example()
        
    except Exception as e:
        log.error(f"Advanced benchmark failed: {e}")
        sys.exit(1)