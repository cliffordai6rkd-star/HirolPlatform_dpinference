#!/usr/bin/env python3
"""
Test New Solvers - Demonstration script for pyroki and curobo integration.

This script demonstrates how to run the IK benchmark with the new pyroki and curobo solvers,
showing the extended capabilities of the HIROL IK benchmark system.
"""

import os
import sys
from pathlib import Path
import glog as log

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from factory.components.robot_factory import RobotFactory
    from factory.ik_benchmark.benchmark_factory import IKBenchmarkFactory
    from hardware.base.utils import dynamic_load_yaml
    from motion.adapters import get_available_adapters
    
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    BENCHMARK_AVAILABLE = False
    _import_error = str(e)


def test_adapter_availability():
    """Test and display adapter availability."""
    print("🔍 Checking Adapter Availability")
    print("-" * 40)
    
    try:
        from motion.adapters import PYROKI_AVAILABLE, CUROBO_AVAILABLE
        
        print(f"Pyroki adapter: {'✓ Available' if PYROKI_AVAILABLE else '✗ Not available'}")
        print(f"CuRobo adapter: {'✓ Available' if CUROBO_AVAILABLE else '✗ Not available'}")
        
        available_adapters = get_available_adapters()
        print(f"Available adapters: {available_adapters}")
        
        return PYROKI_AVAILABLE or CUROBO_AVAILABLE
        
    except ImportError as e:
        print(f"✗ Adapter module not available: {e}")
        return False


def create_test_config():
    """Create a minimal benchmark configuration for testing."""
    return {
        'ik_methods': ['gaussian_newton'],  # Start with basic method
        'test_scenarios': {
            'random_sampling': {
                'count': 10,  # Small number for quick testing
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
            'pyroki': {
                'num_seeds_init': 32,  # Reduced for testing
                'num_seeds_final': 2,
                'total_steps': 8,
                'init_steps': 3,
                'max_iterations': 8
            },
            'curobo': {
                'robot_config_name': "franka.yml",
                'num_seeds': 8,  # Reduced for testing
                'use_gpu': False,  # Force CPU for testing
                'max_iterations': 50
            }
        },
        'output': {
            'directory': 'test_new_solvers_results',
            'formats': ['json'],
            'plots': []
        }
    }


def run_basic_test():
    """Run basic test with traditional methods."""
    print("\n🧪 Running Basic Test (Traditional Methods)")
    print("-" * 50)
    
    try:
        # Create robot configuration
        robot_config = {
            "use_hardware": False,
            "use_simulation": False,
            "robot": "fr3",
            "model_config": {
                "name": "fr3", 
                "cfg": {
                    "fr3": {
                        "urdf_path": "assets/franka_fr3/fr3.urdf",
                        "ee_link": "fr3_hand"
                    }
                }
            }
        }
        
        # Create robot factory
        robot_factory = RobotFactory(robot_config)
        
        # Create test config
        benchmark_config = create_test_config()
        benchmark_config['robot_config'] = robot_config
        
        # Create benchmark factory
        benchmark_factory = IKBenchmarkFactory(robot_factory, benchmark_config)
        
        # Show available methods
        available_methods = benchmark_factory.get_available_ik_methods()
        print(f"Available IK methods: {available_methods}")
        
        # Run basic test with traditional method
        result = benchmark_factory.run_single_test('gaussian_newton', 'random_sampling')
        
        if result.accuracy_result:
            acc = result.accuracy_result
            print(f"✓ Basic test completed:")
            print(f"  Success rate: {acc.convergence_rate:.2%}")
            print(f"  Mean position error: {acc.mean_position_error:.2e} m")
            print(f"  Mean rotation error: {acc.mean_rotation_error:.2e} rad")
        
        return True, available_methods
        
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False, []


def run_extended_test(available_methods):
    """Run extended test with new solvers if available."""
    print("\n🚀 Running Extended Test (New Solvers)")
    print("-" * 50)
    
    # Filter methods to only include available new solvers
    new_methods = [m for m in ['pyroki', 'curobo'] if m in available_methods]
    
    if not new_methods:
        print("✗ No new solvers available for testing")
        return False
    
    try:
        # Create robot configuration
        robot_config = {
            "use_hardware": False,
            "use_simulation": False,
            "robot": "fr3",
            "model_config": {
                "name": "fr3",
                "cfg": {
                    "fr3": {
                        "urdf_path": "assets/franka_fr3/fr3.urdf",
                        "ee_link": "fr3_hand"
                    }
                }
            }
        }
        
        # Create robot factory
        robot_factory = RobotFactory(robot_config)
        
        # Create test config with new methods
        benchmark_config = create_test_config()
        benchmark_config['ik_methods'] = new_methods
        benchmark_config['robot_config'] = robot_config
        
        # Create benchmark factory
        benchmark_factory = IKBenchmarkFactory(robot_factory, benchmark_config)
        
        # Test each new method
        for method in new_methods:
            print(f"\n📊 Testing {method.upper()} solver:")
            try:
                result = benchmark_factory.run_single_test(method, 'random_sampling')
                
                if result.accuracy_result:
                    acc = result.accuracy_result
                    print(f"  ✓ Success rate: {acc.convergence_rate:.2%}")
                    print(f"  ✓ Mean position error: {acc.mean_position_error:.2e} m")
                    print(f"  ✓ Mean rotation error: {acc.mean_rotation_error:.2e} rad")
                else:
                    print(f"  ✗ No accuracy results available")
                
                if result.efficiency_result:
                    eff = result.efficiency_result
                    print(f"  ✓ Mean solve time: {eff.mean_solve_time:.4f} s")
                
            except Exception as e:
                print(f"  ✗ {method} test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Extended test failed: {e}")
        return False


def run_comparison_test(available_methods):
    """Run comparison test between multiple methods."""
    print("\n📈 Running Comparison Test")
    print("-" * 40)
    
    # Select methods for comparison
    comparison_methods = ['gaussian_newton']
    new_methods = [m for m in ['pyroki', 'curobo'] if m in available_methods]
    comparison_methods.extend(new_methods)
    
    if len(comparison_methods) < 2:
        print("✗ Not enough methods available for comparison")
        return False
    
    try:
        # Create robot configuration
        robot_config = {
            "use_hardware": False,
            "use_simulation": False, 
            "robot": "fr3",
            "model_config": {
                "name": "fr3",
                "cfg": {
                    "fr3": {
                        "urdf_path": "assets/franka_fr3/fr3.urdf",
                        "ee_link": "fr3_hand"
                    }
                }
            }
        }
        
        # Create robot factory
        robot_factory = RobotFactory(robot_config)
        
        # Create comparison config
        benchmark_config = create_test_config()
        benchmark_config['ik_methods'] = comparison_methods
        benchmark_config['robot_config'] = robot_config
        benchmark_config['test_scenarios']['random_sampling']['count'] = 50  # More poses for comparison
        
        # Create benchmark factory
        benchmark_factory = IKBenchmarkFactory(robot_factory, benchmark_config)
        
        print(f"Comparing methods: {comparison_methods}")
        
        # Run full benchmark
        report = benchmark_factory.run_full_benchmark(comparison_methods, ['random_sampling'])
        
        # Display results
        print("\n🏆 COMPARISON RESULTS:")
        print("-" * 50)
        
        for method_name, method_result in report.ik_methods_results.items():
            print(f"\n{method_name.upper()}:")
            
            if method_result.accuracy_result:
                acc = method_result.accuracy_result
                print(f"  Accuracy: {acc.convergence_rate:.2%} success rate")
                print(f"  Position error: {acc.mean_position_error:.2e} m")
                print(f"  Rotation error: {acc.mean_rotation_error:.2e} rad")
            
            if method_result.efficiency_result:
                eff = method_result.efficiency_result
                print(f"  Time: {eff.mean_solve_time:.4f}s average")
                print(f"  Convergence: {eff.convergence_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🎯 IK Benchmark Extended Solver Test")
    print("=" * 60)
    
    if not BENCHMARK_AVAILABLE:
        print(f"❌ Benchmark system not available: {_import_error}")
        return False
    
    # Test adapter availability
    adapters_available = test_adapter_availability()
    
    # Run basic test
    basic_success, available_methods = run_basic_test()
    
    if not basic_success:
        print("\n❌ Basic test failed - cannot proceed with extended tests")
        return False
    
    # Run extended tests if new solvers available
    if adapters_available:
        extended_success = run_extended_test(available_methods)
        comparison_success = run_comparison_test(available_methods)
        
        print("\n🏁 TEST SUMMARY")
        print("=" * 30)
        print(f"Basic test: {'✓' if basic_success else '✗'}")
        print(f"Extended test: {'✓' if extended_success else '✗'}")
        print(f"Comparison test: {'✓' if comparison_success else '✗'}")
        
        if basic_success and extended_success and comparison_success:
            print("\n🎉 All tests completed successfully!")
            print("✅ Pyroki and CuRobo adapters are working correctly")
            return True
        else:
            print("\n⚠️  Some tests failed or were skipped")
            return False
    else:
        print("\n⚠️  Extended solvers not available - only basic test completed")
        return basic_success


if __name__ == "__main__":
    # Configure logging
    log.setLevel(log.INFO)
    
    success = main()
    sys.exit(0 if success else 1)