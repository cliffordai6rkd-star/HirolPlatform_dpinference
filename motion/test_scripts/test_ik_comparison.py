"""
Test script to compare performance and success rate between ik and ik_pyroki methods.

This script generates random target poses and tests both inverse kinematics solvers
to compare their success rates, computation times, and solution accuracy.

Usage:
    python test_ik_comparison.py [--robot-config panda|franka] [--num-tests 100]
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# Add parent directory to path to import kinematics module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinematics import PinocchioKinematicsModel


class IKComparisonTest:
    """Test class for comparing IK solvers."""
    
    def __init__(self, urdf_path: str, base_link: str, end_effector_link: str):
        """Initialize the kinematics model for testing."""
        self.model = PinocchioKinematicsModel(urdf_path, base_link, end_effector_link)
        print(f"Initialized kinematics model with {self.model.n_joints} joints")
        print(f"End-effector frame: {self.model.ee_frame_name}")
        
        # Check if pyroki is available
        self.pyroki_available = hasattr(self.model, '_pyroki_robot') and self.model._pyroki_robot is not None
        print(f"PyRoki available: {self.pyroki_available}")
    
    def generate_reachable_poses(self, num_poses: int) -> List[np.ndarray]:
        """Generate reachable poses by using forward kinematics on random joint configurations."""
        poses = []
        
        for _ in range(num_poses):
            # Generate random joint configuration within limits
            q_random = np.random.uniform(
                self.model.joint_lower_limit,
                self.model.joint_upper_limit
            )
            
            # Compute forward kinematics to get reachable pose
            pose = self.model.fk(q_random)
            poses.append(pose)
            
        return poses
    
    def test_ik_method(self, method_name: str, poses: List[np.ndarray], 
                      tolerance: float = 1e-6) -> Dict:
        """Test a specific IK method on given poses."""
        results = {
            'method': method_name,
            'success_count': 0,
            'total_time': 0.0,
            'times': [],
            'errors': [],
            'solutions': []
        }
        
        method = getattr(self.model, method_name)
        
        for i, target_pose in enumerate(poses):
            if i % 50 == 0:
                print(f"  Testing {method_name}: {i}/{len(poses)}")
            
            
            try:
                start_time = time.time()
                converged, solution = method(target_pose, tol=tolerance)
                end_time = time.time()
                
                solve_time = end_time - start_time
                results['times'].append(solve_time)
                results['total_time'] += solve_time
                
                if converged:
                    results['success_count'] += 1
                    # Verify solution accuracy
                    achieved_pose = self.model.fk(solution)
                    error = np.linalg.norm(achieved_pose - target_pose)
                    results['errors'].append(error)
                    results['solutions'].append(solution)
                else:
                    results['errors'].append(float('inf'))
                    results['solutions'].append(solution)
                    
            except Exception as e:
                import traceback
                end_time = time.time()
                solve_time = end_time - start_time
                results['times'].append(solve_time)
                results['total_time'] += solve_time
                results['errors'].append(float('inf'))
                results['solutions'].append(np.zeros(self.model.n_joints))
                print(f"  Error in {method_name} for pose {i}: {e}")
                print(f"  Full traceback: {traceback.format_exc()}")
        
        return results
    
    def run_comparison(self, num_tests: int = 100, use_reachable_poses: bool = True) -> Dict:
        """Run comparison between ik and ik_pyroki methods."""
        print(f"\n=== IK Comparison Test ===")
        print(f"Number of test poses: {num_tests}")
        print(f"Using reachable poses: {use_reachable_poses}")
        
        # Generate test poses
        if use_reachable_poses:
            print("Generating reachable poses using forward kinematics...")
            poses = self.generate_reachable_poses(num_tests)
        else:
            print("Generating random poses in workspace...")
            workspace_bounds = {
                'x': [0.3, 0.8],
                'y': [-0.4, 0.4],
                'z': [0.1, 0.8]
            }
            poses = self.generate_random_poses(num_tests, workspace_bounds)
        
        results = {}
        
        # Test standard IK
        print("\nTesting standard IK method...")
        results['ik'] = self.test_ik_method('ik', poses)
        
        # Test PyRoki IK if available
        if self.pyroki_available:
            print("\nTesting PyRoki IK method...")
            results['ik_pyroki'] = self.test_ik_method('ik_pyroki', poses)
        else:
            print("\nSkipping PyRoki IK test (not available)")
            results['ik_pyroki'] = None
        
        return results
    
    def print_summary(self, results: Dict, num_tests: int):
        """Print comparison summary."""
        print(f"\n=== Results Summary ===")
        
        for method_name, result in results.items():
            if result is None:
                continue
                
            success_rate = (result['success_count'] / num_tests) * 100
            avg_time = result['total_time'] / num_tests * 1000  # Convert to ms
            
            successful_errors = [e for e in result['errors'] if e != float('inf')]
            avg_error = np.mean(successful_errors) if successful_errors else float('inf')
            max_error = np.max(successful_errors) if successful_errors else float('inf')
            
            print(f"\n{method_name.upper()}:")
            print(f"  Success rate: {success_rate:.1f}% ({result['success_count']}/{num_tests})")
            print(f"  Average time: {avg_time:.3f} ms")
            print(f"  Total time: {result['total_time']:.3f} s")
            if successful_errors:
                print(f"  Average error: {avg_error:.8f} ")
                print(f"  Max error: {max_error:.8f} ")
        
        # Comparison
        if results['ik_pyroki'] is not None:
            ik_success_rate = (results['ik']['success_count'] / num_tests) * 100
            pyroki_success_rate = (results['ik_pyroki']['success_count'] / num_tests) * 100
            
            ik_avg_time = results['ik']['total_time'] / num_tests * 1000
            pyroki_avg_time = results['ik_pyroki']['total_time'] / num_tests * 1000
            
            print(f"\n=== Comparison ===")
            print(f"Success rate improvement: {pyroki_success_rate - ik_success_rate:+.1f}%")
            print(f"Speed improvement: {(ik_avg_time / pyroki_avg_time - 1) * 100:+.1f}%")
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot comparison results."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('IK Methods Comparison', fontsize=16)
            
            methods = [name for name, result in results.items() if result is not None]
            colors = ['blue', 'red', 'green', 'orange']
            
            # Success rates
            success_rates = []
            for method in methods:
                rate = (results[method]['success_count'] / len(results[method]['times'])) * 100
                success_rates.append(rate)
            
            axes[0, 0].bar(methods, success_rates, color=colors[:len(methods)])
            axes[0, 0].set_title('Success Rate (%)')
            axes[0, 0].set_ylabel('Success Rate (%)')
            
            # Average computation time
            avg_times = []
            for method in methods:
                avg_time = np.mean(results[method]['times']) * 1000  # Convert to ms
                avg_times.append(avg_time)
            
            axes[0, 1].bar(methods, avg_times, color=colors[:len(methods)])
            axes[0, 1].set_title('Average Computation Time (ms)')
            axes[0, 1].set_ylabel('Time (ms)')
            
            # Time distribution
            for i, method in enumerate(methods):
                times_ms = np.array(results[method]['times']) * 1000
                axes[1, 0].hist(times_ms, bins=30, alpha=0.7, label=method, color=colors[i])
            axes[1, 0].set_title('Computation Time Distribution')
            axes[1, 0].set_xlabel('Time (ms)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            
            # Error distribution (for successful solutions only)
            for i, method in enumerate(methods):
                successful_errors = [e for e in results[method]['errors'] if e != float('inf')]
                if successful_errors:
                    axes[1, 1].hist(successful_errors, bins=30, alpha=0.7, label=method, color=colors[i])
            axes[1, 1].set_title('Position Error Distribution (Successful Solutions)')
            axes[1, 1].set_xlabel('Position Error (m)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            else:
                plt.savefig('/tmp/ik_comparison.png', dpi=300, bbox_inches='tight')
                print(f"Plot saved to: /tmp/ik_comparison.png")
            
        except ImportError:
            print("Matplotlib not available. Skipping plots.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare IK methods performance')
    parser.add_argument('--robot-config', choices=['panda', 'franka'], default='panda',
                       help='Robot configuration to test')
    parser.add_argument('--num-tests', type=int, default=100,
                       help='Number of test poses to generate')
    parser.add_argument('--random-poses', action='store_true',
                       help='Use random workspace poses instead of reachable poses (not recommended for rigorous testing)')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save comparison plot')
    
    args = parser.parse_args()
    
    # Robot configurations
    robot_configs = {
        'panda': {
            'urdf_path': 'assets/franka_fr3/fr3.urdf',
            'base_link': 'fr3_link0',
            'end_effector_link': 'fr3_hand'
        },
        'franka': {
            'urdf_path': 'assets/franka_fr3/fr3.urdf',
            'base_link': 'fr3_link0', 
            'end_effector_link': 'fr3_hand'
        }
    }
    
    config = robot_configs[args.robot_config]
    
    # Initialize test
    tester = IKComparisonTest(
        config['urdf_path'],
        config['base_link'],
        config['end_effector_link']
    )
    
    # Run comparison
    results = tester.run_comparison(
        num_tests=args.num_tests,
        use_reachable_poses=not args.random_poses
    )
    
    # Print results
    tester.print_summary(results, args.num_tests)
    
    # Plot results
    if args.save_plot:
        tester.plot_results(results, args.save_plot)
    else:
        tester.plot_results(results)


if __name__ == "__main__":
    main()