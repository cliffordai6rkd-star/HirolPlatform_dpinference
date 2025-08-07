#!/usr/bin/env python3
"""
Test script to benchmark the speed of ik_pyroki with random poses.

This script generates random target poses and tests the ik_pyroki method
to measure its computation time and success rate.

Usage:
    python test_ik_pyroki_speed.py [--robot-config panda|franka] [--num-tests 1000]
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import statistics

# Add parent directory to path to import kinematics module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinematics import PinocchioKinematicsModel


class IKPyrokiSpeedTest:
    """Test class for benchmarking ik_pyroki speed."""
    
    def __init__(self, urdf_path: str, base_link: str, end_effector_link: str):
        """Initialize the kinematics model for testing."""
        self.model = PinocchioKinematicsModel(urdf_path, base_link, end_effector_link)
        print(f"Initialized kinematics model with {self.model.n_joints} joints")
        print(f"End-effector frame: {self.model.ee_frame_name}")
        
        # Check if pyroki is available
        self.pyroki_available = hasattr(self.model, '_pyroki_robot') and self.model._pyroki_robot is not None
        if not self.pyroki_available:
            raise RuntimeError("PyRoki is not available. Please install pyroki and pyroki_snippets.")
        
        print("PyRoki is available and initialized")
    
    def generate_random_poses(self, num_poses: int, workspace_bounds: Dict = None) -> List[np.ndarray]:
        """Generate random poses within workspace bounds."""
        if workspace_bounds is None:
            # Default workspace bounds for Franka robot
            workspace_bounds = {
                'x': [0.2, 0.8],
                'y': [-0.4, 0.4], 
                'z': [0.1, 0.8]
            }
        
        poses = []
        for _ in range(num_poses):
            # Generate random position within workspace
            position = np.array([
                np.random.uniform(workspace_bounds['x'][0], workspace_bounds['x'][1]),
                np.random.uniform(workspace_bounds['y'][0], workspace_bounds['y'][1]),
                np.random.uniform(workspace_bounds['z'][0], workspace_bounds['z'][1])
            ])
            
            # Generate random orientation (random quaternion)
            # Method: generate random unit quaternion
            u1, u2, u3 = np.random.random(3)
            q = np.array([
                np.sqrt(1-u1) * np.sin(2*np.pi*u2),
                np.sqrt(1-u1) * np.cos(2*np.pi*u2),
                np.sqrt(u1) * np.sin(2*np.pi*u3),
                np.sqrt(u1) * np.cos(2*np.pi*u3)
            ])
            
            # Convert quaternion to rotation matrix
            qw, qx, qy, qz = q
            R = np.array([
                [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
            ])
            
            # Create homogeneous transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = position
            poses.append(pose)
            
        return poses
    
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
    
    def benchmark_ik_pyroki(self, poses: List[np.ndarray], tolerance: float = 1e-6) -> Dict:
        """Benchmark ik_pyroki method on given poses."""
        results = {
            'success_count': 0,
            'total_time': 0.0,
            'times': [],
            'errors': [],
            'solutions': [],
            'failed_poses': []
        }
        
        print(f"Benchmarking ik_pyroki on {len(poses)} poses...")
        
        for i, target_pose in enumerate(poses):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(poses)}")
            
            try:
                start_time = time.time()
                converged, solution = self.model.ik_pyroki(target_pose, tol=tolerance)
                end_time = time.time()
                
                solve_time = end_time - start_time
                results['times'].append(solve_time)
                results['total_time'] += solve_time
                
                if converged:
                    results['success_count'] += 1
                    # Verify solution accuracy using forward kinematics
                    achieved_pose = self.model.fk(solution[:self.model.n_joints])
                    error = np.linalg.norm(achieved_pose - target_pose)
                    results['errors'].append(error)
                    results['solutions'].append(solution)
                else:
                    results['errors'].append(float('inf'))
                    results['solutions'].append(solution)
                    results['failed_poses'].append(i)
                    
            except Exception as e:
                end_time = time.time()
                solve_time = end_time - start_time
                results['times'].append(solve_time)
                results['total_time'] += solve_time
                results['errors'].append(float('inf'))
                results['solutions'].append(np.zeros(self.model.n_joints))
                results['failed_poses'].append(i)
                print(f"  Error for pose {i}: {e}")
        
        return results
    
    def print_detailed_statistics(self, results: Dict, num_tests: int):
        """Print detailed statistics about the benchmark results."""
        print(f"\n=== Detailed ik_pyroki Benchmark Results ===")
        print(f"Total poses tested: {num_tests}")
        print(f"Successful solutions: {results['success_count']}")
        print(f"Failed solutions: {num_tests - results['success_count']}")
        print(f"Success rate: {(results['success_count'] / num_tests) * 100:.2f}%")
        
        if results['times']:
            times_ms = np.array(results['times']) * 1000  # Convert to milliseconds
            print(f"\n=== Timing Statistics ===")
            print(f"Total computation time: {results['total_time']:.3f} seconds")
            print(f"Average time per pose: {np.mean(times_ms):.3f} ms")
            print(f"Median time: {np.median(times_ms):.3f} ms")
            print(f"Min time: {np.min(times_ms):.3f} ms")
            print(f"Max time: {np.max(times_ms):.3f} ms")
            print(f"Standard deviation: {np.std(times_ms):.3f} ms")
            
            # Percentiles
            print(f"\n=== Time Percentiles ===")
            for p in [25, 50, 75, 90, 95, 99]:
                print(f"{p}th percentile: {np.percentile(times_ms, p):.3f} ms")
        
        # Error statistics for successful solutions
        successful_errors = [e for e in results['errors'] if e != float('inf')]
        if successful_errors:
            print(f"\n=== Error Statistics (Successful Solutions) ===")
            print(f"Average error: {np.mean(successful_errors):.8f} m")
            print(f"Median error: {np.median(successful_errors):.8f} m")
            print(f"Max error: {np.max(successful_errors):.8f} m")
            print(f"Min error: {np.min(successful_errors):.8f} m")
            print(f"Standard deviation: {np.std(successful_errors):.8f} m")
        
        # Speed analysis
        if results['times']:
            poses_per_second = num_tests / results['total_time']
            print(f"\n=== Speed Analysis ===")
            print(f"Poses solved per second: {poses_per_second:.2f}")
            print(f"Frequency: {1000/np.mean(times_ms):.2f} Hz")
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot benchmark results."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ik_pyroki Speed Benchmark Results', fontsize=16)
            
            # Convert times to milliseconds
            times_ms = np.array(results['times']) * 1000
            
            # 1. Time distribution histogram
            axes[0, 0].hist(times_ms, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Computation Time Distribution')
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(times_ms), color='red', linestyle='--', label=f'Mean: {np.mean(times_ms):.2f} ms')
            axes[0, 0].axvline(np.median(times_ms), color='green', linestyle='--', label=f'Median: {np.median(times_ms):.2f} ms')
            axes[0, 0].legend()
            
            # 2. Time series plot
            axes[0, 1].plot(times_ms, alpha=0.7, color='blue')
            axes[0, 1].set_title('Computation Time Over Test Sequence')
            axes[0, 1].set_xlabel('Test Number')
            axes[0, 1].set_ylabel('Time (ms)')
            axes[0, 1].axhline(np.mean(times_ms), color='red', linestyle='--', label=f'Mean: {np.mean(times_ms):.2f} ms')
            axes[0, 1].legend()
            
            # 3. Box plot of times
            axes[1, 0].boxplot(times_ms, vert=True)
            axes[1, 0].set_title('Computation Time Box Plot')
            axes[1, 0].set_ylabel('Time (ms)')
            axes[1, 0].set_xticklabels(['ik_pyroki'])
            
            # 4. Error distribution for successful solutions
            successful_errors = [e for e in results['errors'] if e != float('inf')]
            if successful_errors:
                axes[1, 1].hist(successful_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
                axes[1, 1].set_title('Position Error Distribution (Successful Solutions)')
                axes[1, 1].set_xlabel('Position Error (m)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(np.mean(successful_errors), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(successful_errors):.2e} m')
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'No successful solutions', ha='center', va='center', 
                               transform=axes[1, 1].transAxes, fontsize=16)
                axes[1, 1].set_title('Position Error Distribution')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            else:
                plt.savefig('/tmp/ik_pyroki_benchmark.png', dpi=300, bbox_inches='tight')
                print(f"Plot saved to: /tmp/ik_pyroki_benchmark.png")
            
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
    
    def run_benchmark(self, num_tests: int = 1000, use_reachable_poses: bool = True, 
                     workspace_bounds: Dict = None) -> Dict:
        """Run the full benchmark."""
        print(f"\n=== ik_pyroki Speed Benchmark ===")
        print(f"Number of test poses: {num_tests}")
        print(f"Using reachable poses: {use_reachable_poses}")
        
        # Generate test poses
        if use_reachable_poses:
            print("Generating reachable poses using forward kinematics...")
            poses = self.generate_reachable_poses(num_tests)
        else:
            print("Generating random poses in workspace...")
            poses = self.generate_random_poses(num_tests, workspace_bounds)
        
        # Run benchmark
        results = self.benchmark_ik_pyroki(poses)
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Benchmark ik_pyroki speed with random poses')
    parser.add_argument('--robot-config', choices=['panda', 'franka'], default='panda',
                       help='Robot configuration to test')
    parser.add_argument('--num-tests', type=int, default=1000,
                       help='Number of test poses to generate')
    parser.add_argument('--random-poses', action='store_true',
                       help='Use random workspace poses instead of reachable poses')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save benchmark plot')
    parser.add_argument('--workspace-x', nargs=2, type=float, default=[0.2, 0.8],
                       help='Workspace x bounds [min, max]')
    parser.add_argument('--workspace-y', nargs=2, type=float, default=[-0.4, 0.4],
                       help='Workspace y bounds [min, max]')
    parser.add_argument('--workspace-z', nargs=2, type=float, default=[0.1, 0.8],
                       help='Workspace z bounds [min, max]')
    
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
    
    # Workspace bounds
    workspace_bounds = {
        'x': args.workspace_x,
        'y': args.workspace_y,
        'z': args.workspace_z
    }
    
    # Initialize test
    tester = IKPyrokiSpeedTest(
        config['urdf_path'],
        config['base_link'],
        config['end_effector_link']
    )
    
    # Run benchmark
    results = tester.run_benchmark(
        num_tests=args.num_tests,
        use_reachable_poses=not args.random_poses,
        workspace_bounds=workspace_bounds if args.random_poses else None
    )
    
    # Print results
    tester.print_detailed_statistics(results, args.num_tests)
    
    # Plot results
    tester.plot_results(results, args.save_plot)


if __name__ == "__main__":
    main()