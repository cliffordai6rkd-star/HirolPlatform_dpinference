"""
Solvability metric evaluation for IK algorithms.

Evaluates convergence rate, workspace coverage, and singularity avoidance
of IK algorithms.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import glog as log


@dataclass
class SolvabilityResult:
    """Results from solvability evaluation."""
    convergence_rate: float
    workspace_coverage: float
    singular_avoidance_rate: float  
    reachable_poses: int
    unreachable_poses: int
    workspace_heatmap: Optional[np.ndarray] = None


class SolvabilityMetric:
    """Evaluates solvability characteristics of IK algorithms."""
    
    def __init__(self, workspace_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Initialize solvability metric evaluator.
        
        Args:
            workspace_bounds: Tuple of (min_bounds, max_bounds) for workspace
        """
        self._workspace_bounds = workspace_bounds
    
    def evaluate(self, ik_results: List[Tuple[bool, np.ndarray]], 
                test_poses: List[np.ndarray],
                grid_resolution: int = 50) -> SolvabilityResult:
        """
        Evaluate solvability of IK algorithm.
        
        Args:
            ik_results: List of (converged_flag, solution) tuples
            test_poses: List of target 4x4 transformation matrices
            grid_resolution: Resolution for workspace coverage grid
            
        Returns:
            SolvabilityResult containing solvability metrics
        """
        assert len(ik_results) == len(test_poses), \
            "IK results and test poses must have same length"
        
        # Count converged solutions
        converged_flags = [result[0] for result in ik_results]
        converged_count = sum(converged_flags)
        total_count = len(ik_results)
        
        convergence_rate = converged_count / total_count if total_count > 0 else 0.0
        
        # Compute workspace coverage
        workspace_coverage = self._compute_workspace_coverage(
            test_poses, converged_flags, grid_resolution
        )
        
        # Compute singularity avoidance (simplified metric)
        singular_avoidance_rate = self._estimate_singular_avoidance(
            test_poses, converged_flags
        )
        
        # Generate workspace heatmap
        heatmap = self._generate_workspace_heatmap(
            test_poses, converged_flags, grid_resolution
        )
        
        log.info(f"Solvability evaluation: {converged_count}/{total_count} "
                f"converged ({convergence_rate:.2%})")
        
        return SolvabilityResult(
            convergence_rate=convergence_rate,
            workspace_coverage=workspace_coverage,
            singular_avoidance_rate=singular_avoidance_rate,
            reachable_poses=converged_count,
            unreachable_poses=total_count - converged_count,
            workspace_heatmap=heatmap
        )
    
    def _compute_workspace_coverage(self, poses: List[np.ndarray], 
                                  converged_flags: List[bool],
                                  grid_resolution: int) -> float:
        """
        Compute workspace coverage as ratio of reachable workspace volume.
        
        Args:
            poses: List of target poses
            converged_flags: Convergence flags for each pose
            grid_resolution: Grid resolution for discretization
            
        Returns:
            Workspace coverage ratio (0.0 to 1.0)
        """
        if not poses:
            return 0.0
        
        # Extract positions from poses
        positions = np.array([pose[:3, 3] for pose in poses])
        
        # Determine workspace bounds
        if self._workspace_bounds is None:
            min_bounds = np.min(positions, axis=0) - 0.1
            max_bounds = np.max(positions, axis=0) + 0.1
        else:
            min_bounds, max_bounds = self._workspace_bounds
        
        # Create 3D grid
        x = np.linspace(min_bounds[0], max_bounds[0], grid_resolution)
        y = np.linspace(min_bounds[1], max_bounds[1], grid_resolution)
        z = np.linspace(min_bounds[2], max_bounds[2], grid_resolution)
        
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        
        # Count tested and reachable grid cells
        tested_cells = set()
        reachable_cells = set()
        
        for i, (pos, converged) in enumerate(zip(positions, converged_flags)):
            # Find closest grid cell
            distances = np.linalg.norm(grid_points - pos, axis=1)
            closest_idx = np.argmin(distances)
            
            cell_coord = tuple(grid_points[closest_idx])
            tested_cells.add(cell_coord)
            
            if converged:
                reachable_cells.add(cell_coord)
        
        coverage = len(reachable_cells) / len(tested_cells) if tested_cells else 0.0
        return coverage
    
    def _estimate_singular_avoidance(self, poses: List[np.ndarray],
                                   converged_flags: List[bool]) -> float:
        """
        Estimate singularity avoidance rate.
        
        This is a simplified metric based on convergence near workspace boundaries.
        A more sophisticated implementation would analyze Jacobian condition numbers.
        
        Args:
            poses: List of target poses
            converged_flags: Convergence flags
            
        Returns:
            Estimated singularity avoidance rate
        """
        if not poses:
            return 0.0
        
        positions = np.array([pose[:3, 3] for pose in poses])
        
        # Identify boundary positions (simplified approach)
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        boundary_threshold = np.percentile(distances, 80)  # Top 20% furthest points
        
        boundary_mask = distances >= boundary_threshold
        boundary_convergence = np.array(converged_flags)[boundary_mask]
        
        if len(boundary_convergence) == 0:
            return 1.0
        
        # Singularity avoidance rate as convergence rate near boundaries
        return np.mean(boundary_convergence)
    
    def _generate_workspace_heatmap(self, poses: List[np.ndarray],
                                  converged_flags: List[bool],
                                  resolution: int) -> np.ndarray:
        """
        Generate 2D heatmap of workspace reachability (XY projection).
        
        Args:
            poses: List of target poses
            converged_flags: Convergence flags
            resolution: Heatmap resolution
            
        Returns:
            2D numpy array representing reachability heatmap
        """
        if not poses:
            return np.zeros((resolution, resolution))
        
        positions = np.array([pose[:3, 3] for pose in poses])
        
        # Project to XY plane
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]
        
        # Create 2D histogram
        x_range = [np.min(x_pos) - 0.05, np.max(x_pos) + 0.05]
        y_range = [np.min(y_pos) - 0.05, np.max(y_pos) + 0.05]
        
        # Count total attempts in each bin
        total_hist, x_edges, y_edges = np.histogram2d(
            x_pos, y_pos, bins=resolution, range=[x_range, y_range]
        )
        
        # Count successful attempts in each bin
        success_x = x_pos[converged_flags] if any(converged_flags) else []
        success_y = y_pos[converged_flags] if any(converged_flags) else []
        
        if len(success_x) > 0:
            success_hist, _, _ = np.histogram2d(
                success_x, success_y, bins=[x_edges, y_edges]
            )
        else:
            success_hist = np.zeros_like(total_hist)
        
        # Compute success rate heatmap
        heatmap = np.divide(success_hist, total_hist, 
                           out=np.zeros_like(success_hist), 
                           where=total_hist != 0)
        
        return heatmap.T  # Transpose for proper orientation