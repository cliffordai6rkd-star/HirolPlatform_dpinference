"""
Utilities for Hand-Eye Calibration

Provides helper functions for:
- Workspace pose generation
- Calibration solving (AX=XB)
- Data management
"""

from .workspace_generator import generate_grid_poses, random_quaternion_perturbation
from .calibration_solver import solve_eye_in_hand, solve_eye_to_hand, compute_diagnostics, verify_calibration
from .data_manager import DataManager

__all__ = [
    'generate_grid_poses',
    'random_quaternion_perturbation',
    'solve_eye_in_hand',
    'solve_eye_to_hand',
    'compute_diagnostics',
    'verify_calibration',
    'DataManager',
]
