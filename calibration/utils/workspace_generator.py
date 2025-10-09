"""
Workspace Pose Generator

Generates calibration poses in robot workspace.
"""

from typing import List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_grid_poses(
    center: np.ndarray,
    grid_size: List[int],
    spacing: List[float],
    orientation_randomness: float = 30.0,
    base_orientation: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Generate 3D grid of poses for calibration

    Args:
        center: Grid center position [x, y, z] in base frame (meters)
        grid_size: Grid dimensions [nx, ny, nz]
        spacing: Grid spacing [dx, dy, dz] (meters)
        orientation_randomness: Random rotation range (degrees)
        base_orientation: Base orientation quaternion [qx, qy, qz, qw]
                         Default: [1, 0, 0, 0] (no rotation)

    Returns:
        List of 7D poses [x, y, z, qx, qy, qz, qw]

    Note:
        - Poses generated in Z->Y->X order (Z varies slowest)
        - Each pose has random orientation perturbation around base orientation
        - Good calibration requires:
          * Wide spatial distribution
          * Diverse orientations
          * ~15-30 samples minimum
    """
    if base_orientation is None:
        base_orientation = np.array([1, 0, 0, 0])  # Identity quaternion

    poses = []
    nx, ny, nz = grid_size
    dx, dy, dz = spacing

    # Compute starting point (lower-left-front corner)
    start_x = center[0] - (nx - 1) * dx / 2
    start_y = center[1] - (ny - 1) * dy / 2
    start_z = center[2] - (nz - 1) * dz / 2

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Position
                x = start_x + ix * dx
                y = start_y + iy * dy
                z = start_z + iz * dz

                # Orientation: base + random perturbation
                random_quat = random_quaternion_perturbation(
                    base_orientation,
                    max_angle_deg=orientation_randomness
                )

                pose = np.array([x, y, z, *random_quat])
                poses.append(pose)

    return poses


def random_quaternion_perturbation(
    base_quat: np.ndarray,
    max_angle_deg: float
) -> np.ndarray:
    """
    Apply random rotation perturbation to quaternion

    Args:
        base_quat: Base quaternion [qx, qy, qz, qw]
        max_angle_deg: Maximum rotation angle (degrees)

    Returns:
        Perturbed quaternion [qx, qy, qz, qw]

    Note:
        - Generates uniformly distributed rotation on SO(3)
        - Rotation axis is random
        - Rotation angle is uniform in [-max_angle, max_angle]
    """
    # Generate random axis (uniform on unit sphere)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    # Generate random angle
    angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180

    # Create random rotation
    random_rot = R.from_rotvec(axis * angle)
    base_rot = R.from_quat(base_quat)

    # Compose rotations: random * base
    final_rot = random_rot * base_rot

    return final_rot.as_quat()


def generate_hemisphere_poses(
    target_point: np.ndarray,
    radius: float,
    num_samples: int,
    min_elevation_deg: float = 20.0,
    max_elevation_deg: float = 80.0,
    orientation_randomness: float = 15.0
) -> List[np.ndarray]:
    """
    Generate poses on hemisphere looking at target point

    Useful for eye-to-hand calibration where camera is fixed and
    robot moves calibration board.

    Args:
        target_point: Point to look at [x, y, z] (meters)
        radius: Hemisphere radius (meters)
        num_samples: Number of poses to generate
        min_elevation_deg: Minimum elevation angle (degrees from horizontal)
        max_elevation_deg: Maximum elevation angle (degrees from horizontal)
        orientation_randomness: Additional rotation randomness (degrees)

    Returns:
        List of 7D poses [x, y, z, qx, qy, qz, qw]

    Note:
        - Poses are uniformly distributed on hemisphere
        - Each pose's Z-axis points toward target_point
        - Useful for achieving good observability in eye-to-hand setup
    """
    poses = []

    # Convert elevation angles to radians
    min_elev = np.deg2rad(min_elevation_deg)
    max_elev = np.deg2rad(max_elevation_deg)

    for _ in range(num_samples):
        # Generate random spherical coordinates
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(min_elev, max_elev)

        # Convert to Cartesian coordinates
        x = target_point[0] + radius * np.cos(elevation) * np.cos(azimuth)
        y = target_point[1] + radius * np.cos(elevation) * np.sin(azimuth)
        z = target_point[2] + radius * np.sin(elevation)

        position = np.array([x, y, z])

        # Compute orientation: Z-axis points toward target
        z_axis = target_point - position
        z_axis /= np.linalg.norm(z_axis)

        # Choose X-axis perpendicular to Z
        # Use world-up as reference, unless Z is aligned with it
        world_up = np.array([0, 0, 1])
        if np.abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([1, 0, 0])

        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # Y-axis completes right-handed frame
        y_axis = np.cross(z_axis, x_axis)

        # Build rotation matrix
        R_base = np.column_stack([x_axis, y_axis, z_axis])
        base_quat = R.from_matrix(R_base).as_quat()

        # Add random perturbation
        final_quat = random_quaternion_perturbation(
            base_quat, orientation_randomness
        )

        pose = np.array([*position, *final_quat])
        poses.append(pose)

    return poses
