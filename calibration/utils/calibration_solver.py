"""
Hand-Eye Calibration Solver

Solves AX=XB problem for hand-eye calibration.
"""

from typing import List, Dict, Tuple
import numpy as np
import cv2
import glog as log


def solve_eye_in_hand(
    T_base_ee_list: List[np.ndarray],
    T_camera_board_list: List[np.ndarray],
    method: str = 'Tsai'
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve eye-in-hand calibration (AX = XB)

    Camera is mounted on robot end-effector. Solves for transformation
    from end-effector to camera (T_ee_camera).

    Args:
        T_base_ee_list: List of end-effector poses in base frame (4x4)
        T_camera_board_list: List of board poses in camera frame (4x4)
        method: Solver method, one of:
            - 'Tsai': Tsai-Lenz method (default, robust)
            - 'Park': Park-Martin method
            - 'Horaud': Horaud method
            - 'Andreff': Andreff method
            - 'Daniilidis': Daniilidis method (quaternion-based)

    Returns:
        (T_ee_camera, diagnostics)
        - T_ee_camera: Transformation from end-effector to camera (4x4)
        - diagnostics: Quality metrics
          * condition_number: Data conditioning (lower is better)
          * mean_residual: Mean residual error (meters)

    Raises:
        ValueError: Input data size mismatch or invalid dimension
        RuntimeError: Solver failed

    Note:
        OpenCV's calibrateHandEye solves:
            R_gripper2base[i] * R_cam2gripper * R_target2cam[i] = R_cam2gripper * R_gripper2base[i]
        Returns R_cam2gripper, t_cam2gripper
        We invert to get T_ee_camera (T_gripper2cam)
    """
    if len(T_base_ee_list) != len(T_camera_board_list):
        raise ValueError(
            f"Data size mismatch: {len(T_base_ee_list)} vs {len(T_camera_board_list)}"
        )

    if len(T_base_ee_list) < 3:
        raise ValueError(f"Insufficient samples: {len(T_base_ee_list)} < 3")

    # Extract rotation and translation
    R_gripper2base = [T[:3, :3] for T in T_base_ee_list]
    t_gripper2base = [T[:3, 3:4] for T in T_base_ee_list]  # shape (3, 1)

    R_target2cam = [T[:3, :3] for T in T_camera_board_list]
    t_target2cam = [T[:3, 3:4] for T in T_camera_board_list]

    # Select solver method
    method_map = {
        'Tsai': cv2.CALIB_HAND_EYE_TSAI,
        'Park': cv2.CALIB_HAND_EYE_PARK,
        'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown method: {method}. Choose from {list(method_map.keys())}"
        )

    # Call OpenCV solver
    try:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method_map[method]
        )
    except cv2.error as e:
        raise RuntimeError(f"OpenCV calibrateHandEye failed: {e}") from e

    # Build T_cam2gripper  cam coordinate in gripper frame
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

    # Invert to get T_ee_camera (T_gripper2cam)
    T_ee_camera = np.linalg.inv(T_cam2gripper)

    # Compute diagnostics
    diagnostics = compute_diagnostics(
        T_base_ee_list, T_camera_board_list, T_ee_camera, 'eye_in_hand'
    )

    log.info(f"Eye-in-hand calibration completed using {method} method")
    log.info(f"  Condition number: {diagnostics['condition_number']:.2f}")
    log.info(f"  Mean residual: {diagnostics['mean_residual']:.4f} m")

    return T_ee_camera, diagnostics


def solve_eye_to_hand(
    T_base_ee_list: List[np.ndarray],
    T_camera_board_list: List[np.ndarray],
    method: str = 'Tsai'
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve eye-to-hand calibration (AX = ZB)

    Camera is fixed in world. Solves for transformation from base to camera
    (T_base_camera).

    Args:
        T_base_ee_list: List of end-effector poses in base frame (4x4)
        T_camera_board_list: List of board poses in camera frame (4x4)
        method: Solver method (same options as solve_eye_in_hand)

    Returns:
        (T_base_camera, diagnostics)
        - T_base_camera: Transformation from robot base to camera (4x4)
        - diagnostics: Quality metrics

    Raises:
        ValueError: Input data validation failed
        RuntimeError: Solver failed

    Note:
        OpenCV's calibrateRobotWorldHandEye solves:
            R_base2world * R_world2cam = R_gripper2base[i] * R_cam2gripper
        We use it to find T_base_camera (T_world2cam with world=base)
    """
    if len(T_base_ee_list) != len(T_camera_board_list):
        raise ValueError(
            f"Data size mismatch: {len(T_base_ee_list)} vs {len(T_camera_board_list)}"
        )

    if len(T_base_ee_list) < 3:
        raise ValueError(f"Insufficient samples: {len(T_base_ee_list)} < 3")

    # Extract rotation and translation
    R_gripper2base = [T[:3, :3] for T in T_base_ee_list]
    t_gripper2base = [T[:3, 3:4] for T in T_base_ee_list]

    R_target2cam = [T[:3, :3] for T in T_camera_board_list]
    t_target2cam = [T[:3, 3:4] for T in T_camera_board_list]

    # Select solver method
    method_map = {
        'Tsai': cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,  # Most stable for eye-to-hand
        'Park': cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        'Horaud': cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        'Andreff': cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        'Daniilidis': cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
    }

    # Call OpenCV solver
    try:
        R_base2cam, t_base2cam, R_gripper2board, t_gripper2board = cv2.calibrateRobotWorldHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method_map[method]
        )
    except cv2.error as e:
        raise RuntimeError(f"OpenCV calibrateRobotWorldHandEye failed: {e}") from e

    # Build T_base_camera
    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = R_base2cam
    T_base_camera[:3, 3] = t_base2cam.flatten()

    # Compute diagnostics
    diagnostics = compute_diagnostics(
        T_base_ee_list, T_camera_board_list, T_base_camera, 'eye_to_hand'
    )

    log.info(f"Eye-to-hand calibration completed using {method} method")
    log.info(f"  Condition number: {diagnostics['condition_number']:.2f}")
    log.info(f"  Mean residual: {diagnostics['mean_residual']:.4f} m")

    return T_base_camera, diagnostics


def compute_diagnostics(
    T_base_ee_list: List[np.ndarray],
    T_camera_board_list: List[np.ndarray],
    T_result: np.ndarray,
    calibration_type: str
) -> Dict[str, float]:
    """
    Compute calibration quality diagnostics

    Args:
        T_base_ee_list: End-effector poses
        T_camera_board_list: Board poses in camera frame
        T_result: Calibration result
            - eye_in_hand: T_ee_camera
            - eye_to_hand: T_base_camera
        calibration_type: 'eye_in_hand' or 'eye_to_hand'

    Returns:
        Dictionary with metrics:
        - condition_number: Motion conditioning (lower is better, <50 is good)
        - mean_residual: Mean position residual (meters)
    """
    # Compute condition number (simplified: based on position range)
    positions = np.array([T[:3, 3] for T in T_base_ee_list])
    pos_range = np.ptp(positions, axis=0)  # [x_range, y_range, z_range]

    # Condition number: reciprocal of minimum range
    # Well-conditioned: wide range in all directions
    min_range = np.min(pos_range)
    condition_number = 1.0 / (min_range + 1e-6)  # Avoid division by zero

    # Compute residuals
    residuals = []
    for T_base_ee, T_camera_board in zip(T_base_ee_list, T_camera_board_list):
        if calibration_type == 'eye_in_hand':
            # Predicted board pose in base frame:
            # T_base_board = T_base_ee * T_ee_camera * T_camera_board
            T_base_board = T_base_ee @ T_result @ T_camera_board
        else:  # eye_to_hand
            # Predicted board pose in base frame:
            # T_base_board = T_base_camera * T_camera_board
            T_base_board = T_result @ T_camera_board

        # Use first sample as reference
        if len(residuals) == 0:
            T_base_board_ref = T_base_board.copy()

        # Compute position error relative to reference
        pos_error = np.linalg.norm(T_base_board[:3, 3] - T_base_board_ref[:3, 3])
        residuals.append(pos_error)

    # Skip first residual (it's always 0 as reference)
    mean_residual = np.mean(residuals[1:]) if len(residuals) > 1 else 0.0

    return {
        'condition_number': float(condition_number),
        'mean_residual': float(mean_residual)
    }


def verify_calibration(
    T_base_ee_list: List[np.ndarray],
    T_camera_board_list: List[np.ndarray],
    T_result: np.ndarray,
    calibration_type: str
) -> Dict[str, any]:
    """
    Verify calibration by computing per-sample errors

    Args:
        T_base_ee_list: End-effector poses
        T_camera_board_list: Board poses
        T_result: Calibration result
        calibration_type: 'eye_in_hand' or 'eye_to_hand'

    Returns:
        Dictionary with verification results:
        - position_errors: List of position errors (meters)
        - rotation_errors: List of rotation errors (degrees)
        - mean_position_error: Mean position error
        - std_position_error: Standard deviation
        - max_position_error: Maximum error
    """
    position_errors = []
    rotation_errors = []

    # Use first sample as reference
    T_base_ee_ref = T_base_ee_list[0]
    T_camera_board_ref = T_camera_board_list[0]

    for T_base_ee, T_camera_board in zip(T_base_ee_list[1:], T_camera_board_list[1:]):
        if calibration_type == 'eye_in_hand':
            # Predicted relative motion
            T_base_board = T_base_ee @ T_result @ T_camera_board
            T_base_board_ref_pred = T_base_ee_ref @ T_result @ T_camera_board_ref
        else:
            T_base_board = T_result @ T_camera_board
            T_base_board_ref_pred = T_result @ T_camera_board_ref

        # Position error
        pos_error = np.linalg.norm(T_base_board[:3, 3] - T_base_board_ref_pred[:3, 3])
        position_errors.append(pos_error)

        # Rotation error (angle between rotation matrices)
        R_error = T_base_board[:3, :3].T @ T_base_board_ref_pred[:3, :3]
        angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        rotation_errors.append(np.rad2deg(angle_error))

    return {
        'position_errors': position_errors,
        'rotation_errors': rotation_errors,
        'mean_position_error': float(np.mean(position_errors)),
        'std_position_error': float(np.std(position_errors)),
        'max_position_error': float(np.max(position_errors)),
        'mean_rotation_error': float(np.mean(rotation_errors)),
        'max_rotation_error': float(np.max(rotation_errors))
    }
