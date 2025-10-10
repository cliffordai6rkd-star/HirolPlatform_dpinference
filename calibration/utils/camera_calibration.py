"""
Camera Intrinsic Calibration

Uses ChArUco board data collected during hand-eye calibration
to simultaneously calibrate camera intrinsic parameters.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import cv2.aruco as aruco
import glog as log


def calibrate_camera_intrinsics(
    charuco_corners_list: List[np.ndarray],
    charuco_ids_list: List[np.ndarray],
    charuco_board: aruco.CharucoBoard,
    image_size: Tuple[int, int],
    initial_camera_matrix: Optional[np.ndarray] = None,
    initial_dist_coeffs: Optional[np.ndarray] = None,
    flags: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calibrate camera intrinsic parameters using ChArUco board detections

    Args:
        charuco_corners_list: List of detected ChArUco corners for each image
                              Each element shape: (N, 1, 2) where N is number of corners
        charuco_ids_list: List of ChArUco corner IDs for each image
                          Each element shape: (N, 1) where N is number of corners
        charuco_board: ChArUco board object
        image_size: Image dimensions (width, height)
        initial_camera_matrix: Initial camera matrix guess (3x3), None for default
        initial_dist_coeffs: Initial distortion coefficients (5,), None for zero
        flags: Calibration flags (e.g., cv2.CALIB_FIX_ASPECT_RATIO)
               Default 0 = calibrate all parameters

    Returns:
        (camera_matrix, dist_coeffs, reprojection_error)
        - camera_matrix: Calibrated camera intrinsics (3x3)
        - dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3)
        - reprojection_error: RMS reprojection error in pixels

    Raises:
        ValueError: Insufficient data or invalid parameters
        RuntimeError: Calibration failed
    """
    if len(charuco_corners_list) != len(charuco_ids_list):
        raise ValueError(
            f"Data mismatch: {len(charuco_corners_list)} corners vs "
            f"{len(charuco_ids_list)} IDs"
        )

    if len(charuco_corners_list) < 3:
        raise ValueError(
            f"Insufficient images for calibration: {len(charuco_corners_list)} < 3"
        )

    # Filter out images with too few corners
    min_corners = 8
    filtered_corners = []
    filtered_ids = []

    for corners, ids in zip(charuco_corners_list, charuco_ids_list):
        if corners is not None and len(corners) >= min_corners:
            filtered_corners.append(corners)
            filtered_ids.append(ids)

    if len(filtered_corners) < 3:
        raise ValueError(
            f"Insufficient valid images after filtering: {len(filtered_corners)} < 3"
        )

    log.info(f"Using {len(filtered_corners)} images for camera calibration")
    log.info(f"  Image size: {image_size[0]}×{image_size[1]}")
    log.info(f"  Corners per image: min={min([len(c) for c in filtered_corners])}, "
             f"max={max([len(c) for c in filtered_corners])}, "
             f"mean={np.mean([len(c) for c in filtered_corners]):.1f}")

    # Initialize camera matrix if not provided
    if initial_camera_matrix is None:
        # Default: focal length = image width, principal point = image center
        fx = fy = image_size[0]
        cx = image_size[0] / 2
        cy = image_size[1] / 2
        initial_camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

    # Initialize distortion coefficients if not provided
    if initial_dist_coeffs is None:
        initial_dist_coeffs = np.zeros(5, dtype=np.float64)

    # Convert ChArUco detections to object points and image points for calibration
    # OpenCV 4.7+ compatible approach: use cv2.calibrateCamera instead of deprecated calibrateCameraCharuco
    object_points_list = []
    image_points_list = []

    for corners, ids in zip(filtered_corners, filtered_ids):
        # Get 3D object points for detected ChArUco corners
        try:
            # OpenCV 4.7+: use getChessboardCorners()
            all_obj_points = charuco_board.getChessboardCorners()
        except AttributeError:
            # Fallback for older OpenCV
            all_obj_points = charuco_board.chessboardCorners

        # Extract object points for detected corner IDs
        obj_points = all_obj_points[ids.flatten()]

        # Extract image points (reshape from (N, 1, 2) to (N, 2))
        img_points = corners.reshape(-1, 2).astype(np.float32)

        # Ensure proper shape: (N, 3) for object points, (N, 2) for image points
        object_points_list.append(obj_points.astype(np.float32))
        image_points_list.append(img_points)

    # Perform camera calibration using standard cv2.calibrateCamera
    try:
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=object_points_list,
            imagePoints=image_points_list,
            imageSize=image_size,
            cameraMatrix=initial_camera_matrix,
            distCoeffs=initial_dist_coeffs,
            flags=flags
        )
    except cv2.error as e:
        raise RuntimeError(f"Camera calibration failed: {e}") from e

    # retval is the RMS reprojection error
    reprojection_error = retval

    log.info(f"Camera calibration completed:")
    log.info(f"  RMS reprojection error: {reprojection_error:.3f} pixels")
    log.info(f"  fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}")
    log.info(f"  cx={camera_matrix[0, 2]:.2f}, cy={camera_matrix[1, 2]:.2f}")
    log.info(f"  Distortion coeffs: k1={dist_coeffs[0]:.6f}, k2={dist_coeffs[1]:.6f}, "
             f"p1={dist_coeffs[2]:.6f}, p2={dist_coeffs[3]:.6f}, k3={dist_coeffs[4]:.6f}")

    return camera_matrix, dist_coeffs, reprojection_error


def evaluate_calibration_quality(
    charuco_corners_list: List[np.ndarray],
    charuco_ids_list: List[np.ndarray],
    charuco_board: aruco.CharucoBoard,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Dict[str, any]:
    """
    Evaluate camera calibration quality

    Args:
        charuco_corners_list: List of detected ChArUco corners
        charuco_ids_list: List of ChArUco corner IDs
        charuco_board: ChArUco board object
        camera_matrix: Calibrated camera matrix (3x3)
        dist_coeffs: Distortion coefficients (5,)

    Returns:
        Dictionary with evaluation metrics:
        - per_image_errors: List of RMS errors for each image (pixels)
        - mean_error: Mean RMS error across all images
        - max_error: Maximum RMS error
        - good_image_ratio: Fraction of images with error < 1.0 pixel
    """
    per_image_errors = []

    for corners, ids in zip(charuco_corners_list, charuco_ids_list):
        if corners is None or len(corners) < 4:
            continue

        # Get 3D object points for detected corners
        try:
            obj_points = charuco_board.getChessboardCorners()[ids.flatten()]
        except AttributeError:
            # Fallback for older OpenCV
            obj_points = charuco_board.chessboardCorners[ids.flatten()]

        # Project points
        projected_points, _ = cv2.projectPoints(
            obj_points,
            np.zeros((3, 1)),  # Identity rotation
            np.zeros((3, 1)),  # Zero translation
            camera_matrix,
            dist_coeffs
        )

        # Compute reprojection error for this image
        errors = np.linalg.norm(
            corners.reshape(-1, 2) - projected_points.reshape(-1, 2),
            axis=1
        )
        rms_error = np.sqrt(np.mean(errors ** 2))
        per_image_errors.append(rms_error)

    if not per_image_errors:
        return {
            'per_image_errors': [],
            'mean_error': float('inf'),
            'max_error': float('inf'),
            'good_image_ratio': 0.0
        }

    mean_error = float(np.mean(per_image_errors))
    max_error = float(np.max(per_image_errors))
    good_image_ratio = float(np.sum(np.array(per_image_errors) < 1.0) / len(per_image_errors))

    return {
        'per_image_errors': per_image_errors,
        'mean_error': mean_error,
        'max_error': max_error,
        'good_image_ratio': good_image_ratio
    }


def compare_calibrations(
    initial_camera_matrix: np.ndarray,
    initial_dist_coeffs: np.ndarray,
    calibrated_camera_matrix: np.ndarray,
    calibrated_dist_coeffs: np.ndarray
) -> None:
    """
    Print comparison between initial and calibrated camera parameters

    Args:
        initial_camera_matrix: Initial camera matrix (3x3)
        initial_dist_coeffs: Initial distortion coeffs (5,)
        calibrated_camera_matrix: Calibrated camera matrix (3x3)
        calibrated_dist_coeffs: Calibrated distortion coeffs (5,)
    """
    log.info("=" * 70)
    log.info("Camera Calibration Comparison")
    log.info("=" * 70)

    log.info("Focal Length:")
    log.info(f"  Initial:    fx={initial_camera_matrix[0,0]:.2f}, fy={initial_camera_matrix[1,1]:.2f}")
    log.info(f"  Calibrated: fx={calibrated_camera_matrix[0,0]:.2f}, fy={calibrated_camera_matrix[1,1]:.2f}")
    log.info(f"  Change:     Δfx={(calibrated_camera_matrix[0,0]-initial_camera_matrix[0,0]):.2f}, "
             f"Δfy={(calibrated_camera_matrix[1,1]-initial_camera_matrix[1,1]):.2f}")

    log.info("\nPrincipal Point:")
    log.info(f"  Initial:    cx={initial_camera_matrix[0,2]:.2f}, cy={initial_camera_matrix[1,2]:.2f}")
    log.info(f"  Calibrated: cx={calibrated_camera_matrix[0,2]:.2f}, cy={calibrated_camera_matrix[1,2]:.2f}")
    log.info(f"  Change:     Δcx={(calibrated_camera_matrix[0,2]-initial_camera_matrix[0,2]):.2f}, "
             f"Δcy={(calibrated_camera_matrix[1,2]-initial_camera_matrix[1,2]):.2f}")

    log.info("\nDistortion Coefficients:")
    log.info(f"  Initial:    {initial_dist_coeffs.flatten()}")
    log.info(f"  Calibrated: {calibrated_dist_coeffs.flatten()}")
    log.info(f"  Change:     {(calibrated_dist_coeffs - initial_dist_coeffs).flatten()}")

    log.info("=" * 70)
