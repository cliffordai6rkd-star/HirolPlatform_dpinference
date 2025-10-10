"""
ArUco Marker Detector

Detects ArUco markers and estimates their pose.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import cv2
import cv2.aruco as aruco
import glog as log

from .base_detector import BoardDetectorBase


class ArucoDetector(BoardDetectorBase):
    """
    ArUco marker detector

    Detects single ArUco markers for simple calibration scenarios.
    Use ChArUco for better accuracy.
    """

    def __init__(self, board_config: Dict, intrinsics: Dict):
        """
        Initialize ArUco detector

        Args:
            board_config: Board configuration
                - marker_length: float (meters)
                - aruco_dict: str (e.g., 'DICT_5X5_250')
                - marker_id: int (optional, specific marker to detect)
            intrinsics: Camera intrinsics
        """
        super().__init__(board_config, intrinsics)

        # Marker parameters
        self._marker_length = board_config['marker_length']
        self._target_marker_id = board_config.get('marker_id', None)

        # Create ArUco dictionary
        dict_name = board_config['aruco_dict']
        dict_attr = getattr(aruco, dict_name, None)
        if dict_attr is None:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        self._aruco_dict = aruco.getPredefinedDictionary(dict_attr)

        # Create detector
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._detector = aruco.ArucoDetector(self._aruco_dict, detector_params)

        log.info(f"ArucoDetector initialized: marker_length={self._marker_length}m, "
                f"target_id={self._target_marker_id}")

    def detect(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Detect ArUco marker and estimate pose

        Args:
            image: Input BGR image

        Returns:
            (success, T_camera_board, reprojection_error)

        Note:
            If target_marker_id is specified, only that marker is used.
            Otherwise, uses the first detected marker.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return False, None, float('inf')

        # Find target marker
        marker_idx = 0
        if self._target_marker_id is not None:
            # Search for specific marker
            marker_indices = np.where(ids.flatten() == self._target_marker_id)[0]
            if len(marker_indices) == 0:
                return False, None, float('inf')
            marker_idx = marker_indices[0]

        # Estimate pose for selected marker
        rvecs, tvecs, obj_points = aruco.estimatePoseSingleMarkers(
            [corners[marker_idx]],
            self._marker_length,
            self._camera_matrix,
            self._dist_coeffs
        )

        rvec = rvecs[0]
        tvec = tvecs[0]

        # Compute reprojection error
        reproj_error = self._compute_reprojection_error(
            corners[marker_idx], rvec, tvec
        )

        # Check reprojection error threshold
        if reproj_error > self._max_reprojection_error:
            log.warning(f"Reprojection error {reproj_error:.2f}px exceeds "
                       f"threshold {self._max_reprojection_error}px")
            return False, None, reproj_error

        # Convert to homogeneous transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T_camera_board = np.eye(4)
        T_camera_board[:3, :3] = R
        T_camera_board[:3, 3] = tvec.flatten()

        return True, T_camera_board, reproj_error

    def _compute_reprojection_error(self,
                                    corners: np.ndarray,
                                    rvec: np.ndarray,
                                    tvec: np.ndarray) -> float:
        """
        Compute reprojection error for detected marker

        Args:
            corners: Detected corner positions (4 corners)
            rvec: Rotation vector
            tvec: Translation vector

        Returns:
            Mean reprojection error in pixels
        """
        # 3D object points for marker corners (in marker frame)
        half_size = self._marker_length / 2
        obj_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        # Project 3D points to image
        projected_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec,
            self._camera_matrix, self._dist_coeffs
        )

        # Compute error
        errors = np.linalg.norm(
            corners.reshape(-1, 2) - projected_points.reshape(-1, 2),
            axis=1
        )
        mean_error = np.mean(errors)

        return float(mean_error)

    def draw_detection(self, image: np.ndarray,
                       T_camera_board: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw detection results on image

        Args:
            image: Input BGR image
            T_camera_board: Detected pose (None if detection failed)

        Returns:
            Image with visualization
        """
        display_image = image.copy()

        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = self._detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Draw detected markers
            aruco.drawDetectedMarkers(display_image, corners, ids)

        # Draw coordinate frame if pose is valid
        if T_camera_board is not None:
            R = T_camera_board[:3, :3]
            t = T_camera_board[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(3, 1)

            # Draw axes (2x marker size)
            axis_length = self._marker_length * 2
            cv2.drawFrameAxes(
                display_image, self._camera_matrix, self._dist_coeffs,
                rvec, tvec, axis_length
            )

            # Display translation
            pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
            cv2.putText(display_image, pose_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return display_image
