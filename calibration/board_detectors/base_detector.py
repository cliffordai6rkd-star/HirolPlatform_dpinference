"""
Base Detector for Calibration Boards

Abstract interface for calibration board detection.
"""

import abc
from typing import Dict, Optional, Tuple
import numpy as np


class BoardDetectorBase(abc.ABC):
    """
    Abstract base class for calibration board detectors

    Provides unified interface for detecting calibration boards (ChArUco, ArUco, etc.)
    and estimating their pose relative to camera.
    """

    def __init__(self, board_config: Dict, intrinsics: Dict):
        """
        Initialize board detector

        Args:
            board_config: Board configuration
                - type: str ('charuco' or 'aruco')
                - square_length: float (meters)
                - marker_length: float (meters)
                - board_size: List[int] [cols, rows]
                - aruco_dict: str (e.g., 'DICT_5X5_250')
            intrinsics: Camera intrinsics
                - fx, fy: float (focal lengths)
                - cx, cy: float (principal point)
                - coeffs: np.ndarray (distortion coefficients)
                - width, height: int (image resolution)
        """
        self._board_config = board_config
        self._intrinsics = intrinsics

        # Build camera matrix
        self._camera_matrix = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float32)

        self._dist_coeffs = np.array(intrinsics['coeffs'], dtype=np.float32)

        # Detection parameters (can be overridden in config)
        self._min_markers = board_config.get('min_markers', 4)
        self._max_reprojection_error = board_config.get('max_reprojection_error', 2.0)

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Detect calibration board and estimate pose

        Args:
            image: Input image (BGR format)

        Returns:
            (success, T_camera_board, reprojection_error)
            - success: bool, whether detection succeeded
            - T_camera_board: np.ndarray shape=(4,4), homogeneous transformation matrix
              from board frame to camera frame (None if detection failed)
            - reprojection_error: float, reprojection error in pixels

        Note:
            Implementations must validate:
            - Sufficient markers/corners detected
            - Reprojection error within threshold
        """
        raise NotImplementedError

    @abc.abstractmethod
    def draw_detection(self, image: np.ndarray,
                       T_camera_board: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw detection results on image

        Args:
            image: Input image (BGR format)
            T_camera_board: Detected pose (None if detection failed)

        Returns:
            Image with detection visualization (copy of input)
        """
        raise NotImplementedError

    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera intrinsics

        Returns:
            (camera_matrix, dist_coeffs)
        """
        return self._camera_matrix, self._dist_coeffs
