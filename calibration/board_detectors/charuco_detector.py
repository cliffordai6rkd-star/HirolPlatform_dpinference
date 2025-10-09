"""
ChArUco Board Detector

Detects ChArUco calibration boards and estimates their pose.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import cv2
import cv2.aruco as aruco
import glog as log

from .base_detector import BoardDetectorBase


class CharucoDetector(BoardDetectorBase):
    """
    ChArUco board detector

    ChArUco boards combine chessboard and ArUco markers for robust detection.
    Advantages over pure ArUco:
    - Subpixel corner accuracy
    - More stable pose estimation
    - Partial occlusion tolerance
    """

    def __init__(self, board_config: Dict, intrinsics: Dict):
        """
        Initialize ChArUco detector

        Args:
            board_config: Board configuration
                - square_length: float (meters)
                - marker_length: float (meters)
                - board_size: List[int] [cols, rows]
                - aruco_dict: str (e.g., 'DICT_5X5_250')
                - min_corner_count: int (default: 8)
            intrinsics: Camera intrinsics
        """
        super().__init__(board_config, intrinsics)

        # Store last detection for camera calibration
        self._last_charuco_corners = None
        self._last_charuco_ids = None

        # Board parameters
        self._square_length = board_config['square_length']
        self._marker_length = board_config['marker_length']
        self._board_size = tuple(board_config['board_size'])  # (cols, rows)

        # Minimum corner count for pose estimation
        self._min_corner_count = board_config.get('min_corner_count', 8)

        # Create ArUco dictionary
        dict_name = board_config['aruco_dict']
        dict_attr = getattr(aruco, dict_name, None)
        if dict_attr is None:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        self._aruco_dict = aruco.getPredefinedDictionary(dict_attr)

        # Create ChArUco board
        self._charuco_board = aruco.CharucoBoard(
            self._board_size,
            self._square_length,
            self._marker_length,
            self._aruco_dict
        )

        # Create detector (OpenCV 4.7+ uses CharucoDetector)
        try:
            # OpenCV >= 4.7: Use CharucoDetector with CharucoParameters
            charuco_params = aruco.CharucoParameters()
            detector_params = aruco.DetectorParameters()
            detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
            refine_params = aruco.RefineParameters()
            self._detector = aruco.CharucoDetector(
                self._charuco_board, charuco_params, detector_params, refine_params
            )
            self._use_charuco_detector = True
        except (TypeError, AttributeError):
            # OpenCV < 4.7: Use ArucoDetector (fallback)
            detector_params = aruco.DetectorParameters()
            detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
            self._detector = aruco.ArucoDetector(self._aruco_dict, detector_params)
            self._use_charuco_detector = False

        log.info(f"CharucoDetector initialized: board_size={self._board_size}, "
                f"square_length={self._square_length}m, marker_length={self._marker_length}m")

    def detect(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Detect ChArUco board and estimate pose

        Args:
            image: Input BGR image

        Returns:
            (success, T_camera_board, reprojection_error)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ChArUco corners based on detector type
        if self._use_charuco_detector:
            # OpenCV >= 4.7: Use CharucoDetector.detectBoard()
            charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(gray)
            result = len(charuco_ids) if charuco_ids is not None else 0
        else:
            # OpenCV < 4.7: Detect ArUco markers first, then interpolate corners
            corners, ids, rejected = self._detector.detectMarkers(gray)

            if ids is None or len(ids) == 0:
                return False, None, float('inf')

            # Interpolate ChArUco corners from ArUco markers
            result, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, self._charuco_board,
                cameraMatrix=self._camera_matrix,
                distCoeffs=self._dist_coeffs
            )

        # Check if enough corners detected
        if charuco_ids is None or result < self._min_corner_count:
            self._last_charuco_corners = None
            self._last_charuco_ids = None
            return False, None, float('inf')

        # Store detection for camera calibration
        self._last_charuco_corners = charuco_corners
        self._last_charuco_ids = charuco_ids

        # Estimate pose using OpenCV 4.7+ API
        try:
            # OpenCV >= 4.7: Use board.matchImagePoints + cv2.solvePnP
            obj_points, img_points = self._charuco_board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            if obj_points is None or len(obj_points) < 4:
                return False, None, float('inf')

            valid, rvec, tvec = cv2.solvePnP(
                obj_points, img_points,
                self._camera_matrix, self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not valid:
                return False, None, float('inf')

        except AttributeError:
            # OpenCV < 4.7: Use deprecated estimatePoseCharucoBoard
            try:
                # API version 1: returns (success, rvec, tvec)
                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self._charuco_board,
                    self._camera_matrix, self._dist_coeffs, None, None
                )
            except TypeError:
                # API version 2: modifies rvec, tvec in-place
                rvec = np.zeros((3, 1), dtype=np.float32)
                tvec = np.zeros((3, 1), dtype=np.float32)
                valid = aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self._charuco_board,
                    self._camera_matrix, self._dist_coeffs, rvec, tvec
                )

            if not valid:
                return False, None, float('inf')

        # Compute reprojection error
        reproj_error = self._compute_reprojection_error(
            charuco_corners, charuco_ids, rvec, tvec
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

    def get_last_detection(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get ChArUco corners and IDs from last successful detection

        Returns:
            (charuco_corners, charuco_ids)
            - charuco_corners: Detected corner positions, shape (N, 1, 2)
            - charuco_ids: Corner IDs, shape (N, 1)
            - Both None if no recent detection or detection failed
        """
        return self._last_charuco_corners, self._last_charuco_ids

    def get_board(self):
        """
        Get ChArUco board object for camera calibration

        Returns:
            CharucoBoard object
        """
        return self._charuco_board

    def _compute_reprojection_error(self,
                                    charuco_corners: np.ndarray,
                                    charuco_ids: np.ndarray,
                                    rvec: np.ndarray,
                                    tvec: np.ndarray) -> float:
        """
        Compute reprojection error for detected corners

        Args:
            charuco_corners: Detected corner positions
            charuco_ids: Corner IDs
            rvec: Rotation vector
            tvec: Translation vector

        Returns:
            Mean reprojection error in pixels
        """
        # Get 3D object points for detected corners
        obj_points = self._charuco_board.getChessboardCorners()[charuco_ids.flatten()]

        # Project 3D points to image
        projected_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec,
            self._camera_matrix, self._dist_coeffs
        )

        # Compute error
        errors = np.linalg.norm(
            charuco_corners - projected_points.reshape(-1, 1, 2),
            axis=2
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

            # Interpolate ChArUco corners
            result, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, self._charuco_board,
                cameraMatrix=self._camera_matrix,
                distCoeffs=self._dist_coeffs
            )

            if result > 0:
                # Draw ChArUco corners
                aruco.drawDetectedCornersCharuco(
                    display_image, charuco_corners, charuco_ids
                )

        # Draw coordinate frame if pose is valid
        if T_camera_board is not None:
            R = T_camera_board[:3, :3]
            t = T_camera_board[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(3, 1)

            # Draw axes (20cm length)
            axis_length = 0.2 if self._square_length > 0.05 else self._square_length * 4
            cv2.drawFrameAxes(
                display_image, self._camera_matrix, self._dist_coeffs,
                rvec, tvec, axis_length
            )

            # Display translation
            pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
            cv2.putText(display_image, pose_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return display_image
