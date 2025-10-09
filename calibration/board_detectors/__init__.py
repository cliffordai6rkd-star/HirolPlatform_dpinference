"""
Board Detectors for Hand-Eye Calibration

Provides detection interfaces for various calibration boards:
- ChArUco boards
- ArUco markers
"""

from .base_detector import BoardDetectorBase
from .charuco_detector import CharucoDetector
from .aruco_detector import ArucoDetector

__all__ = [
    'BoardDetectorBase',
    'CharucoDetector',
    'ArucoDetector',
]
