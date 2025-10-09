"""
Hand-Eye Calibration using RobotMotion Framework

Unified calibration tool for:
- Eye-in-Hand / Eye-to-Hand
- ChArUco / ArUco calibration boards
- Multiple robot platforms (FR3, Monte01, UnitreeG1, etc.)

Author: Refactored from legacy code
Date: 2025-10-09
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import glog as log

# Add HIROLRobotPlatform to path
platform_path = Path(__file__).parent.parent
if str(platform_path) not in sys.path:
    sys.path.insert(0, str(platform_path))

from factory.tasks.robot_motion import RobotMotion
from calibration.board_detectors import BoardDetectorBase, CharucoDetector, ArucoDetector
from calibration.utils import (
    generate_grid_poses,
    solve_eye_in_hand,
    solve_eye_to_hand,
    verify_calibration,
    DataManager
)
from calibration.utils.camera_calibration import (
    calibrate_camera_intrinsics,
    evaluate_calibration_quality,
    compare_calibrations
)
from hardware.base.utils import dynamic_load_yaml


class HandEyeCalibration:
    """
    Hand-Eye Calibration System

    Supports:
    - Eye-in-Hand: Camera mounted on robot end-effector
    - Eye-to-Hand: Camera fixed in world, robot moves board

    Features:
    - Automatic grid-based data collection
    - Manual data collection (press 'r' to record)
    - Multiple calibration board types
    - Robot-agnostic (uses RobotMotion factory)
    """

    def __init__(self, config_path: str):
        """
        Initialize calibration system

        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        self._config = self._load_and_validate_config(config_path)
        log.info(f"Loaded configuration from {config_path}")

        # Initialize RobotMotion
        motion_config_path = self._config['robot_motion_config']
        self._robot_motion = RobotMotion(motion_config_path, auto_initialize=True)
        log.info("RobotMotion initialized")

        # Get camera intrinsics
        self._camera_name = self._config['calibration']['camera']['name']
        self._intrinsics = self._get_camera_intrinsics()
        log.info(f"Camera intrinsics loaded: fx={self._intrinsics['fx']:.1f}, "
                f"fy={self._intrinsics['fy']:.1f}")

        # Create board detector
        self._detector = self._create_detector()
        log.info(f"Board detector created: {self._config['calibration']['board']['type']}")

        # Data management
        save_path = self._config['calibration']['data_save_path']
        self._data_manager = DataManager(save_path)
        self._samples = []

        # Calibration type
        self._calibration_type = self._config['calibration']['type']
        log.info(f"Calibration type: {self._calibration_type}")

        # Camera intrinsic calibration tracking
        self._calibrate_intrinsics = self._config['calibration']['camera'].get('calibrate_intrinsics', False)
        self._charuco_detections = []  # Store (corners, ids, image) for camera calibration
        self._calibrated_camera_matrix = None
        self._calibrated_dist_coeffs = None

        log.info("=" * 60)
        log.info(" HandEyeCalibration initialized successfully")
        log.info("=" * 60)

    def _load_and_validate_config(self, config_path: str) -> Dict:
        """Load and validate configuration file with !include support"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Use dynamic_load_yaml to support !include tags
        config = dynamic_load_yaml(config_path)

        # Validate required fields
        required_fields = ['robot_motion_config', 'calibration']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")

        calib_config = config['calibration']
        required_calib_fields = ['type', 'board', 'camera', 'solver', 'data_save_path']
        for field in required_calib_fields:
            if field not in calib_config:
                raise ValueError(f"Missing required calibration field: {field}")

        # Validate calibration type
        if calib_config['type'] not in ['eye_in_hand', 'eye_to_hand']:
            raise ValueError(f"Invalid calibration type: {calib_config['type']}")

        return config

    def _get_camera_intrinsics(self) -> Dict:
        """
        Get camera intrinsics from RobotMotion

        Returns:
            Dictionary with fx, fy, cx, cy, coeffs, width, height

        Raises:
            RuntimeError: If camera not found or doesn't support intrinsics

        Note:
            Camera is matched by name (exact match).
            When RobotMotion has multiple cameras, specify exact camera name.
        """
        # Get cameras from RobotFactory sensors
        robot_system = self._robot_motion._robot_system

        if not hasattr(robot_system, '_sensors') or 'camera' not in robot_system._sensors:
            raise RuntimeError(
                "No cameras configured in RobotMotion.\n"
                "Hint: Add cameras to motion_config's sensor_dicts.cameras"
            )

        cameras = robot_system._sensors['camera']  # List of {'name': str, 'object': CameraBase}

        # List all available cameras for debugging
        available_cameras = [cam['name'] for cam in cameras]
        log.info(f"Available cameras in RobotMotion: {available_cameras}")

        # Find target camera by exact name match
        target_camera = None
        for cam in cameras:
            cam_name = cam['name']
            if cam_name == self._camera_name:
                target_camera = cam['object']
                log.info(f"Selected camera: '{cam_name}' (exact match)")
                break

        if target_camera is None:
            raise RuntimeError(
                f"Camera '{self._camera_name}' not found.\n"
                f"Available cameras: {available_cameras}\n"
                f"Hint: Update 'camera.name' in calibration config to match one of the above.\n"
                f"      Camera names are defined in motion_config's sensor_dicts.cameras[].name"
            )

        # Get intrinsics (RealSense camera)
        if not hasattr(target_camera, '_intrinsics'):
            raise RuntimeError(
                f"Camera '{self._camera_name}' does not support intrinsics query.\n"
                f"Only RealSense cameras have _intrinsics attribute.\n"
                f"Camera type: {type(target_camera).__name__}"
            )

        rs_intrinsics = target_camera._intrinsics

        return {
            'fx': rs_intrinsics.fx,
            'fy': rs_intrinsics.fy,
            'cx': rs_intrinsics.ppx,
            'cy': rs_intrinsics.ppy,
            'coeffs': np.array(rs_intrinsics.coeffs, dtype=np.float32),
            'width': rs_intrinsics.width,
            'height': rs_intrinsics.height
        }

    def _create_detector(self) -> BoardDetectorBase:
        """Create board detector based on configuration"""
        board_config = self._config['calibration']['board']
        camera_config = self._config['calibration']['camera']

        # Merge configs
        detector_config = {**board_config, **camera_config}

        board_type = board_config['type']
        if board_type == 'charuco':
            return CharucoDetector(detector_config, self._intrinsics)
        elif board_type == 'aruco':
            return ArucoDetector(detector_config, self._intrinsics)
        else:
            raise ValueError(f"Unknown board type: {board_type}")

    def _get_camera_image(self) -> Optional[np.ndarray]:
        """Get current camera image from RobotMotion"""
        cameras_data = self._robot_motion._robot_system.get_cameras_infos()

        if cameras_data is None or len(cameras_data) == 0:
            return None

        # Find target camera image
        for cam_data in cameras_data:
            if self._camera_name in cam_data['name']:
                return cam_data['img']

        return None

    def _get_robot_pose(self) -> np.ndarray:
        """
        Get current robot end-effector pose

        Returns:
            T_base_ee: 4x4 homogeneous transformation matrix
        """
        state = self._robot_motion.get_state()
        pose_7d = state['pose']  # [x, y, z, qx, qy, qz, qw]

        # Convert to 4x4 matrix
        from scipy.spatial.transform import Rotation as R
        T_base_ee = np.eye(4)
        T_base_ee[:3, 3] = pose_7d[:3]
        T_base_ee[:3, :3] = R.from_quat(pose_7d[3:]).as_matrix()

        return T_base_ee

    def collect_data(self, auto: bool = True) -> int:
        """
        Collect calibration data

        Args:
            auto: True for automatic grid collection, False for manual

        Returns:
            Number of samples collected
        """
        if auto:
            return self._auto_collect()
        else:
            return self._manual_collect()

    def _auto_collect(self) -> int:
        """Automatic grid-based data collection"""
        log.info("=" * 60)
        log.info(" Automatic Grid Data Collection")
        log.info("=" * 60)

        # Enable hardware
        self._robot_motion.enable_hardware()
        time.sleep(1.0)

        # Generate workspace poses
        workspace_cfg = self._config['calibration'].get('workspace')
        if workspace_cfg is None:
            raise ValueError("workspace configuration required for auto collection")

        poses = generate_grid_poses(
            center=np.array(workspace_cfg['center']),
            grid_size=workspace_cfg['grid_size'],
            spacing=workspace_cfg['spacing'],
            orientation_randomness=workspace_cfg['orientation_randomness']
        )

        log.info(f"Generated {len(poses)} target poses")

        success_count = 0
        for i, pose in enumerate(poses):
            log.info(f"Moving to pose {i+1}/{len(poses)}")

            try:
                # Send motion command
                self._robot_motion.send_pose_command(pose)
                time.sleep(2.5)  # Wait for motion to stabilize

                # Get image
                image = self._get_camera_image()
                if image is None:
                    log.warning(f"Failed to get image at pose {i+1}, skipping")
                    continue

                # Detect board
                success, T_camera_board, reproj_error = self._detector.detect(image)

                if not success:
                    log.warning(f"Detection failed at pose {i+1}, skipping")
                    continue

                # Get robot pose
                T_base_ee = self._get_robot_pose()

                # Save image
                image_path = self._data_manager.save_image(image, len(self._samples))

                # Record sample
                self._samples.append({
                    'T_base_ee': T_base_ee,
                    'T_camera_board': T_camera_board,
                    'image_path': image_path,
                    'reprojection_error': reproj_error
                })

                # Save ChArUco detection for camera calibration (if enabled)
                if self._calibrate_intrinsics and hasattr(self._detector, 'get_last_detection'):
                    corners, ids = self._detector.get_last_detection()
                    if corners is not None and ids is not None:
                        self._charuco_detections.append((corners, ids))

                success_count += 1
                log.info(f"Sample {success_count} collected "
                        f"(reproj_error={reproj_error:.2f}px)")

            except Exception as e:
                log.error(f"Error at pose {i+1}: {e}")
                continue

        log.info("=" * 60)
        log.info(f" Data collection completed: {success_count}/{len(poses)} samples")
        log.info("=" * 60)

        return success_count

    def _manual_collect(self) -> int:
        """Manual data collection (user presses 'r' to record)"""
        log.info("=" * 60)
        log.info(" Manual Data Collection")
        log.info(" Press 'r' to record, 'q' to quit")
        log.info("=" * 60)

        # Enable hardware
        self._robot_motion.enable_hardware()

        cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)

        success_count = 0
        while True:
            # Get image
            image = self._get_camera_image()
            if image is None:
                time.sleep(0.01)
                continue

            # Detect board
            success, T_camera_board, reproj_error = self._detector.detect(image)

            # Draw detection
            display_image = self._detector.draw_detection(
                image, T_camera_board if success else None
            )

            # Display status
            status_text = f"Samples: {success_count} | "
            if success:
                status_text += f"Detected (error={reproj_error:.2f}px)"
            else:
                status_text += "Not detected"

            cv2.putText(display_image, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Calibration', display_image)
            key = cv2.waitKey(1)

            if key == ord('r') and success:
                # Record sample
                T_base_ee = self._get_robot_pose()
                image_path = self._data_manager.save_image(image, len(self._samples))

                self._samples.append({
                    'T_base_ee': T_base_ee,
                    'T_camera_board': T_camera_board,
                    'image_path': image_path,
                    'reprojection_error': reproj_error
                })

                success_count += 1
                log.info(f"Sample {success_count} recorded "
                        f"(reproj_error={reproj_error:.2f}px)")

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        log.info(f"Manual collection completed: {success_count} samples")

        return success_count

    def calibrate_camera(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate camera intrinsic parameters using collected ChArUco detections

        Returns:
            (camera_matrix, dist_coeffs, reprojection_error)

        Raises:
            ValueError: Insufficient data or camera calibration disabled
            RuntimeError: Calibration failed
        """
        if not self._calibrate_intrinsics:
            raise ValueError("Camera intrinsic calibration is disabled in config")

        if len(self._charuco_detections) < 3:
            raise ValueError(
                f"Insufficient images for camera calibration: {len(self._charuco_detections)} < 3"
            )

        log.info("=" * 70)
        log.info(" Camera Intrinsic Calibration")
        log.info("=" * 70)

        # Extract corners and IDs
        corners_list = [det[0] for det in self._charuco_detections]
        ids_list = [det[1] for det in self._charuco_detections]

        # Get ChArUco board from detector
        if not hasattr(self._detector, 'get_board'):
            raise RuntimeError("Detector does not support camera calibration")

        charuco_board = self._detector.get_board()

        # Get image size from intrinsics
        image_size = (self._intrinsics['width'], self._intrinsics['height'])

        # Initial camera matrix and distortion coeffs
        initial_camera_matrix = np.array([
            [self._intrinsics['fx'], 0, self._intrinsics['cx']],
            [0, self._intrinsics['fy'], self._intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float64)

        initial_dist_coeffs = np.array(self._intrinsics['coeffs'], dtype=np.float64)

        # Calibration flags
        calib_flags = self._config['calibration']['camera'].get('calibration_flags', 0)

        # Perform calibration
        camera_matrix, dist_coeffs, reprojection_error = calibrate_camera_intrinsics(
            corners_list, ids_list, charuco_board, image_size,
            initial_camera_matrix, initial_dist_coeffs, calib_flags
        )

        # Store calibrated parameters
        self._calibrated_camera_matrix = camera_matrix
        self._calibrated_dist_coeffs = dist_coeffs

        # Print comparison
        compare_calibrations(
            initial_camera_matrix, initial_dist_coeffs,
            camera_matrix, dist_coeffs
        )

        # Save camera intrinsics to file
        import json
        from datetime import datetime
        intrinsics_file = self._data_manager._save_path / "camera_intrinsics.json"
        intrinsics_data = {
            "timestamp": datetime.now().isoformat(),
            "camera_name": self._camera_name,
            "image_size": {
                "width": image_size[0],
                "height": image_size[1]
            },
            "initial_intrinsics": {
                "fx": float(initial_camera_matrix[0, 0]),
                "fy": float(initial_camera_matrix[1, 1]),
                "cx": float(initial_camera_matrix[0, 2]),
                "cy": float(initial_camera_matrix[1, 2]),
                "k1": float(initial_dist_coeffs[0]),
                "k2": float(initial_dist_coeffs[1]),
                "p1": float(initial_dist_coeffs[2]),
                "p2": float(initial_dist_coeffs[3]),
                "k3": float(initial_dist_coeffs[4])
            },
            "calibrated_intrinsics": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
                "k1": float(dist_coeffs[0]),
                "k2": float(dist_coeffs[1]),
                "p1": float(dist_coeffs[2]),
                "p2": float(dist_coeffs[3]),
                "k3": float(dist_coeffs[4])
            },
            "calibration_quality": {
                "rms_reprojection_error": float(reprojection_error),
                "num_images": len(corners_list)
            }
        }

        with open(intrinsics_file, 'w') as f:
            json.dump(intrinsics_data, f, indent=2)

        log.info(f"Saved camera intrinsics to {intrinsics_file}")

        return camera_matrix, dist_coeffs, reprojection_error

    def calibrate(self) -> Tuple[np.ndarray, Dict]:
        """
        Solve hand-eye calibration

        Returns:
            (T_result, diagnostics)
            - T_result: Calibration transformation (4x4)
            - diagnostics: Quality metrics
        """
        min_samples = self._config['calibration']['solver']['min_samples']
        if len(self._samples) < min_samples:
            raise RuntimeError(
                f"Insufficient samples: {len(self._samples)} < {min_samples}"
            )

        # First, calibrate camera intrinsics if enabled
        if self._calibrate_intrinsics and len(self._charuco_detections) >= 3:
            log.info("")
            try:
                camera_matrix, dist_coeffs, _ = self.calibrate_camera()

                # Update detector with calibrated intrinsics
                log.info("Updating detector with calibrated camera intrinsics...")
                self._detector._camera_matrix = camera_matrix
                self._detector._dist_coeffs = dist_coeffs

                # Re-detect all boards with calibrated intrinsics
                log.info("Re-detecting boards with calibrated intrinsics...")
                for i, sample in enumerate(self._samples):
                    # Construct full image path (relative path is relative to data manager save path)
                    image_relative_path = sample['image_path']
                    image_full_path = self._data_manager._save_path / image_relative_path
                    image = cv2.imread(str(image_full_path))
                    if image is not None:
                        success, T_camera_board, reproj_error = self._detector.detect(image)
                        if success:
                            self._samples[i]['T_camera_board'] = T_camera_board
                            self._samples[i]['reprojection_error'] = reproj_error

                log.info("Board re-detection completed")
                log.info("")
            except Exception as e:
                log.warning(f"Camera calibration failed: {e}")
                log.warning("Proceeding with original camera intrinsics")
                log.info("")

        log.info("=" * 60)
        log.info(f" Hand-Eye Calibration with {len(self._samples)} samples")
        log.info("=" * 60)

        # Extract data
        T_base_ee_list = [s['T_base_ee'] for s in self._samples]
        T_camera_board_list = [s['T_camera_board'] for s in self._samples]

        # Call solver
        method = self._config['calibration']['solver']['method']

        if self._calibration_type == 'eye_in_hand':
            T_result, diagnostics = solve_eye_in_hand(
                T_base_ee_list, T_camera_board_list, method
            )
            log.info("Result: T_ee_camera (end-effector to camera)")
        elif self._calibration_type == 'eye_to_hand':
            T_result, diagnostics = solve_eye_to_hand(
                T_base_ee_list, T_camera_board_list, method
            )
            log.info("Result: T_base_camera (robot base to camera)")
        else:
            raise ValueError(f"Unknown calibration type: {self._calibration_type}")

        # Print result
        log.info("Transformation matrix:")
        log.info("\n" + str(T_result))
        log.info(f"Translation: {T_result[:3, 3]}")

        return T_result, diagnostics

    def verify(self, T_result: np.ndarray) -> Dict:
        """Verify calibration accuracy"""
        log.info("=" * 60)
        log.info(" Verification")
        log.info("=" * 60)

        T_base_ee_list = [s['T_base_ee'] for s in self._samples]
        T_camera_board_list = [s['T_camera_board'] for s in self._samples]

        verification = verify_calibration(
            T_base_ee_list, T_camera_board_list,
            T_result, self._calibration_type
        )

        log.info(f"Mean position error: {verification['mean_position_error']*1000:.2f} mm")
        log.info(f"Std position error: {verification['std_position_error']*1000:.2f} mm")
        log.info(f"Max position error: {verification['max_position_error']*1000:.2f} mm")
        log.info(f"Mean rotation error: {verification['mean_rotation_error']:.2f} deg")
        log.info(f"Max rotation error: {verification['max_rotation_error']:.2f} deg")

        return verification

    def save_result(self, T_result: np.ndarray, diagnostics: Dict):
        """Save calibration result"""
        # Use calibrated intrinsics if available, otherwise use original
        intrinsics_to_save = self._intrinsics.copy()
        if self._calibrated_camera_matrix is not None:
            intrinsics_to_save['fx'] = self._calibrated_camera_matrix[0, 0]
            intrinsics_to_save['fy'] = self._calibrated_camera_matrix[1, 1]
            intrinsics_to_save['cx'] = self._calibrated_camera_matrix[0, 2]
            intrinsics_to_save['cy'] = self._calibrated_camera_matrix[1, 2]
            intrinsics_to_save['coeffs'] = self._calibrated_dist_coeffs
            intrinsics_to_save['_calibrated'] = True
        else:
            intrinsics_to_save['_calibrated'] = False

        self._data_manager.save_calibration_result(
            T_result=T_result,
            diagnostics=diagnostics,
            samples=self._samples,
            intrinsics=intrinsics_to_save,
            config=self._config
        )
        log.info("Calibration result saved")

    def close(self):
        """Close resources"""
        self._robot_motion.close()
        cv2.destroyAllWindows()
        log.info("HandEyeCalibration closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hand-Eye Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Eye-in-hand calibration with ChArUco board (auto collection)
  python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml

  # Manual collection mode
  python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml --manual

  # Eye-to-hand calibration
  python hand_eye_calibration.py --config config/eye_to_hand_fr3_charuco.yaml
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--manual', action='store_true',
                       help='Use manual collection mode (press \'r\' to record)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Skip collection, load existing data and verify')

    args = parser.parse_args()

    calib = None
    try:
        # Initialize
        calib = HandEyeCalibration(args.config)

        if not args.verify_only:
            # Collect data
            num_samples = calib.collect_data(auto=not args.manual)

            if num_samples == 0:
                log.error("No samples collected, exiting")
                return

        # Calibrate
        T_result, diagnostics = calib.calibrate()

        # Verify
        verification = calib.verify(T_result)

        # Save
        calib.save_result(T_result, diagnostics)

        log.info("=" * 60)
        log.info(" Calibration completed successfully!")
        log.info("=" * 60)

    except Exception as e:
        log.error(f"Calibration failed: {e}")
        raise

    finally:
        if calib is not None:
            calib.close()


if __name__ == "__main__":
    main()
