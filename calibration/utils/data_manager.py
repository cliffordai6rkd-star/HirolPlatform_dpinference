"""
Data Manager for Calibration

Handles data persistence for calibration samples and results.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
import cv2
import glog as log


class DataManager:
    """
    Manages calibration data storage and retrieval

    Responsibilities:
    - Save/load calibration samples
    - Save/load calibration results
    - Image file management
    """

    def __init__(self, save_path: str):
        """
        Initialize data manager

        Args:
            save_path: Base directory for saving data
        """
        self._save_path = Path(save_path)
        self._images_path = self._save_path / "images"
        self._data_file = self._save_path / "calibration_data.json"
        self._result_file = self._save_path / "calibration_result.json"

        # Create directories
        os.makedirs(self._save_path, exist_ok=True)
        os.makedirs(self._images_path, exist_ok=True)

        log.info(f"DataManager initialized at {self._save_path}")

    def save_image(self, image: np.ndarray, index: int) -> str:
        """
        Save calibration image

        Args:
            image: BGR image
            index: Sample index

        Returns:
            Relative path to saved image
        """
        filename = f"sample_{index:04d}.png"
        filepath = self._images_path / filename
        cv2.imwrite(str(filepath), image)
        return f"images/{filename}"

    def save_samples(self, samples: List[Dict], metadata: Dict):
        """
        Save calibration samples to JSON

        Args:
            samples: List of sample dictionaries
                Each sample contains:
                - T_base_ee: np.ndarray (4x4)
                - T_camera_board: np.ndarray (4x4)
                - image_path: str
                - reprojection_error: float
            metadata: Metadata dictionary
                - calibration_type: str
                - robot_type: str
                - board_type: str
                - timestamp: str
        """
        data = {
            "metadata": metadata,
            "samples": []
        }

        for i, sample in enumerate(samples):
            data["samples"].append({
                "index": i,
                "image_path": sample['image_path'],
                "T_base_ee": sample['T_base_ee'].tolist(),
                "T_camera_board": sample['T_camera_board'].tolist(),
                "reprojection_error": float(sample['reprojection_error'])
            })

        with open(self._data_file, 'w') as f:
            json.dump(data, f, indent=2)

        log.info(f"Saved {len(samples)} samples to {self._data_file}")

    def load_samples(self) -> tuple:
        """
        Load calibration samples from JSON

        Returns:
            (samples, metadata)
        """
        if not self._data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self._data_file}")

        with open(self._data_file, 'r') as f:
            data = json.load(f)

        samples = []
        for sample_data in data["samples"]:
            samples.append({
                "T_base_ee": np.array(sample_data["T_base_ee"]),
                "T_camera_board": np.array(sample_data["T_camera_board"]),
                "image_path": sample_data["image_path"],
                "reprojection_error": sample_data["reprojection_error"]
            })

        return samples, data["metadata"]

    def save_calibration_result(self,
                                T_result: np.ndarray,
                                diagnostics: Dict,
                                samples: List[Dict],
                                intrinsics: Dict,
                                config: Dict):
        """
        Save complete calibration result

        Args:
            T_result: Calibration result transformation matrix (4x4)
            diagnostics: Diagnostic information
            samples: List of samples used
            intrinsics: Camera intrinsics
            config: Configuration used
        """
        # Prepare metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "calibration_type": config['calibration']['type'],
            "board_type": config['calibration']['board']['type'],
            "solver_method": config['calibration']['solver']['method'],
            "num_samples": len(samples)
        }

        # Save samples separately
        self.save_samples(samples, metadata)

        # Save result
        result_data = {
            "metadata": metadata,
            "camera_intrinsics": {
                "fx": float(intrinsics['fx']),
                "fy": float(intrinsics['fy']),
                "cx": float(intrinsics['cx']),
                "cy": float(intrinsics['cy']),
                "coeffs": intrinsics['coeffs'].tolist(),
                "width": int(intrinsics['width']),
                "height": int(intrinsics['height'])
            },
            "result": {
                "T_result": T_result.tolist(),
                "condition_number": float(diagnostics['condition_number']),
                "mean_residual": float(diagnostics['mean_residual'])
            }
        }

        with open(self._result_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        log.info(f"Saved calibration result to {self._result_file}")

    def load_calibration_result(self) -> tuple:
        """
        Load calibration result

        Returns:
            (T_result, diagnostics, metadata, intrinsics)
        """
        if not self._result_file.exists():
            raise FileNotFoundError(f"Result file not found: {self._result_file}")

        with open(self._result_file, 'r') as f:
            data = json.load(f)

        T_result = np.array(data["result"]["T_result"])
        diagnostics = {
            "condition_number": data["result"]["condition_number"],
            "mean_residual": data["result"]["mean_residual"]
        }
        metadata = data["metadata"]
        intrinsics = {
            k: np.array(v) if k == 'coeffs' else v
            for k, v in data["camera_intrinsics"].items()
        }

        return T_result, diagnostics, metadata, intrinsics
