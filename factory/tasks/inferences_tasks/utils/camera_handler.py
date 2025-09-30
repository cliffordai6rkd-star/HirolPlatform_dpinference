#!/usr/bin/env python3
"""
Camera Handler for ACT Inference
Handles camera observations, image processing, and action image saving
"""

import time
import cv2
import numpy as np
import torch
import glog as log
from typing import Dict, Any, Optional
from pathlib import Path


class CameraHandler:
    """Handles camera observations and image processing for ACT inference"""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        """
        Initialize camera handler

        Args:
            config: Configuration dictionary
            device: PyTorch device for tensor operations
        """
        self._device = device
        self._robot_type = config.get("robot_type", "fr3")

        # Action image saving configuration
        self.last_gripper_action_type = None
        self._clear_action_images_directory()

        log.info("✅ Camera handler initialized")

    def extract_camera_observations(self, gym_obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract camera observations and convert to tensors"""
        camera_obs = {}

        for camera_name, img in gym_obs.get('colors', {}).items():
            if img is not None and len(img.shape) == 3:
                # Normalize image
                if img.dtype == np.uint8:
                    processed_img = img.astype(np.float32) / 255.0
                else:
                    processed_img = np.clip(img.astype(np.float32), 0.0, 1.0)

                # Convert to tensor format [C, H, W]
                img_tensor = torch.from_numpy(np.transpose(processed_img, (2, 0, 1))).float().to(self._device)
                camera_obs[camera_name] = img_tensor

        return camera_obs

    def check_and_save_action_images(self, gym_action: Dict[str, Any], gym_robot) -> None:
        """检测夹爪动作并保存相机图像"""
        # Get gripper command value
        gripper_raw = gym_action.get('tool', [1.0])[0] if 'tool' in gym_action else 1.0

        # Normalize command value
        max_gripper_width = 0.08
        if gripper_raw <= max_gripper_width:
            normalized_command = min(1.0, max(0.0, gripper_raw / max_gripper_width))
        else:
            normalized_command = gripper_raw

        # Detect action type based on value thresholds
        action_detected = None
        if normalized_command < 0.02:
            action_detected = "close"
        elif normalized_command > 0.95:
            action_detected = "open"

        # Save images only when action type changes (new action marker will be drawn)
        if action_detected and action_detected != self.last_gripper_action_type:
            self._save_action_camera_images(action_detected, gym_robot)
            self.last_gripper_action_type = action_detected
        elif action_detected is None:
            # Reset when no action is detected (neutral state)
            self.last_gripper_action_type = None

    def _get_current_camera_observations(self, gym_robot) -> Optional[Dict[str, np.ndarray]]:
        """
        获取当前相机观测数据

        Returns:
            Optional[Dict[str, np.ndarray]]: 相机名称到图像数据的映射，失败时返回None
        """
        # 获取当前观测
        current_obs = gym_robot.get_observation()

        # 检查是否包含相机数据
        if "colors" not in current_obs or not current_obs["colors"]:
            log.debug("No camera observations available in current observation")
            return None

        # 返回相机观测
        return current_obs["colors"]

    def _create_action_grid_image(self, camera_obs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        从相机观测创建四宫格拼接图像

        Args:
            camera_obs: 相机观测字典 {"camera_name": image_array}

        Returns:
            Optional[np.ndarray]: 拼接后的图像数组(BGR格式)，失败时返回None
        """
        if not camera_obs:
            log.debug("Empty camera observations, cannot create grid image")
            return None

        # 收集有效的图像
        valid_images = []
        for cam_name, img in camera_obs.items():
            if isinstance(img, np.ndarray) and len(img.shape) == 3:
                valid_images.append(img)
            else:
                log.debug(f"Invalid image format for camera {cam_name}: {type(img)}")

        if not valid_images:
            log.debug("No valid images found for grid creation")
            return None

        # 创建2x2或1xN网格布局
        num_images = len(valid_images)

        if num_images == 1:
            return valid_images[0]
        elif num_images == 2:
            # 1x2布局（水平拼接）
            return np.hstack(valid_images)
        elif num_images == 3:
            # 2x2布局，第四个位置放黑色占位图
            h, w = valid_images[0].shape[:2]
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            valid_images.append(placeholder)
            # 创建2x2网格
            top_row = np.hstack(valid_images[:2])
            bottom_row = np.hstack(valid_images[2:4])
            return np.vstack([top_row, bottom_row])
        else:  # 4个或更多
            # 使用前4个图像创建2x2网格
            top_row = np.hstack(valid_images[:2])
            bottom_row = np.hstack(valid_images[2:4])
            return np.vstack([top_row, bottom_row])

    def _save_action_camera_images(self, action_name: str, gym_robot) -> None:
        """
        保存夹爪动作时的实时相机图像四宫格拼接

        Args:
            action_name: 动作名称 ("close" 或 "open")
        """
        # 获取当前相机观测
        camera_obs = self._get_current_camera_observations(gym_robot)
        if camera_obs is None:
            log.warning(f"⚠️ Failed to get current camera observations for {action_name} action")
            return

        # 创建拼接图像
        grid_image = self._create_action_grid_image(camera_obs)
        if grid_image is None:
            log.warning(f"⚠️ Failed to create grid image for {action_name} action")
            return

        # Create timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds

        # Create save directory
        save_dir = Path("./logs/action_images")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save image with action name and timestamp
        filename = f"{action_name}_action_{timestamp}.png"
        filepath = save_dir / filename

        cv2.imwrite(str(filepath), grid_image)
        log.info(f"📸 Saved {action_name} action images (realtime): {filepath}")

    def _clear_action_images_directory(self) -> None:
        """Clear action images directory before starting inference"""
        import shutil
        from pathlib import Path

        action_images_dir = Path("./logs/action_images")

        if action_images_dir.exists():
            # Remove entire directory
            shutil.rmtree(action_images_dir)
            log.info(f"🗑️ Removed action images directory: {action_images_dir}")

        # Create fresh directory
        action_images_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"📁 Created fresh action images directory: {action_images_dir}")