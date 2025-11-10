#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Orientation helpers shared by LeRobot inference utilities."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _get_orientation_dim(rotation_repr: str) -> int:
    """Return the dimensionality for the requested rotation representation."""
    repr_lower = rotation_repr.lower()
    if repr_lower == "euler":
        return 3
    if repr_lower in {"quaternion", "quat"}:
        return 4
    if repr_lower in {"rotvec", "rotation_vector", "axis_angle"}:
        return 3
    raise NotImplementedError(f"Orientation representation '{rotation_repr}' is not supported.")


def _orientation_to_rotation(values: np.ndarray, rotation_repr: str) -> R:
    """Convert values encoded in the requested representation into a Rotation."""
    repr_lower = rotation_repr.lower()
    if repr_lower == "euler":
        return R.from_euler("xyz", values, degrees=False)
    if repr_lower in {"quaternion", "quat"}:
        return R.from_quat(values)
    if repr_lower in {"rotvec", "rotation_vector", "axis_angle"}:
        return R.from_rotvec(values)
    raise NotImplementedError(f"Orientation representation '{rotation_repr}' is not supported.")


def _split_pose_and_gripper(action_vec: np.ndarray, rotation_repr: str) -> tuple[np.ndarray, np.ndarray]:
    """Split a pose vector (position + orientation) from optional tool segments."""
    orient_dim = _get_orientation_dim(rotation_repr)
    pose_dim = 3 + orient_dim
    if action_vec.size < pose_dim:
        raise ValueError(
            f"Expected at least {pose_dim} values for pose with '{rotation_repr}' orientation, got {action_vec.size}"
        )
    pose_values = action_vec[:pose_dim]
    remainder = action_vec[pose_dim:]
    return pose_values, remainder


def _pose_to_representation(
    pose: np.ndarray, rotation_repr: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position, requested orientation representation, and quaternion fallback."""
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.size < 7:
        raise ValueError(f"Pose must contain position + quaternion (expected >=7 values, got {pose_arr.size})")
    position = pose_arr[:3]
    quat = pose_arr[3:7]
    repr_lower = rotation_repr.lower()
    if repr_lower == "euler":
        orientation = R.from_quat(quat).as_euler("xyz", degrees=False)
    elif repr_lower in {"quaternion", "quat"}:
        orientation = quat
    elif repr_lower in {"rotvec", "rotation_vector", "axis_angle"}:
        orientation = R.from_quat(quat).as_rotvec()
    else:
        raise NotImplementedError(f"Unsupported observation orientation representation '{rotation_repr}'.")
    return position, orientation, quat


__all__ = [
    "_get_orientation_dim",
    "_orientation_to_rotation",
    "_pose_to_representation",
    "_split_pose_and_gripper",
]
