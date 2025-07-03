import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import json
from pathlib import Path
from datetime import datetime
import threading
from collections import defaultdict, deque
import pickle
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

from teleop.XR.quest3.utils.log import logger
from teleop.XR.quest3.utils.mat_tool import fast_mat_inv
from teleop.XR.quest3.utils.constants import T_corenetic_openxr


def trans_to_world(pose: np.ndarray) -> np.ndarray:
    """Transforms a pose from the VR device's coordinate system to the world coordinate system.

    Args:
        pose: A 4x4 transformation matrix representing the pose in the VR device's coordinate system.
    """
    return T_corenetic_openxr @ pose @ fast_mat_inv(T_corenetic_openxr)


class TeleopDataRecorder:
    """Records teleoperation data for analysis of jitter and performance issues"""

    def __init__(self, device_name: str, max_buffer_size: int = 100000):
        self.device_name = device_name
        self.recording = False
        self.record_thread = None
        self.robot_interface = None
        self.start_time = None
        self.max_buffer_size = max_buffer_size  # Maximum size of data buffers

        # Data storage
        self.data = {
            "vr": {
                "timestamps": [],
                "left_controller": [],
                "right_controller": [],
                "left_button": [],
                "right_button": [],
                "left_arm_active": [],
                "right_arm_active": [],
                "frame_rates": [],
            },
            "robot": {
                "timestamps": [],
                "left_arm_target": [],
                "right_arm_target": [],
                "left_arm_command": [],
                "right_arm_command": [],
                "left_end_pose": [],
                "right_end_pose": [],
                "left_arm_control_latency": [],
                "right_arm_control_latency": [],
            },
            "metadata": {
                "start_time": 0,
                "end_time": 0,
                "device_name": device_name,
                "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        }

        # Create data directory
        self.data_dir = Path("teleoperation_data")
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Last record time for rate control
        self.last_vr_record_time = 0
        self.last_robot_record_time = 0
        self.vr_record_interval = 1 / 100  # Record at 100Hz
        self.robot_record_interval = 1 / 100  # Record at 100Hz

    def start_recording(self, vr, robot_interface=None):
        """Start recording teleoperation data"""
        if self.recording:
            logger.warning("Recording already in progress")
            return
        self.start_time = time.perf_counter()
        self.vr_device = vr  # Store reference to VR device
        self._reset_buffers()
        self.robot_interface = robot_interface
        self.data["metadata"]["start_time"] = time.perf_counter()
        self.recording = True

        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.daemon = True
        self.record_thread.start()

        logger.info(f"Started teleoperation data recording (Session ID: {self.data['metadata']['session_id']})")

    def stop_recording(self):
        """Stop recording teleoperation data"""
        if not self.recording:
            logger.warning("No recording in progress to stop")
            return

        self.recording = False
        self.data["metadata"]["end_time"] = time.perf_counter()

        if self.record_thread:
            self.record_thread.join(timeout=2.0)

        # Save data
        self._save_data()
        logger.info(f"Stopped teleoperation data recording (Session ID: {self.data['metadata']['session_id']})")

    def _record_loop(self):
        """Main recording loop that captures data at specified intervals"""
        while self.recording:
            current_time = time.perf_counter()

            # Record VR data at specified interval
            if current_time - self.last_vr_record_time >= self.vr_record_interval:
                self._record_vr_data()
                self.last_vr_record_time = current_time

            # Record robot data if available at specified interval
            if self.robot_interface and current_time - self.last_robot_record_time >= self.robot_record_interval:
                self._record_robot_data()
                self.last_robot_record_time = current_time

            # Sleep to avoid high CPU usage
            time.sleep(0.001)

    def _reset_buffers(self):
        """Reset data buffers to empty lists"""
        self.data = {
            "vr": {
                "timestamps": [],
                "left_controller": [],
                "right_controller": [],
                "left_button": [],
                "right_button": [],
                "left_arm_active": [],
                "right_arm_active": [],
                "frame_rates": [],
            },
            "robot": {
                "timestamps": [],
                "left_arm_target": [],
                "right_arm_target": [],
                "left_arm_command": [],
                "right_arm_command": [],
                "left_end_pose": [],
                "right_end_pose": [],
                "left_arm_control_latency": [],
                "right_arm_control_latency": [],
            },
            "metadata": {
                "start_time": 0,
                "end_time": 0,
                "device_name": self.device_name,
                "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        }

    def _trim_buffers(self):
        """Trim buffers to maximum size"""
        for key in self.data:
            for subkey in self.data[key]:
                if isinstance(self.data[key][subkey], list):
                    self.data[key][subkey] = self.data[key][subkey][-self.max_buffer_size:]

    def _record_vr_data(self):
        """Record VR device data"""
        try:
            timestamp = time.perf_counter()
            self.data["vr"]["timestamps"].append(timestamp - self.start_time)

            # Record controller poses
            self.data["vr"]["left_controller"].append(
                self.vr_device.left_controller.copy() if hasattr(self.vr_device, 'left_controller') else np.eye(4))
            self.data["vr"]["right_controller"].append(
                self.vr_device.right_controller.copy() if hasattr(self.vr_device, 'right_controller') else np.eye(4))

            # Record button states
            self.data["vr"]["left_button"].append(
                self.vr_device.left_button.copy() if hasattr(self.vr_device, 'left_button') else np.zeros(4))
            self.data["vr"]["right_button"].append(
                self.vr_device.right_button.copy() if hasattr(self.vr_device, 'right_button') else np.zeros(4))

            # Record arm active states
            self.data["vr"]["left_arm_active"].append(
                self.vr_device.left_arm_active if hasattr(self.vr_device, 'left_arm_active') else False)
            self.data["vr"]["right_arm_active"].append(
                self.vr_device.right_arm_active if hasattr(self.vr_device, 'right_arm_active') else False)

            # Record frame rates
            self.data["vr"]["frame_rates"].append(
                self.vr_device.controller_fps.copy() if hasattr(self.vr_device, 'controller_fps') else 0)
            # Trim buffers to avoid excessive memory usage
            self._trim_buffers()

        except Exception as e:
            logger.error(f"Error recording VR data: {e}")

    def _record_robot_data(self):
        """Record robot data if available"""
        try:
            vr = self.vr_device
            if not vr:
                return

            timestamp = time.perf_counter()
            self.data["robot"]["timestamps"].append(timestamp - self.start_time)

            # Record arm target and command poses
            self.data["robot"]["left_arm_target"].append(
                vr.left_arm_target.copy() if vr.left_arm_target is not None else np.eye(4))
            self.data["robot"]["right_arm_target"].append(
                vr.right_arm_target.copy() if vr.right_arm_target is not None else np.eye(4))
            self.data["robot"]["left_arm_command"].append(
                vr.left_arm_command.copy() if vr.left_arm_command is not None else np.eye(4))
            self.data["robot"]["right_arm_command"].append(
                vr.right_arm_command.copy() if vr.right_arm_command is not None else np.eye(4))
            # Record end pose
            self.data["robot"]["left_end_pose"].append(
                vr.left_arm_end_pose.copy() if vr.left_arm_end_pose is not None else np.eye(4))
            self.data["robot"]["right_end_pose"].append(
                vr.right_arm_end_pose.copy() if vr.right_arm_end_pose is not None else np.eye(4))

            # Record control latency
            self.data["robot"]["left_arm_control_latency"].append(
                vr.left_arm_control_latency if hasattr(vr, 'left_arm_control_latency') else 0.0)
            self.data["robot"]["right_arm_control_latency"].append(
                vr.right_arm_control_latency if hasattr(vr, 'right_arm_control_latency') else 0.0)

        except Exception as e:
            logger.error(f"Error recording robot data: {e}")

    def _get_vr_device_reference(self):
        """Get reference to the VR device this recorder is attached to"""
        # This is a bit of a hack - the recorder is attached to the VR device as an attribute
        # So we need to find the reference to "self" from the vr device
        import gc
        for obj in gc.get_objects():
            if hasattr(obj, 'data_recorder') and obj.data_recorder is self:
                return obj
        return None

    def _save_data(self):
        """Save recorded data to disk"""
        try:
            # Generate filename with session ID
            filename = self.data_dir / f"teleop_data_{self.data['metadata']['session_id']}.pkl"

            # Create a copy of the data with numpy arrays converted to lists for serialization
            data_to_save = {
                "vr": {
                    "timestamps": self.data["vr"]["timestamps"],
                    "left_controller": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                        self.data["vr"]["left_controller"]],
                    "right_controller": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                         self.data["vr"]["right_controller"]],
                    "left_button": [b.tolist() if isinstance(b, np.ndarray) else b for b in
                                    self.data["vr"]["left_button"]],
                    "right_button": [b.tolist() if isinstance(b, np.ndarray) else b for b in
                                     self.data["vr"]["right_button"]],
                    "left_arm_active": self.data["vr"]["left_arm_active"],
                    "right_arm_active": self.data["vr"]["right_arm_active"],
                    "frame_rates": self.data["vr"]["frame_rates"],
                },
                "robot": {
                    "timestamps": self.data["robot"]["timestamps"],
                    "left_arm_target": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                        self.data["robot"]["left_arm_target"]],
                    "right_arm_target": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                         self.data["robot"]["right_arm_target"]],
                    "left_arm_command": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                         self.data["robot"]["left_arm_command"]],
                    "right_arm_command": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                          self.data["robot"]["right_arm_command"]],
                    "left_end_pose": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                      self.data["robot"]["left_end_pose"]],
                    "right_end_pose": [m.tolist() if isinstance(m, np.ndarray) else m for m in
                                       self.data["robot"]["right_end_pose"]],
                    "left_arm_control_latency": self.data["robot"]["left_arm_control_latency"],
                    "right_arm_control_latency": self.data["robot"]["right_arm_control_latency"],
                },
                "metadata": self.data["metadata"],
            }

            # Save to pickle file
            with open(filename, 'wb') as f:
                pickle.dump(data_to_save, f)

            logger.info(f"Saved teleoperation data to {filename}")

        except Exception as e:
            logger.error(f"Error saving teleoperation data: {e}")


class TeleopDataAnalyzer:
    """Analyzes recorded teleoperation data to identify issues and performance metrics"""

    def __init__(self, data_file=None, output_dir="teleop_reports"):
        self.data = None
        self.analysis_results = {}
        self.figures = {}
        self.root = Path(__file__).parent.parent
        if data_file:
            self.load_data(data_file)
        if output_dir is not None:
            self.output_dir = self.root.joinpath(output_dir)
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.output_dir = self.root.joinpath("teleop_reports")
            self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_data(self, data_file):
        """Load teleoperation data from file"""
        try:
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            logger.info(f"Loaded teleoperation data from {data_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading teleoperation data: {e}")
            return False

    def analyze_all(self):
        """Run all analysis methods."""
        if self.data is None:
            logger.error("No data loaded for analysis")
            return False

        self.analyze_frame_rates()
        self.analyze_trajectory_smoothness()
        self.analyze_control_latency()
        self.analyze_trajectory_tracking()

        return True

    def calculate_derivatives(self, positions, timestamps):
        """Calculate velocity, acceleration, and jerk with proper filtering"""
        # Determine appropriate window size for filter (must be odd)
        window_length = min(51, len(positions) - 2 if len(positions) % 2 == 0 else len(positions) - 1)
        window_length = max(5, window_length)
        if window_length % 2 == 0:
            window_length -= 1

        # Filter positions first
        filtered_positions = np.zeros_like(positions)
        for dim in range(positions.shape[1]):
            filtered_positions[:, dim] = savgol_filter(positions[:, dim], window_length, 3)

        # Calculate velocities from filtered positions
        dt = np.diff(timestamps)
        dt = np.maximum(dt, 0.001)  # Avoid division by zero
        velocities = np.zeros_like(positions)
        velocities[1:] = np.diff(filtered_positions, axis=0) / dt.reshape(-1, 1)

        # Filter velocities before calculating accelerations
        filtered_velocities = np.zeros_like(velocities)
        for dim in range(velocities.shape[1]):
            filtered_velocities[:, dim] = savgol_filter(velocities[:, dim], window_length, 3)

        # Calculate accelerations from filtered velocities
        accelerations = np.zeros_like(positions)
        if len(dt) > 1:
            dt1 = np.diff(timestamps[:-1])
            dt1 = np.maximum(dt1, 0.001)
            accelerations[2:] = np.diff(filtered_velocities[1:], axis=0) / dt1.reshape(-1, 1)

            # Filter accelerations before calculating jerk
            filtered_accelerations = np.zeros_like(accelerations)
            for dim in range(accelerations.shape[1]):
                filtered_accelerations[:, dim] = savgol_filter(accelerations[:, dim], window_length, 3)

            # Calculate jerk from filtered accelerations
            jerks = np.zeros_like(positions)
            if len(dt1) > 1:
                dt2 = np.diff(timestamps[:-2])
                dt2 = np.maximum(dt2, 0.001)
                jerks[3:] = np.diff(filtered_accelerations[2:], axis=0) / dt2.reshape(-1, 1)
        else:
            filtered_accelerations = accelerations
            jerks = np.zeros_like(positions)

        return filtered_velocities, filtered_accelerations, jerks

    def analyze_frame_rates(self):
        """Analyze VR system frame rates from the recorded data."""
        vr_data = self.data['vr']
        timestamps = vr_data['timestamps']

        if len(timestamps) < 2:
            logger.warning("Not enough timestamps for frame rate analysis")
            return

        # Calculate time differences between consecutive frames
        time_diffs = np.diff(timestamps)
        frame_rates = 1.0 / time_diffs  # Convert to Hz

        # Calculate statistics
        mean_fps = np.mean(frame_rates)
        min_fps = np.min(frame_rates)
        max_fps = np.max(frame_rates)
        std_fps = np.std(frame_rates)
        p5_fps = np.percentile(frame_rates, 5)
        p95_fps = np.percentile(frame_rates, 95)
        dropped_frames = np.sum(frame_rates < 30)

        # Prepare figure for visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Time series plot with annotations
        axes[0].plot(timestamps[1:], frame_rates)
        axes[0].set_title('VR Data Frame Rate')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frame Rate (Hz)')

        # Add horizontal line for mean
        axes[0].axhline(y=mean_fps, color='r', linestyle='-', alpha=0.7)
        axes[0].text(timestamps[1], mean_fps, f'Mean: {mean_fps:.2f} Hz',
                     va='bottom', ha='left', backgroundcolor='w', fontsize=9)

        # Mark min and max values
        min_idx = np.argmin(frame_rates)
        max_idx = np.argmax(frame_rates)
        min_time = timestamps[min_idx + 1]  # +1 because frame_rates is calculated from diff
        max_time = timestamps[max_idx + 1]

        # Mark min value with vertical line
        axes[0].axvline(x=min_time, color='blue', linestyle='--', alpha=0.7)
        axes[0].plot(min_time, min_fps, 'bo')
        axes[0].text(min_time, min_fps, f'Min: {min_fps:.2f} Hz',
                     va='top', ha='right', fontsize=9, backgroundcolor='w')

        # Mark max value with vertical line
        axes[0].axvline(x=max_time, color='green', linestyle='--', alpha=0.7)
        axes[0].plot(max_time, max_fps, 'go')
        axes[0].text(max_time, max_fps, f'Max: {max_fps:.2f} Hz',
                     va='bottom', ha='right', fontsize=9, backgroundcolor='w')

        # Histogram with statistical annotations
        n, bins, patches = axes[1].hist(frame_rates, bins=50, alpha=0.7,
                                        density=True, label='Frame Rate')
        axes[1].set_title('VR Frame Rate Distribution')
        axes[1].set_xlabel('Frame Rate (Hz)')
        axes[1].set_ylabel('Probability Density')

        # Add a normal distribution curve for reference
        x = np.linspace(mean_fps - 4 * std_fps, mean_fps + 4 * std_fps, 100)
        y = (1 / (std_fps * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_fps) / std_fps) ** 2)
        axes[1].plot(x, y, 'r-', linewidth=2, label='Normal Dist.')

        # Mark mean and standard deviations
        axes[1].axvline(x=mean_fps, color='r', linestyle='-', alpha=0.7)
        axes[1].axvline(x=mean_fps + std_fps, color='orange', linestyle='--', alpha=0.7)
        axes[1].axvline(x=mean_fps - std_fps, color='orange', linestyle='--', alpha=0.7)
        axes[1].axvline(x=mean_fps + 2 * std_fps, color='green', linestyle=':', alpha=0.7)
        axes[1].axvline(x=mean_fps - 2 * std_fps, color='green', linestyle=':', alpha=0.7)

        # Add statistical information as text box
        stats_text = '\n'.join([
            f'Mean: {mean_fps:.2f} Hz',
            f'Std Dev: {std_fps:.2f} Hz',
            f'Min: {min_fps:.2f} Hz',
            f'Max: {max_fps:.2f} Hz',
            f'5th %: {p5_fps:.2f} Hz',
            f'95th %: {p95_fps:.2f} Hz',
            f'Dropped frames: {dropped_frames}'
        ])

        # Position text box in the upper right corner
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right', bbox=props)

        axes[1].legend()

        # Store calculated statistics
        self.analysis_results['frame_rates'] = {
            'mean_fps': mean_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'std_fps': std_fps,
            'p5_fps': p5_fps,
            'p95_fps': p95_fps,
            'dropped_frames': dropped_frames
        }

        # Log results
        logger.info(f"VR Frame Rate - Mean: {mean_fps:.2f} Hz, Min: {min_fps:.2f} Hz, Max: {max_fps:.2f} Hz")
        logger.info(f"Dropped frames: {dropped_frames}")

        plt.tight_layout()
        self.figures['frame_rates'] = fig

    def analyze_trajectory_smoothness(self):
        """Analyze smoothness of VR and robot trajectories."""
        if self.data is None:
            logger.warning("No data available for trajectory smoothness analysis")
            return

        arms = ['left', 'right']
        fig, axes = plt.subplots(len(arms), 3, figsize=(24, 10 * len(arms)))

        # Make axes accessible even with a single arm
        if len(arms) == 1:
            axes = np.array([axes]).reshape(1, -1)

        for i, arm in enumerate(arms):
            # Check if we have robot data
            has_robot_data = ('robot' in self.data and
                              f'{arm}_end_pose' in self.data['robot'] and
                              len(self.data['robot'][f'{arm}_end_pose']) > 10)

            # Get VR controller data and transform to world/robot coordinate system
            vr_poses = [np.array(pose) for pose in self.data['vr'][f'{arm}_controller']]
            try:
                vr_positions = np.array([trans_to_world(pose)[:3, 3] for pose in vr_poses])
            except Exception as e:
                logger.error(f"Error transforming VR controller poses: {e}")
                vr_positions = np.array([pose[:3, 3] for pose in vr_poses])

            vr_timestamps = np.array(self.data['vr']['timestamps'][:len(vr_positions)])

            if len(vr_positions) > 10:
                # First subplot: Controller motion profiles
                ax = axes[i, 0]

                # Calculate filtered derivatives
                vr_velocities, vr_accelerations, vr_jerks = self.calculate_derivatives(vr_positions, vr_timestamps)

                # Calculate magnitudes
                vr_vel_mag = np.linalg.norm(vr_velocities, axis=1)
                vr_acc_mag = np.linalg.norm(vr_accelerations, axis=1)
                vr_jerk_mag = np.linalg.norm(vr_jerks, axis=1)

                # Calculate statistics
                vr_vel_mean = np.mean(vr_vel_mag[1:])
                vr_vel_max = np.max(vr_vel_mag[1:])
                vr_acc_mean = np.mean(vr_acc_mag[2:])
                vr_acc_max = np.max(vr_acc_mag[2:])
                vr_jerk_mean = np.mean(vr_jerk_mag[3:])
                vr_jerk_max = np.max(vr_jerk_mag[3:])
                vr_jerk_min = np.min(vr_jerk_mag[3:])
                vr_jerk_std = np.std(vr_jerk_mag[3:])

                # Plot controller positions over time (converted to mm for readability)
                ax.plot(vr_timestamps, vr_positions[:, 0] * 1000, 'r-', linewidth=1.5, label='X', alpha=0.7)
                ax.plot(vr_timestamps, vr_positions[:, 1] * 1000, 'g-', linewidth=1.5, label='Y', alpha=0.7)
                ax.plot(vr_timestamps, vr_positions[:, 2] * 1000, 'b-', linewidth=1.5, label='Z', alpha=0.7)

                # If robot data is available, plot it for comparison
                if has_robot_data:
                    robot_poses = [np.array(pose) for pose in self.data['robot'][f'{arm}_end_pose']]
                    robot_positions = np.array([pose[:3, 3] for pose in robot_poses])
                    robot_timestamps = np.array(self.data['robot']['timestamps'][:len(robot_positions)])

                    if len(robot_timestamps) > 2 and len(robot_positions) > 2:
                        # Interpolate robot positions to match VR timestamps for better comparison
                        from scipy.interpolate import interp1d

                        # Create interpolation functions
                        try:
                            interp_x = interp1d(robot_timestamps, robot_positions[:, 0],
                                                bounds_error=False, kind='linear',
                                                fill_value="extrapolate")
                            interp_y = interp1d(robot_timestamps, robot_positions[:, 1],
                                                bounds_error=False, kind='linear',
                                                fill_value="extrapolate")
                            interp_z = interp1d(robot_timestamps, robot_positions[:, 2],
                                                bounds_error=False, kind='linear',
                                                fill_value="extrapolate")

                            # Get interpolated positions at VR timestamps
                            robot_x = interp_x(vr_timestamps)
                            robot_y = interp_y(vr_timestamps)
                            robot_z = interp_z(vr_timestamps)

                            # Plot robot trajectories as dotted lines
                            ax.plot(vr_timestamps, robot_x * 1000, 'r:', linewidth=2, label='X (Robot)')
                            ax.plot(vr_timestamps, robot_y * 1000, 'g:', linewidth=2, label='Y (Robot)')
                            ax.plot(vr_timestamps, robot_z * 1000, 'b:', linewidth=2, label='Z (Robot)')
                        except Exception as e:
                            logger.error(f"Error interpolating robot positions: {e}")

                ax.set_title(f'{arm.capitalize()} Arm Trajectory (mm)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Position (mm)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Second subplot: VR Controller jerk analysis
                ax = axes[i, 1]
                ax.plot(vr_timestamps[3:], vr_jerk_mag[3:], 'b-', alpha=0.7)

                # Mark mean jerk
                ax.axhline(y=vr_jerk_mean, color='r', linestyle='-', alpha=0.7)
                ax.text(vr_timestamps[3], vr_jerk_mean, f'Mean: {vr_jerk_mean:.2f} m/s³',
                        va='bottom', ha='left', fontsize=9, backgroundcolor='w')

                # Mark minimum jerk
                min_idx = np.argmin(vr_jerk_mag[3:]) + 3
                ax.axvline(x=vr_timestamps[min_idx], color='b', linestyle='--', alpha=0.7)
                ax.plot(vr_timestamps[min_idx], vr_jerk_mag[min_idx], 'bo')
                ax.text(vr_timestamps[min_idx], vr_jerk_mag[min_idx], f'Min: {vr_jerk_min:.2f} m/s³',
                        va='top', ha='right', fontsize=9, backgroundcolor='w')

                # Mark maximum jerk
                max_idx = np.argmax(vr_jerk_mag[3:]) + 3
                ax.axvline(x=vr_timestamps[max_idx], color='g', linestyle='--', alpha=0.7)
                ax.plot(vr_timestamps[max_idx], vr_jerk_mag[max_idx], 'go')
                ax.text(vr_timestamps[max_idx], vr_jerk_mag[max_idx], f'Max: {vr_jerk_max:.2f} m/s³',
                        va='bottom', ha='right', fontsize=9, backgroundcolor='w')

                # Add statistical information
                stats_text = '\n'.join([
                    f'Controller Jerk:',
                    f'  Mean: {vr_jerk_mean:.2f} m/s³',
                    f'  Max: {vr_jerk_max:.2f} m/s³',
                    f'  Min: {vr_jerk_min:.2f} m/s³',
                    f'  Std: {vr_jerk_std:.2f} m/s³',
                    f'% Smooth Motion: {np.mean(vr_jerk_mag[3:] < 5.0) * 100:.1f}%'
                ])

                # Add text box with statistics
                props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                        va='top', bbox=props)

                ax.set_title(f'{arm.capitalize()} Controller Jerk Analysis')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Jerk Magnitude (m/s³)')
                ax.grid(True, alpha=0.3)

                # Set y-axis limit to focus on relevant jerk range (filter extreme values)
                jerk_99percentile = np.percentile(vr_jerk_mag[3:], 99)
                ax.set_ylim([0, min(jerk_99percentile * 1.5, vr_jerk_max)])

                # Third subplot: Robot jerk analysis (if available)
                ax = axes[i, 2]
                if has_robot_data and len(robot_positions) > 10:
                    robot_velocities, robot_accelerations, robot_jerks = self.calculate_derivatives(robot_positions,
                                                                                                    robot_timestamps)
                    robot_jerk_mag = np.linalg.norm(robot_jerks, axis=1)

                    # Calculate statistics
                    robot_jerk_mean = np.mean(robot_jerk_mag[3:])
                    robot_jerk_max = np.max(robot_jerk_mag[3:])
                    robot_jerk_min = np.min(robot_jerk_mag[3:])
                    robot_jerk_std = np.std(robot_jerk_mag[3:])

                    # Plot robot jerk
                    ax.plot(robot_timestamps[3:], robot_jerk_mag[3:], 'r-', alpha=0.7)

                    # Mark mean robot jerk
                    ax.axhline(y=robot_jerk_mean, color='r', linestyle='-', alpha=0.7)
                    ax.text(robot_timestamps[3], robot_jerk_mean, f'Mean: {robot_jerk_mean:.2f} m/s³',
                            va='bottom', ha='left', fontsize=9, backgroundcolor='w')

                    # Mark minimum robot jerk
                    min_idx = np.argmin(robot_jerk_mag[3:]) + 3
                    ax.axvline(x=robot_timestamps[min_idx], color='b', linestyle='--', alpha=0.7)
                    ax.plot(robot_timestamps[min_idx], robot_jerk_mag[min_idx], 'bo')
                    ax.text(robot_timestamps[min_idx], robot_jerk_mag[min_idx],
                            f'Min: {robot_jerk_min:.2f} m/s³',
                            va='top', ha='right', fontsize=9, backgroundcolor='w')

                    # Mark maximum robot jerk
                    max_idx = np.argmax(robot_jerk_mag[3:]) + 3
                    ax.axvline(x=robot_timestamps[max_idx], color='g', linestyle='--', alpha=0.7)
                    ax.plot(robot_timestamps[max_idx], robot_jerk_mag[max_idx], 'go')
                    ax.text(robot_timestamps[max_idx], robot_jerk_mag[max_idx],
                            f'Max: {robot_jerk_max:.2f} m/s³',
                            va='bottom', ha='right', fontsize=9, backgroundcolor='w')

                    # Add robot statistics
                    stats_text = '\n'.join([
                        f'Robot Jerk:',
                        f'  Mean: {robot_jerk_mean:.2f} m/s³',
                        f'  Max: {robot_jerk_max:.2f} m/s³',
                        f'  Min: {robot_jerk_min:.2f} m/s³',
                        f'  Std: {robot_jerk_std:.2f} m/s³',
                        f'% Smooth Motion: {np.mean(robot_jerk_mag[3:] < 5.0) * 100:.1f}%'
                    ])

                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                            va='top', bbox=props)

                    ax.set_title(f'{arm.capitalize()} Robot Jerk Analysis')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Jerk Magnitude (m/s³)')
                    ax.grid(True, alpha=0.3)

                    # Set y-axis limit to focus on relevant jerk range
                    jerk_99percentile = np.percentile(robot_jerk_mag[3:], 99)
                    ax.set_ylim([0, min(jerk_99percentile * 1.5, robot_jerk_max)])

                    # Store robot smoothness results
                    self.analysis_results.setdefault('smoothness', {})[f'robot_{arm}'] = {
                        'mean_jerk': robot_jerk_mean,
                        'max_jerk': robot_jerk_max,
                        'min_jerk': robot_jerk_min,
                        'std_jerk': robot_jerk_std,
                        'smooth_motion_percentage': np.mean(robot_jerk_mag[3:] < 5.0) * 100
                    }
                else:
                    ax.text(0.5, 0.5, "No robot data available",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{arm.capitalize()} Robot Jerk Analysis')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Jerk Magnitude (m/s³)')

                # Store controller smoothness results
                self.analysis_results.setdefault('smoothness', {})[f'vr_{arm}'] = {
                    'mean_jerk': vr_jerk_mean,
                    'max_jerk': vr_jerk_max,
                    'min_jerk': vr_jerk_min,
                    'std_jerk': vr_jerk_std,
                    'smooth_motion_percentage': np.mean(vr_jerk_mag[3:] < 5.0) * 100
                }
            else:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, "Not enough trajectory data",
                                    ha='center', va='center', transform=axes[i, j].transAxes)

        plt.tight_layout()
        self.figures['trajectory_smoothness'] = fig

    def analyze_control_latency(self):
        """Analyze control latency between VR commands and robot execution."""
        if 'robot' not in self.data or len(self.data['robot']['timestamps']) < 2:
            logger.warning("No robot data available for latency analysis")
            return

        arms = ['left', 'right']
        fig, axes = plt.subplots(len(arms), 2, figsize=(15, 6 * len(arms)))

        # Make axes accessible even with a single arm
        if len(arms) == 1:
            axes = np.array([axes])

        for i, arm in enumerate(arms):
            # Skip if no robot data for this arm
            if f'{arm}_arm_control_latency' not in self.data['robot'] or len(
                    self.data['robot'][f'{arm}_arm_control_latency']) == 0:
                for j in range(2):
                    axes[i, j].text(0.5, 0.5, f"No latency data for {arm} arm",
                                    ha='center', va='center', transform=axes[i, j].transAxes)
                continue

            # Get recorded control latency
            latency = np.array(self.data['robot'][f'{arm}_arm_control_latency'])
            timestamps = self.data['robot']['timestamps'][:len(latency)]

            # Calculate statistics
            mean_latency = np.mean(latency) * 1000  # Convert to ms
            min_latency = np.min(latency) * 1000
            max_latency = np.max(latency) * 1000
            std_latency = np.std(latency) * 1000
            p5_latency = np.percentile(latency, 5) * 1000
            p95_latency = np.percentile(latency, 95) * 1000

            # Plot latency over time
            ax = axes[i, 0]
            ax.plot(timestamps, latency * 1000)  # Convert to milliseconds
            ax.set_title(f'{arm.capitalize()} Arm Control Latency')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Latency (ms)')
            ax.grid(True, alpha=0.3)

            # Add horizontal line for mean
            ax.axhline(y=mean_latency, color='r', linestyle='-', alpha=0.7)
            ax.text(timestamps[0], mean_latency, f'Mean: {mean_latency:.2f} ms',
                    va='bottom', ha='left', backgroundcolor='w', fontsize=9)

            # Mark minimum latency
            min_idx = np.argmin(latency)
            min_time = timestamps[min_idx]
            ax.axvline(x=min_time, color='b', linestyle='--', alpha=0.7)
            ax.plot(min_time, min_latency, 'bo')
            ax.text(min_time, min_latency, f'Min: {min_latency:.2f} ms',
                    va='top', ha='right', fontsize=9, backgroundcolor='w')

            # Mark maximum latency
            max_idx = np.argmax(latency)
            max_time = timestamps[max_idx]
            ax.axvline(x=max_time, color='g', linestyle='--', alpha=0.7)
            ax.plot(max_time, max_latency, 'go')
            ax.text(max_time, max_latency, f'Max: {max_latency:.2f} ms',
                    va='bottom', ha='right', fontsize=9, backgroundcolor='w')

            # Plot latency histogram with statistical annotations
            ax = axes[i, 1]
            n, bins, patches = ax.hist(latency * 1000, bins=50, alpha=0.7,
                                       density=True, label='Latency')
            ax.set_title(f'{arm.capitalize()} Arm Latency Distribution')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Probability Density')

            # Add a normal distribution curve for reference
            if std_latency > 1e-10:  # Only plot normal distribution if there's variation
                x = np.linspace(mean_latency - 4 * std_latency, mean_latency + 4 * std_latency, 100)
                y = (1 / (std_latency * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_latency) / std_latency) ** 2)
                ax.plot(x, y, 'r-', linewidth=2, label='Normal Dist.')
            else:
                # Just add a vertical line at the mean if no variation
                ax.axvline(x=mean_latency, color='r', linewidth=2, label='Mean (No Variation)')
                ax.text(mean_latency, ax.get_ylim()[1] * 0.9, f'Single value: {mean_latency:.2f} ms',
                        va='top', ha='center', fontsize=9, backgroundcolor='w')

            # Mark mean and standard deviations
            ax.axvline(x=mean_latency, color='r', linestyle='-', alpha=0.7)
            ax.axvline(x=mean_latency + std_latency, color='orange', linestyle='--', alpha=0.7)
            ax.axvline(x=mean_latency - std_latency, color='orange', linestyle='--', alpha=0.7)
            ax.axvline(x=mean_latency + 2 * std_latency, color='green', linestyle=':', alpha=0.7)
            ax.axvline(x=mean_latency - 2 * std_latency, color='green', linestyle=':', alpha=0.7)

            # Add statistical information as text box
            stats_text = '\n'.join([
                f'Mean: {mean_latency:.2f} ms',
                f'Std Dev: {std_latency:.2f} ms',
                f'Min: {min_latency:.2f} ms',
                f'Max: {max_latency:.2f} ms',
                f'5th %: {p5_latency:.2f} ms',
                f'95th %: {p95_latency:.2f} ms',
                f'% < 10ms: {np.mean(latency < 0.01) * 100:.1f}%',
                f'% < 5ms: {np.mean(latency < 0.005) * 100:.1f}%'
            ])

            # Position text box in the upper right corner
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            ax.legend()

            # Store calculated statistics
            self.analysis_results.setdefault('control_latency', {})[arm] = {
                'mean_ms': mean_latency,
                'min_ms': min_latency,
                'max_ms': max_latency,
                'std_ms': std_latency,
                'p5_ms': p5_latency,
                'p95_ms': p95_latency,
                'percent_under_20ms': np.mean(latency < 0.02) * 100,
                'percent_under_50ms': np.mean(latency < 0.05) * 100
            }

            # Log results
            logger.info(f"{arm.capitalize()} Arm Control Latency - Mean: {mean_latency:.2f} ms, "
                        f"Min: {min_latency:.2f} ms, Max: {max_latency:.2f} ms")

        plt.tight_layout()
        self.figures['control_latency'] = fig

    def analyze_trajectory_tracking(self):
        """Analyze how well the robot trajectory follows the commanded trajectory."""
        if 'robot' not in self.data or len(self.data['robot']['timestamps']) < 2:
            logger.warning("No robot data available for trajectory tracking analysis")
            return

        arms = ['left', 'right']
        fig, axes = plt.subplots(len(arms), 3, figsize=(18, 6 * len(arms)))

        # Make axes accessible even with a single arm
        if len(arms) == 1:
            axes = np.array([axes]).reshape(1, -1)

        for i, arm in enumerate(arms):
            # Skip if no robot data for this arm
            if f'{arm}_arm_command' not in self.data['robot'] or f'{arm}_end_pose' not in self.data['robot']:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, f"No trajectory data for {arm} arm",
                                    ha='center', va='center', transform=axes[i, j].transAxes)
                continue

            # Get command and actual positions
            command_poses = [np.array(pose) for pose in self.data['robot'][f'{arm}_arm_command']]
            actual_poses = [np.array(pose) for pose in self.data['robot'][f'{arm}_end_pose']]
            timestamps = np.array(self.data['robot']['timestamps'])

            # Truncate to the same length
            min_length = min(len(command_poses), len(actual_poses), len(timestamps))
            command_poses = command_poses[:min_length]
            actual_poses = actual_poses[:min_length]
            timestamps = timestamps[:min_length]

            # Extract positions
            command_positions = np.array([pose[:3, 3] for pose in command_poses])
            actual_positions = np.array([pose[:3, 3] for pose in actual_poses])

            # Calculate position error (in mm for better readability)
            position_errors = np.linalg.norm(command_positions - actual_positions, axis=1) * 1000  # convert to mm

            # Calculate error statistics
            mean_error = np.mean(position_errors)
            max_error = np.max(position_errors)
            min_error = np.min(position_errors)
            std_error = np.std(position_errors)
            p95_error = np.percentile(position_errors, 95)

            # Plot position error over time with annotations
            ax = axes[i, 0]
            ax.plot(timestamps, position_errors)
            ax.set_title(f'{arm.capitalize()} Arm Position Error')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position Error (mm)')
            ax.grid(True, alpha=0.3)

            # Add horizontal line for mean error
            ax.axhline(y=mean_error, color='r', linestyle='-', alpha=0.7)
            ax.text(timestamps[0], mean_error, f'Mean: {mean_error:.2f} mm',
                    va='bottom', ha='left', backgroundcolor='w', fontsize=9)

            # Mark minimum error
            min_idx = np.argmin(position_errors)
            min_time = timestamps[min_idx]
            ax.axvline(x=min_time, color='b', linestyle='--', alpha=0.7)
            ax.plot(min_time, min_error, 'bo')
            ax.text(min_time, min_error, f'Min: {min_error:.2f} mm',
                    va='top', ha='right', fontsize=9, backgroundcolor='w')

            # Mark maximum error
            max_idx = np.argmax(position_errors)
            max_time = timestamps[max_idx]
            ax.axvline(x=max_time, color='g', linestyle='--', alpha=0.7)
            ax.plot(max_time, max_error, 'go')
            ax.text(max_time, max_error, f'Max: {max_error:.2f} mm',
                    va='bottom', ha='right', fontsize=9, backgroundcolor='w')

            # Add statistical information as text box
            stats_text = '\n'.join([
                f'Mean: {mean_error:.2f} mm',
                f'Std Dev: {std_error:.2f} mm',
                f'Min: {min_error:.2f} mm',
                f'Max: {max_error:.2f} mm',
                f'95th %: {p95_error:.2f} mm',
                f'% < 5mm: {np.mean(position_errors < 5.0) * 100:.1f}%',
                f'% < 10mm: {np.mean(position_errors < 10.0) * 100:.1f}%'
            ])

            # Position text box in the upper right corner
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            # Plot error histogram with annotations
            ax = axes[i, 1]
            n, bins, patches = ax.hist(position_errors, bins=40, alpha=0.7, density=True)
            ax.set_title(f'{arm.capitalize()} Position Error Distribution')
            ax.set_xlabel('Position Error (mm)')
            ax.set_ylabel('Probability Density')

            # Add a normal distribution curve for reference if variance > 0
            if std_error > 1e-6:
                x = np.linspace(max(0, mean_error - 4 * std_error), mean_error + 4 * std_error, 100)
                y = (1 / (std_error * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_error) / std_error) ** 2)
                ax.plot(x, y, 'r-', linewidth=2, label='Normal Dist.')

                # Mark mean and standard deviations
                ax.axvline(x=mean_error, color='r', linestyle='-', alpha=0.7, label='Mean')
                ax.axvline(x=mean_error + std_error, color='orange', linestyle='--', alpha=0.7, label='±1σ')
                ax.axvline(x=mean_error - std_error, color='orange', linestyle='--', alpha=0.7)
                ax.axvline(x=mean_error + 2 * std_error, color='green', linestyle=':', alpha=0.7, label='±2σ')
                ax.axvline(x=mean_error - 2 * std_error, color='green', linestyle=':', alpha=0.7)

                ax.legend(loc='lower right')
            else:
                ax.axvline(x=mean_error, color='r', linewidth=2, label='Mean (No Variation)')
                ax.text(mean_error, ax.get_ylim()[1] * 0.9, f'Single value: {mean_error:.2f} mm',
                        va='top', ha='center', fontsize=9, backgroundcolor='w')

            # Display the same statistics box in histogram plot
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            # 3D trajectory visualization with error amplification
            ax = axes[i, 2]
            ax = fig.add_subplot(len(arms), 3, i * 3 + 3, projection='3d')

            # Plot commanded trajectory
            ax.plot(command_positions[:, 0], command_positions[:, 1], command_positions[:, 2],
                    'b-', linewidth=1.5, label='Target', alpha=0.7)

            # Plot actual trajectory
            ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2],
                    'r-', linewidth=1.5, label='Actual', alpha=0.7)

            # Add connecting lines between corresponding points to show error
            # Only show every Nth point to avoid cluttering (adjust N based on data density)
            N = max(1, len(timestamps) // 30)
            for j in range(0, len(timestamps), N):
                # Use color gradient based on error magnitude
                error_norm = position_errors[j] / max_error  # Normalize to [0,1]
                color = plt.cm.jet(error_norm)  # Use color map

                ax.plot([command_positions[j, 0], actual_positions[j, 0]],
                        [command_positions[j, 1], actual_positions[j, 1]],
                        [command_positions[j, 2], actual_positions[j, 2]],
                        'k', alpha=0.4, linewidth=1, color=color)

            # Create a separate scatter plot with error magnitude as color
            sc = ax.scatter(command_positions[::N, 0], command_positions[::N, 1], command_positions[::N, 2],
                            c=position_errors[::N], cmap='jet', s=10, alpha=0.8)

            # Add color bar to show error scale
            cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
            cbar.set_label('Error (mm)')

            # Set equal aspect ratio to correctly visualize distances
            max_range = np.max([
                np.ptp(np.concatenate([command_positions[:, 0], actual_positions[:, 0]])),
                np.ptp(np.concatenate([command_positions[:, 1], actual_positions[:, 1]])),
                np.ptp(np.concatenate([command_positions[:, 2], actual_positions[:, 2]]))
            ])

            # Calculate the mid points
            mid_x = np.mean(np.concatenate([command_positions[:, 0], actual_positions[:, 0]]))
            mid_y = np.mean(np.concatenate([command_positions[:, 1], actual_positions[:, 1]]))
            mid_z = np.mean(np.concatenate([command_positions[:, 2], actual_positions[:, 2]]))

            # Set plot limits to be centered and equal range
            ax.set_xlim([mid_x - max_range / 2, mid_x + max_range / 2])
            ax.set_ylim([mid_y - max_range / 2, mid_y + max_range / 2])
            ax.set_zlim([mid_z - max_range / 2, mid_z + max_range / 2])

            ax.set_title(f'{arm.capitalize()} 3D Trajectory')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.legend()

            # Store tracking performance results
            self.analysis_results.setdefault('tracking', {})[arm] = {
                'mean_error_mm': mean_error,
                'max_error_mm': max_error,
                'min_error_mm': min_error,
                'std_error_mm': std_error,
                'p95_error_mm': p95_error,
                'percent_under_5mm': np.mean(position_errors < 5.0) * 100,
                'percent_under_10mm': np.mean(position_errors < 10.0) * 100
            }

            # Log results
            logger.info(f"{arm.capitalize()} Arm Tracking Error - Mean: {mean_error:.2f} mm, "
                        f"Max: {max_error:.2f} mm, 95th percentile: {p95_error:.2f} mm")

        plt.tight_layout()
        self.figures['trajectory_tracking'] = fig

    def generate_report(self):
        """Generate a comprehensive report of the analysis results."""
        if not self.analysis_results:
            logger.warning("No analysis results available")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"teleop_analysis_report_{timestamp}.html"
        output_path = self.output_dir.joinpath(output_path)
        # Create HTML report
        html = ["<html><head><title>Teleoperation Analysis Report</title>",
                "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse;width:100%}",
                "th,td{text-align:left;padding:8px;border:1px solid #ddd}",
                "th{background-color:#f2f2f2}</style></head><body>",
                "<h1>Teleoperation Analysis Report</h1>"]

        # Add metadata
        if 'metadata' in self.data:
            html.append("<h2>Session Information</h2>")
            html.append("<table>")
            html.append("<tr><th>Property</th><th>Value</th></tr>")
            for key, value in self.data['metadata'].items():
                html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            html.append("</table>")

        # Add analysis results
        html.append("<h2>Analysis Results</h2>")

        # Frame rates
        if 'frame_rates' in self.analysis_results:
            html.append("<h3>Frame Rate Analysis</h3>")
            html.append("<table>")
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            for key, value in self.analysis_results['frame_rates'].items():
                html.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
            html.append("</table>")

        # Latency
        if 'latency' in self.analysis_results:
            html.append("<h3>Control Latency Analysis</h3>")
            for arm, metrics in self.analysis_results['latency'].items():
                html.append(f"<h4>{arm.capitalize()} Arm</h4>")
                html.append("<table>")
                html.append("<tr><th>Metric</th><th>Value</th></tr>")
                for key, value in metrics.items():
                    html.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
                html.append("</table>")

        # Tracking
        if 'tracking' in self.analysis_results:
            html.append("<h3>Trajectory Tracking Analysis</h3>")
            for arm, metrics in self.analysis_results['tracking'].items():
                html.append(f"<h4>{arm.capitalize()} Arm</h4>")
                html.append("<table>")
                html.append("<tr><th>Metric</th><th>Value</th></tr>")
                for key, value in metrics.items():
                    html.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
                html.append("</table>")

        # Jitter
        if 'jitter' in self.analysis_results:
            html.append("<h3>Jitter Analysis</h3>")
            for device, metrics in self.analysis_results['jitter'].items():
                html.append(f"<h4>{device}</h4>")
                html.append("<table>")
                html.append("<tr><th>Metric</th><th>Value</th></tr>")
                for key, value in metrics.items():
                    html.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
                html.append("</table>")

        # Close HTML
        html.append("</body></html>")

        # Write to file
        try:
            with open(str(output_path), 'w') as f:
                f.write("\n".join(html))
            logger.info(f"Report generated at {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing report: {e}")
            return False

    def save_figures(self, ):
        """Save all generated figures to the specified directory."""
        if not self.figures:
            logger.warning("No figures available to save")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"teleop_analysis_figures_{timestamp}"
        output_dir = self.output_dir.joinpath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each figure
        for name, fig in self.figures.items():
            output_path = output_dir.joinpath(f"{name}.png")
            try:
                fig.savefig(str(output_path), dpi=150)
                logger.info(f"Saved figure to {output_path}")
            except Exception as e:
                logger.error(f"Error saving figure {name}: {e}")

        return True


if __name__ == "__main__":
    # Example usage
    recorder = TeleopDataAnalyzer()
    recorder.load_data("/home/sillyman/Project/teleoperation/teleoperation_data/teleop_data_20250610_200301.pkl")
    recorder.analyze_all()
    recorder.generate_report()
    recorder.save_figures()
