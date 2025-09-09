"""
Animation plotter for real-time visualization of joint states and actions
"""

import os
import time
import threading
import collections
import queue
import json
from typing import Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glog as log


class AnimationPlotter:
    """Multi-canvas real-time animation plotter for joint states and actions."""

    def __init__(
        self,
        joint_state_names: list[str],
        action_names: list[str],
        max_points: int = 50,
        figsize: tuple[int, int] = (10, 6),
        cols: int = 3,
        is_debug = False,
        update_frequency = 100
    ) -> None:
        """Initialize multi-canvas animation plotter.

        Args:
            joint_state_names: Names of joint states to plot
            action_names: Names of actions to plot
            max_points: Maximum display points (sliding window buffer)
            figsize: Overall figure size
            cols: Number of subplot columns
            is_debug: Enable debug logging
            update_frequency: Update frequency for plots (Hz)
        """
        assert len(joint_state_names) == len(action_names), f"len of joint name{len(joint_state_names)} != len action name {len(action_names)}"

        self.joint_state_names = joint_state_names
        self.action_names = action_names
        self.signal_count = len(joint_state_names)
        self.max_points = max_points
        self.figsize = figsize
        self.is_debug = is_debug
        self._has_gui = False  # Will be set in _init_plots()
        self._update_frequency = update_frequency

        # Smart column calculation: don't use more columns than signals
        self.cols = min(cols, self.signal_count)
        self.rows = (self.signal_count + self.cols - 1) // self.cols

        # Simplified thread safety
        self._lock = threading.Lock()
        self._active = False  # Single state flag
        self._update_timer = None
        self._plot_queue: queue.Queue = queue.Queue(maxsize=100)

        # Data storage with fixed-size buffers
        self.joint_data: dict[int, collections.deque] = {}
        self.action_data: dict[int, collections.deque] = {}
        self.timestamps: collections.deque = collections.deque(maxlen=max_points)

        for i in range(self.signal_count):
            self.joint_data[joint_state_names[i]] = collections.deque(maxlen=max_points)
            self.action_data[action_names[i]] = collections.deque(maxlen=max_points)

        # Matplotlib objects
        self.fig = None
        self.axes = None
        self.joint_lines = None
        self.action_lines = None

        log.info(f"AnimationPlotter initialized with {self.signal_count} signals, "
                f"max_points={max_points}, layout={self.rows}x{cols}")

    def _setup_matplotlib_backend(self) -> bool:
        """Simplified backend setup."""
        current_backend = matplotlib.get_backend()

        # If not using Agg, assume it works
        if current_backend != 'Agg':
            log.info(f"Using existing backend: {current_backend}")
            return True

        # Simple DISPLAY check
        if os.environ.get('DISPLAY'):
            # Try TkAgg first (most common)
            matplotlib.use('TkAgg')
            log.info("Using GUI backend: TkAgg")
            return True

        # Headless mode
        matplotlib.use('Agg')
        log.warning("Using headless mode (Agg backend)")
        return False

    def _init_plots(self) -> None:
        """Initialize matplotlib subplots with simplified backend selection."""
        # Setup backend
        self._has_gui = self._setup_matplotlib_backend()

        if self._has_gui:
            plt.ion()  # Enable interactive mode for GUI

        self.fig, self.axes = plt.subplots(
            self.rows, self.cols,
            figsize=self.figsize,
            facecolor='white'
        )

        # Handle different subplot configurations
        if self.signal_count == 1 and self.rows == 1 and self.cols == 1:
            # Single subplot case
            self.axes = [self.axes]
        else:
            # Multiple subplots - always flatten to get a consistent list
            self.axes = self.axes.flatten()

        self.joint_lines = []
        self.action_lines = []

        for i in range(self.signal_count):
            ax = self.axes[i]
            ax.set_title(f"plot_{i}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time Steps", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)

            # 确保线条样式正确：joint_state实线，action虚线
            joint_line, = ax.plot([], [], 'b-', linewidth=2, label=self.joint_state_names[i], alpha=0.9)
            action_line, = ax.plot([], [], 'r--', linewidth=2, label=self.action_names[i], alpha=0.9)

            self.joint_lines.append(joint_line)
            self.action_lines.append(action_line)

            ax.legend(loc='upper right', fontsize=9)

            # 设置初始轴限制，避免空白
            ax.set_xlim(0, 10)
            ax.set_ylim(-2, 2)

        # Hide unused subplots
        for i in range(self.signal_count, len(self.axes)):
            self.axes[i].set_visible(False)

        plt.tight_layout()
        self._cur_time_stamp = 0
        log.info("Matplotlib subplots initialized")

    def update_signal(
        self,
        joint_states: Union[list, np.ndarray],
        actions: Union[list, np.ndarray]
    ) -> None:
        """Thread-safe update of signal data via queue mechanism.

        This method can be safely called from any thread. The actual plotting
        is handled by the main thread updater.

        Args:
            joint_states: Joint state data, length must match signal_names
            actions: Model output action data, length must match signal_names
        """
        # Convert to numpy arrays for consistent handling
        if not isinstance(joint_states, np.ndarray):
            joint_states = np.array(joint_states)
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)

        self._queue_plot_update(joint_states, actions)

    def _queue_plot_update(self, joint_states: np.ndarray, actions: np.ndarray) -> None:
        """Queue plot data for thread-safe processing.

        Args:
            joint_states: Joint state data to plot
            actions: Action data to plot
        """
        try:
            current_time = time.time()
            plot_data = {
                'timestamp': current_time,
                'joint_states': joint_states.copy(),
                'actions': actions.copy()
            }

            # Non-blocking put with LRU policy
            try:
                self._plot_queue.put_nowait(plot_data)
            except queue.Full:
                # Remove oldest item and add new one (LRU policy)
                try:
                    self._plot_queue.get_nowait()
                    self._plot_queue.put_nowait(plot_data)
                except queue.Empty:
                    pass  # Queue was emptied by another thread

        except Exception as e:
            log.warning(f"Failed to queue plot data: {e}")

    def _process_plot_queue(self) -> None:
        """Simplified queue processing (main thread only)."""
        if not self._active or self.fig is None:
            return

        # Process all available queue items
        updates = 0
        while not self._plot_queue.empty():
            plot_data = self._plot_queue.get_nowait()
            self._update_data_buffers(plot_data)
            updates += 1

        if updates > 0:
            self._update_plots_directly()

    def _update_data_buffers(self, plot_data: dict) -> None:
        """Update internal data buffers with queued data.

        Args:
            plot_data: Dictionary containing timestamp, joint_states, actions
        """
        joint_states = plot_data['joint_states']
        actions = plot_data['actions']

        # Dimension validation
        if len(joint_states) != self.signal_count or len(actions) != self.signal_count:
            log.warning(f"Data dimension mismatch: joints={len(joint_states)}, actions={len(actions)}, expected={self.signal_count}")
            return

        with self._lock:
            # Add timestamp (using sequential index)
            len_data = len(self.timestamps)
            need_to_pop = False
            if len_data >= self.max_points:
                self.timestamps.popleft()
                need_to_pop = True
            self.timestamps.append(self._cur_time_stamp)
            self._cur_time_stamp += 1

            # Add data points for each signal
            for i in range(self.signal_count):
                if need_to_pop:
                    self.joint_data[self.joint_state_names[i]].popleft()
                    self.action_data[self.action_names[i]].popleft()
                self.joint_data[self.joint_state_names[i]].append(float(joint_states[i]))
                self.action_data[self.action_names[i]].append(float(actions[i]))

            # Debug: Log data update occasionally
            if self._cur_time_stamp % 10 == 0 and self.is_debug:
                log.info(f"📈 Data update {self._cur_time_stamp}: joints={[f'{float(joint_states[j]):.3f}' for j in range(min(2, len(joint_states)))]},"
                         f" actions={[f'{float(actions[j]):.3f}' for j in range(min(2, len(actions)))]}")

    def _update_plots_directly(self) -> None:
        """Update matplotlib plots directly (main thread only)."""
        with self._lock:
            # Check if we have data to plot
            if len(self.timestamps) == 0:
                return

            # Get current data
            x_data = list(self.timestamps)

            # Update each signal subplot
            for i in range(self.signal_count):
                joint_key = self.joint_state_names[i]
                action_key = self.action_names[i]
                if (i >= len(self.joint_lines) or
                    len(self.joint_data[joint_key]) <= 0 or
                    len(self.action_data[action_key]) <= 0):
                    log.warning(f"Skipping update for subplot {i} due to insufficient data or missing lines")
                    continue

                # Get y data for this signal
                joint_y = list(self.joint_data[joint_key])
                action_y = list(self.action_data[action_key])
                # Update line data in main thread
                self.joint_lines[i].set_data(x_data, joint_y)
                self.action_lines[i].set_data(x_data, action_y)

                # Update axis limits for visibility
                if x_data:
                    x_min, x_max = min(x_data), max(x_data)
                    if x_max > x_min:
                        self.axes[i].set_xlim(x_min - 1, x_max + 1)

                if joint_y or action_y:
                    all_y = joint_y + action_y
                    if all_y:
                        y_min, y_max = min(all_y), max(all_y)
                        if y_max > y_min:
                            y_margin = max(0.2, (y_max - y_min) * 0.15)
                            self.axes[i].set_ylim(y_min - y_margin, y_max + y_margin)
                        else:
                            self.axes[i].set_ylim(y_min - 1, y_min + 1)

            # Redraw in main thread
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def start_animation(self) -> None:
        """Start simplified real-time plotting display."""
        if self._active:
            log.warning("Animation already running")
            return

        try:
            # Initialize plots
            self._init_plots()

            # Start interactive mode
            plt.ion()
            plt.show(block=False)

            # Try to bring window to front
            try:
                self.fig.canvas.manager.window.raise_()
                self.fig.canvas.manager.window.wm_attributes('-topmost', 1)
                self.fig.canvas.manager.window.wm_attributes('-topmost', 0)
            except (AttributeError, RuntimeError):
                log.info("Window focus adjustment not supported on this system")

            # Initial draw
            self.fig.canvas.draw()
            plt.pause(0.01)

            log.info("Animation started with queue-based updates")
        except Exception as e:
            log.error(f"Failed to start animation: {e}")
            self._active = False

    def start_main_thread_updater(self) -> None:
        """Start the main thread plot updater timer.

        Must be called from the main thread.

        Raises:
            RuntimeError: If not called from main thread
        """
        assert threading.current_thread() == threading.main_thread(), "Must be called from main thread"

        if self._active:
            log.warning("Updater already running")
            return

        self._active = True

        def _timer_callback():
            if self._active:
                self._process_plot_queue()

        if self._has_gui and self.fig is not None:
            # Use matplotlib timer
            self._update_timer = self.fig.canvas.new_timer(interval=1000/self._update_frequency)
            self._update_timer.add_callback(_timer_callback)
            self._update_timer.start()
        log.info("Main thread plot updater started")

    def stop_animation(self) -> None:
        """Stop animation and cleanup."""
        self._active = False

        # Stop timer
        if self._update_timer is not None:
            self._update_timer.stop()
            self._update_timer = None

        # Clear queue
        while not self._plot_queue.empty():
            self._plot_queue.get_nowait()

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        log.info("Animation stopped")

    def clear_data(self) -> None:
        """Clear all stored data and queue."""
        with self._lock:
            self.timestamps.clear()
            for key in self.joint_data:
                self.joint_data[key].clear()
            for key in self.action_data:
                self.action_data[key].clear()
            self._cur_time_stamp = 0

        # Clear plot queue
        while not self._plot_queue.empty():
            try:
                self._plot_queue.get_nowait()
            except queue.Empty:
                break

        log.info("Animation data and queue cleared")

    def save_data(self, filepath: str) -> None:
        """Save current trajectory data to JSON file.

        Args:
            filepath: Output file path
        """
        with self._lock:
            data = {
                'joint_state_names': self.joint_state_names,
                'action_names': self.action_names,
                'timestamps': list(self.timestamps),
                'joint_data': {i: list(self.joint_data[self.joint_state_names[i]]) for i in range(self.signal_count)},
                'action_data': {i: list(self.action_data[self.action_names[i]]) for i in range(self.signal_count)},
                'max_points': self.max_points
            }

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            log.info(f"Trajectory data saved to {filepath}")
        except Exception as e:
            log.error(f"Failed to save data to {filepath}: {e}")
            raise