"""
Utilities for inference tasks.

Keep imports lightweight and robust so headless environments (no X server) can
still run inference even if optional GUI/keyboard backends are unavailable.
"""

from .gripper_controller import GripperStateLogger
from .display import display_images, create_image_grid, calculate_grid_layout
from .plotter import AnimationPlotter
from .gripper_visualizer import GripperVisualizer
from .gripper_visualization_wrapper import GripperVisualizationWrapper
from .camera_handler import CameraHandler
from .performance_monitor import PerformanceMonitor
from .state_processor import StateProcessor

# Keyboard handler depends on `pynput`, which may fail to initialize on
# headless servers (e.g. missing X server). Import it defensively and provide
# a no-op stub so everything else can still be imported.
try:  # pragma: no cover - small defensive import shim
    from .keyboard_handler import KeyboardHandler
except Exception as e:  # ImportError or backend-specific errors
    import glog as log

    log.warning(
        "KeyboardHandler could not be imported (%s). "
        "Running without interactive keyboard controls.", e
    )

    class KeyboardHandler:  # type: ignore[no-redef]
        """Fallback stub when real KeyboardHandler is unavailable."""

        def __init__(self, *_, **__) -> None:
            log.warning(
                "KeyboardHandler stub in use; keyboard input is disabled "
                "(no X server / pynput backend)."
            )

        def wait_for_key_press(self, prompt_message: str, valid_keys=None) -> str:
            log.warning(
                "wait_for_key_press() called on stub KeyboardHandler; "
                "always returning 'q'. Prompt was: %s", prompt_message
            )
            # Default to a safe "quit" semantics for callers that expect a key.
            return "q"

        def cleanup(self) -> None:
            # Nothing to clean up in the stub.
            return


__all__ = [
    "GripperStateLogger",
    "display_images",
    "create_image_grid",
    "calculate_grid_layout",
    "AnimationPlotter",
    "GripperVisualizer",
    "GripperVisualizationWrapper",
    "CameraHandler",
    "PerformanceMonitor",
    "KeyboardHandler",
    "StateProcessor",
]
