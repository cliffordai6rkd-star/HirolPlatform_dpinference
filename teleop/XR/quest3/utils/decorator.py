import asyncio
from collections import deque
import functools
import time

import numpy as np
from teleop.XR.quest3.utils.state_code import *
from teleop.XR.quest3.utils.log import logger


def device_is_connected(func):
    """Decorator to check if the Device is connected before executing the function."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if self.state_manager.connect_state == ConnectState.CONNECTED:
            return await func(self, *args, **kwargs)
        else:
            # logger.error(f'Device {self.device_name} is not connected')
            return await asyncio.sleep(0)

    return wrapper


def device_is_ready(func):
    """Decorator to check if the Device is ready before executing the function."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if self.state_manager.current_state != OperationState.OPERATING:
            # logger.warning("Device is not ready to operate")
            return await asyncio.sleep(0)
        return await func(self, *args, **kwargs)

    return wrapper


def motion_is_enable(func):
    """Decorator to check if the motion is enabled before executing the function."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.disable_motion:
            logger.warning(f"Motion is disabled for {self.robot_name}")
            return None
        return func(self, *args, **kwargs)

    return wrapper


def timer(func):
    """Decorator to measure the execution time of a function."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution time of {func.__name__}: {end_time - start_time:.5f} seconds")
        return result

    return wrapper


def fps_statistics(queue_size=30, fps_threshold=50):
    """Decorator to track frame rate statistics for controller and hand movement methods.

    Args:
        queue_size: Maximum number of timestamps to keep for calculating frame rate
        fps_threshold: Frame rate threshold below which a warning is logged
    """

    def decorator(func):
        timestamps = deque(maxlen=queue_size)

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            current_time = time.perf_counter()
            # Calculate time difference from previous frame if available
            if timestamps and len(timestamps) > 0:
                # Calculate interval between consecutive frames
                time_diff = current_time - timestamps[-1]
                if time_diff > 0.001:  # Avoid division by zero
                    instantaneous_fps = 1.0 / time_diff

                    # Store the frame rate in the instance
                    method_name = func.__name__
                    if method_name == "on_controller_move":
                        self._controller_fps_shared.value = instantaneous_fps
                    # Check for low frame rate
                    if instantaneous_fps < fps_threshold:
                        logger.warning(f"Low frame rate detected in {method_name}: {instantaneous_fps:.2f} FPS")

            # Add current timestamp to the queue
            timestamps.append(current_time)
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator
