import abc
import threading
import copy
import time
import warnings
from typing import Dict, Optional, Any, Tuple
import numpy as np
import glog as log
from hardware.base.utils import PaxiniState


class TactileBase(abc.ABC, metaclass=abc.ABCMeta):
    """
    Abstract base class for tactile sensors.
    
    Provides unified interface for different tactile sensor implementations:
    - Paxini tactile sensors (serial/network)
    - Other tactile sensor brands
    - Simulated tactile sensors
    
    Similar to CameraBase design pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = threading.Lock()
        self._is_initialized = False
        
        # Tactile sensor data (similar to camera's _image_data)
        self._tactile_data = None
        self._time_stamp = time.perf_counter()
        
        # Basic configuration parameters
        self._n_taxels = config.get("taxel_nums", 120)
        self._update_frequency = config.get("update_frequency", 100)
        self._sensor_type = config.get("type", "tactile_sensor")
        
        # Initialize sensor (implemented by subclass)
        self._is_initialized = self.initialize()

    def print_state(self) -> None:
        """Print current tactile sensor state"""
        if not self._is_initialized:
            warnings.warn("Tactile sensor is not initialized")
            return
            
        if self._tactile_data is not None:
            log.info(f"Tactile sensor state:")
            log.info(f"  Sensor type: {self._sensor_type}")
            log.info(f"  Data shape: {self._tactile_data.shape}")
            log.info(f"  Timestamp: {self._time_stamp}")
            log.info(f"  Update frequency: {self._update_frequency} Hz")
        else:
            warnings.warn("No tactile data available")

    def capture_all_tactile_data(self) -> Dict[str, Any]:
        """
        Capture all tactile data (similar to camera's capture_all_data)
        
        Returns:
            Dict containing tactile data and metadata
        """
        if not self._is_initialized:
            raise RuntimeError("Tactile sensor is not initialized, cannot capture data.")
        
        _, tactile_data, timestamp = self.read_tactile_data()
        
        return {
            'tactile_data': tactile_data,
            'timestamp': timestamp,
            'sensor_type': self._sensor_type
        }

    def read_tactile_data(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read tactile data in a thread-safe manner (similar to camera's read_image)
        
        Returns:
            Tuple[bool, Optional[np.ndarray], float]: (success, data, timestamp)
        """
        if not self._is_initialized or self._tactile_data is None:
            warnings.warn(f"The sensor is not initialized {self._is_initialized} or "
                         f"still not retrieve the tactile data {self._tactile_data is None}")
            return False, None, 0.0
        
        self._lock.acquire()
        tactile_data = copy.deepcopy(self._tactile_data)
        timestamp = copy.deepcopy(self._time_stamp)
        self._lock.release()
        
        return True, tactile_data, timestamp

    def get_tactile_state(self) -> PaxiniState:
        """
        Get current tactile state as PaxiniState object
        
        Returns:
            PaxiniState: Current tactile state
            
        Note: Using PaxiniState for now, should be renamed to TactileState in future
        """
        state = PaxiniState()
        
        if self._tactile_data is not None:
            self._lock.acquire()
            state._tactile_data = copy.deepcopy(self._tactile_data)
            state._time_stamp = copy.deepcopy(self._time_stamp)
            state._is_connected = True
            
            # Set module information from data shape
            if len(self._tactile_data.shape) >= 2:
                state._n_modules = self._tactile_data.shape[0]
                state._module_ids = list(range(state._n_modules))
            
            self._lock.release()
        
        return state

    def is_sensor_connected(self) -> bool:
        """Check if sensor is connected and responding"""
        return self._is_initialized and self._tactile_data is not None

    def get_sensor_info(self) -> Dict[str, Any]:
        """Get sensor information and configuration"""
        return {
            "sensor_type": self._sensor_type,
            "n_taxels": self._n_taxels,
            "is_connected": self.is_sensor_connected(),
            "update_frequency": self._update_frequency,
            "data_shape": self._tactile_data.shape if self._tactile_data is not None else None
        }

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get tactile sensor resolution in terms of number of modules and taxels per module.
        
        Returns:
            Tuple[int, int]: (n_modules, n_taxels_per_module)
        """
        if self._tactile_data is not None and len(self._tactile_data.shape) >= 2:
            return self._tactile_data.shape[0], self._tactile_data.shape[1]
        return 0, self._n_taxels

    # ========== Abstract Methods (Implementation-Specific) ==========
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the tactile sensor connection.
        
        Returns:
            bool: Initialization success status
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> bool:
        """
        Close the sensor connection and cleanup resources.
        
        Returns:
            bool: Success status
        """
        raise NotImplementedError