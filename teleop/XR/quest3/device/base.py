import abc
from typing import Dict, Any


class TeleopDevice(abc.ABC):
    """
    Abstract base class for teleoperation devices.
    """

    def __init__(self, name: str):
        self.name = name
        self._connected = False
        self._ready = False
        self.last_state = None

    @abc.abstractmethod
    def connect(self) -> bool:
        """Connects to the teleoperation device.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> bool:
        """Disconnects from the teleoperation device.
        """
        pass

    @abc.abstractmethod
    def get_device_state(self) -> Dict[str, Any]:
        """Gets the current state of the teleoperation device.
        """
        pass

    @abc.abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Gets the information of the teleoperation device.
        """
        pass

    @property
    def is_connected(self):
        """Checks if the teleoperation device is connected.
        """
        return self._connected

    @property
    def is_ready(self):
        """Checks if the teleoperation device is ready.
        """
        return self._ready
