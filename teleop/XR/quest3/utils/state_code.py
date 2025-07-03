import enum


class ConnectState(enum.Enum):
    """Enum for connection states."""
    CONNECTING = 0
    CONNECTED = 1
    DISCONNECTED = 2


class OperationState(enum.Enum):
    """Enum for operation states.
    """
    IDLE = 0
    OPERATING = 1
    PAUSED = 2
    TERMINATED = 3
    ERROR = 4


class OperationMode(enum.Enum):
    """Enum for operation modes.
    """
    SINGLE_ARM = 0
    DUAL_ARM = 1
    WHOLE_BODY_CONTROL = 2


class ArmSide(enum.Enum):
    """Enum for arm side.
    """
    LEFT = 1
    RIGHT = 2


class ControlTarget(enum.Enum):
    """Enum for control modes."""
    NONE = 0
    LEFT_ARM = 1
    RIGHT_ARM = 2
    CHASSIS = 3
