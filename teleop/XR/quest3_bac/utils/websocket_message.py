from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import json


class ActionType(Enum):
    CONNECT_DEVICE = "connect_device"
    DISCONNECT_DEVICE = "disconnect_device"
    GET_AVAILABLE_DEVICES = "get_available_devices"
    EMERGENCY_STOP = "emergency_stop"
    RESUME = "resume"
    CONTROL = "control"


class DeviceType(Enum):
    # GAMEPAD = "gamepad"
    VR = "vr"
    EXOSKELETON = "exoskeleton"
    KEYBOARD = "keyboard"
    WEB_UI = "web_ui"


class ControlTarget(Enum):
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    HEAD = "head"
    TORSO = "torso"
    LEFT_GRIPPER = "left_gripper"
    RIGHT_GRIPPER = "right_gripper"
    BASE = "base"


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    STATE_UPDATE = "robot_state_update"
    CONTROL = "control"


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


@dataclass
class Position:
    x: float
    y: float
    z: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


@dataclass
class Orientation:
    x: float
    y: float
    z: float
    w: float

    def to_list(self) -> List[float]:
        return [self.w, self.x, self.y, self.z]


@dataclass
class Pose:
    position: List[float]
    orientation: List[float]


@dataclass
class BaseMessage:
    type: MessageType

    def to_json(self) -> str:
        return json.dumps(asdict(self), cls=EnumEncoder)


@dataclass
class RequestMessage(BaseMessage):
    action: ActionType
    device_type: Optional[DeviceType] = None

    def __post_init__(self):
        self.type = MessageType.REQUEST


@dataclass
class ResponseMessage(BaseMessage):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    devices: Optional[Dict[str, Dict[str, bool]]] = None

    def __post_init__(self):
        self.type = MessageType.RESPONSE


@dataclass
class RobotStateMessage(BaseMessage):
    robot_state: Dict[str, Any]

    def __post_init__(self):
        self.type = MessageType.STATE_UPDATE


@dataclass
class ControlMessage(BaseMessage):
    # Each field is optional
    arm_joints: Optional[Dict[str, List[float]]] = None
    arm_poses: Optional[Dict[str, Pose]] = None
    gripper_positions: Optional[Dict[str, float]] = None
    head_position: Optional[List[float]] = None
    torso_position: Optional[List[float]] = None
    base_velocity: Optional[List[float]] = None
    base_rotation: Optional[float] = None

    def __post_init__(self):
        self.type = MessageType.CONTROL


def parse_message(json_str: str) -> Union[
    RequestMessage, ResponseMessage, RobotStateMessage, ControlMessage]:
    data = json.loads(json_str)
    msg_type_str = data.get("type")

    # Create a copy of data to modify
    processed_data = data.copy()

    # Handle common conversions
    if msg_type_str == MessageType.REQUEST.value:
        # Convert action string to enum
        if "action" in processed_data:
            processed_data["action"] = ActionType(processed_data["action"])

        # Convert device_type if present
        if processed_data.get("device_type"):
            processed_data["device_type"] = DeviceType(processed_data["device_type"])

        return RequestMessage(**processed_data)

    elif msg_type_str == MessageType.RESPONSE.value:
        return ResponseMessage(**processed_data)

    elif msg_type_str == MessageType.STATE_UPDATE.value:
        return RobotStateMessage(**processed_data)

    elif msg_type_str == MessageType.CONTROL.value:
        return ControlMessage(**processed_data)

    else:
        raise ValueError(f"Unknown message type: {msg_type_str}")
