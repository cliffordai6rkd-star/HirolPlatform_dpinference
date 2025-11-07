#py310 is more stable, but should use ubuntu22.04 x ros2 humble, which can achieve by a docker container.
ROBOTLIB_SO_PATH='dependencies/monte02_sdk/build'


ARM_ENABLE=1
ARM_DENABLE=2
ENABLE=1
DENABLE=2

GRIPPER_ENABLE = 1
GRIPPER_DENABLE = 0

COM_TYPE_LEFT=1
COM_TYPE_RIGHT=2

# ARM_MODE_POSITION_CTRL = 0
ARM_MODE_SERVO_MOTION = 1
ARM_MODE_JOINT_TEACHING = 2

GRIPPER_MODE_POSITION_CTRL = 1
GRIPPER_MODE_TORQUE_CTRL = 2

ARM_STATE_SPORT=0
ARM_STATE_PAUSE=1
ARM_STATE_STOP=2

TRUNK_JOINT_MODE_PROFILE = 0
TRUNK_JOINT_MODE_SERVOJ = 1

import glog as log
import inspect
CORENETIC_GRIPPER_MAX_POSITION = 0.074  # Maximum position for Corenetic gripper (0 to 0.074 meters)

def CHECK(result: bool, operation_desc: str = "") -> bool:
    """
    Check if operation result is successful, log error with call location if failed.

    Args:
        result: The boolean result to check
        operation_desc: Description of the operation being checked

    Returns:
        bool: The result value (pass-through)

    Example:
        CHECK(self._robot.clean_arm_err_warn_code(COM_TYPE_LEFT), "Left arm clean_arm_err_warn_code")
    """
    if not result:
        # Get caller information from stack
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name

        # Format error message with call location
        error_msg = f"Operation FAILED at {filename}:{line_number} in {function_name}()"
        if operation_desc:
            error_msg += f" - {operation_desc}"

        log.warn(error_msg)

    return result