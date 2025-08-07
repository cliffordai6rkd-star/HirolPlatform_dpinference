from teleop.XR.quest3.utils.state_code import OperationState, OperationMode, ControlTarget, ConnectState


class StateManager:
    """StateManager class to manage the state of the Teleoperation device from external control.
    """

    def __init__(self):
        self._current_state = None
        self._current_mode = None
        self._current_target = None
        self._connect_state = ConnectState.DISCONNECTED
        self.feedback_callbacks = []

    def register_state_callback(self, callback, *args, **kwargs):
        """Register a callback function to be called when the state or mode changes.

        Args:
            callback: Function to call on state change
            *args, **kwargs: Fixed arguments to pass to callback
        """
        self.feedback_callbacks.append((callback, args, kwargs))

    @property
    def current_state(self):
        """Get the current operation state of the robot."""
        return self._current_state

    @current_state.setter
    def current_state(self, state: OperationState):
        """Set the current state of the robot and notify all registered callbacks.

        Args:
            state: The new state of the robot
        """
        old_state = self._current_state
        self._current_state = state
        print(f"Setting state to: {state}")
        if old_state != state:
            for callback, args, kwargs in self.feedback_callbacks:
                callback(*args, **kwargs)

    @property
    def current_mode(self):
        """Get the current mode of the robot."""
        return self._current_mode

    @current_mode.setter
    def current_mode(self, mode: OperationMode):
        """Set the current mode of the robot and notify all registered callbacks.

        Args:
            mode: The new mode of the robot
        """
        old_mode = self._current_mode
        self._current_mode = mode
        print(f"Setting mode to: {mode}")
        if old_mode != mode:
            for callback, args, kwargs in self.feedback_callbacks:
                callback(*args, **kwargs)

    @property
    def current_target(self):
        """Get the current target of the robot."""
        return self._current_target

    @current_target.setter
    def current_target(self, target: ControlTarget):
        """Set the current target of the robot and notify all registered callbacks.

        Args:
            target: The new target of the robot
        """
        old_target = self._current_target
        self._current_target = target
        print(f"Setting target to: {target}")
        if old_target != target:
            for callback, args, kwargs in self.feedback_callbacks:
                callback(*args, **kwargs)

    @property
    def connect_state(self):
        """Get the current connection state of the robot."""
        return self._connect_state

    @connect_state.setter
    def connect_state(self, state: ConnectState):
        """Set the current connection state of the robot and notify all registered callbacks.

        Args:
            state: The new connection state of the robot
        """
        old_state = self._connect_state
        self._connect_state = state
        print(f"Setting connection state to: {state}")
        if old_state != state:
            for callback, args, kwargs in self.feedback_callbacks:
                callback(*args, **kwargs)
