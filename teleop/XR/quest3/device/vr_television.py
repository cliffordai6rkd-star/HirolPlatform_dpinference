import copy
from collections import defaultdict
from pathlib import Path
import time
import threading
import asyncio
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Text, Sequence
from collections import deque
from enum import Enum
from multiprocessing import context
from multiprocessing import Array, Process, shared_memory

Value = context._default_context.Value

from vuer import Vuer
from vuer.schemas import MotionControllers, Hands
from scipy.spatial.transform import Rotation, Slerp

from teleop.XR.quest3.device.base import TeleopDevice
from teleop.XR.quest3.device.TTSPlayer import TTSPlayer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from data_types.se3 import Transform
from teleop.XR.quest3.utils.decorator import fps_statistics
from teleop.XR.quest3.utils.log import logger
from teleop.XR.quest3.utils.constants import T_corenetic_openxr
from teleop.XR.quest3.utils.mat_tool import mat_update, fast_mat_inv
from teleop.XR.quest3.utils.teleop_data_recorder import TeleopDataRecorder
from hardware.monte01.agent import Agent

class ControlMode(Enum):
    DUAL_ARM = 0
    WBC = 1


class TeleopMode(Enum):
    Inference = 0
    DataCollection = 1


class ArmSide(Enum):
    LEFT = 'left'
    RIGHT = 'right'


class VRTelevision(TeleopDevice):
    def __init__(self, robot_interface:Agent=None):
        logger.info("Initializing VRTelevision teleoperation device")
        super().__init__("Quest3")
        self.robot_interface = robot_interface
        self._controller_fps_shared = Value('d', 0.0, lock=True)
        self.left_controller_shared = Array('d', 16, lock=True)
        self.right_controller_shared = Array('d', 16, lock=True)
        self.left_controller_state_shared = Array('b', 3, lock=True)
        self.right_controller_state_shared = Array('b', 3, lock=True)
        self.left_button_shared = Array('d', 4, lock=True)
        self.right_button_shared = Array('d', 4, lock=True)
        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 16 * 24, lock=True)
        self.right_landmarks_shared = Array('d', 16 * 24, lock=True)
        self.left_hand_state_value_shared = Array('d', 3, lock=True)
        self.right_hand_state_value_shared = Array('d', 3, lock=True)
        self.left_hand_state_shared = Array('b', 3, lock=True)
        self.right_hand_state_shared = Array('b', 3, lock=True)

        # Control state
        self.control_mode = ControlMode.DUAL_ARM
        self.emergency_stop = False
        self.teleop_mode = TeleopMode.DataCollection
        self.left_grip_engaged = False
        self.right_grip_engaged = False
        self.left_grip_initial_press_time = 0
        self.right_grip_initial_press_time = 0

        # Teleoperations status for each arm
        self.left_arm_active = False
        self.right_arm_active = False

        # Origin poses when teleop starts
        self.left_controller_origin = None
        self.right_controller_origin = None
        self.left_arm_origin = None
        self.right_arm_origin = None
        self.left_rotation_offset = None
        self.right_rotation_offset = None

        # Robot arm reference poses
        self.left_arm_reference = np.eye(4)
        self.right_arm_reference = np.eye(4)

        # Target poses for robot arms - updated by controller thread
        self.left_arm_target = None
        self.right_arm_target = None

        # Command poses for robot arms - sent to robot at higher frequency
        self.left_arm_command = self.left_arm_end_pose
        self.right_arm_command = self.right_arm_end_pose

        # Button timing for double-click detection
        init_time = time.perf_counter()
        self.last_button_press = {
            'left_grip': init_time,
            'right_grip': init_time,
            'a_button': init_time,
            'b_button': init_time,
            'thumbstick': init_time,
            'left_trigger': init_time,
            'right_trigger': init_time
        }
        self.button_press_count = {k: 0 for k in self.last_button_press}
        self.button_states_prev = {
            'left_grip': 0,
            'right_grip': 0,
            'a_button': 0,
            'b_button': 0,
            'thumbstick': 0,
            'left_trigger': 0,
            'right_trigger': 0,
            'left_thumbstick_x': 0,
            'right_thumbstick_x': 0
        }
        self.trigger_pending = {'left_trigger': False, 'right_trigger': False, 'b_button': False, 'thumbstick': False}
        self.trigger_pending_time = {'left_trigger': 0, 'right_trigger': 0, 'b_button': 0, 'thumbstick': 0}
        self.double_click_wait_time = 0.3  # 300ms double-click detection window

        # Scaling factors for arm movement (default 1:1 ratio)
        self.left_arm_scale = 1.0
        self.right_arm_scale = 1.0

        # Minimum scale factor for precise operations
        self.min_scale_factor = 0.5
        # Maximum scale factor
        self.max_scale_factor = 2.0
        # Step size for scale changes
        self.scale_step = 0.1

        # Gripper states
        self.left_gripper_pos = 0.05
        self.right_gripper_pos = 0.05
        # Gripper control parameters
        self.gripper_min = 0.0
        self.gripper_max = 1.0
        self.gripper_step = 0.03  # Reduced for smoother control (was 0.05)
        self.is_send_gripper_command = False

        # Robot end pose smoothing - use more frames for smoother movement
        self.position_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }
        self.rotation_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }

        # Robot end velocity tracking for prediction
        self.position_velocity = {
            'left': np.zeros(3),
            'right': np.zeros(3)
        }
        self.last_position = {
            'left': np.zeros(3),
            'right': np.zeros(3)
        }
        self.last_timestamp = {
            'left': time.perf_counter(),
            'right': time.perf_counter()
        }

        # Previous arm commands for safety checks
        self._previous_left_arm_command = None
        self._previous_right_arm_command = None
        self.max_position_change = 0.01  # 1cm maximum change per control cycle
        self.max_rotation_change = 0.1  # ~5.7 degrees maximum rotation per cycle
        self.max_safe_position_change = 0.05  # 5cm maximum change for safety

        # Threading
        self.running = False
        self.controller_thread = None
        self.robot_command_thread = None

        # Records data for analysis system
        self.is_recording = False
        self.data_recorder = None
        self._left_arm_control_latency = 0.
        self._right_arm_control_latency = 0.

        # Data collection mode
        self.is_data_collection = False
        self.is_reset = False  # Whether the robot arm is reset to the initial pose
        # Add long press detection variables
        self.button_hold_start = {
            'a_button': 0,
            'b_button': 0
        }
        self.button_hold_active = {
            'a_button': False,
            'b_button': False
        }
        self.long_press_threshold = 2.0  # 3 seconds for long press

        self._init()
        logger.info("VRTelevision teleoperation device initialized successfully")

    def _init(self):
        self._init_path()
        self._init_vuer()
        self._init_tts()
        if self.is_recording:
            self.data_recorder = TeleopDataRecorder(self.name)

    def _init_path(self):
        self.root = Path(__file__).parent.parent
        self.certificate_path = self.root.joinpath("certification")
        self.certificate_path.mkdir(parents=True, exist_ok=True)
        self.cert_file = self.certificate_path.joinpath("cert.pem")
        self.key_file = self.certificate_path.joinpath("key.pem")
        if not self.cert_file.exists() or not self.key_file.exists():
            raise FileNotFoundError(
                f"Certificate files not found. Please generate them first and place them under: {self.certificate_path}")

    def _init_vuer(self):
        if self.is_connected:
            logger.warning("Vuer is already initialized.")
            return
        self.vuer = Vuer(host='0.0.0.0', cert=self.cert_file, key=self.key_file, queries=dict(grid=False), queue_len=3)
        self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)
        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.spawn(start=False)(self.run_hands_server)
        self.vuer.spawn(start=False)(self.run_controllers_server)

    def _init_tts(self):
        try:
            self.tts_client = TTSPlayer()
        except OSError as e:
            print(f"Warning: TTS initialization failed: {e}")
            self.tts_client = None

    def _reset_values_by_side(self, side: ArmSide):
        """Resets values for a specific arm side.

        Args:
            side: The side of the arm to reset (LEFT or RIGHT).
        """
        current_time = time.perf_counter()
        if side == ArmSide.LEFT:
            # Clear previous position history
            self.position_history['left'].clear()
            self.rotation_history['left'].clear()
            # Initialize position and rotation with current robot arm pose
            if self.robot_interface is not None:
                pose = self.robot_interface.get_tcp_pose("left")
                pose = pose['left'].matrix.copy()
                self.left_arm_reference = pose
                self.position_history['left'].append(pose[:3, 3])
                self.rotation_history['left'].append(pose[:3, :3])
                self.last_position['left'] = pose[:3, 3]
                self.last_timestamp['left'] = current_time
                self.left_gripper_pos = self.robot_interface.get_gripper_state("left")[0]
                self._previous_left_arm_command = pose.copy()  # Stores initial command
                self.robot_interface.enable_servo_control("left")
                # self.robot_interface.set_gripper_mode("left", mode=1)
            else:
                self.left_arm_reference = self.trans_to_world(self.left_controller)
                self.position_history['left'].append(self.left_arm_reference[:3, 3])
                self.rotation_history['left'].append(self.left_arm_reference[:3, :3])
                self.left_gripper_pos = 0.05  # Default gripper position
                self._previous_left_arm_command = self.trans_to_world(self.left_controller)
            self.left_arm_command = self._previous_left_arm_command
            # Resets velocity tracking
            self.position_velocity['left'] = np.zeros(3)
        elif side == ArmSide.RIGHT:
            # Clear previous position history
            self.position_history['right'].clear()
            self.rotation_history['right'].clear()
            # Initialize position and rotation with current robot arm pose
            if self.robot_interface is not None:
                pose = self.robot_interface.get_tcp_pose("right")
                pose = pose['right'].matrix.copy()
                self.right_arm_reference = pose
                self.position_history['right'].append(pose[:3, 3])
                self.rotation_history['right'].append(pose[:3, :3])
                self.last_position['right'] = pose[:3, 3]
                self.last_timestamp['right'] = current_time
                self.right_gripper_pos = self.robot_interface.get_gripper_state("right")[0]
                self._previous_right_arm_command = pose.copy()
                self.robot_interface.enable_servo_control("right")
                # self.robot_interface.set_gripper_mode("right", mode=1)
            else:
                self.right_arm_reference = self.trans_to_world(self.right_controller)
                self.position_history['right'].append(self.right_arm_reference[:3, 3])
                self.rotation_history['right'].append(self.right_arm_reference[:3, :3])
                self.right_gripper_pos = 0.05
                self._previous_right_arm_command = self.trans_to_world(self.right_controller)
            self.right_arm_command = self._previous_right_arm_command
            # Resets velocity tracking
            self.position_velocity['right'] = np.zeros(3)
        else:
            raise ValueError("Invalid arm side. Use ArmSide.LEFT or ArmSide.RIGHT.")

    @fps_statistics(queue_size=30)
    async def on_controller_move(self, event, session):
        try:
            self.left_controller_shared[:] = event.value["left"]
            self.right_controller_shared[:] = event.value["right"]
            left_thumbstick_x = event.value["leftState"]["thumbstickValue"][0]
            right_thumbstick_x = event.value["rightState"]["thumbstickValue"][0]
            
            self.left_button_shared[:] = (event.value["leftState"]["triggerValue"],
                                          event.value["leftState"]["squeezeValue"],
                                          # event.value["leftState"]["aButtonValue"],
                                          # event.value["leftState"]["bButtonValue"],
                                          left_thumbstick_x,
                                          event.value["leftState"]["thumbstickValue"][1],
                                          )
            self.right_button_shared[:] = (event.value["rightState"]["triggerValue"],
                                           event.value["rightState"]["squeezeValue"],
                                           # event.value["rightState"]["aButtonValue"],
                                           # event.value["rightState"]["bButtonValue"],
                                           right_thumbstick_x,
                                           event.value["rightState"]["thumbstickValue"][1],
                                           )
            self.left_controller_state_shared[:] = (event.value["leftState"]["aButtonValue"],
                                                    event.value["leftState"]["bButtonValue"],
                                                    event.value["leftState"]["thumbstick"]
                                                    )
            self.right_controller_state_shared[:] = (event.value["rightState"]["aButtonValue"],
                                                     event.value["rightState"]["bButtonValue"],
                                                     event.value["rightState"]["thumbstick"]
                                                     )


        except Exception as e:
            pass

    # @fps_statistics(queue_size=30)
    async def on_hand_move(self, event, session):
        try:
            left_data = event.value["left"]
            right_data = event.value["right"]
            self.left_hand_shared[:] = left_data[:16]
            self.right_hand_shared[:] = right_data[:16]
            self.left_landmarks_shared[:] = left_data[16:]
            self.right_landmarks_shared[:] = right_data[16:]
            self.left_hand_state_value_shared[:] = (event.value["leftState"]["pinchValue"],
                                                    event.value["leftState"]["squeezeValue"],
                                                    event.value["leftState"]["tapValue"])
            self.right_hand_state_value_shared[:] = (event.value["rightState"]["pinchValue"],
                                                     event.value["rightState"]["squeezeValue"],
                                                     event.value["rightState"]["tapValue"])
            self.left_hand_state_shared[:] = (event.value["rightState"]["pinch"],
                                              event.value["rightState"]["squeeze"],
                                              event.value["rightState"]["tap"])
            self.right_hand_state_shared[:] = (event.value["leftState"]["pinch"],
                                               event.value["leftState"]["squeeze"],
                                               event.value["leftState"]["tap"])
        except Exception as e:
            pass

    async def run_hands_server(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        while True:
            await asyncio.sleep(0.01)

    async def run_controllers_server(self, session, fps=120):
        session.upsert @ MotionControllers(stream=True, key="motion-controller", fps=fps)
        while True:
            await asyncio.sleep(0.01)

    def _robot_command_loop(self):
        """Thread for sending smoothed commands to robot at 100Hz"""
        logger.info("Robot command thread started")
        control_hz = 100
        last_command_time = time.perf_counter()

        while self.running:
            try:
                # Maintain 100Hz command rate
                current_time = time.perf_counter()
                elapsed = current_time - last_command_time
                if elapsed < 1 / control_hz:
                    time.sleep(1 / control_hz - elapsed)

                last_command_time = time.perf_counter()

                # Skip if emergency stop is active
                if self.emergency_stop:
                    continue

                # Generate smoothed commands for robot arms
                self._generate_smooth_commands()

                # Here we would send the commands to the actual robot
                # This is a placeholder for the robot control API
                if self.left_arm_active:
                    self._send_arm_command('left', self.left_arm_command)
                    if self.left_arm_command is not None:
                        self._previous_left_arm_command = self.left_arm_command.copy()

                if self.right_arm_active:
                    self._send_arm_command('right', self.right_arm_command)
                    if self.right_arm_command is not None:
                        self._previous_right_arm_command = self.right_arm_command.copy()
            except Exception as e:
                logger.error(f"Error in robot command loop: {e}")
                time.sleep(0.01)  # Prevent tight loop if there's an error

    def _robot_gripper_command_loop(self):
        """Thread for sending gripper commands at 60Hz"""
        logger.info("Robot gripper command thread started")
        control_hz = 60
        last_command_time = time.perf_counter()

        while self.running:
            try:
                current_time = time.perf_counter()
                elapsed = current_time - last_command_time
                if elapsed < 1 / control_hz:
                    time.sleep(1 / control_hz - elapsed)

                last_command_time = time.perf_counter()

                # Skip if emergency stop is active
                if self.emergency_stop:
                    continue

                # Send gripper commands to the robot
                if self.left_arm_active and self.is_send_gripper_command:
                    self._send_gripper_command('left', self.left_gripper_pos)

                if self.right_arm_active and self.is_send_gripper_command:
                    self._send_gripper_command('right', self.right_gripper_pos)

            except Exception as e:
                logger.error(f"Error in robot gripper command loop: {e}")
                time.sleep(1)

    def _send_arm_command(self, arm: Text, command: np.ndarray):
        """Placeholder function to send arm command to the robot.

        Args:
            arm: The side of the arm (left or right).
            command: The target pose for the arm.
        """
        if self.control_mode == ControlMode.DUAL_ARM:
            if self.robot_interface is not None and command is not None:
                curt_time = time.perf_counter()
                self.robot_interface.set_arm_servo_flange_pose({arm: Transform(matrix=command)})
                if arm == 'left':
                    self._left_arm_control_latency = time.perf_counter() - curt_time
                elif arm == 'right':
                    self._right_arm_control_latency = time.perf_counter() - curt_time
            else:
                pass
        else:
            pass

    def _send_gripper_command(self, arm: Text, gripper_pos: float):
        """Placeholder function to send gripper command to the robot.

        Args:
            arm: The side of the arm (left or right).
            gripper_pos: The target position for the gripper.
        """
        if self.robot_interface is not None:
            self.robot_interface.set_gripper_state(arm, gripper_pos)
        else:
            pass

    async def _voice_feedback_async(self, message: str):
        """Provide voice feedback from the message asynchronously.

        Args:
            message (str): The message to be spoken.
        """
        if not self.tts_client or not hasattr(self.tts_client, "speak"):
            return
        await self.tts_client.speak(message)

    def _voice_feedback(self, message: str):
        """Synchronous wrapper for voice feedback.

        Args:
            message (str): The message to be spoken.
        """
        try:
            # Create a new event loop for this thread if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Run the async function to completion
            loop.run_until_complete(self._voice_feedback_async(message))
            loop.close()
        except Exception as e:
            logger.error(f"Error in voice feedback: {e}")

    def _controller_processing_loop(self):
        """Thread for processing controller data at 60Hz.
        """
        logger.info("Controller processing thread started")
        control_hz = 60
        last_process_time = time.perf_counter()
        while self.running:
            try:
                current_time = time.perf_counter()
                elapsed = current_time - last_process_time
                if elapsed < 1 / control_hz:
                    time.sleep(1 / control_hz - elapsed)
                last_process_time = time.perf_counter()
                # Processes button presses for control functions
                self._process_button_presses()
                # Processes controller movements for robot arm control
                self._process_controller_movement()
                # Send gripper commands directly in main loop for higher frequency
                self._send_gripper_commands_if_needed()
            except Exception as e:
                logger.error(f"Error in controller processing loop: {e}")
                time.sleep(0.05)  # Prevents tight loop if there's an error

    def _process_button_presses(self):
        """Processes button presses for control functions"""
        # Gets current button states
        current_buttons = {
            'left_grip': self.left_button[1] > 0.5,  # squeeze value
            'right_grip': self.right_button[1] > 0.5,  # squeeze value
            'a_button': self.left_button_state[0] or self.right_button_state[0],  # A button on either controller
            'b_button': self.left_button_state[1] or self.right_button_state[1],  # B button on either controller
            "thumbstick": self.left_button_state[2] or self.right_button_state[2],
            # thumbstick pressed on either controller
            'left_trigger': self.left_button[0] > 0.5,  # left trigger
            'right_trigger': self.right_button[0] > 0.5,  # right trigger
            'left_thumbstick_x': self.left_button[2],  # left thumbstick X
            'right_thumbstick_x': self.right_button[2]  # right thumbstick X                               
        }

        current_time = time.perf_counter()

        if self.teleop_mode == TeleopMode.DataCollection:
            self._handle_data_collection_mode(current_buttons, current_time)
        else:
            self._handle_inference_mode(current_buttons, current_time)
        # Processes left and right triggers and b_button because they have reuse functionality
        for button in ["left_trigger", "right_trigger", 'b_button', 'thumbstick']:
            if current_buttons[button] and not self.button_states_prev[button]:
                if self.trigger_pending[button]:
                    self._handle_double_click(button)
                    self.trigger_pending[button] = False
                else:
                    # Sets pending state for double click detection
                    self.trigger_pending[button] = True
                    self.trigger_pending_time[button] = current_time
            if self.trigger_pending[button] and current_time - self.trigger_pending_time[
                button] > self.double_click_wait_time:
                # If the button is still pressed after the double click window, reset pending state
                self._handle_single_click(button)
                self.trigger_pending[button] = False
        # Double-click detection for a_button,thumbstick
        for button in ['a_button', 'thumbstick']:
            if current_buttons[button] and not self.button_states_prev[button]:
                # Button just pressed
                if current_time - self.last_button_press[button] < 1.0:
                    # Double click detected (two presses within 1 second)
                    self.button_press_count[button] += 1
                    if self.button_press_count[button] >= 2:
                        self._handle_double_click(button)
                        self.button_press_count[button] = 0
                else:
                    # First click in a potential double click
                    self.button_press_count[button] = 1
                self.last_button_press[button] = current_time

        # Reset button press counter if time elapsed is more than double-click window
        for button in self.button_press_count:
            if current_time - self.last_button_press[button] > 1.0 and self.button_press_count[button] == 1:
                self.button_press_count[button] = 0

        # Process thumbstick for gripper control
        self._process_thumbstick_for_gripper(current_buttons)

        # Update previous button states
        self.button_states_prev = current_buttons

    def _handle_data_collection_mode(self, current_buttons: Dict[str, np.ndarray], current_time: float):
        """Handles button presses in Data Collection mode.Double-clicking left or right grip toggles teleoperation for that arm.
            If teleoperation is activated, it initializes the controller origin and rotation offset, clears previous position history,
            and initializes position and rotation with the current robot arm pose.

        Args:
            current_buttons: Current state of the buttons.
            current_time: Current timestamp for click.
        """
        if current_buttons["left_grip"] and not self.button_states_prev["left_grip"]:
            # Left grip pressed
            if current_time - self.last_button_press['left_grip'] < 1.0:
                # Double click detected
                self.left_arm_active = not self.left_arm_active
                if self.left_arm_active:
                    if self.is_recording:
                        self.data_recorder.start_recording(self, self.robot_interface)
                    self.left_controller_origin = self.left_controller.copy()
                    R_world_hand = self.left_controller_origin[:3, :3]
                    R_world_robot = self.left_arm_reference[:3, :3]
                    self.left_rotation_offset = R_world_robot @ R_world_hand.T
                    if self.robot_interface is not None:
                        self.left_arm_origin = self.robot_interface.get_tcp_pose("left")['left'].matrix.copy()
                    else:
                        self.left_arm_origin = self.trans_to_world(self.left_controller_origin)
                    self._reset_values_by_side(ArmSide.LEFT)
                    
                    # Debug info for initial alignment
                    controller_world = self.trans_to_world(self.left_controller_origin)
                    logger.info("Left arm teleoperation activated (data collection mode)")
                    logger.info(f"Left controller origin (OpenXR): {self.left_controller_origin[:3, 3]}")
                    logger.info(f"Left controller origin (world): {controller_world[:3, 3]}")
                    logger.info(f"Left arm origin (robot): {self.left_arm_origin[:3, 3]}")
                    logger.info(f"Initial position difference: {self.left_arm_origin[:3, 3] - controller_world[:3, 3]}")
                    self._voice_feedback("激活左臂遥操作")
                else:
                    if self.is_recording:
                        logger.info(f"Stopping data recording for left arm teleoperation")
                        self.data_recorder.stop_recording()
                    self.left_controller_origin = None
                    self.left_arm_origin = None
                    self.left_rotation_offset = None
                    logger.info("Left arm teleoperation deactivated (data collection mode)")
                    self._voice_feedback("停止左臂遥操作")
            self.last_button_press['left_grip'] = current_time
        if current_buttons["right_grip"] and not self.button_states_prev["right_grip"]:
            # Right grip pressed
            if current_time - self.last_button_press['right_grip'] < 1.0:
                # Double click detected
                self.right_arm_active = not self.right_arm_active
                if self.right_arm_active:
                    if self.is_recording:
                        self.data_recorder.start_recording(self.robot_interface)
                    self.right_controller_origin = self.right_controller.copy()
                    R_world_hand = self.right_controller_origin[:3, :3]
                    R_world_robot = self.right_arm_reference[:3, :3]
                    self.right_rotation_offset = R_world_robot @ R_world_hand.T
                    if self.robot_interface is not None:
                        self.right_arm_origin = self.robot_interface.get_tcp_pose("right")['right'].matrix.copy()
                    else:
                        self.right_arm_origin = self.trans_to_world(self.right_controller_origin)
                    self._reset_values_by_side(ArmSide.RIGHT)
                    
                    # Debug info for initial alignment
                    controller_world = self.trans_to_world(self.right_controller_origin)
                    logger.info("Right arm teleoperation activated (data collection mode)")
                    logger.info(f"Right controller origin (OpenXR): {self.right_controller_origin[:3, 3]}")
                    logger.info(f"Right controller origin (world): {controller_world[:3, 3]}")
                    logger.info(f"Right arm origin (robot): {self.right_arm_origin[:3, 3]}")
                    logger.info(f"Initial position difference: {self.right_arm_origin[:3, 3] - controller_world[:3, 3]}")
                    self._voice_feedback("激活右臂遥操作")
                else:
                    if self.is_recording:
                        self.data_recorder.stop_recording()
                    self.right_controller_origin = None
                    self.right_arm_origin = None
                    self.right_rotation_offset = None
                    logger.info("Right arm teleoperation deactivated (data collection mode)")
                    self._voice_feedback("停止右臂遥操作")
            self.last_button_press['right_grip'] = current_time

    def _handle_inference_mode(self, current_buttons: Dict[str, np.ndarray], current_time: float):
        """Handles button presses in Inference mode.Keeps pressing left or right grip toggle teleoperation for that arm.

        Args:
            current_buttons: Current state of the buttons.
            current_time: Current timestamp for click.
        """
        if current_buttons["left_grip"]:
            if not self.button_states_prev["left_grip"]:
                if self.is_recording:
                    self.data_recorder.start_recording(self.robot_interface)
                self.left_grip_initial_press_time = current_time
                self.left_grip_engaged = True
                self.left_controller_origin = self.left_controller.copy()
                R_world_hand = self.left_controller_origin[:3, :3]
                R_world_robot = self.left_arm_reference[:3, :3]
                self.left_rotation_offset = R_world_robot @ R_world_hand.T
                if self.robot_interface is not None:
                    self.left_arm_origin = self.robot_interface.get_tcp_pose("left")['left'].matrix.copy()
                else:
                    self.left_arm_origin = self.trans_to_world(self.left_controller_origin)
                self.left_arm_active = True
                self._reset_values_by_side(ArmSide.LEFT)
                
                # Debug info for initial alignment
                controller_world = self.trans_to_world(self.left_controller_origin)
                logger.info("Left arm teleoperation activated (inference mode)")
                logger.info(f"Left controller origin (OpenXR): {self.left_controller_origin[:3, 3]}")
                logger.info(f"Left controller origin (world): {controller_world[:3, 3]}")
                logger.info(f"Left arm origin (robot): {self.left_arm_origin[:3, 3]}")
                logger.info(f"Initial position difference: {self.left_arm_origin[:3, 3] - controller_world[:3, 3]}")
                self._voice_feedback("激活左臂遥操作")
            self.left_arm_active = True
        else:
            if self.is_recording:
                self.data_recorder.stop_recording()
            # Release to deactivate left arm teleoperation
            if self.button_states_prev["left_grip"]:
                self.left_arm_active = False
                self.left_grip_engaged = False
                self.left_controller_origin = None
                self.left_rotation_offset = None
                self.left_arm_origin = None
                logger.info("Left arm teleoperation deactivated (inference mode)")
                self._voice_feedback(f"停止左臂遥操作")
        if current_buttons["right_grip"]:
            if self.is_recording:
                self.data_recorder.start_recording(self.robot_interface)
            if not self.button_states_prev["right_grip"]:
                self.right_grip_initial_press_time = current_time
                self.right_grip_engaged = True
                self.right_controller_origin = self.right_controller.copy()
                R_world_hand = self.right_controller_origin[:3, :3]
                R_world_robot = self.right_arm_reference[:3, :3]
                self.right_rotation_offset = R_world_robot @ R_world_hand.T
                if self.robot_interface is not None:
                    self.right_arm_origin = self.robot_interface.get_tcp_pose("right")['right'].matrix.copy()
                else:
                    self.right_arm_origin = self.trans_to_world(self.right_controller_origin)
                self.right_arm_active = True
                self._reset_values_by_side(ArmSide.RIGHT)
                
                # Debug info for initial alignment
                controller_world = self.trans_to_world(self.right_controller_origin)
                logger.info("Right arm teleoperation activated (inference mode)")
                logger.info(f"Right controller origin (OpenXR): {self.right_controller_origin[:3, 3]}")
                logger.info(f"Right controller origin (world): {controller_world[:3, 3]}")
                logger.info(f"Right arm origin (robot): {self.right_arm_origin[:3, 3]}")
                logger.info(f"Initial position difference: {self.right_arm_origin[:3, 3] - controller_world[:3, 3]}")
                self._voice_feedback("激活右臂遥操作")
            self.right_arm_active = True
        else:
            if self.is_recording:
                self.data_recorder.stop_recording()
            # Release to deactivate right arm teleoperation
            if self.button_states_prev["right_grip"]:
                self.right_arm_active = False
                self.right_grip_engaged = False
                self.right_controller_origin = None
                self.right_rotation_offset = None
                self.right_arm_origin = None
                logger.info("Right arm teleoperation deactivated (inference mode)")
                self._voice_feedback(f"停止右臂遥操作")

    def _handle_double_click(self, button: Text):
        """Handles double-click events for buttons.

        Args:
            button: The button that was double-clicked.
        """
        if button == "a_button":
            # Toggles emergency stop
            self.emergency_stop = not self.emergency_stop
            logger.warning(f"Emergency stop {'activated' if self.emergency_stop else 'deactivated'}")
            if self.emergency_stop:
                self.left_arm_active = False
                self.right_arm_active = False
                self._voice_feedback("急停已激活")
            else:
                self._voice_feedback("急停已解除")
        elif button == "b_button":
            if self.teleop_mode == TeleopMode.DataCollection:
                self.teleop_mode = TeleopMode.Inference
                self.left_arm_active = False
                self.right_arm_active = False
                logger.info("Switched to Inference teleoperation mode")
                self._voice_feedback("切换到推理干预模式")
            else:
                self.teleop_mode = TeleopMode.DataCollection
                self.left_arm_active = False
                self.right_arm_active = False
                logger.info("Switched to Data Collection teleoperation mode")
                self._voice_feedback(f"切换到数据采集模式")
        elif button == "left_trigger":
            # Increases left arm scale factor
            self.left_arm_scale = min(self.left_arm_scale + self.scale_step, self.max_scale_factor)
            logger.info(f"Left arm scale factor increased to {self.left_arm_scale:.2f}")
            self._voice_feedback(f"左臂缩放因子增加到 {self.left_arm_scale:.2f}")
        elif button == "right_trigger":
            # Increases right arm scale factor
            self.right_arm_scale = min(self.right_arm_scale + self.scale_step, self.max_scale_factor)
            logger.info(f"Right arm scale factor increased to {self.right_arm_scale:.2f}")
            self._voice_feedback(f"右臂缩放因子增加到 {self.right_arm_scale:.2f}")
        elif button == "thumbstick":
            # Toggle reset
            self.left_arm_active = False
            self.right_arm_active = False
            self.is_reset = True
            logger.info("Reset activated")
            self._voice_feedback("复位激活,停止左右臂遥操作")

    def _handle_single_click(self, button: Text):
        """Handles single-click events for buttons.

        Args:
            button: The button that was clicked.
        """
        if button == "b_button":
            if self.control_mode == ControlMode.DUAL_ARM:
                self.control_mode = ControlMode.WBC
                logger.info("Switched to WBC control mode")
                self._voice_feedback("切换到双臂控制")
            else:
                self.control_mode = ControlMode.DUAL_ARM
                logger.info("Switched to Dual Arm control mode")
                self._voice_feedback("切换到全身控制")
        elif button == "left_trigger":
            self.left_arm_scale = max(self.left_arm_scale - self.scale_step, self.min_scale_factor)
            logger.info(f"Left arm scale factor decreased to {self.left_arm_scale:.2f}")
            self._voice_feedback(f"左臂缩放因子减少到 {self.left_arm_scale:.2f}")
        elif button == "right_trigger":
            self.right_arm_scale = max(self.right_arm_scale - self.scale_step, self.min_scale_factor)
            logger.info(f"Right arm scale factor decreased to {self.right_arm_scale:.2f}")
            self._voice_feedback(f"右臂缩放因子减少到 {self.right_arm_scale:.2f}")
        elif button == "thumbstick":
            # Toggle data recording
            self.is_data_collection = not self.is_data_collection
            logger.info(f"Data recording {'started' if self.is_data_collection else 'stopped'}")
            if self.is_data_collection is False:
                self.left_arm_active = False
                self.right_arm_active = False
                self._voice_feedback(f"数据录制停止，停止左右臂遥操作")
            else:
                self._voice_feedback("数据录制开始")

    def _process_thumbstick_for_gripper(self, current_buttons: Dict[str, np.ndarray]):
        """Processes thumbstick input for gripper control.

        Args:
            current_buttons: Current state of the buttons including thumbstick values.
        """
        left_stick_x = current_buttons['left_thumbstick_x']
        right_stick_x = current_buttons['right_thumbstick_x']
        # Both thumbsticks are not pushed, no need to send gripper command
        if abs(left_stick_x) < 0.1 and abs(right_stick_x) < 0.1:
            self.is_send_gripper_command = False
            return
        
        if abs(left_stick_x) > 0.2:  # Add deadzone
            # Negative X is left (open), positive X is right (close)
            self.left_gripper_pos += left_stick_x * self.gripper_step
            self.left_gripper_pos = np.clip(self.left_gripper_pos, self.gripper_min, self.gripper_max)
            
        if abs(right_stick_x) > 0.2:  # Add deadzone
            # Negative X is left (open), positive X is right (close)
            self.right_gripper_pos += right_stick_x * self.gripper_step
            self.right_gripper_pos = np.clip(self.right_gripper_pos, self.gripper_min, self.gripper_max)
            
        self.is_send_gripper_command = True

    def _send_gripper_commands_if_needed(self):
        """Send gripper commands if needed at high frequency (60Hz)"""
        if not self.is_send_gripper_command:
            return
            
        # Skip if emergency stop is active
        if self.emergency_stop:
            return

        # Send gripper commands to the robot
        if self.left_arm_active:
            self._send_gripper_command('left', self.left_gripper_pos)

        if self.right_arm_active:
            self._send_gripper_command('right', self.right_gripper_pos)

    def _process_controller_movement(self):
        """Processes controller movements to update robot arm targets.
        """
        if self.emergency_stop:
            return
        current_time = time.perf_counter()
        # Processes left controller movement
        if self.left_arm_active and self.left_controller_origin is not None and self.left_arm_origin is not None:
            # Gets current_controller pose
            current_pose = self.left_controller.copy()
            world_left_controller_origin = self.trans_to_world(self.left_controller_origin)
            world_current_pose = self.trans_to_world(current_pose)
            # Calculate relative movement from origin
            target_pose = np.eye(4)
            target_pose[:3, :3] = world_current_pose[:3, :3] @ np.linalg.inv(
                world_left_controller_origin[:3, :3]) @ self.left_arm_origin[:3, :3]
            target_pose[:3, 3] = (world_current_pose[:3, 3] - world_left_controller_origin[:3,
                                                              3]) * self.left_arm_scale + self.left_arm_origin[:3, 3]
            # Adds sto history for smoothing
            pos = target_pose[:3, 3]
            rot = target_pose[:3, :3]
            # Calculates velocity for prediction
            if len(self.position_history['left']) > 0:
                dt = current_time - self.last_timestamp['left']
                if dt > 0:
                    # Calculates velocity
                    vel = (pos - self.last_position['left']) / dt
                    # Applies exponential smoothing to velocity
                    # logger.debug(f"Left arm velocity: {vel}|| position: {pos}, last_position: {self.last_position['left']}, dt: {dt}")
                    alpha = 0.3  # Smoothing factor
                    self.position_velocity['left'] = alpha * vel + (1 - alpha) * self.position_velocity['left']
            self.last_position['left'] = pos
            self.last_timestamp['left'] = current_time
            # Adds to smoothing history
            self.position_history['left'].append(pos)
            self.rotation_history['left'].append(rot)

            # Updates target pose
            self.left_arm_target = target_pose

        # Processes right controller movement
        if self.right_arm_active and self.right_controller_origin is not None and self.right_arm_origin is not None:
            # Gets current controller pose
            current_pose = self.right_controller.copy()
            world_right_controller_origin = self.trans_to_world(self.right_controller_origin)
            world_current_pose = self.trans_to_world(current_pose)
            # Calculate relative movement from origin
            target_pose = np.eye(4)
            target_pose[:3, :3] = world_current_pose[:3, :3] @ np.linalg.inv(
                world_right_controller_origin[:3, :3]) @ self.right_arm_origin[:3, :3]
            target_pose[:3, 3] = (world_current_pose[:3, 3] - world_right_controller_origin[:3,
                                                              3]) * self.right_arm_scale + self.right_arm_origin[:3, 3]
            # Adds to history for smoothing
            pos = target_pose[:3, 3]
            rot = target_pose[:3, :3]
            # Calculate velocity for prediction
            if len(self.position_history['right']) > 0:
                dt = current_time - self.last_timestamp['right']
                if dt > 0:
                    # Calculates velocity
                    vel = (pos - self.last_position['right']) / dt
                    # Applies exponential smoothing to velocity
                    alpha = 0.3  # Smoothing factor
                    self.position_velocity['right'] = alpha * vel + (1 - alpha) * self.position_velocity['right']

            self.last_position['right'] = pos
            self.last_timestamp['right'] = current_time
            # Adds to smoothing history
            self.position_history['right'].append(pos)
            self.rotation_history['right'].append(rot)
            # Updates target pose
            self.right_arm_target = target_pose

    def _generate_smooth_commands(self):
        """Generate smoothed command poses for robot arms"""
        # Left arm smoothing
        if self.left_arm_active and len(
                self.position_history['left']) > 0 and self._previous_left_arm_command is not None:
            # Applies position smoothing with weighted average
            positions = list(self.position_history['left'])
            weights = [i + 1 for i in range(len(positions))]
            sum_weights = sum(weights)

            # Calculate weighted average position
            avg_pos = np.zeros(3)
            for i, pos in enumerate(positions):
                avg_pos += pos * (weights[i] / sum_weights)

            # Applies velocity prediction
            prediction_time = 0.02  # Predict 20ms ahead
            predicted_pos = avg_pos + self.position_velocity['left'] * prediction_time
            # logger.debug(
            # f"Predicted position for left arm: {predicted_pos}|| avg_pos {avg_pos}, Velocity: {self.position_velocity['left']}")
            # Applies jitter removal - ignore very small movements
            jitter_threshold = 0.0005  # 0.5mm threshold
            if np.linalg.norm(predicted_pos - self._previous_left_arm_command[:3, 3]) < jitter_threshold:
                predicted_pos = self._previous_left_arm_command[:3, 3]

            # Smooth rotation using SLERP
            rotations = list(self.rotation_history['left'])
            # Converts rotation matrices to quaternions
            quats = [Rotation.from_matrix(r).as_quat() for r in rotations]

            # Uses the most recent rotation if we don't have enough for smoothing
            if len(quats) >= 2:
                # Weight toward most recent rotation
                weights = [i + 1 for i in range(len(quats))]
                sum_weights = sum(weights)

                # Calculates weighted average quaternion (simple approximation)
                avg_quat = np.zeros(4)
                for i, quat in enumerate(quats):
                    avg_quat += quat * (weights[i] / sum_weights)
                avg_quat = avg_quat / np.linalg.norm(avg_quat)

                # Converts back to rotation matrix
                smoothed_rot = Rotation.from_quat(avg_quat).as_matrix()
            else:
                smoothed_rot = rotations[-1]

            # Combines smoothed position and rotation into a pose
            smoothed_pose = np.eye(4)
            smoothed_pose[:3, :3] = smoothed_rot
            smoothed_pose[:3, 3] = predicted_pos
            # if self._previous_left_arm_command is not None:
            # Limits the maximum change in position to avoid large jumps
            pos_diff = smoothed_pose[:3, 3] - self._previous_left_arm_command[:3, 3]
            pose_diff_norm = np.linalg.norm(pos_diff)
            if pose_diff_norm > self.max_safe_position_change:
                logger.error(
                    f"Position change too large: {pose_diff_norm:.4f}, smoothed_pose: {smoothed_pose[:3, 3]} || pre: {self._previous_left_arm_command[:3, 3]},disabling left arm for safety")
                if self.is_recording:
                    self.data_recorder.stop_recording()
                self.left_arm_active = False
                return
            if pose_diff_norm > self.max_position_change:
                # Scale down the position change
                scale_factor = self.max_position_change / pose_diff_norm
                smoothed_pose[:3, 3] = self._previous_left_arm_command[:3, 3] + pos_diff * scale_factor
                logger.warning(
                    f"Position change too large: pose_diff: {pose_diff_norm} || norm: {pose_diff_norm:.4f}, scaling down to {self.max_position_change:.3f}, cmd: {smoothed_pose[:3, 3]}, prev: {self._previous_left_arm_command[:3, 3]}")
            # Limits the maximum change in rotation to avoid large jumps
            prev_rot = Rotation.from_matrix(self._previous_left_arm_command[:3, :3])
            current_rot = Rotation.from_matrix(smoothed_pose[:3, :3])
            rot_diff = prev_rot.inv() * current_rot
            angle = np.linalg.norm(rot_diff.as_rotvec())
            if angle > self.max_rotation_change:
                # Scale down the rotation change
                scale_factor = self.max_rotation_change / angle
                limited_rot = prev_rot * Rotation.from_rotvec(rot_diff.as_rotvec() * scale_factor)
                smoothed_pose[:3, :3] = limited_rot.as_matrix()
                logger.warning(
                    f"Rotation change too large: {angle:.4f}, scaling down to {self.max_rotation_change:.4f}")
            # Updates command pose
            self.left_arm_command = smoothed_pose

        # Right arm smoothing (similar logic)
        if self.right_arm_active and len(
                self.position_history['right']) > 0 and self._previous_right_arm_command is not None:
            # Apply position smoothing with weighted average
            positions = list(self.position_history['right'])
            weights = [i + 1 for i in range(len(positions))]
            sum_weights = sum(weights)

            # Calculate weighted average position
            avg_pos = np.zeros(3)
            for i, pos in enumerate(positions):
                avg_pos += pos * (weights[i] / sum_weights)

            # Apply velocity prediction
            prediction_time = 0.02  # Predict 20ms ahead
            predicted_pos = avg_pos + self.position_velocity['right'] * prediction_time

            # Apply jitter removal - ignore very small movements
            jitter_threshold = 0.0005  # 0.5mm threshold
            if np.linalg.norm(predicted_pos - self._previous_right_arm_command[:3, 3]) < jitter_threshold:
                predicted_pos = self._previous_right_arm_command[:3, 3]

            # Smooth rotation using SLERP
            rotations = list(self.rotation_history['right'])
            # Convert rotation matrices to quaternions
            quats = [Rotation.from_matrix(r).as_quat() for r in rotations]

            # Use the most recent rotation if we don't have enough for smoothing
            if len(quats) >= 2:
                # Weight toward most recent rotation
                weights = [i + 1 for i in range(len(quats))]
                sum_weights = sum(weights)

                # Calculate weighted average quaternion (simple approximation)
                avg_quat = np.zeros(4)
                for i, quat in enumerate(quats):
                    avg_quat += quat * (weights[i] / sum_weights)
                avg_quat = avg_quat / np.linalg.norm(avg_quat)

                # Convert back to rotation matrix
                smoothed_rot = Rotation.from_quat(avg_quat).as_matrix()
            else:
                smoothed_rot = rotations[-1]

            # Combine smoothed position and rotation into a pose
            smoothed_pose = np.eye(4)
            smoothed_pose[:3, :3] = smoothed_rot
            smoothed_pose[:3, 3] = predicted_pos
            # if self._previous_right_arm_command is not None:
            # Limits the maximum change in position to avoid large jumps
            pos_diff = smoothed_pose[:3, 3] - self._previous_right_arm_command[:3, 3]
            pose_diff_norm = np.linalg.norm(pos_diff)
            if pose_diff_norm > self.max_safe_position_change:
                logger.error(
                    f"Position change too large: {pose_diff_norm:.4f}, smoothed_pose: {smoothed_pose[:3, 3]} || pre: {self._previous_right_arm_command[:3, 3]},disabling right arm for safety")
                if self.is_recording:
                    self.data_recorder.stop_recording()
                self.right_arm_active = False
                return
            if pose_diff_norm > self.max_position_change:
                # Scales down the position change
                scale_factor = self.max_position_change / pose_diff_norm
                smoothed_pose[:3, 3] = self._previous_right_arm_command[:3, 3] + pos_diff * scale_factor
                logger.warning(
                    f"Position change too large: {pose_diff_norm:.4f}, scaling down to {self.max_position_change:.4f}")
            # Limits the maximum change in rotation to avoid large jumps
            prev_rot = Rotation.from_matrix(self._previous_right_arm_command[:3, :3])
            current_rot = Rotation.from_matrix(smoothed_pose[:3, :3])
            rot_diff = prev_rot.inv() * current_rot
            angle = np.linalg.norm(rot_diff.as_rotvec())
            if angle > self.max_rotation_change:
                # Scales down the rotation change
                scale_factor = self.max_rotation_change / angle
                limited_rot = prev_rot * Rotation.from_rotvec(rot_diff.as_rotvec() * scale_factor)
                smoothed_pose[:3, :3] = limited_rot.as_matrix()
                logger.warning(
                    f"Rotation change too large: {angle:.4f}, scaling down to {self.max_rotation_change:.4f}")
            # Updates command pose
            self.right_arm_command = smoothed_pose

    def vuer_run(self):
        self.vuer.run()

    def connect(self) -> bool:
        """Connect to the VR device and start processing threads"""
        if self._connected:
            return True

        try:
            logger.info("Starting VR Television connection...")
            self.process = Process(target=self.vuer_run)
            self.process.daemon = True
            self.process.start()
            self.running = True
            self.controller_thread = threading.Thread(target=self._controller_processing_loop)
            self.controller_thread.daemon = True
            self.controller_thread.start()

            self.robot_command_thread = threading.Thread(target=self._robot_command_loop)
            self.robot_command_thread.daemon = True
            self.robot_command_thread.start()


            self._connected = True
            self._ready = True
            logger.info("VR Television connected successfully")
            self._voice_feedback("遥操作连接成功，可以开始操作")
            return True
        except Exception as e:
            logger.error(f"Failed to connect VR Television: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from the VR device and stop processing"""
        if not self._connected:
            return True

        try:
            logger.info("Disconnecting VR Television...")
            self.running = False

            if self.controller_thread:
                self.controller_thread.join(timeout=2.0)
            if self.robot_command_thread:
                self.robot_command_thread.join(timeout=2.0)
            if self.process:
                self.process.terminate()
                self.process.join(timeout=2.0)
            self._connected = False
            self._ready = False
            logger.info("VR Television disconnected successfully")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting VR Television: {e}")
            return False

    def get_device_state(self) -> Dict[str, Any]:
        return {
            "left_hand": self.left_hand.tolist(),
            "right_hand": self.right_hand.tolist(),
            "left_controller": self.left_controller.tolist(),
            "right_controller": self.right_controller.tolist(),
            "left_arm_active": self.left_arm_active,
            "right_arm_active": self.right_arm_active,
            "left_gripper": self.left_gripper_pos,
            "right_gripper": self.right_gripper_pos,
            "emergency_stop": self.emergency_stop,
            "control_mode": self.control_mode.name,
            "left_arm_scale": self.left_arm_scale,
            "right_arm_scale": self.right_arm_scale
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the VR device"""
        return {
            "name": self.name,
            "type": "VR Headset",
            "model": "Quest 3",
            "control_modes": [mode.name for mode in ControlMode]
        }

    def trans_to_world(self, pose: np.ndarray) -> np.ndarray:
        """Transforms a pose from the VR device's coordinate system to the world coordinate system.

        Args:
            pose: A 4x4 transformation matrix representing the pose in the VR device's coordinate system.
        """
        return T_corenetic_openxr @ pose @ fast_mat_inv(T_corenetic_openxr)

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def left_hand_joints(self):
        return np.array(self.left_landmarks_shared[:]).reshape(24, 4, 4, order="F")

    @property
    def right_hand_joints(self):
        return np.array(self.right_landmarks_shared[:]).reshape(24, 4, 4, order="F")

    @property
    def left_hand_state_value(self):
        return np.array(self.left_hand_state_value_shared[:])

    @property
    def right_hand_state_value(self):
        return np.array(self.right_hand_state_value_shared[:])

    @property
    def left_hand_state(self):
        return np.array(self.left_hand_state_shared[:], dtype=bool)

    @property
    def right_hand_state(self):
        return np.array(self.right_hand_state_shared[:], dtype=bool)

    @property
    def left_controller(self):
        return np.array(self.left_controller_shared[:]).reshape(4, 4, order="F")

    @property
    def right_controller(self):
        return np.array(self.right_controller_shared[:]).reshape(4, 4, order="F")

    @property
    def left_button(self):
        return np.array(self.left_button_shared[:])

    @property
    def right_button(self):
        return np.array(self.right_button_shared[:])

    @property
    def left_button_state(self):
        return np.array(self.left_controller_state_shared[:], dtype=bool)

    @property
    def right_button_state(self):
        return np.array(self.right_controller_state_shared[:], dtype=bool)

    @property
    def left_arm_end_pose(self):
        if self.robot_interface is not None:
            return self.robot_interface.get_tcp_pose("left")["left"].matrix.copy()
        else:
            return None

    @property
    def right_arm_end_pose(self):
        if self.robot_interface is not None:
            return self.robot_interface.get_tcp_pose("right")["right"].matrix.copy()
        else:
            return None

    @property
    def left_arm_control_latency(self):
        return self._left_arm_control_latency

    @property
    def right_arm_control_latency(self):
        return self._right_arm_control_latency

    @property
    def controller_fps(self):
        return np.array(self._controller_fps_shared.value)

    @property
    def left_intervene(self):
        return self.left_arm_active

    @property
    def right_intervene(self):
        return self.right_arm_active


if __name__ == "__main__":
    vr_television = VRTelevision()
    vr_television.connect()
    while True:
        # logger.info(f"{vr_television.get_device_state()}")
        time.sleep(1)
