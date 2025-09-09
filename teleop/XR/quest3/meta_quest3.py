"""
    # Credit to unitree xr_teleoperation objects
    # Modified by zyx
"""

from teleop.base.teleoperation_base import TeleoperationDeviceBase
from hardware.base.utils import convert_homo_2_7D_pose, negate_transform, transform_quat, fast_mat_inv, pose_diff
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
from teleop.base.utils import RisingEdgeDetector
from typing import Optional
import numpy as np
import asyncio
import cv2, copy
import os
from pathlib import Path
import glog as log

def safe_mat_update(prev_mat, mat):
    # Return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0).
    det = np.linalg.det(mat)
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return prev_mat, False
    return mat, True

def safe_rot_update(prev_rot_array, rot_array):
    dets = np.linalg.det(rot_array)
    if not np.all(np.isfinite(dets)) or np.any(np.isclose(dets, 0.0, atol=1e-6)):
        return prev_rot_array, False
    return rot_array, True

def init_image_shared_mem(cfg):
    # cfg 补充
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    img_shape = cfg['image_shape']
    if (img_shape[1] / img_shape[0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    cfg['binocular'] = BINOCULAR
        
    if BINOCULAR and not (img_shape[1] / img_shape[0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_shape[0], img_shape[1] * 2, 3)
    else:
        tv_img_shape = (img_shape[0], img_shape[1], 3)
    cfg['image_shape'] = tv_img_shape
    
    img_shm = shared_memory.SharedMemory(create=True, 
                    size = np.prod(tv_img_shape) * np.uint8().itemsize)
    cfg["img_shm_name"] = img_shm.name
    return img_shm

class MetaQuest3(TeleoperationDeviceBase):
    # @TODO: check with init situation
    T_ROBOT_OPENXR = np.array([[ 0, 0,-1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 0, 1]])

    T_OPENXR_ROBOT = np.array([[ 0,-1, 0, 0],
                           [ 0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 0, 1]])
    CONST_RIGHT_ARM_POSE = np.eye(4)
    CONST_LEFT_ARM_POSE = np.eye(4)
    CONST_HEAD_POSE = np.eye(4)
    CONST_HAND_ROT = np.tile(np.eye(3)[None, :, :], (25, 1, 1))
    ROT_THRESHOLD = np.pi / 2.0
    INIT_TARGET_POSE = [np.eye(4), np.eye(4)]
    
    def __init__(self, config):
        self._binocular = config["binocular"]
        self._output_left = config.get("output_left", False)
        self._output_right = config.get("output_right", True)
        log.info(f'MetaQuest3 config: output_left={self._output_left}, output_right={self._output_right}')
        self._use_hand_tracking = config["use_hand_tracking"]
        if self._use_hand_tracking:
            self._relative_finger_pose = config.get("relative_finger_pose", True)
        else:
            # tool control, binary -> controlled by trigger; continous -> controlled by trigger and squeeze
            self._tool_control_mode = config["tool_mode"]
            self._rising_edges = [RisingEdgeDetector(), RisingEdgeDetector()]
            self._tool_step_sizes = config.get("incremental_step", {})
            self._last_tool_values = {"left": 1.0, "right": 1.0}
        self._img_shm_name = config["img_shm_name"]
        log.info(f'Image shared memory name: {self._img_shm_name}')
        self._img_shape = config["image_shape"]
        self._init_pose = config.get('init_pose', None)
        self._last_quat = [np.array([0,0,0,1]), np.array([0,0,0,1])]
        if not self._init_pose is None:
            self._init_pose_rot = [self._init_pose["initial_pose_left"][3:], 
                                    self._init_pose["initial_pose_right"][3:]]
            self._last_quat = self._init_pose_rot
            self._init_pose_trans = [self._init_pose["initial_pose_left"][:3], 
                                    self._init_pose["initial_pose_right"][:3]]
        self._img_height = self._img_shape[0]
        if self._binocular:
            self._img_width  = self._img_shape[1] // 2
        else:
            self._img_width  = self._img_shape[1]
        
        self._cert_file = config.get("cert_file", None)
        
        self._key_file = config.get("key_file", None)
        self._ngork = config.get("ngork", False)
        self._webrtc = config.get("webrtc", False)
        self._img_shm: Optional[shared_memory.SharedMemory] = None
        self._device_enabled = False
        
        # initialize and thread starting
        super().__init__(config)

    def initialize(self) -> bool:
        if self._is_initialized:
            return True
        
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if self._cert_file is None:
            self._cert_file = os.path.join(cur_path, 'certifications', "meta_quest3_cert.pem")
        else:
            self._cert_file = os.path.join(cur_path, "../../..", self._cert_file)
        if self._key_file is None:
            self._key_file = os.path.join(cur_path, 'certifications', "meta_quest3_key.pem")
        else:
            self._key_file = os.path.join(cur_path, "../../..", self._key_file)
            
        if self._ngork:
            self._vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self._vuer = Vuer(host='0.0.0.0', cert=self._cert_file, 
                                key=self._key_file, queries=dict(grid=False), queue_len=3)
            
        self._vuer.add_handler("CAMERA_MOVE")(self._on_cam_move)
        if self._use_hand_tracking:
            self._vuer.add_handler("HAND_MOVE")(self._on_hand_move)
        else:
            self._vuer.add_handler("CONTROLLER_MOVE")(self._on_controller_move)

        # Connect to the shared memory we just created
        existing_shm = shared_memory.SharedMemory(name=self._img_shm_name)
        self._img_array = np.ndarray(self._img_shape, dtype=np.uint8, buffer=existing_shm.buf)
        
        if self._binocular and not self._webrtc:
            self._vuer.spawn(start=False)(self._main_image_binocular)
        elif not self._binocular and not self._webrtc:
            self._vuer.spawn(start=False)(self._main_image_monocular)
        elif self._webrtc:
            self._vuer.spawn(start=False)(self._main_image_webrtc)

        self._head_pose_shared = Array('d', 16, lock=True)
        self._left_arm_pose_shared = Array('d', 16, lock=True)
        self._right_arm_pose_shared = Array('d', 16, lock=True)
        if self._use_hand_tracking:
            self._left_hand_position_shared = Array('d', 75, lock=True)
            self._right_hand_position_shared = Array('d', 75, lock=True)
            self._left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self._right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self._left_pinch_state_shared = Value('b', False, lock=True)
            self._left_pinch_value_shared = Value('d', 0.0, lock=True)
            self._left_squeeze_state_shared = Value('b', False, lock=True)
            self._left_squeeze_value_shared = Value('d', 0.0, lock=True)

            self._right_pinch_state_shared = Value('b', False, lock=True)
            self._right_pinch_value_shared = Value('d', 0.0, lock=True)
            self._right_squeeze_state_shared = Value('b', False, lock=True)
            self._right_squeeze_value_shared = Value('d', 0.0, lock=True)
        else:
            self._left_trigger_state_shared = Value('b', False, lock=True)
            self._left_trigger_value_shared = Value('d', 0.0, lock=True)
            self._left_squeeze_state_shared = Value('b', False, lock=True)
            self._left_squeeze_value_shared = Value('d', 0.0, lock=True)
            self._left_thumbstick_state_shared = Value('b', False, lock=True)
            self._left_thumbstick_value_shared = Array('d', 2, lock=True)
            self._left_aButton_shared = Value('b', False, lock=True)
            self._left_bButton_shared = Value('b', False, lock=True)

            self._right_trigger_state_shared = Value('b', False, lock=True)
            self._right_trigger_value_shared = Value('d', 0.0, lock=True)
            self._right_squeeze_state_shared = Value('b', False, lock=True)
            self._right_squeeze_value_shared = Value('d', 0.0, lock=True)
            self._right_thumbstick_state_shared = Value('b', False, lock=True)
            self._right_thumbstick_value_shared = Array('d', 2, lock=True)
            self._right_aButton_shared = Value('b', False, lock=True)
            self._right_bButton_shared = Value('b', False, lock=True)

        self._process = Process(target=self.read_data)
        self._process.daemon = True
        self._process.start()
        return True

    def read_data(self):
        self._vuer.run()
        
    def close(self):
        # Clean up shared memory if it exists
        if hasattr(self, '_img_shm') and self._img_shm is not None:
            try:
                self._img_shm.close()
                self._img_shm.unlink()
                log.info(f"Cleaned up shared memory: {self._img_shm.name}")
            except FileNotFoundError:
                # Already cleaned up
                pass
            except Exception as e:
                log.warning(f"Failed to clean up shared memory: {e}")
            finally:
                self._img_shm = None
        
        return self._vuer.close_ws()
    
    def print_data(self):
        data = self.parse_data_2_robot_target("absolute")
        log.info(f'pose: {data[1]}')
        
    def apply_init_offset(self, pose, id):
        new_pose = copy.deepcopy(pose)
        # basis convertion (mainly for rot)
        if self._init_pose is not None:
            init_rot = self._init_pose_rot[id]
            new_pose[3:] = transform_quat(pose[3:], init_rot)
        
        # @TODO: try to add a threshold 
        # no need posi offset means need head posi to calib
        # if not need_posi_offset:
        #     # filter of the pose
        #     last_pose = np.hstack((np.array([0, 0, 0]), np.array(self._last_quat[id])))
        #     pose_diff = compute_pose_diff(new_pose, last_pose)
        #     if np.linalg.norm(pose_diff[3:]) < self.ROT_THRESHOLD:
        #         self._last_quat[id] = new_pose[3:]
        #     else:
        #         new_pose[3:] = self._last_quat[id]
        pose[:7] = new_pose[:7] # update step for sallow copy
        return new_pose
    
    def parse_data_2_robot_target(self, mode):
        if not self._is_initialized:
            log.warn("The meta quest 3 xr device is not initialized!!!")
            return False, None, None
        
        if mode != "absolute" and mode != "absolute_delta":
            log.warn(f'meta quest3 only support absolute related mode but get {mode}!!!!')
            return False, None, None
        
        pose_target = {}
        world2head_trans, head_valid = safe_mat_update(self.CONST_HEAD_POSE, self.head_pose)
        world2left_arm_trans, left_arm_valid = safe_mat_update(self.CONST_LEFT_ARM_POSE, self.left_arm_pose)
        world2right_arm_trans, right_arm_valid = safe_mat_update(self.CONST_RIGHT_ARM_POSE, self.right_arm_pose)
        # Check validity - require at least head and one arm to be valid
        if not head_valid:
            log.warn(f'Head pose is not valid!!!')
            return False, None, None
        
        if not left_arm_valid and not right_arm_valid:
            log.warn(f'Both arm poses are not valid!!!')
            return False, None, None
        
        # Log validity status for debugging
        # log.info(f'[DEBUG] Pose validity: head={head_valid}, left_arm={left_arm_valid}, right_arm={right_arm_valid}')
        # log.info(f'[DEBUG] Output config: output_left={self._output_left}, output_right={self._output_right}')
        
        # basis change
        robot_world2head_trans = self.T_ROBOT_OPENXR @ world2head_trans @ self.T_OPENXR_ROBOT if head_valid else np.eye(4)
        robot_world2left_arm_trans = self.T_ROBOT_OPENXR @ world2left_arm_trans @ self.T_OPENXR_ROBOT if left_arm_valid else np.eye(4)
        robot_world2right_arm_trans = self.T_ROBOT_OPENXR @ world2right_arm_trans @ self.T_OPENXR_ROBOT if right_arm_valid else np.eye(4)
        pose_left = convert_homo_2_7D_pose(robot_world2left_arm_trans)
        pose_right = convert_homo_2_7D_pose(robot_world2right_arm_trans)
        #  init convertion and rotation clipping
        pose_left = self.apply_init_offset(pose_left, 0)
        pose_right = self.apply_init_offset(pose_right, 1)
        debug = [pose_left, pose_right]
        
        if mode == "absolute":
            head_posi = robot_world2head_trans[:3, 3]
            # head posi offset
            pose_left[:2] = pose_left[:2] - head_posi[:2]
            pose_left[2] = 1.0 - (head_posi[2] - pose_left[2])
            pose_right[:2] = pose_right[:2] - head_posi[:2]
            pose_right[2] = 1.0 - (head_posi[2] - pose_right[2])
            # if self._init_pose is not None: # To support config based position offset
            #     pose_left[:3] += self._init_pose_trans[0]
            #     pose_right[:3] += self._init_pose_trans[1]
        # Generate pose targets based on config and data validity
        elif mode == "absolute_delta":
            if self._device_enabled:
                pose_left = self._get_diff_trans(pose_left, 0)
                pose_right = self._get_diff_trans(pose_right, 1)
        
        if not self._output_left:
            if right_arm_valid: pose_target['single'] = pose_right
            else: 
                log.info(f'Output right arm does not get right arm pose ')
                return False, None, None
        elif not self._output_right:
            if left_arm_valid: pose_target['single'] = pose_left
            else: 
                log.info(f'Output left arm doe not get left arm pose')
                return False, None, None
        else:
            if not left_arm_valid or not right_arm_valid:
                log.info(f'Output dual arms: left_valid={left_arm_valid}, right_valid={right_arm_valid}')
                return False, None, None

            # Both arms enabled and valid
            if left_arm_valid:
                pose_target['left'] = pose_left
            if right_arm_valid:
                pose_target['right'] = pose_right
        
        tool_target = {}
        if self._use_hand_tracking:
            #  hand tracking for finger joint pose, [4,9,14,19,24] for finger end
            left_hand_position = self.left_hand_positions
            left_hand_orientation, left_hand_valid = safe_rot_update(self.CONST_HAND_ROT, self.left_hand_orientations)
            right_hand_orientation, right_hand_valid = safe_rot_update(self.CONST_HAND_ROT, self.right_hand_orientations)
            if not left_hand_valid or not right_hand_valid:
                log.warn(f'Hand rotation is not valid!!!')
                return False, None, None
            left_hand_orientation = np.einsum('ij, nij -> nij', self.T_ROBOT_OPENXR[:3, :3], left_hand_orientation)
            left_hand_orientation = np.einsum('nij, ij -> nij', left_hand_orientation, self.T_OPENXR_ROBOT[:3, :3])
            right_hand_orientation = np.einsum('ij, nij -> nij', self.T_ROBOT_OPENXR[:3, :3], right_hand_orientation)
            right_hand_orientation = np.einsum('nij, ij -> nij', right_hand_orientation, self.T_OPENXR_ROBOT[:3, :3])
            
            n, _ = left_hand_position.shape 
            last_row = np.array([0,0,0,1] * n).reshape(n, 1, 4)
            left_hand_pose = np.concatenate((left_hand_orientation, left_hand_position[:, :, None]), axis=2)
            left_hand_pose = np.concatenate((left_hand_pose, last_row), axis=1)
            right_hand_position = self.right_hand_positions
            right_hand_pose = np.concatenate((right_hand_orientation, right_hand_position[:, :, None]), axis=2)
            right_hand_pose = np.concatenate((right_hand_pose, last_row), axis=1)
            
            if self._relative_finger_pose:
                world2leftwrist_trans = left_hand_pose[0]
                leftwrist2world_trans = negate_transform(world2leftwrist_trans)
                left_hand_pose = np.einsum('ij, nij -> nij', leftwrist2world_trans, left_hand_pose)
                world2rightwrist_trans = right_hand_pose[0]
                rightwrist2world_trans = negate_transform(world2rightwrist_trans)
                right_hand_pose = np.einsum('ij, nij -> nij', rightwrist2world_trans, right_hand_pose)
            left_hand_pose_7d = np.zeros((n, 7))
            right_hand_pose_7d = np.zeros((n, 7))
            # 4 9 14 19 24 for right fingertip from thumb to the right
            for i in range(n):
                left_hand_pose_7d[i, :] = convert_homo_2_7D_pose(left_hand_pose[i])
                right_hand_pose_7d[i, :] = convert_homo_2_7D_pose(right_hand_pose[i])
                # basis changing
                self.apply_init_offset(left_hand_pose_7d[i], 0)
                self.apply_init_offset(right_hand_pose_7d[i], 1) 
            tool_left = left_hand_pose_7d
            tool_right = left_hand_pose_7d
            # @TODO: find a way to init the pose target
            self._device_enabled = True
        else:
            # controller tracking for gripper control
            # state (bool) & value (float): contains triggers , squezzes, thumb sticks, buttons (bool)
            left_trigger_state = self.left_controller_trigger_state 
            left_trigger_value = self.left_controller_trigger_value 
            left_squeeze_state = self.left_controller_squeeze_state
            left_squeeze_value = self.left_controller_squeeze_value
            left_thumb_stick_state = self.left_controller_thumbstick_state
            left_thumb_stick_value = self.left_controller_thumbstick_value # x, y float values
            left_a_button = self.left_controller_aButton
            left_b_button = self.left_controller_bButton
            left_data = np.hstack((left_trigger_state, left_trigger_value, left_squeeze_state,
                                   left_squeeze_value, left_a_button, left_b_button, 
                                   left_thumb_stick_state, left_thumb_stick_value))
            right_trigger_state = self.right_controller_trigger_state 
            right_trigger_value = self.right_controller_trigger_value 
            right_squeeze_state = self.right_controller_squeeze_state
            right_squeeze_value = self.right_controller_squeeze_value
            right_thumb_stick_state = self.right_controller_thumbstick_state
            right_thumb_stick_value = self.right_controller_thumbstick_value # x, y float values
            right_a_button = self.right_controller_aButton
            right_b_button = self.right_controller_bButton
            right_data = np.hstack((right_trigger_state, right_trigger_value, right_squeeze_state,
                                   right_squeeze_value, right_a_button, right_b_button, 
                                   right_thumb_stick_state, right_thumb_stick_value))
            # update init_pose target
            if mode == "absolute_delta":
                left_data = np.hstack((left_data, left_a_button))
                right_data = np.hstack((right_data, left_a_button))
            if left_a_button and left_arm_valid and right_arm_valid and not self._device_enabled:
                self.INIT_TARGET_POSE[0] = pose_left
                self.INIT_TARGET_POSE[1] = pose_right
                self._device_enabled = True
                log.info(f"{'='*10}INIT TARGET POSE is successfully updated, and could start teleoperation{'='*10}")
                return False, None, None
            if right_a_button and self._device_enabled: 
                log.info(f"{'='*10}Disable the output from meta quest3{'='*10}")
                self._device_enabled = False
            
            # data: (state, value combination), trigger, squeeze, a,b button, thumb stick
            for i, (key, tool_mode) in enumerate(self._tool_control_mode.items()):
                cur_trigger = left_trigger_state if i == 0 else right_trigger_state
                cur_tool_data = left_data if i == 0 else right_data
                if tool_mode == "binary":
                    control_value = self._rising_edges[i].update(float(cur_trigger))
                    if control_value:
                        self._last_tool_values[key] = not self._last_tool_values[key]
                    cur_tool_data[0] = float(self._last_tool_values[key])
                else:
                    cur_squezze = left_squeeze_state if i == 0 else right_squeeze_state
                    if cur_trigger:
                        self._last_tool_values[key] += self._tool_step_sizes[key]
                    elif cur_squezze : 
                        self._last_tool_values[key] -= self._tool_step_sizes[key]
                    cur_tool_data[0] = self._last_tool_values[key]
                    
            tool_left = left_data
            tool_right = right_data
        if not self._output_left:
            tool_target['single'] = tool_right
        elif not self._output_right:
            tool_target['single'] = tool_left
        else:
            tool_target['left'] = tool_left
            tool_target['right'] = tool_right
        tool_target["reset"] = np.array([left_b_button, right_b_button, 
            right_thumb_stick_state, left_thumb_stick_state])
        
        if (self._device_enabled and mode == "absolute_delta") or mode == "absolute":
            return True, pose_target, tool_target
        else: 
            log.warn(f'mode: {mode}, device enabled: {self._device_enabled}')
            return False, None, None
        
    
    # helper functions
    def _get_diff_trans(self, cur_pose, index):
        diff_pose = pose_diff(cur_pose, self.INIT_TARGET_POSE[index])
        return diff_pose
    
    # related to vuer xr helper function
    async def _on_cam_move(self, event, session, fps=60):
        # print(f'triggered cam move!!!!!')
        # print(f'cam key: {event.key} value: {event.value}')
        try:
            with self._head_pose_shared.get_lock():
                self._head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def _on_controller_move(self, event, session, fps=60):
        # log.info(f'triggered controller move!!!!!')
        try:
            with self._left_arm_pose_shared.get_lock():
                self._left_arm_pose_shared[:] = event.value["left"]
            with self._right_arm_pose_shared.get_lock():
                self._right_arm_pose_shared[:] = event.value["right"]

            left_controller_state = event.value["leftState"]
            right_controller_state = event.value["rightState"]
            # log.info(f'left value: {left_controller_state} \n right value: {right_controller_state}')

            def extract_controller_states(state_dict, prefix):
                # log.info(f'extract controller statets for {prefix}, state_dict: {state_dict}')
                # trigger
                with getattr(self, f"_{prefix}_trigger_state_shared").get_lock():
                    getattr(self, f"_{prefix}_trigger_state_shared").value = bool(state_dict.get("trigger", False))
                with getattr(self, f"_{prefix}_trigger_value_shared").get_lock():
                    getattr(self, f"_{prefix}_trigger_value_shared").value = float(state_dict.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"_{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"_{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"_{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"_{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"_{prefix}_thumbstick_state_shared").get_lock():
                    getattr(self, f"_{prefix}_thumbstick_state_shared").value = bool(state_dict.get("thumbstick", False))
                with getattr(self, f"_{prefix}_thumbstick_value_shared").get_lock():
                    getattr(self, f"_{prefix}_thumbstick_value_shared")[:] = state_dict.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"_{prefix}_aButton_shared").get_lock():
                    getattr(self, f"_{prefix}_aButton_shared").value = bool(state_dict.get("aButton", False))
                with getattr(self, f"_{prefix}_bButton_shared").get_lock():
                    getattr(self, f"_{prefix}_bButton_shared").value = bool(state_dict.get("bButton", False))

            extract_controller_states(left_controller_state, "left")
            # log.info('finished left gripper state extraction')
            extract_controller_states(right_controller_state, "right")
            # log.info('finished both gripper state extraction')
        except:
            pass

    async def _on_hand_move(self, event, session, fps=60):
        # print(f'trigered hand move!!!')
        try:
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            left_hand_state = event.value["leftState"]
            right_hand_state = event.value["rightState"]

            def extract_hand_poses(hand_data, arm_pose_shared, hand_position_shared, hand_orientation_shared):
                with arm_pose_shared.get_lock():
                    arm_pose_shared[:] = hand_data[0:16]

                with hand_position_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_position_shared[i * 3: i * 3 + 3] = [hand_data[base + 12], hand_data[base + 13], hand_data[base + 14]]

                with hand_orientation_shared.get_lock():
                    for i in range(25):
                        base = i * 16
                        hand_orientation_shared[i * 9: i * 9 + 9] = [
                            hand_data[base + 0], hand_data[base + 1], hand_data[base + 2],
                            hand_data[base + 4], hand_data[base + 5], hand_data[base + 6],
                            hand_data[base + 8], hand_data[base + 9], hand_data[base + 10],
                        ]
                        
            def extract_hand_states(state_dict, prefix):
                # pinch
                with getattr(self, f"{prefix}_pinch_state_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_state_shared").value = bool(state_dict.get("pinch", False))
                with getattr(self, f"{prefix}_pinch_value_shared").get_lock():
                    getattr(self, f"{prefix}_pinch_value_shared").value = float(state_dict.get("pinchValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
                with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
                    getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))

            extract_hand_poses(left_hand_data, self._left_arm_pose_shared, self._left_hand_position_shared, self._left_hand_orientation_shared)
            extract_hand_poses(right_hand_data, self._right_arm_pose_shared, self._right_hand_position_shared, self._right_hand_orientation_shared)
            extract_hand_states(left_hand_state, "left")
            extract_hand_states(right_hand_state, "right")

        except:
            pass
                        
    async def _main_image_binocular(self, session, fps=60):
        if self._use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False
                ),
                # to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                # to="bgChildren",
            )

        while True:
            display_image = cv2.cvtColor(self._img_array, cv2.COLOR_BGR2RGB)
            # @TODO: 后续采集数据需要打开
            aspect_ratio = self._img_width / self._img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image[:, :self._img_width],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=100,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        display_image[:, self._img_width:],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=100,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                # to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)
            
    async def _main_image_monocular(self, session, fps=60):
        if self._use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=False,
                    hideRight=False
                ),
                # to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                # to="bgChildren",
            )

        while True:
            display_image = cv2.cvtColor(self._img_array, cv2.COLOR_BGR2RGB)
            aspect_ratio = self._img_width / self._img_height
            # session.upsert(
            #     [
            #         ImageBackground(
            #             display_image,
            #             aspect=1.778,
            #             height=1,
            #             distanceToCamera=1,
            #             format="jpeg",
            #             quality=50,
            #             key="background-mono",
            #             interpolate=True,
            #         ),
            #     ],
            #     # to="bgChildren",
            # )
            await asyncio.sleep(0.016)

    async def _main_image_webrtc(self, session, fps=60):
        if self._use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    showLeft=True,
                    showRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    showLeft=True,
                    showRight=True,
                )
            )
    
        session.upsert(
            WebRTCVideoPlane(
            # WebRTCStereoVideoPlane(
                src="https://10.0.7.49:8080/offer",
                iceServer={},
                key="webrtc",
                aspect=1.778,
                height = 7,
            ),
            to="bgChildren",
        )
        while True:
            await asyncio.sleep(1)
            
    # ==================== common data ====================
    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self._head_pose_shared.get_lock():
            return np.array(self._head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self._left_arm_pose_shared.get_lock():
            return np.array(self._left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self._right_arm_pose_shared.get_lock():
            return np.array(self._right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self._left_hand_position_shared.get_lock():
            return np.array(self._left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self._right_hand_position_shared.get_lock():
            return np.array(self._right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self, order = 'F'):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, default: column-major)."""
        with self._left_hand_orientation_shared.get_lock():
            return np.array(self._left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order=order)

    @property
    def right_hand_orientations(self, order = 'F'):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, default: column-major)."""
        with self._right_hand_orientation_shared.get_lock():
            return np.array(self._right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order=order)

    @property
    def left_hand_pinch_state(self):
        """bool, whether left hand is pinching."""
        with self._left_pinch_state_shared.get_lock():
            return self._left_pinch_state_shared.value

    @property
    def left_hand_pinch_value(self):
        """float, pinch strength of left hand."""
        with self._left_pinch_value_shared.get_lock():
            return self._left_pinch_value_shared.value

    @property
    def left_hand_squeeze_state(self):
        """bool, whether left hand is squeezing."""
        with self._left_squeeze_state_shared.get_lock():
            return self._left_squeeze_state_shared.value

    @property
    def left_hand_squeeze_value(self):
        """float, squeeze strength of left hand."""
        with self._left_squeeze_value_shared.get_lock():
            return self._left_squeeze_value_shared.value

    @property
    def right_hand_pinch_state(self):
        """bool, whether right hand is pinching."""
        with self._right_pinch_state_shared.get_lock():
            return self._right_pinch_state_shared.value

    @property
    def right_hand_pinch_value(self):
        """float, pinch strength of right hand."""
        with self._right_pinch_value_shared.get_lock():
            return self._right_pinch_value_shared.value

    @property
    def right_hand_squeeze_state(self):
        """bool, whether right hand is squeezing."""
        with self._right_squeeze_state_shared.get_lock():
            return self._right_squeeze_state_shared.value

    @property
    def right_hand_squeeze_value(self):
        """float, squeeze strength of right hand."""
        with self._right_squeeze_value_shared.get_lock():
            return self._right_squeeze_value_shared.value

    # ==================== Controller Data ====================
    @property
    def left_controller_trigger_state(self):
        """bool, left controller trigger pressed or not."""
        with self._left_trigger_state_shared.get_lock():
            return self._left_trigger_state_shared.value

    @property
    def left_controller_trigger_value(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self._left_trigger_value_shared.get_lock():
            return self._left_trigger_value_shared.value

    @property
    def left_controller_squeeze_state(self):
        """bool, left controller squeeze pressed or not."""
        with self._left_squeeze_state_shared.get_lock():
            return self._left_squeeze_state_shared.value

    @property
    def left_controller_squeeze_value(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self._left_squeeze_value_shared.get_lock():
            return self._left_squeeze_value_shared.value

    @property
    def left_controller_thumbstick_state(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self._left_thumbstick_state_shared.get_lock():
            return self._left_thumbstick_state_shared.value

    @property
    def left_controller_thumbstick_value(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self._left_thumbstick_value_shared.get_lock():
            return np.array(self._left_thumbstick_value_shared[:])

    @property
    def left_controller_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self._left_aButton_shared.get_lock():
            return self._left_aButton_shared.value

    @property
    def left_controller_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self._left_bButton_shared.get_lock():
            return self._left_bButton_shared.value

    @property
    def right_controller_trigger_state(self):
        """bool, right controller trigger pressed or not."""
        with self._right_trigger_state_shared.get_lock():
            return self._right_trigger_state_shared.value

    @property
    def right_controller_trigger_value(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self._right_trigger_value_shared.get_lock():
            return self._right_trigger_value_shared.value

    @property
    def right_controller_squeeze_state(self):
        """bool, right controller squeeze pressed or not."""
        with self._right_squeeze_state_shared.get_lock():
            return self._right_squeeze_state_shared.value

    @property
    def right_controller_squeeze_value(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self._right_squeeze_value_shared.get_lock():
            return self._right_squeeze_value_shared.value

    @property
    def right_controller_thumbstick_state(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self._right_thumbstick_state_shared.get_lock():
            return self._right_thumbstick_state_shared.value

    @property
    def right_controller_thumbstick_value(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self._right_thumbstick_value_shared.get_lock():
            return np.array(self._right_thumbstick_value_shared[:])

    @property
    def right_controller_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self._right_aButton_shared.get_lock():
            return self._right_aButton_shared.value

    @property
    def right_controller_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self._right_bButton_shared.get_lock():
            return self._right_bButton_shared.value
        
    
