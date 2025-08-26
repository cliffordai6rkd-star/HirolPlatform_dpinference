try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None

from hardware.base.arm import ArmBase
from hardware.monte01.xarm_api_manager import XArmAPIManager
import glog as log
import threading, time
import numpy as np

from hardware.monte01.xarm_defs import *
from hardware.monte01.coordinate_transforms import JP_XARM2CORENETIC

class Xarm7Arm(ArmBase):
    def __init__(self, config):
        self._ip = config["ip"]
        # Get shared XArmAPI instance
        self._robot = XArmAPIManager.get_instance(self._ip)
        
        if self._robot is not None:
            log.info(f"XArm7 using shared API instance for {self._ip}")
        else:
            log.error(f"Failed to get shared XArmAPI instance for {self._ip}")

        self._filter_coefficient = config.get("filter_coeff", None)
        self._collision_behaviour = config.get("collision_behaviour", None)
        self._control_mode = None
        self._thread = threading.Thread(target=self.update_robot_state_thread)
        self._last_velocities = np.zeros(7)
        self._state = None
        self._state_update_flag = False
        self._thread_running = True

        super().__init__(config)

    def update_robot_state_thread(self):
        read_frequency = 500
        target_period = 1.0 / read_frequency
        
        last_read_time = time.perf_counter()
        first_iteration = True
        
        while self._thread_running:
            current_time = time.perf_counter()
            dt = current_time - last_read_time
            
            # Get joint states
            code, self._state = self._robot.get_joint_states(is_radian=True)
            if code != XARM_SUCCESS:
                log.error(f"Failed to get joint states: {code}")
                time.sleep(0.001)  # Short sleep on error
                continue
            
            self._state_update_flag = True
            current_velocity = np.array(self._state[1])  # velocity from get_joint_states return
            
            # Calculate acceleration (skip first iteration to avoid invalid dt)
            if not first_iteration and dt > 0:
                self._joint_states._accelerations = (current_velocity - self._last_velocities) / dt
            else:
                self._joint_states._accelerations = np.zeros_like(current_velocity)
                first_iteration = False
            
            self._last_velocities = current_velocity

            # Update state
            self._lock.acquire()
            self.update_arm_states()
            self._lock.release()
            
            # Sleep control with proper timing
            loop_end_time = time.perf_counter()
            elapsed_time = loop_end_time - current_time
            
            if elapsed_time < target_period:
                sleep_time = target_period - elapsed_time
                time.sleep(sleep_time)
            elif elapsed_time > target_period * 2:
                log.debug(f"XArm7 state update is slower than target frequency "
                           f"{read_frequency}Hz, actual: {1.0 / elapsed_time:.1f}Hz")
            
            last_read_time = time.perf_counter()
            
        log.info(f'XArm7 with ip {self._ip} stopped its thread!!!')

    def initialize(self):
        if self._control_mode is None:
            self._control_mode = "position"
            # self._robot.set_collision_tool_model(tool_type=TOOL_TYPE_XARM_GRIPPER)
        if not self._is_initialized:
            # if self._collision_behaviour is not None:
            #     self.set_collision_threshold(self._collision_behaviour["torque_min"],
            #                                  self._collision_behaviour["torque_max"],
            #                                  self._collision_behaviour["force_min"],
            #                                  self._collision_behaviour["force_max"])
            self._thread.start()
        self._state_update_flag = False
        while not self._state_update_flag:
            pass
        log.info(f'Xarm7 robot with ip {self._ip} is successfully updated!!!')
        
        # Initialize safety checker state with current joint positions
        if self._state and len(self._state) > 0:
            self.init_safety_state(np.array(self._state[0]))
            log.info("Safety checker initialized with current joint positions")
        
        return True

    def close(self):
        self._thread_running = False
        self._thread.join()
        if self._robot:
            # Don't close the shared instance, just stop this component
            log.info(f'XArm7 arm component for {self._ip} is closed!!')
        else:
            log.info(f'Robot with ip {self._ip} is closed!!')

    def update_arm_states(self):
        if not self._state_update_flag:
            log.warn(f'The fr3 state is still not ready for robot state to update!')
            return 
        
        self._joint_states._positions = np.array(self._state[0] + JP_XARM2CORENETIC)
        # print(f'posi: {self._joint_states._positions}')
        self._joint_states._velocities = np.array(self._state[1])
        # self._joint_states._torques = np.array(self._fr3_state.tau_J)
        self._joint_states._time_stamp = time.perf_counter()
        
        # Update safety checker with current joint positions
        self.update_safety_state(np.array(self._state[0]))
    
    def set_joint_command(self, mode, command):
        # controller setting or controller change
        if not self._is_initialized or self._control_mode != mode:
            self._control_mode = mode
            self._is_initialized = self.initialize()

        # set command, checking
        if len(command) != self._dof:
            log.warn(f"the command dimension does not match with the arm dof: "
                    f"expect: {self._dof}, get: {len(command)}")
            return False

        xarm_cmd = command - JP_XARM2CORENETIC  # Adjust command to XARM reference
        # Safety check: validate joint command with joint constraints
        is_safe, reason = self.check_joint_command_safety(xarm_cmd)

        if not is_safe:
            log.warning(f"Joint command safety check failed: {reason}")
            return False
        
        self._robot.clean_error()
        self._robot.motion_enable(enable=True)
        self._robot.set_state(state=XARM_STATE_SPORT)
        # set command
        if mode == 'position':
            self._robot.set_mode(XARM_MODE_SERVO)
            code = self._robot.set_servo_angle_j(angles=xarm_cmd, is_radian=True)
        elif mode == 'velocity':
            self._robot.set_mode(XARM_MODE_VELOCITY)
            code = self._robot.vc_set_joint_velocity(speeds=xarm_cmd, is_radian=True)
        # elif mode == 'torque':
        #     self.set_joint_torque(xarm_cmd)
        if code != XARM_SUCCESS:
            log.error(f"Failed to set joint cmd: mode={mode}, code={code}")
            return False
        
        # Commit safe state after successful command execution
        self.commit_safe_state()
        return True
    
    def set_teaching_mode(self):
        code = self._robot.set_mode(mode=XARM_MODE_TEACHING)
        if code != XARM_SUCCESS:
            log.error(f"Failed to set teaching mode: code={code}")
            return False
        log.info("Teaching mode set successfully")
        return True
    
    def recover(self):
        """
        Recover from error state by resetting the robot.
        Returns True if recovery was successful, False otherwise.
        
        Based on xArm SDK documentation, proper error recovery sequence:
        1. Check robot state and error status
        2. Clean errors and warnings
        3. Enable motion
        4. Set robot to sport state
        5. Verify recovery was successful
        """
        if not self._robot:
            log.error("Robot not initialized, cannot recover")
            return False
            
        try:
            log.info("Starting robot recovery process...")
            
            # Step 1: Get current robot state and error info
            code, state = self._robot.get_state()
            if code != XARM_SUCCESS:
                log.error(f"Failed to get robot state: {code}")
                return False
                
            code, err_warn_code = self._robot.get_err_warn_code()
            if code != XARM_SUCCESS:
                log.error(f"Failed to get error/warning code: {code}")
                return False
                
            log.info(f"Robot state: {state}, Error/Warning code: {err_warn_code}")
            
            # Step 2: Clean errors and warnings
            log.info("Cleaning errors and warnings...")
            clean_error_code = self._robot.clean_error()
            if clean_error_code != XARM_SUCCESS:
                log.error(f"Failed to clean error: {clean_error_code}")
                return False
                
            clean_warn_code = self._robot.clean_warn()
            if clean_warn_code != XARM_SUCCESS:
                log.error(f"Failed to clean warning: {clean_warn_code}")
                return False
            
            # Step 3: Enable motion (required after clean_error according to docs)
            log.info("Enabling motion...")
            motion_enable_code = self._robot.motion_enable(enable=True)
            if motion_enable_code != XARM_SUCCESS:
                log.error(f"Failed to enable motion: {motion_enable_code}")
                return False
            
            # Step 4: Set robot to sport state
            log.info("Setting robot to sport state...")
            set_state_code = self._robot.set_state(state=XARM_STATE_SPORT)
            if set_state_code != XARM_SUCCESS:
                log.error(f"Failed to set sport state: {set_state_code}")
                return False
            
            # Step 5: Wait a bit for state change to take effect
            time.sleep(0.1)
            
            # Step 6: Verify recovery was successful
            code, new_state = self._robot.get_state()
            if code != XARM_SUCCESS:
                log.error(f"Failed to verify robot state after recovery: {code}")
                return False
            
            code, new_err_warn_code = self._robot.get_err_warn_code()
            if code != XARM_SUCCESS:
                log.error(f"Failed to verify error status after recovery: {code}")
                return False
            
            # Check if recovery was successful
            if new_state == XARM_STATE_SPORT and new_err_warn_code == 0:
                log.info(f"Robot recovery successful! State: {new_state}, Error code: {new_err_warn_code}")
                return True
            else:
                log.warning(f"Robot recovery may be incomplete. State: {new_state}, Error code: {new_err_warn_code}")
                # Still return True if state is sport, even if there are minor warnings
                return new_state == XARM_STATE_SPORT
                
        except Exception as e:
            log.error(f"Exception during robot recovery: {e}")
            return False
    
    def emergency_recover(self):
        """
        Emergency recovery function for critical errors.
        Uses more aggressive recovery methods.
        """
        if not self._robot:
            log.error("Robot not initialized, cannot perform emergency recovery")
            return False
            
        try:
            log.info("Starting emergency recovery process...")
            
            # Stop any ongoing motion
            self._robot.emergency_stop()
            time.sleep(0.5)
            
            # Clean all types of errors
            self._robot.clean_error()
            self._robot.clean_warn()
            
            # Try to clean gripper errors if present
            try:
                self._robot.clean_gripper_error()
            except:
                pass  # Gripper might not be present
            
            # Reset robot configuration to defaults if needed
            try:
                self._robot.clean_conf()
            except:
                pass  # May not be supported on all versions
            
            # Enable motion and set to sport state
            self._robot.motion_enable(enable=True)
            self._robot.set_state(state=XARM_STATE_SPORT)
            
            time.sleep(1.0)  # Wait longer for emergency recovery
            
            # Verify recovery
            code, state = self._robot.get_state()
            if code == XARM_SUCCESS and state == XARM_STATE_SPORT:
                log.info("Emergency recovery successful!")
                return True
            else:
                log.error(f"Emergency recovery failed. State: {state}, Code: {code}")
                return False
                
        except Exception as e:
            log.error(f"Exception during emergency recovery: {e}")
            return False
    
    def check_error_state(self):
        """
        Check if robot is in error state.
        Returns (has_error, error_code, state)
        """
        if not self._robot:
            return True, -1, -1
            
        try:
            code, state = self._robot.get_state()
            if code != XARM_SUCCESS:
                return True, code, -1
                
            code, err_warn_code = self._robot.get_err_warn_code()
            if code != XARM_SUCCESS:
                return True, code, state
                
            # Check if in error state (state 4 is error state)
            has_error = (state == 4) or (err_warn_code != 0)
            return has_error, err_warn_code, state
            
        except Exception as e:
            log.error(f"Exception checking error state: {e}")
            return True, -1, -1
    
    def get_ee_pose(self):
        """
        Get end effector pose in homogeneous transformation matrix format
        
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix representing end effector pose
                       Format: [[R11, R12, R13, tx],
                               [R21, R22, R23, ty], 
                               [R31, R32, R33, tz],
                               [0,   0,   0,   1 ]]
        """
        from scipy.spatial.transform import Rotation as R
        
        if not self._robot:
            log.error("Robot not initialized, cannot get end effector pose")
            return np.eye(4)
        
        try:
            # Get end effector pose from xArm API
            # get_position returns: [x, y, z, roll, pitch, yaw] (position in mm, angles in radians)
            code, pose = self._robot.get_position(is_radian=True)
            if code != XARM_SUCCESS:
                log.error(f"Failed to get end effector pose: {code}")
                return np.eye(4)
            
            # Extract position and orientation
            position = np.array(pose[:3]) / MM_PER_M  # Convert from mm to meters
            euler_angles = np.array(pose[3:6])  # [roll, pitch, yaw] in radians
            
            # Convert Euler angles to rotation matrix using scipy
            rotation = R.from_euler('xyz', euler_angles)
            rotation_matrix = rotation.as_matrix()
            
            # Create 4x4 homogeneous transformation matrix
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = rotation_matrix
            homogeneous_matrix[:3, 3] = position
            
            return homogeneous_matrix
            
        except Exception as e:
            log.error(f"Exception getting end effector pose: {e}")
            return np.eye(4)
    
    def move_to_start(self, joint_commands = None):
        if self._init_joint_positions is None:
            log.error("Initial joint positions not set, cannot move to start position")
        else:
            # self.set_joint_command('position', self._init_joint_positions)
            log.warn(f"TODO: Move to start position not implemented for XArm7 {self._init_joint_positions}")
            self._robot.clean_error()
            time.sleep(0.005)
            self._robot.motion_enable(enable=True)
            time.sleep(0.005)
            self._robot.set_mode(XARM_MODE_POSITION)
            time.sleep(0.005)
            self._robot.set_state(state=XARM_STATE_SPORT)
            time.sleep(0.005)
            # set command
            code = self._robot.set_servo_angle(angle=self._init_joint_positions, is_radian=True, wait=True)
            if code != XARM_SUCCESS:
                log.error(f"Failed to move to start position: code={code}")
            else:
                log.info("Robot moved to start position successfully")