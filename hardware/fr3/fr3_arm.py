from hardware.base.arm import ArmBase
import numpy as np
import warnings
import panda_py
from panda_py import controllers 
# import panda_py.constants
from panda_py import Panda
import threading
import time
from scipy.spatial.transform import Rotation as R
import copy

class Fr3Arm(ArmBase):
    def __init__(self, config):
        self._ip = config["ip"]
        self._fr3_robot = Panda(self._ip)
        self._damping = config.get("damping", None)
        self._stiffness = config.get("stiffness", None)
        self._filter_coefficient = config.get("filter_coeff", None)
        self._control_mode = None
        self._thread = threading.Thread(target=self.update_robot_state_thread)
        self._last_velocities = np.zeros(7)
        self._fr3_state = None
        self._fr3_state_update_flag = False
        self._thread_running = True
        
        # init
        super().__init__(config)
    
    def update_robot_state_thread(self):
        read_frequency = 800
        
        last_read_time = time.time()
        while self._thread_running:
            dt = time.time() - last_read_time
            last_read_time = time.time()
            
            # diff acceleration
            self._fr3_state = self._fr3_robot.get_state()
            self._fr3_state_update_flag = True
            self._joint_states._accelerations = (np.array(self._fr3_state.dq) - self._last_velocities) / dt
            self._last_velocities = np.array(self._fr3_state.dq)

            # update state
            self._lock.acquire()
            self.update_arm_states()
            self._lock.release()
            
            if dt > 1.2 / read_frequency:
                warnings.warn(f"Reading fr3 robot state is slower than the read frequency "
                              f"{read_frequency}Hz, actual: {1.0 / dt}Hz")
            elif dt < 1.0 / read_frequency:
                sleep_time = (1.0 / read_frequency) - dt
                time.sleep(sleep_time)
        print(f'Fr3 with ip {self._ip} stopped its thread!!!')
            
    def initialize(self):
        if self._control_mode is None:
            self._panda_py_controller = controllers.JointPosition()
            self._control_mode = "position"

        # pandapy low level controller parameters
        if not self._damping is None: 
            self._panda_py_controller.set_damping(self._damping)
        if not self._stiffness is None and \
            not isinstance(self._panda_py_controller, controllers.AppliedTorque):
            self._panda_py_controller.set_stiffness(self._stiffness)
        if not self._filter_coefficient is None and \
            not isinstance(self._panda_py_controller, controllers.IntegratedVelocity):
            self._panda_py_controller.set_filter(self._filter_coefficient)
        # controller start
        self._fr3_robot.start_controller(self._panda_py_controller)    

        if not self._is_initialized:
            self._thread.start()
        while not self._fr3_state_update_flag:
            pass

        print(f'Fr3 robot with ip {self._ip} is successfully updated!!!')
        return True
    
    def close(self):
        self._thread_running = False
        self._thread.join()
        self._fr3_robot.stop_controller()
        print(f'Fr3 robot with ip {self._ip} is closed!!')        
                   
    def update_arm_states(self):
        if not self._fr3_state_update_flag:
            warnings.warn(f'The fr3 state is still not ready for robot state to update!')
            return 
        
        self._joint_states._positions = np.array(self._fr3_state.q)
        # print(f'posi: {self._joint_states._positions}')
        self._joint_states._velocities = np.array(self._fr3_state.dq)
        self._joint_states._torques = np.array(self._fr3_state.tau_J)

        
        # tcp_pose = np.array(self._fr3_state.O_T_EE)
        # tcp_pose = np.reshape(tcp_pose, (4, 4))
        # self._tcp_pose[:3] = tcp_pose[:3, 3]
        # self._tcp_pose[3:] = R.from_matrix(tcp_pose[:3, :3]).as_quat()
    
    def set_joint_command(self, mode, command):
        # controller setting or controller change
        if not self._is_initialized or self._control_mode != mode:
            if mode == 'position':
                self._panda_py_controller = controllers.JointPosition()
            elif mode == 'velocity':
                self._panda_py_controller = controllers.IntegratedVelocity()
            elif mode == 'torque':
                self._panda_py_controller = controllers.AppliedTorque()
            else:
                warnings.warn(f"Unsupported control mode: {mode}")
                return False
            self._control_mode = mode
            self._is_initialized = self.initialize()

        # set command, checking
        if len(command) != self._dof:
            warnings.warn(f"the command dimension does not match with the arm dof: "
                    f"expect: {self._dof}, get: {len(command)}")
            return False
        if not mode is None and mode != self._control_mode:
            warnings.warn(f"the controller mode is not consistent with the previous one: "
                    f"expect: {self._control_mode}, get: {mode}")
            return False
        
        # set command
        if mode == 'position':
            self.set_joint_position(command)
        elif mode == 'velocity':
            self.set_joint_velocity(command)
        elif mode == 'torque':
            self.set_joint_torque(command)
        else:
            warnings.warn(f'fr3 did not support control mode {mode}')
            return False
        return True
            
            
    def set_joint_position(self, position):
        # @TODO: wait for testing
        self._panda_py_controller.set_control(position)
    
    def set_joint_velocity(self, velocity):
        self._panda_py_controller.set_control(velocity)
    
    def set_joint_torque(self, torque):
        self._panda_py_controller.set_control(torque)

    def set_teaching_mode(self, enable_taching: bool):
        self._fr3_robot.teaching_mode(enable_taching)

    def get_ee_pose(self):
        """
            return the end effector pose in the homogenous format
        """
        pose = np.array(self._fr3_robot.get_pose())
        return pose
    
    def stop_controller(self):
        """
            stop the controller
        """
        self._fr3_robot.stop_controller()
        self._control_mode = None

if __name__ == '__main__':
    import yaml
    import os
    config = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, 'config', 'fr3_cfg.yaml')
    print(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'yaml data: {config}')
    fr3 = Fr3Arm(config['fr3'])
    
    test_mode = "torque"
    test_times = 10
    counter = 0
    joint_id = 6
    test_value = 0.06
    init_state = fr3.get_joint_states()
    while counter <= test_times:
        joint_state = fr3.get_joint_states()
        # print(f'joint acc: {joint_state._accelerations}')
        print(f'tor: {joint_state._torques}')
        # fr3.print_state()
        if test_mode == "position":
            joint_command = init_state._positions
            if counter % 2 == 0:
                joint_command[joint_id] +=  test_value
            else:
                joint_command[joint_id] -= test_value
            init_state._positions = joint_command
        elif test_mode == "torque":
            joint_command = init_state._torques
            if counter % 2 == 0:
                joint_command[joint_id] = test_value
            else:
                joint_command[joint_id] = -test_value
        else:
            raise ValueError(f"Not support for mode {test_mode}")
        
        print(f'command: {joint_command}, mode: {test_mode}')
        fr3.set_joint_command(test_mode, joint_command)
        
        counter += 1     
        time.sleep(0.02)
        
    print('Exit the testing of fr3 arm!!!!')
    