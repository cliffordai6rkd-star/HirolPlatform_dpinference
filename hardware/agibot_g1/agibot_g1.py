from hardware.base.arm import ArmBase
import threading, time, warnings
from hardware.base.utils import RobotJointState
import numpy as np

# 条件导入 a2d_sdk
try:
    from a2d_sdk.robot import RobotDds as Robot
    A2D_SDK_AVAILABLE = True
except ImportError:
    A2D_SDK_AVAILABLE = False
    print("Warning: a2d_sdk not available, using mock implementation")
    
    class MockRobot:
        def __init__(self):
            self._arm_positions = [0.0] * 14
            self._head_positions = [0.0] * 2
            self._waist_positions = [0.0] * 2
            print("[Mock] Robot initialized")
        
        def move_arm(self, positions):
            self._arm_positions = list(positions)
            print(f"[Mock] Moving arm to: {positions[:3]}...")
        
        def move_head(self, positions):
            self._head_positions = list(positions)
            print(f"[Mock] Moving head to: {positions}")
        
        def move_waist(self, positions):
            self._waist_positions = list(positions)
            print(f"[Mock] Moving waist to: {positions}")
        
        def move_wheel(self, linear, angular):
            print(f"[Mock] Moving wheel: linear={linear}, angular={angular}")
        
        def arm_joint_states(self):
            velocities = [0.0] * len(self._arm_positions)
            return self._arm_positions.copy(), velocities
        
        def waist_joint_states(self):
            velocities = [0.0] * len(self._waist_positions)
            return self._waist_positions.copy(), velocities
        
        def head_joint_states(self):
            velocities = [0.0] * len(self._head_positions)
            return self._head_positions.copy(), velocities
        
        def reset(self):
            print("[Mock] Robot reset")
        
        def shutdown(self):
            print("[Mock] Robot shutdown")
    
    Robot = MockRobot

class AgibotG1(ArmBase):
    HEAD_LIMITS_MIN = [-90, -20]
    HEAD_LIMITS_MAX = [90, 20]
    WAIST_LIMITS_MIN = [0, 0]
    WAIST_LIMITS_MAX = [90, 0.5]
    
    def __init__(self, config):
        self._robot = Robot()
        self._control_head = config.get('control_head', False)
        self._control_waist = config.get('control_waist', False)
        self._control_wheel = config.get('control_wheel', False)
        self._thread_running = True
        self._update_thread = threading.Thread(target=self.update_arm_states)
        super().__init__(config)
        
    def initialize(self):
        self._last_posi = np.zeros(self._dof)
        self._last_vel = np.zeros(self._dof)
    
        self._update_thread.start()        
        return True
        
    def update_state_task(self):
        read_frequency = 800

        print(f'The Agibot {self._ip} started update thread!!!')
        last_read_time = time.time()
        while self._thread_running:
            dt = time.time() - last_read_time
            last_read_time = time.time()
            
            # status update
            self._lock.acquire()
            self.update_arm_states()
            self._joint_states._velocities = (self._joint_states._positions - self._last_posi) / dt
            self._joint_states._accelerations = (self._joint_states._velocities - self._last_vel) / dt
            self._lock.release()
            
            if dt < (1.0 / read_frequency):
                sleep_time = (1.0 / read_frequency) - dt
                time.sleep(sleep_time)
            elif dt > 1.2 * (1.0 / read_frequency):
                warnings.warn(f'Read frequency is slow: expected: {1.0 / read_frequency}, '
                              f'actual: {1.0 / dt}')
                
        print(f'The Agibot {self._ip} stopped update thread!!!')
    
    def update_arm_states(self):
        arm_posi = self.get_dual_arm_positions()
        waist_posi = self.get_waist_positions()
        head_posi = self.get_head_positions()
        
        self._joint_states._positions = arm_posi
        if self._control_head and not self._control_waist:
            # dual arm + head
            self._joint_states._positions = np.hstack(((arm_posi, head_posi)))
        elif not self._control_head and self._control_waist:
            self._joint_states._positions = np.hstack(((arm_posi, waist_posi)))
        elif self._control_head and self._control_waist:
            self._joint_states._positions = np.hstack(((arm_posi, head_posi, waist_posi)))

    def set_joint_command(self, mode: list[str], command):
        """
            command: 14 dof: arm joint radians; 16 dof: arm joint + head radians; 
                18 dof: arm joint + head radians + waist(lift(m)+pitch(radian)) 
        """
        for cur_mode in mode:
            if cur_mode != 'position':
                raise ValueError(f'AgibotG1 only supports for position control!!!')
            
        if len(command) != self._dof:
            raise ValueError(f'AgibotG1 joint command length should be {self._dof}, '
                             f'but got {len(command)}')
        self._robot.move_arm(command[:14])
        
        # @TODO: find the way to handle the joint configurations
        if self._dof > 14:
            head_command = command[14:16]
            head_command = head_command / np.pi * 180
            head_command = np.clip(head_command, self.HEAD_LIMITS_MIN, self.HEAD_LIMITS_MAX)
            self._robot.move_head(head_command)
        if self._dof > 16:
            waist_lift = command[16]
            waist_pitch = command[17] / np.pi * 180
            waist_command = np.array([waist_pitch, waist_lift])
            waist_command = np.clip(waist_command, self.WAIST_LIMITS_MIN, self.WAIST_LIMITS_MAX)
            self._robot.move_waist(waist_command)
    
    def close(self):
        self._thread_running = False
        self._update_thread.join()
        self._robot.reset()
        self._robot.shutdown()

    def set_chassis_command(self, chassis_speed):
        if len(chassis_speed) != 2:
            raise ValueError(f'AgibotG1 chassis command should be 2D, but got {len(chassis_speed)}D')
        self._robot.move_wheel(chassis_speed[0], chassis_speed[1])
    
    def get_dual_arm_positions(self):
        arm_posi, _ = self._robot.arm_joint_states()
        return np.array(arm_posi)
    
    def get_waist_positions(self):
        waits_posi, _ = self._robot.waist_joint_states()
        return np.array(waits_posi)
    
    def get_head_positions(self):
        head_posi, _ = self._robot.head_joint_states()
        return np.array(head_posi)
    