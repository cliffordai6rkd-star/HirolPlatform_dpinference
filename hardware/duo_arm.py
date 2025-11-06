from hardware.fr3.fr3_arm import Fr3Arm
from hardware.monte01.xarm7_arm import Xarm7Arm
from hardware.monte02.monte02_arm import Monte02_Arm
from hardware.base.arm import ArmBase
from hardware.base.utils import RobotJointState, combine_two_joint_states
import threading

class DuoArm(ArmBase):
    arm: dict[str, ArmBase]
    def __init__(self, config):
        self.single_arm_classes = {
            'fr3': Fr3Arm,
            'xarm7': Xarm7Arm,
            'monte02arm': Monte02_Arm,
        }
        self.arm = {}
        left_config = config["left"]
        left_type = left_config["type"]
        self.arm['left'] = self.single_arm_classes[left_type](config=left_config)
        right_config = config["right"]
        right_type = right_config["type"]
        self.arm['right'] = self.single_arm_classes[right_type](config=right_config)
        self._is_initialized = self.arm['left']._is_initialized\
                                and self.arm['right']._is_initialized
        self._dof = [self.arm['left']._dof, self.arm['right']._dof]
        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        res_left = self.arm['left'].initialize()
        res_right = self.arm['right'].initialize()
        return res_left and res_right
        
    def print_state(self):
        self.arm['left'].print_state()
        self.arm['right'].print_state()
    
    def get_joint_states(self)-> RobotJointState:
        """
            joint state should contains two arm commands
            np.ndarray, [0-dof['left']]: left arm command
            [dof['left']: dof['left']+dof['right']]: right arm command
        """
        joint_left = self.arm['left'].get_joint_states()
        joint_right = self.arm['right'].get_joint_states()
        combined_state = combine_two_joint_states(joint_left, joint_right)
        return combined_state
    
    def update_arm_states(self):
        pass
    
    def close(self):
        self.arm['left'].close()
        self.arm['right'].close()
        print(f'Duo arm hardware has been successfully closed!!!')
    
    def set_joint_command(self, mode, command):
        """
            command should contains two arm commands
            np.ndarray, [0-dof['left']]: left arm command
            [dof['left']: dof['left']+dof['right']]: right arm command
        """
        if len(mode) != 2:
            raise ValueError(f'The mode for dual arm does not have 2 elements but get {len(mode)}')
        start = [0, self._dof[0]]
        end = [self._dof[0], self._dof[0] + self._dof[1]]
        for i, cur_mode in enumerate(mode):            
            arm = self.arm['left'] if i == 0 else self.arm['right']
            arm.set_joint_command(cur_mode, command[start[i]:end[i]])
            
    def move_to_start(self):
        def move_to_start_async(index):
            self.arm[index].move_to_start()
        threads:list[threading.Thread] = []
        for cur_index in ["left", "right"]:
            cur_thread = threading.Thread(target=move_to_start_async, args=(cur_index,))
            cur_thread.start()
            threads.append(cur_thread)
        
        for cur_thread in threads:
            cur_thread.join()
    
