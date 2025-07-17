from controller.controller_base import ControllerBase, IKController
from controller.impedance_controller import ImpedanceController
from hardware.base.utils import RobotJointState, get_joint_slice_value
import warnings
import numpy as np

def check_joint_state(robot_state: RobotJointState, num_arm: int):
    dim = len(robot_state._positions)
    res = dim > 7 and num_arm == 2
    res = res or dim <= 7 and num_arm == 1
    return res


class DuoController(ControllerBase):
    def __init__(self, config, robot_model: dict):
        """
            @ brief: the duo controller for the left and right arm
            @ params:
                config: file
                robot_model: dict with two elements, left and right for each arm's model
        """
        super().__init__(config, robot_model)
        self.controller_classes = {
            "ik": IKController,
            "impedance": ImpedanceController
        }
        self.controller = dict()
        left_config = config["left"]
        left_type = left_config[type]
        self.controller['left'] = self.controller_classes[left_type](config=left_config,
                                                                     robot_model=robot_model["left"])
        right_config = config["right"]
        right_type = right_config[type]
        self.controller['right'] = self.controller_classes[right_type](config=right_config,
                                                                     robot_model=robot_model["right"])
        
    def compute_controller(self, target, robot_state = None):
        """
            @params:
                target: key: 'letf_' or 'right_' + frame name
        """
        if not check_joint_state(robot_state, 2):
            warnings.warn(f'the joint state dim did not match with duo arm robot')
            raise ValueError("Wrong dim for joint state dim")
        
        if len(target) != 2:
            warnings.warn(f'Did not get two targets for the controller')
            raise ValueError("Wrong length of target for the controller")
        
        joint_left = None, mode_left = None, success_left = False
        joint_right = None, mode_right = None, success_right = False
        for key, value in target.items():
            if 'left' in key:
                success_left, joint_left, mode_left = self._compute_single_controller_output(
                                                    key, value, robot_state, True)
            else:
                success_right, joint_right, mode_right = self._compute_single_controller_output(
                                                    key, value, robot_state, False)
        if not success_left or not success_right:
            return False, None, None
        joint_target = np.hstack((joint_left, joint_right))
        mode = [mode_left[0], mode_right[0]]
        return True, joint_target, mode
                
    def _compute_single_controller_output(self, target_key, target_value, joint_states, is_left = True):
        controller = self.controller['left'] if is_left else self.controller['right']
        frame_name = target_key.split('_', 1)[1]
        target = {frame_name: target_value}
        cur_joint_state = self._slice_robot_joint_states(is_left, joint_states)
        return controller.compute_controller(target, cur_joint_state)
        
    def _slice_robot_joint_states(self, is_left, joint_states: RobotJointState):
        sliced_joint_states = None
        left_dof = self._robot_model["left"].nv
        right_dof = self._robot_model["right"].nv
        if is_left:
            sliced_joint_states = get_joint_slice_value(0, left_dof, joint_states)
        else:
            sliced_joint_states = get_joint_slice_value(left_dof, left_dof+right_dof, joint_states)
        return sliced_joint_states
        