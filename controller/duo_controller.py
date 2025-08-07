from controller.controller_base import ControllerBase, IKController
from controller.impedance_controller import ImpedanceController
from motion.duo_model import DuoRobotModel
from hardware.base.utils import RobotJointState, get_joint_slice_value
import warnings
import numpy as np

def check_joint_state(robot_state: RobotJointState, num_arm: int):
    dim = len(robot_state._positions)
    res = dim > 7 and num_arm == 2
    res = res or dim <= 7 and num_arm == 1
    return res


class DuoController(ControllerBase):
    _controller: dict[str, ControllerBase]
    _robot_model: DuoRobotModel
    def __init__(self, config, robot_model: DuoRobotModel):
        """
            @ brief: the duo controller for the left and right arm
            @ params:
                config: file
                robot_model: duo model
        """
        super().__init__(config, robot_model)
        self.controller_classes = {
            "ik": IKController,
            "impedance": ImpedanceController
        }
        self._controller = dict()
        left_config = config["left"]
        left_type = left_config["type"]
        self._controller['left'] = self.controller_classes[left_type](config=left_config,
                                                    robot_model=robot_model._models["left"])
        right_config = config["right"]
        right_type = right_config["type"]
        self._controller['right'] = self.controller_classes[right_type](config=right_config,
                                                    robot_model=robot_model._models["right"])
        
    def compute_controller(self, target, robot_state = None):
        """
            @params:
                target: list of dict[str, np.ndarray]
        """
        if not check_joint_state(robot_state, 2):
            warnings.warn(f'the joint state dim did not match with duo arm robot')
            raise ValueError("Wrong dim for joint state dim")
        
        if len(target) != 2:
            warnings.warn(f'Did not get two targets for the controller')
            raise ValueError("Wrong length of target for the controller")
        
        joint_left = None; mode_left = None; success_left = False
        joint_right = None; mode_right = None; success_right = False
        for i, cur_target in enumerate(target):
            # print(f'{i}th target: {cur_target}')
            is_left = True if i == 0 else False
            sliced_joint_states = self._slice_robot_joint_states(is_left, robot_state)
            controller = self._controller["left"] if is_left else self._controller["right"]
            if i == 0:
                success_left, joint_left, mode_left = controller.compute_controller(
                                                    [cur_target], sliced_joint_states)
            else:
                success_right, joint_right, mode_right = controller.compute_controller(
                                                    [cur_target], sliced_joint_states)
        if not success_left or not success_right:
            return False, None, None
        joint_target = np.hstack((joint_left, joint_right))
        mode = [mode_left, mode_right]
        return True, joint_target, mode
        
    def _slice_robot_joint_states(self, is_left, joint_states: RobotJointState):
        sliced_joint_states = None
        dofs = self._robot_model.get_model_dof()
        left_dof = dofs[0]
        right_dof = dofs[1]
        if is_left:
            sliced_joint_states = get_joint_slice_value(0, left_dof, joint_states)
        else:
            sliced_joint_states = get_joint_slice_value(left_dof, left_dof+right_dof, joint_states)
        return sliced_joint_states
        