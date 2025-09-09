from controller.controller_base import ControllerBase, IKController
from controller.impedance_controller import ImpedanceController
from motion.duo_model import DuoRobotModel
from hardware.base.utils import RobotJointState, get_joint_slice_value
import warnings
import glog as log
import numpy as np

def check_joint_state(robot_state: RobotJointState, num_arm: int):
    dim = len(robot_state._positions)
    res = dim > 7 and num_arm == 2
    res = res or (dim <= 7 and num_arm == 1)
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
            log.warn(f'the joint state dim did not match with duo arm robot')
            raise ValueError(f"Wrong dim for joint state dim, get ")
        
        if len(target) != 2:
            log.warn(f'Did not get two targets for the controller')
            raise ValueError(f"Wrong length of target for the controller, get len: {len(target)}")
        
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
    
    def reset(self, frame_name: str, robot_state: RobotJointState) -> None:
        """
        Reset the specific sub-controller based on frame_name
        
        Args:
            frame_name: End-effector frame name to reset
            robot_state: Current robot joint state
        """
        ee_links = self._robot_model.get_model_end_links()
        if not isinstance(ee_links, list):
            ee_links = [ee_links]
        
        # Ensure we have exactly 2 end-effector links for duo robot
        if len(ee_links) != 2:
            log.warning(f"Expected 2 end-effector links for duo robot, got {len(ee_links)}: {ee_links}")
            return
        
        # Determine which controller to reset based on frame_name
        if frame_name == ee_links[0]:  # Left arm frame
            left_joint_state = self._slice_robot_joint_states(True, robot_state)
            log.info(f'Resetting left controller with frame: {frame_name}')
            self._controller['left'].reset(frame_name, left_joint_state)
        elif frame_name == ee_links[1]:  # Right arm frame
            right_joint_state = self._slice_robot_joint_states(False, robot_state)
            log.info(f'Resetting right controller with frame: {frame_name}')
            self._controller['right'].reset(frame_name, right_joint_state)
        else:
            log.warning(f"Unknown frame_name '{frame_name}' for duo robot. Expected one of: {ee_links}")
            return
        
        log.info(f'Duo controller reset completed for frame: {frame_name}')
        