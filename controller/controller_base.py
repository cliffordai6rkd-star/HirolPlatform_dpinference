import pinocchio as pin
import abc
import numpy as np
from hardware.base.utils import RobotJointState
from scipy.spatial.transform import Rotation as R
from hardware.base.utils import convert_7D_2_homo
from motion.ik import GaussianNetwon, IK_DLS, IK_LM
from motion.pin_model import RobotModel
import copy

class ControllerBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config, robot_model: RobotModel):
        self._config = config
        self._robot_model = robot_model

    @abc.abstractmethod
    def compute_controller(self, target: dict, 
                           robot_state: RobotJointState | None = None) -> tuple[bool, np.ndarray , str]:
        """
            @brief: Compute the joint command based on the ee pose target
            @params:
                target (dict): 
                    key: the frame name
                    value: the ee pose target in form of [x,y,z,qx,qy,qz,qw]
            @returns:
                is successfully computed the value from controller (bool)
                joint values
                joint controlled mode ['position', 'velocity', 'torque'] (str)
        """
        raise NotImplementedError
    
class IKController(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self._damping_weight = config["damping_weight"]
        self._ik_type = config["ik_type"]
        self._tol = config["tolerance"]
        self._max_iter = config["max_iteration"]
        print(f"ik tol: {self._tol}, type: {type(self._tol)}")
        ik_class = {
            "gaussian_newtown": GaussianNetwon,
            "dls": IK_DLS,
            "lm": IK_LM
        }
        self._ik_object = ik_class[self._ik_type]()

    def compute_controller(self, target: dict, 
                           robot_state: RobotJointState | None = None):
        curr_target = copy.deepcopy(target)
        frame_name, pose_7d = next(iter(curr_target.items()))
        pose_homo = convert_7D_2_homo(pose_7d)
        curr_target[frame_name] = pose_homo
        # self._ik_object = IK_DLS()
        pin_model, pin_data = self._robot_model.get_pin_model_N_data()
        res = self._ik_object.ik(pin_model, pin_data, curr_target,
                                 robot_state._positions, self._tol,
                                 self._max_iter, self._damping_weight)
        # print(f'is ik converged: {res[0]}')
        success, joint_target, mode = res
        return success, joint_target, mode
    