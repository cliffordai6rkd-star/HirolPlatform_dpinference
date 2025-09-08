import pinocchio as pin
import abc
import numpy as np
from hardware.base.utils import RobotJointState
from scipy.spatial.transform import Rotation as R
from hardware.base.utils import convert_7D_2_homo
from motion.ik import GaussianNewton, IK_DLS, IK_LM, IK_PPINK, IK_PYROKI, IK_CUROBO
from motion.pin_model import RobotModel
from hardware.base.utils import object_class_check
import copy

class ControllerBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config, robot_model: RobotModel):
        self._config = config
        self._robot_model = robot_model

    @abc.abstractmethod
    def compute_controller(self, target: list[dict[str, np.ndarray]], 
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
        self._damping_weight = config.get("damping_weight", 0.3)
        self._ik_type = config["ik_type"]
        self._tol = config.get("tolerance", 1e-4)
        self._max_iter = config.get("max_iteration", 1000)
        ik_class = {
            "gaussian_newton": GaussianNewton,
            "dls": IK_DLS,
            "lm": IK_LM,
            "pink": IK_PPINK,
            "pyroki": IK_PYROKI,
            "curobo": IK_CUROBO
        }
        if not object_class_check(ik_class, self._ik_type):
            raise ValueError
        
        # Get solver-specific configuration
        solver_config = {}
        if self._ik_type == "pyroki" and "pyroki_config" in config:
            solver_config = config["pyroki_config"]
        elif self._ik_type == "curobo" and "curobo_config" in config:
            solver_config = config["curobo_config"]
        
        self._ik_object = ik_class[self._ik_type](**solver_config)

    def compute_controller(self, target: list[dict[str, np.ndarray]], 
                           robot_state: RobotJointState | None = None):
        curr_target = copy.deepcopy(target[0])
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
    