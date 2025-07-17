from motion.pin_model import RobotModel
from controller.controller_base import ControllerBase
import casadi
import pinocchio.casadi as cpin
import pinocchio as pin
from hardware.base.utils import RobotJointState, convert_7D_2_homo, convert_quat_to_rot_matrix

class WholeBodyIk(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self.tracking_frames = config["tracking_frames"]
        self.show_debug = config.get("show_debug", False)
        self.translation_weight = config["trans_weight"]
        self.rot_weight = config["rot_weight"]
        self.smooth_weight = config["smooth_weight"]
        self.regularization_weight = config["regularization_weight"]
        
        # init of casadi opti problem
        self._init_casadi_problem()
        
    def compute_controller(self, target, robot_state: RobotJointState | None = None):
        """
            Get the inverse kinematics solution for multiple target
            @params:
                target: Dict[np.ndarray], multiple target, key: frame name, 
                    value: 7D pose, [x,y,z,qx,qy,qz,qw] 
            @return: return the actuated joint positions (dim: nv)
        """
        # init of the problem
        init_q = robot_state._positions
        self.opti.set_initial(self.var_q, init_q)
        
        pin.framesForwardKinematics(self.pin_model, self.pin_data, init_q)
        pin.computeForwardKinematicsDerivatives(self.pin_model, self.pin_data, init_q, 
                                                robot_state._velocities, 
                                                robot_state._accelerations)
        # set parameter values
        self.opti.set_value(self.parameters[0], init_q)
        for i, (frame_name, value) in enumerate(target.items()):
            if self.tracking_frames[i]["name"] != frame_name:
                raise ValueError("target frame name is not matching: "
                                 f"expected: {self.tracking_frames[i]['name']}, actual: {frame_name}")
            if self.tracking_frames[i]["require_trans"] and self.tracking_frames[i]["require_rot"]:
                target_homo = convert_7D_2_homo(value)
                self.opti.set_value(self.parameters[i+1], target_homo)
            elif self.tracking_frames[i]["require_trans"]:
                self.opti.set_value(self.parameters[i+1], value[:3])
            elif self.tracking_frames[i]["require_rot"]:
                rot = convert_quat_to_rot_matrix(value[:4])
                self.opti.set_value(self.parameters[i+1], rot)
        
        # solve optimization
        try:
            soln = self.opti.solve()
            
            q_target = self.opti.value(self.var_q)
            return True, q_target, "position"
        except Exception as e:
            print(f"ERROR in convergence{e}")
            return False, None, "position"
    
    
    def _init_casadi_problem(self):
        self.pin_model, self.pin_data = self._robot_model.get_pin_model_N_data()
        self.cmodel = cpin.Model(self.pin_model)
        self.cdata = self.cmodel.createData()
        
        # define symbolic variables
        self.variables = [casadi.SX.sym("q", self._robot_model.nq, 1)]
        for tracking_frame in self.tracking_frames:
            # parse the frame target dimension
            self.variables.append(casadi.SX.sym(tracking_frame["name"],
                                  *tracking_frame["dim"]))        
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.variables[0])
        
        # cost functions
        trans_errors = []
        trans_variables = []
        rotation_errros = []
        rotation_variables = []
        for i in range(len(self.tracking_frames)):
            if self.tracking_frames[i]["require_trans"]:
                trans_errors.append(self._trans_err(self.tracking_frames[i]["name"],
                                                self.variables[1+i], 
                                                self.tracking_frames[i]["trans_scalar"]))
                trans_variables.append(self.variables[i])
            if self.tracking_frames[i]["require_rot"]:
                rotation_errros.append(self._rot_err(self.tracking_frames[i]["name"],
                                                self.variables[1+i],
                                                self.tracking_frames[i]["rot_scalar"]))
                rotation_variables.append(self.variables[i])

        # @TODO: related variables
        self.translation_func = casadi.Function(
            "translation_error",
            [self.variables[i] for i in range(len(self.variables))],
            [casadi.vertcat(
                    *trans_errors
                )],
        )
        self.rot_func = casadi.Function(
            "rotation_error",
            [self.variables[i] for i in range(len(self.variables))],
            [casadi.vertcat(
                *rotation_errros
                )],
        )
        
        # casadi opti
        self.opti = casadi.Opti()
        # variables
        self.var_q = self.opti.variable(self._robot_model.nq)
        # first parameters is the last q
        self.parameters = [self.opti.parameter(self._robot_model.nq)]
        trans_params = []
        rot_params = []
        for i in range(len(self.tracking_frames)):
            curr_parameter = self.opti.parameter(*self.tracking_frames[i]["dim"])
            self.parameters.append(curr_parameter)
            if self.tracking_frames[i]["require_trans"]:
                trans_params.append(curr_parameter)
            if self.tracking_frames[i]["require_rot"]:
                rot_params.append(curr_parameter)

        # costs
        self.translation_cost = casadi.sumsqr(self.translation_func(self.var_q, *trans_params))
        self.rotation_cost = casadi.sumsqr(self.rot_func(self.var_q, *rot_params))
        self.smooth_cost = casadi.sumsqr(self.var_q - self.parameters[0])
        self.regularization_cost = casadi.sumsqr(self.var_q)
        
        # optimization constraints and objectives
        self.opti.subject_to(self.opti.bounded(
            self.pin_model.lowerPositionLimit,
            self.var_q,
            self.pin_model.upperPositionLimit)
        )
        # @TODO: add the constraint if there is mimic joints
        # self.opti.
        self.opti.minimize(self.translation_weight * self.translation_cost 
                           + self.rot_weight * self.rotation_cost 
                           + self.regularization_weight * self.regularization_cost 
                           + self.smooth_weight * self.smooth_cost
                          )
        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50, # 50
                'tol':1e-3
            },
            'print_time':self.show_debug,# print or not
            'calc_lam_p':False 
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)
        
    def _trans_err(self, frame_name, varaibale, scalar = 1.0):
        frame_id = self._robot_model.frames_name2id[frame_name]
        return scalar * self.cdata.oMf[frame_id].translation - varaibale[:3, 3]
    
    def _rot_err(self, frame_name, variable, scalar = 1.0):
        frame_id = self._robot_model.frames_name2id[frame_name]
        return scalar * cpin.log3(variable[:3, :3] @ self.cdata.oMf[frame_id].rotation.T)
        
   