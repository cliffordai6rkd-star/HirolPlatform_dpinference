import pinocchio as pin
import os
import numpy as np
import casadi
import pinocchio.casadi as cpin
from motion.model_base import ModelBase
from hardware.base.utils import RobotJointState, convert_7D_2_homo, convert_quat_to_rot_matrix

def get_joint_ids_between_frames(model: pin.Model, base_frame_id: str, 
                                 end_frame_id: str)-> list[int]:
    """
        @brief: get the joint ids from base frame to end frame
        @params: 
            model: pinocchino model
            base_frame: base frame id
            end_frame: end frame id
        @return: list of int store the joint ids
    """
    joint_ids = []
    curr_joint = model.frames[end_frame_id].parentJoint
    end_joint = model.frames[base_frame_id].parentJoint
    while True:
        joint_ids.append(curr_joint)
        print(f'joint name: {model.names[curr_joint]}')
        curr_joint = model.parents[curr_joint]
        if model.parents[end_joint] == model.parents[curr_joint] or curr_joint < 0:
            break
    
    return np.array(joint_ids).flatten()
    

# @Notice: pin pose: {x, y, z, qw, qx, qy, qz}
class RobotModel(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.urdf_path = config["urdf_path"]
        self.mesh_dir_offset = config["mesh_offset"]
        self.frame_names = config["frames"]
        self.frames_name2id = dict()
        self.base_link = config["base_link"]
        self.ee_link = config["ee_link"]
        self.lock_joint_info = config.get("lock_joints", None)
        self.fixed_base = config["fixed_base"]
        self.tracking_frames = config.get("tracking_frames", None)
        if self.tracking_frames is not None:
            self.show_debug = config.get("show_debug", False)
            self.translation_weight = config["trans_weight"]
            self.rot_weight = config["rot_weight"]
            self.smooth_weight = config["smooth_weight"]
            self.regularization_weight = config["regularization_weight"]
        
        # model building
        cur_path = cur_path = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(cur_path, "..", self.urdf_path)
        if self.fixed_base:
            self.model = pin.buildModelFromUrdf(self.urdf_path)
            mesh_dir = os.path.join(cur_path, "..", self.mesh_dir_offset)
            # self.model, collision_model, visual_model = \
            #     pin.buildModelsFromUrdf(self.urdf_path, package_dirs=mesh_dir)
        else:
            self.model = pin.buildModelFromUrdf(self.urdf_path, 
                                                root_joint = pin.JointModelFreeFlyer())
            # self.model = pin.buildModelFromUrdf(self.urdf_path, 
                                                # root_joint = pin.JointModelFreeFlyer(),
                                                # package_dirs=mesh_dir)
        print(f'full model, nq:{self.model.nq}, nv: {self.model.nv}')
        # update frames info:
        for frame_name in self.frame_names:
            if not self.model.existFrame(frame_name):
                raise ValueError(f"The pin model could not find frame {frame_name}")
            else:
                frame_id = self.model.getFrameId(frame_name)
                self.frames_name2id[frame_name] = frame_id
                if frame_name == self.base_link:
                    self.base_id = frame_id
                if frame_name == self.ee_link:
                    self.ee_id = frame_id
        neutral_q = pin.neutral(self.model)
            
        # reduced model
        jointsToLockIDs = []
        if self.lock_joint_info is not None:
            for lock_info in self.lock_joint_info:
                joint_ids = get_joint_ids_between_frames(self.model, 
                                                        self.frames_name2id[lock_info["base"]],
                                                        self.frames_name2id[lock_info["end"]])
                jointsToLockIDs = np.hstack((jointsToLockIDs, joint_ids)).astype(np.int32)
            jointsToLockIDs = jointsToLockIDs.tolist()
        # geom_models = [visual_model, collision_model]
        self.model = pin.buildReducedModel(self.model, jointsToLockIDs, neutral_q)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        print(f'nq:{self.nq}, nv: {self.nv}')
                    
        # init ik casadi optimization
        if self.tracking_frames is not None:
            self._init_casadi_problem()
            
    def get_pin_model_N_data(self):
        return self.model, self.data

    def get_model_dof(self):
        return self.nv

    def update_kinematics(self, joint_positions, joint_vel = None,
                              joint_acc = None):
        if joint_vel is not None and joint_acc is not None:
            pin.forwardKinematics(self.model, self.data, joint_positions,
                                  joint_vel, joint_acc)
        elif joint_vel is not None:
            pin.forwardKinematics(self.model, self.data, joint_positions, joint_vel)
        else:
            pin.forwardKinematics(self.model, self.data, joint_positions)
        pin.updateFramePlacements(self.model, self.data)
    
    def get_frame_pose(self, frame_name, joint_positions: np.ndarray | None = None,
                       need_update: bool = False, model_type = "single"):
        """
            @brief: return the specific frame transformation (fk),
                in the format of 4x4 homogenous matrix
            @ params:
                need_update: if set false, you need to update the pinocchio
                joint state first by calling `update_kinematics`
        """
        # @TODO @ yx
        # if model_type != "single":
        #     raise ValueError("This is a single urdf model, please check your model type!!!")
        
        if need_update:
            if joint_positions is None:
                raise ValueError("The joint position should not be None to update"
                                 "pin model state")
            else:
                self.update_kinematics(joint_positions)
                
        frame_id = self.frames_name2id[frame_name]
        return self.data.oMf[frame_id].homogeneous
        
    def get_frame_twist(self, frame_name, joint_position = None, 
                        joint_velocity = None, reference_frame = pin.LOCAL_WORLD_ALIGNED,
                        need_update: bool = False, model_type = "single"):
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        if need_update:
            if joint_position is None or joint_velocity is None:
                raise ValueError("THe joint posi, vel is needed to update!")
            self.update_kinematics(joint_position, joint_velocity)
        twist = pin.getFrameVelocity(self.model, self.data, self.frames_name2id[frame_name],
                                     reference_frame=reference_frame)
        return twist
    
    def get_frame_acc(self, frame_name, joint_position = None, joint_velocity = None, 
                        joint_acceleration = None, reference_frame = pin.LOCAL_WORLD_ALIGNED,
                        need_update: bool = False, model_type = "single"):
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        if need_update:
            if joint_position is None or joint_velocity is None \
                or joint_acceleration is None:
                raise ValueError("THe joint posi, vel is needed to update!")
            self.update_kinematics(joint_position, joint_velocity, joint_acceleration)
        acc = pin.getFrameAcceleration(self.model, self.data, self.frames_name2id[frame_name],
                                    reference_frame=reference_frame)
        return acc
    
    def get_jacobian(self, frame_name, joint_position, dim = None,
                     reference_frame = pin.LOCAL_WORLD_ALIGNED, model_type = "single"):
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        J = pin.computeFrameJacobian(self.model, self.data, joint_position,
                                     self.frames_name2id[frame_name], reference_frame)
        if dim is not None:
            J = J[:, dim]
        return J
        
    def get_inertial_matrix(self, joint_positions, dims=None, model_type = "single"):
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        M = pin.crba(self.model, self.data, 
                    joint_positions)
        if dims is not None:
            M = M[np.ix_(dims, dims)]
        return M
    
    def get_coriolis_matrix(self, joint_position, joint_velocity, dims=None, model_type = "single"):
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        C = pin.computeCoriolisMatrix(self.model, self.data, 
                                      joint_position, joint_velocity)
        if dims is not None:
            C = C[np.ix_(dims, dims)]
        return C
    
    def get_gravity_vector(self, joint_positions, dims, model_type = "single"):
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        g = pin.computeGeneralizedGravity(self.model, self.data,joint_positions)
        if dims is not None:
            g = g[dims]
        return g
    
    def get_dynamic_paras(self, posi, vel, model_type = "single") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            @brief: return all dynamic parameters
            @returns: tuples(M(q), C(q,dot(q)), G(q))
        """
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        self.update_kinematics(posi, vel)
        M = pin.crba(self.model, self.data, posi)
        C = pin.computeCoriolisMatrix(self.model, self.data, posi, vel)
        g = pin.computeGeneralizedGravity(self.model, self.data, posi)
        return M ,C, g
        
    def _init_casadi_problem(self):
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        
        # define symbolic variables
        self.variables = [casadi.SX.sym("q", self.nq, 1)]
        for tracking_frame in self.tracking_frames:
            # parse the frame target dimension
            self.variables.append(casadi.SX.sym(tracking_frame["name"],
                                  *tracking_frame["dim"]))        
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.variables[0])
        
        # cost functions
        trans_errors = []
        rotation_errros = []
        for i in range(len(self.tracking_frames)):
            if self.tracking_frames[i]["require_trans"]:
                trans_errors.append(self._trans_err(self.tracking_frames[i]["name"],
                                                self.variables[1+i], 
                                                self.tracking_frames[i]["trans_scalar"]))
            if self.tracking_frames[i]["require_rot"]:
                rotation_errros.append(self._rot_err(self.tracking_frames[i]["name"],
                                                self.variables[1+i],
                                                self.tracking_frames[i]["rot_scalar"]))

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
        self.var_q = self.opti.variable(self.nq)
        # first parameters is the last q
        self.parameters = [self.opti.parameter(self.nq)]
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
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
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
        return scalar * self.cdata.oMf[self.frames_name2id[frame_name]].translation - varaibale[:3, 3]
    
    def _rot_err(self, frame_name, variable, scalar = 1.0):
        return scalar * cpin.log3(variable[:3, :3] @ self.cdata.oMf[self.frames_name2id[frame_name]].rotation.T)
        
    def ik(self, targets: dict, robot_joint_states: RobotJointState):
        """
            Get the inverse kinematics solution for multiple targets
            @params:
                targets: Dict[np.ndarray], multiple targets, key: frame name, 
                    value: 7D pose, [x,y,z,qx,qy,qz,qw] 
            @return: return the actuated joint positions (dim: nv)
        """
        # init of the problem
        init_q = robot_joint_states._positions
        self.opti.set_initial(self.var_q, init_q)
        
        pin.framesForwardKinematics(self.model, self.data, init_q)
        pin.computeForwardKinematicsDerivatives(self.model, self.data, init_q, 
                                                robot_joint_states._velocities, 
                                                robot_joint_states._accelerations)
        # set parameter values
        self.opti.set_value(self.parameters[0], init_q)
        for i, (frame_name, value) in enumerate(targets.items()):
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
            
    def id(self, joint_positions, joint_velocity, joint_accleration, model_type = "single"):
        if model_type != "single":
            raise ValueError("This is a single urdf model, please check your model type!!!")
        
        tau = pin.rnea(self.model, self.data, joint_positions, 
                       joint_velocity, joint_accleration)
        return tau
    
    
if __name__ == '__main__':
    import yaml, time
    from simulation.mujoco.mujoco_sim import MujocoSim
    config = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, 'config/robot_model_cfg_template.yaml')
    print(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    robot_model = RobotModel(config)
    
    mujoco_config = "simulation/config/mujoco_g1_29dof_with_hand.yaml"
    mujoco_config = os.path.join(cur_path, "..", mujoco_config)
    with open(mujoco_config, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'mujoco: {config}')
    mujoco = MujocoSim(config["mujoco"])
    
    mocap_names = ["lh", "rh", "lf", "rf"]
    while True:
        target = {}
        pevis_pose = mujoco.get_site_pose("pelvis_site", quat_seq="xyzw")
        for i, mocap in enumerate(mocap_names):
            name = robot_model.tracking_frames[i]["name"]
            pose_7d = mujoco.get_site_pose(mocap + "_site", quat_seq="xyzw")
            pose_7d[:3] -= pevis_pose[:3]
            target[name] = pose_7d 
        print(f'target: {target}')
        joint_states = mujoco.get_joint_states()
        start = time.time()
        pose_cur = robot_model.get_frame_pose("left_hand_palm_link", joint_states._positions,
                                                 need_update=True)
        print(f'pose_cur: {pose_cur}')
        success, command, mode = robot_model.ik(target, joint_states)
        print(f'used time for ik: {time.time() - start}')
        if success:
            mujoco.set_joint_command([mode]*robot_model.nv, command)
        time.sleep(0.006)
        