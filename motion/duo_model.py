from __future__ import annotations

from motion.pin_model import RobotModel
from motion.model_base import ModelBase
import numpy as np
import pinocchio as pin
import copy

class DuoRobotModel(ModelBase):
    _models: dict[str, RobotModel]
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._models = {}
        left_cfg = config["left"]
        self._models["left"] = RobotModel(left_cfg)
        right_cfg = config["right"]
        self._models["right"] = RobotModel(right_cfg)
        self.ee_link = [self._models['left'].ee_link, self._models['right'].ee_link]

    def get_pin_model_N_data(self):
        pin_model = {}
        pin_data = {}
        pin_model['left'], pin_data['left'] = self._models['left'].get_pin_model_N_data()
        pin_model['left'], pin_data['left'] = self._models['left'].get_pin_model_N_data()
        return pin_model, pin_data

    def update_kinematics(self, joint_positions, joint_vel = None,
                              joint_acc = None):
        left_model = self._models['left']
        left_dof = left_model.nv
        left_joint_positions = joint_positions[0:left_dof]
        left_joint_vel = joint_vel[0:left_dof] if not joint_vel is None else None
        left_joint_acc = joint_acc[0:left_dof] if not joint_acc is None else None
        
        right_model = self._models['right']
        right_dof = right_model.nv
        right_joint_positions = joint_positions[left_dof:right_dof]
        right_joint_vel = joint_vel[left_dof:right_dof] if not joint_vel is None else None
        right_joint_acc = joint_acc[left_dof:right_dof] if not joint_acc is None else None
        
        left_model.update_kinematics(left_joint_positions, left_joint_vel, left_joint_acc)
        right_model.update_kinematics(right_joint_positions, right_joint_vel, right_joint_acc)
    
    def get_frame_pose(self, frame_name, joint_positions: np.ndarray | None = None,
                       need_update: bool = False, model_type = "left"):
        """
            @brief: return the specific frame transformation (fk),
                in the format of 4x4 homogenous matrix
            @ params:
                need_update: if set false, you need to update the pinocchio
                joint state first by calling `update_kinematics`
        """
        #@TODO: @yx
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right,"
                             f"but get: {model_type}!!!")
        
        sliced_joint_position = self.get_sliced_joint_feedback(joint_positions, model_type)
        pose_homo = self._models[model_type].get_frame_pose(frame_name, sliced_joint_position, need_update)
        return pose_homo
        
    def get_frame_twist(self, frame_name, joint_position = None, 
                        joint_velocity = None, reference_frame = pin.LOCAL_WORLD_ALIGNED,
                        need_update: bool = False, model_type = "left"):
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")

        sliced_joint_position = self.get_sliced_joint_feedback(joint_position, model_type)
        sliced_joint_velocity = self.get_sliced_joint_feedback(joint_velocity, model_type)
        twist = self._models[model_type].get_frame_twist(frame_name, sliced_joint_position, 
                                                         sliced_joint_velocity, 
                                                         reference_frame, need_update)
        return twist

    def get_frame_acc(self, frame_name, joint_position = None, joint_velocity = None, 
                        joint_acceleration = None, reference_frame = pin.LOCAL_WORLD_ALIGNED,
                        need_update: bool = False, model_type = "left"):
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")

        sliced_joint_position = self.get_sliced_joint_feedback(joint_position, model_type)
        sliced_joint_velocity = self.get_sliced_joint_feedback(joint_velocity, model_type)
        sliced_joint_acc = self.get_sliced_joint_feedback(joint_acceleration, model_type)
        acc = self._models[model_type].get_frame_acc(frame_name, sliced_joint_position, sliced_joint_velocity, 
                                                         sliced_joint_acc, reference_frame, need_update)
        return acc

    def get_jacobian(self, frame_name, joint_position, dim = None,
                     reference_frame = pin.LOCAL_WORLD_ALIGNED, model_type = "left"):
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")
        
        sliced_joint_position = self.get_sliced_joint_feedback(joint_position, model_type)
        jacobian = self._models[model_type].get_jacobian(frame_name, sliced_joint_position, dim, reference_frame)
        return jacobian
    
    def get_inertial_matrix(self, joint_positions, dims=None, model_type = "left"):
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")
        
        sliced_joint_position = self.get_sliced_joint_feedback(joint_positions, model_type)
        model = self._models[model_type]
        return model.get_inertial_matrix(sliced_joint_position, dims)
        
    def get_coriolis_matrix(self, joint_position, joint_velocity, dims=None, model_type = "left"):
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")
        
        sliced_joint_position = self.get_sliced_joint_feedback(joint_position, model_type)
        sliced_joint_velocity = self.get_sliced_joint_feedback(joint_velocity, model_type)
        model = self._models[model_type]
        return model.get_coriolis_matrix(sliced_joint_position, sliced_joint_velocity, dims)
    
    def get_gravity_vector(self, joint_positions, dims, model_type = "left"):
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")

        sliced_joint_position = self.get_sliced_joint_feedback(joint_positions, model_type)
        model = self._models[model_type]
        return model.get_gravity_vector(sliced_joint_position, dims)
    
    def get_dynamic_paras(self, posi, vel, model_type = "left") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            @brief: return all dynamic parameters
            @returns: tuples(M(q), C(q,dot(q)), G(q))
        """
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")

        sliced_joint_position = self.get_sliced_joint_feedback(posi, model_type)
        sliced_joint_velocity = self.get_sliced_joint_feedback(vel, model_type)
        model = self._models[model_type]
        return model.get_dynamic_paras(sliced_joint_position, sliced_joint_velocity)
    
    def id(self, joint_positions, joint_velocity, joint_accleration, model_type = "left"):
        if not model_type in ["left", "right"]:
            raise ValueError("This is a double urdf model, model type only contain left & right!!!")

        sliced_joint_position = self.get_sliced_joint_feedback(joint_positions, model_type)
        sliced_joint_velocity = self.get_sliced_joint_feedback(joint_velocity, model_type)
        sliced_joint_acc = self.get_sliced_joint_feedback(joint_accleration, model_type)
        model = self._models[model_type]
        return model.id(sliced_joint_position, sliced_joint_velocity, sliced_joint_acc)
    
    def get_model_dof(self):
        return [self._models["left"].nv, self._models["left"].nv]
    
    def get_sliced_joint_feedback(self, joint_feedback, model_type):
        left_dof = self._models["left"].nv
        right_dof = self._models["right"].nv
        sliced_joint = copy.deepcopy(joint_feedback)
        if model_type == 'left':
            sliced_joint = joint_feedback[0:left_dof]
        else:
            sliced_joint = joint_feedback[left_dof:left_dof+right_dof]
        return sliced_joint
    