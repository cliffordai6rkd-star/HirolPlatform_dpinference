from __future__ import annotations

import abc
import pinocchio as pin
import numpy as np

class ModelBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self.ee_link = None
        pass
    
    @abc.abstractmethod    
    def get_pin_model_N_data(self):
        raise NotImplementedError

    @abc.abstractmethod    
    def update_kinematics(self, joint_positions, joint_vel = None,
                              joint_acc = None):
        raise NotImplementedError
    
    @abc.abstractmethod    
    def get_frame_pose(self, frame_name, joint_positions: np.ndarray | None = None,
                       need_update: bool = False, model_type = "single"):
        """
            @brief: return the specific frame transformation (fk),
                in the format of 4x4 homogenous matrix
            @ params:
                need_update: if set false, you need to update the pinocchio
                joint state first by calling `update_kinematics`
        """
        raise NotImplementedError
    
    @abc.abstractmethod        
    def get_frame_twist(self, frame_name, joint_position = None, 
                        joint_velocity = None, reference_frame = pin.LOCAL_WORLD_ALIGNED,
                        need_update: bool = False, model_type = "single"):
        raise NotImplementedError
    
    @abc.abstractmethod    
    def get_frame_acc(self, frame_name, joint_position = None, joint_velocity = None, 
                        joint_acceleration = None, reference_frame = pin.LOCAL_WORLD_ALIGNED,
                        need_update: bool = False, model_type = "single"):
       raise NotImplementedError
    
    @abc.abstractmethod    
    def get_jacobian(self, frame_name, joint_position, dim = None,
                     reference_frame = pin.LOCAL_WORLD_ALIGNED, model_type = "single"):
        raise NotImplementedError
    
    @abc.abstractmethod    
    def get_inertial_matrix(self, joint_positions, dims=None, model_type = "single"):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_coriolis_matrix(self, joint_position, joint_velocity, dims=None, model_type = "single"):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_gravity_vector(self, joint_positions, dims, model_type = "single"):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_dynamic_paras(self, posi, vel, model_type = "single") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            @brief: return all dynamic parameters
            @returns: tuples(M(q), C(q,dot(q)), G(q))
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def id(self, joint_positions, joint_velocity, joint_accleration, model_type = "single"):
        raise NotImplementedError
   
    @abc.abstractmethod
    def get_model_dof(self):
        raise NotImplementedError
     
    def get_model_end_links(self):
        return self.ee_link
    
    