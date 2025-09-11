import numpy as np
from typing import Union, Optional, Tuple, List
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
import yaml
import warnings
from collections import deque
import copy, time
from enum import Enum

class RobotJointState:
    _positions: np.ndarray
    _velocities: np.ndarray
    _accelerations: np.ndarray
    _torques: np.ndarray
    _time_stamp: float
    def __init__(self):
        self._positions = np.zeros_like(7)
        self._velocities = np.zeros_like(7)
        self._accelerations = np.zeros_like(7)
        self._torques = np.zeros_like(7)
        self._time_stamp = time.perf_counter()
    
def get_joint_slice_value(start, end, joint_state: RobotJointState):
    new_state = RobotJointState()
    new_state._positions = joint_state._positions[start:end]
    new_state._velocities = joint_state._velocities[start:end]
    new_state._accelerations = joint_state._accelerations[start:end]
    new_state._torques = joint_state._torques[start:end]
    new_state._time_stamp = joint_state._time_stamp
    return new_state

def combine_two_joint_states(joint_state1: RobotJointState, joint_state2: RobotJointState):
    new_state = RobotJointState()
    new_state._positions = np.hstack((joint_state1._positions, joint_state2._positions))
    new_state._velocities = np.hstack((joint_state1._velocities, joint_state2._velocities))
    new_state._accelerations = np.hstack((joint_state1._accelerations, joint_state2._accelerations))
    new_state._torques = np.hstack((joint_state1._torques, joint_state2._torques))
    new_state._time_stamp = (joint_state1._time_stamp + joint_state2._time_stamp) / 2
    return new_state

class ToolType(Enum):
    GRIPPER = 0
    SUCTION = 1
    HAND = 2

class ToolControlMode(Enum):
    BINARY = "binary"
    INCREMENTAL = "incremental"
    HAND = "hand"

class ToolState:
    _position: np.float32 = 0.0
    # contact force
    _force: Union[np.float32, np.ndarray] = 0.0
    # grasp status
    _is_grasped: bool = False
    _tool_type: ToolType = ToolType.GRIPPER
    _time_stamp: float = 0.0

class PaxiniState:
    def __init__(self):
        self._tactile_data: np.ndarray = np.zeros((0, 120, 3))  # Shape: (n_modules, n_taxels, 3)
        self._module_ids: List[int] = []  # Active sensor module connection IDs
        self._n_modules: int = 0  # Number of connected modules
        self._n_taxels: int = 120  # Taxels per module (fixed)
        self._time_stamp: float = time.perf_counter()
        self._is_connected: bool = False
    
class TrajectoryState:
    # size: [num_state, dim_traj]
    _zero_order_values: np.ndarray # position
    _first_order_values: np.ndarray # vel
    _second_order_values: np.ndarray # acc

# FIFO buffer
class Buffer:
    _size: float
    _dim: float
    _data: deque
    _time_stamp: deque
    def __init__(self, size, dim):
        self._size = size
        self._dim = dim
        self._data = deque()
        self._time_stamp = deque()
        
    def push_data(self, data, stamp):
        if len(data) != self._dim:
            warnings.warn(f"The data dim: {len(data)} is not matched with buffer data dim {self._dim}")
            return False
        
        if len(self._data) == self._size:
            self.pop_data()
        self._data.append(data)
        self._time_stamp.append(stamp)
        return True
        
    def pop_data(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
            Pop the data from the buffer
            @returns
                bool for succeessfully poped or not
                np.array with size _dim
        """
        if len(self._data) == 0:
            return False, None, None
        
        poped_data = self._data.popleft()
        poped_stamp = self._time_stamp.popleft()
        return True, poped_data, poped_stamp
    
    def size(self):
        return len(self._data)
    
    def clear(self):
        self._data.clear()
        self._time_stamp.clear()
        
    def clear_outdated_data(self, cur_stamp):
        while True:
            success, data, stamp = self.pop_data()
            if not success: return
            criterion = (cur_stamp - stamp) < 0.01
            if criterion: return
            self.pop_data()
    
def check_traj_size(traj: TrajectoryState, size: int) -> bool:
    if len(traj._zero_order_values[0]) != size:
        return False
    if len(traj._first_order_values[0]) != size:
        return False
    if len(traj._second_order_values[0]) != size:
        return False
    return True
    
def include_constructor(loader, node):
    with open(node.value) as f:
        return yaml.safe_load(f)
    
def dynamic_load_yaml(main_yaml_path):
    yaml.SafeLoader.add_constructor('!include', include_constructor)
    with open(main_yaml_path) as f:
        data = yaml.safe_load(f)
    return data
    
def object_class_check(classes, object_str):
    if not object_str in classes:
        warnings.warn(f'The class does not support the object {object_str}')
        return False
    return True
    
# math utils
def quaternion_error(q1, q2):
    """
        @brief: compute the error between two quaternions
        @params:
            q1: the first quaternion
            q2: the second quaternion
        @return: the quaternion error
    """
    pass
    quat1 = R.from_quat(q1)
    quat2 = R.from_quat(q2)
    conjugate_quat2 = quat2.inv()
    quat_error = quat1 * conjugate_quat2
    return np.array(quat_error.as_quat())

def compute_pose_diff(pose1, pose2):
    """
        @brief: compute the difference between two poses (pose1 - pose2)
        @params:
            pose1 & pose2: format is in [x,y,z,qx,qy,qz,qw]
        @return: the 6D numpy array of two pose difference
    """
    diff = np.zeros(6)
    diff[:3] = pose1[:3] - pose2[:3]
    
    quat_error = quaternion_error(pose1[3:], pose2[3:])
    ori_error = np.array([quat_error[0], quat_error[1], quat_error[2]])

    norm = np.linalg.norm(ori_error)
    if norm < 1e-15:
        ori_error = np.array([1, 0, 0])
    else:
        ori_error = (1 / norm) * ori_error
    angle = 2 * np.arctan2(norm, quat_error[3])
    # angle = 2 * np.atan2(norm, quat_error[3])
    if (angle > np.pi):
        angle -= 2 * np.pi
    ori_error = angle * ori_error

    diff[3:] = ori_error
    return diff

def convert_rot_matrix_to_quat(rot_matrix):
    """
        @brief: convert a rotation matrix to a quaternion
        @params:
            rot_matrix: the rotation matrix
        @return: the quaternion in [qx, qy, qz, qw] format
    """
    quat = R.from_matrix(rot_matrix).as_quat()
    return quat  # [qx, qy, qz, qw] 

def convert_quat_to_rot_matrix(quat):
    rot = R.from_quat(quat).as_matrix()
    return rot

def convert_se3_2_7D_pose(se3: pin.SE3):
    """
        @brief: convert a pinocchio SE3 to a 6D pose
        @params:
            se3: the pinocchio SE3 object
        @return: the 6D pose in [x, y, z, qx, qy, qz, qw] format
    """
    position = se3.translation
    rotation = convert_rot_matrix_to_quat(se3.rotation)
    return np.concatenate((position, rotation))  # [x, y, z, qx, qy, qz, qw]

def convert_homo_2_7D_pose(homo):
    pose_7d = np.zeros(7)
    pose_7d[:3] = homo[:3, 3]
    pose_7d[3:] = R.from_matrix(homo[:3, :3]).as_quat()
    return pose_7d

def convert_7D_2_homo(pose_7d):
    homo = np.eye(4)
    homo[:3, 3] = pose_7d[:3]
    homo[:3, :3] = R.from_quat(pose_7d[3:]).as_matrix()
    return homo

def matrix_sqrt(matrix: np.ndarray):
    eig_val, eig_vec = np.linalg.eig(matrix)
    sqrt_eigenvalue = np.sqrt(eig_val)
    sqrt_matrix = eig_vec @ sqrt_eigenvalue @ eig_vec.T
    return sqrt_matrix

def negate_transform(trans):
    """
        negate the transformation in homogenous format
    """
    result = copy.deepcopy(trans)
    rot = result[:3, :3]
    rot = R.from_matrix(rot)
    posi = result[:3, 3]
    
    inv_rot = rot.inv()
    inv_posi = -inv_rot.apply(posi)
    
    result[:3, :3] = inv_rot.as_matrix()
    result[:3 , 3] = inv_posi
    return np.array(result)

def negate_pose(pose):
    """
        negate the 7D pose, format: [x,y,z,qx,qy,qz,qw]
    """
    result = copy.deepcopy(pose)
    pos = result[:3]
    quat = result[3:]
    
    rot = R.from_quat(quat)
    inv_rot = rot.inv()
    inv_pos = -inv_rot.apply(pos)
    
    result[:3] = inv_pos
    result[3:] = inv_rot.as_quat()
    return result

def transform_quat(quat1, quat2):
    """
        @ brief: assuming quat1 is q_ab and quat2 is q_bc,
            this function return the quat of q_ac
        @ params: 
            quat1: q_ab
            quat2: q_bc
        @ return: the final quat q_ac
    """
    rot_ab = R.from_quat(quat1)
    rot_bc = R.from_quat(quat2)
    rot_ac = rot_ab * rot_bc  # R_ac = R_ab * R_bc
    return rot_ac.as_quat()  # [qx, qy, qz, qw]
    
def transform_pose(pose1, pose2, posi_translation=True):
    """
        @ brief: assuming pose1 is T_ab pose 2 is T_bc,
            this function return the pose of T_ac, format"[x,y,z,qx,qy,qz,qw]
        @ params: 
            pose1: T_ab
            pose2: T_bc
        @ return: the final pose T_ac
    """
    t_ab = pose1[:3]
    q_ab = pose1[3:]
    
    t_bc = pose2[:3]
    q_bc = pose2[3:]

    # rotation
    rot_ab = R.from_quat(q_ab)
    rot_bc = R.from_quat(q_bc)
    rot_ac = rot_ab * rot_bc  # R_ac = R_ab * R_bc

    # translation
    if posi_translation:
        t_ac = t_ab + rot_ab.apply(t_bc)
    else: t_ac = t_ab + t_bc
    
    # result
    T_ac = np.concatenate([t_ac, rot_ac.as_quat()])
    return T_ac

def pose_diff(pose1, pose2):
    assert pose1.shape == (7,), f"pose1 must be 7D array, got shape {pose1.shape}"
    assert pose2.shape == (7,), f"pose2 must be 7D array, got shape {pose2.shape}"
    
    res = np.zeros(7)
    res[:3] = pose1[:3] - pose2[:3]
    rot1 = R.from_quat(pose1[3:])
    rot2 = R.from_quat(pose2[3:])
    rot = rot2.inv() * rot1
    res[3:] = rot.as_quat()
    return res

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret
