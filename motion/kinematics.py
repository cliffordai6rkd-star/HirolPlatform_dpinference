"""
@File    : kinematics
@Author  : Haotian Liang
@Time    : 2025/4/25 14:33
@Email   :Haotianliang10@gmail.com
"""

import numpy as np
import pinocchio as pin
from typing import Optional, Tuple, Dict, List
from abc import ABC, abstractmethod
import numpy.linalg as LA

import glog as log
import os
import time
from threading import Lock

# --- URDF模型管理器 (共享加载机制) ---
class UrdfModelManager:
    """
    URDF模型管理器，用于共享加载URDF模型，避免重复加载相同的URDF文件。
    支持线程安全的单例模式。
    """
    _instance = None
    _lock = Lock()
    _loaded_models = {}  # 缓存已加载的模型 {urdf_path: (full_model, collision_model, visual_model)}
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def get_full_model(self, urdf_path: str):
        """
        获取完整的URDF模型，如果已经加载过则直接返回缓存的版本。
        
        Args:
            urdf_path: URDF文件路径
            
        Returns:
            tuple: (full_model, collision_model, visual_model)
        """
        urdf_path = os.path.abspath(urdf_path)
        
        with self._lock:
            if urdf_path not in self._loaded_models:
                log.info(f"首次加载URDF模型: {urdf_path}")
                t_start = time.time()
                try:
                    package_dir = str(os.path.dirname(urdf_path))
                    full_model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, package_dir)
                    self._loaded_models[urdf_path] = (full_model, collision_model, visual_model)
                    t_end = time.time()
                    log.info(f"URDF模型加载完成，耗时: {t_end - t_start:.3f} 秒")
                except Exception as e:
                    raise IOError(f"无法从 {urdf_path} 载入 URDF。请检查路径和档案内容。错误: {e}")
            else:
                log.info(f"使用缓存的URDF模型: {urdf_path}")
                
            return self._loaded_models[urdf_path]

# --- 輔助函式 (解決舊版 Pinocchio 的相容性問題) ---
def get_chain_joint_names_from_frame(model: pin.Model, frame_id: int) -> List[str]:
    """
    從指定的 frame 開始，向上追溯其父關節的名稱，直到模型的根部。
    """
    joint_ids = []
    current_joint_id = model.frames[frame_id].parentJoint
    while current_joint_id > 0:
        joint_ids.append(current_joint_id)
        current_joint_id = model.parents[current_joint_id]
    joint_ids.reverse()
    
    # 修正：根據 joint ID 從 model.names 獲取關節名稱
    joint_names = [model.names[jid] for jid in joint_ids if model.joints[jid].nq > 0]
    return joint_names


# Define the abstract base class
class BaseKinematicsModel(ABC):
    """
    Abstract interface for robot kinematics modeling.
    Provides methods for forward kinematics, inverse kinematics,
    Jacobian calculation, and inverse velocity kinematics.
    """

    @abstractmethod
    def fk(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Calculate the end-effector pose from given joint positions.

        Args:
            joint_positions: Joint positions array of shape (n,) where n is the number of joints

        Returns:
            pose: End-effector pose as a homogeneous transformation matrix of shape (4, 4)
                 or as a pose vector (position and orientation)
        """
        pass

    @abstractmethod
    def ik(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calculate joint positions to achieve a target end-effector pose.

        Args:
            target_pose: Target end-effector pose as a homogeneous transformation matrix
                        of shape (4, 4) or as a pose vector
            seed: Initial guess for joint positions, shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)

        Returns:
            joint_positions: Calculated joint positions of shape (n,)
        """
        pass

# Implement the Pinocchio-based model
class PinocchioKinematicsModel(BaseKinematicsModel):

    def __init__(self, urdf_path: str, base_link: str, end_effector_link: str):
        """
        從 URDF 檔案初始化 Pinocchio 運動學模型，
        並將其縮減為從 base_link 到 end_effector_link 的特定運動鏈。

        Args:
            urdf_path: URDF 檔案的路徑。
            base_link: 期望的運動鏈的基座連結名稱。
            end_effector_link: 期望的運動鏈的末端連結名稱。
        """
        t_start = time.time()
        super().__init__()

        # 1. 使用共享模型管理器載入完整的模型
        model_manager = UrdfModelManager()

        full_model, collision_model, visual_model = model_manager.get_full_model(urdf_path)

        if not full_model.existFrame(base_link):
            raise ValueError(f"Base link '{base_link}' 在模型中未找到！")
        if not full_model.existFrame(end_effector_link):
            raise ValueError(f"End-effector link '{end_effector_link}' 在模型中未找到！")

        ee_frame_id = full_model.getFrameId(end_effector_link)
        base_frame_id = full_model.getFrameId(base_link)

        # 2. 確定需要保留的關節名稱
        joints_in_ee_chain = get_chain_joint_names_from_frame(full_model, ee_frame_id)
        joints_in_base_chain = get_chain_joint_names_from_frame(full_model, base_frame_id)
        joints_to_keep_names = list(set(joints_in_ee_chain) - set(joints_in_base_chain))

        # 3. 確定需要鎖定的關節
        # 修正：正確地遍歷關節 ID 來獲取可動關節的名稱
        all_movable_joint_names = [
            full_model.names[jid] for jid in range(1, full_model.njoints) if full_model.joints[jid].nq > 0
        ]
        
        joints_to_lock_names = list(set(all_movable_joint_names) - set(joints_to_keep_names))
        
        # 獲取要鎖定的關節的 ID
        joints_to_lock_ids = [full_model.getJointId(name) for name in joints_to_lock_names]
        log.debug(f"將鎖定 {len(joints_to_lock_ids)} 個關節: {joints_to_lock_names}")

        # 4. 建立縮減模型
        reference_configuration = pin.neutral(full_model)
        
        geom_models = [visual_model, collision_model]
        self.model, geometric_models_reduced = pin.buildReducedModel(
            full_model,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=joints_to_lock_ids,
            reference_configuration=reference_configuration,
        )
        self.visual_model, self.collision_model = geometric_models_reduced

        # 5. 基於新的、縮減後的模型更新所有屬性
        self.data = self.model.createData()
        self.n_joints = self.model.nq
        self.joint_lower_limit = self.model.lowerPositionLimit
        self.joint_upper_limit = self.model.upperPositionLimit

        if not self.model.existFrame(end_effector_link):
            raise RuntimeError(f"BUG: 末端連結 '{end_effector_link}' 在縮減模型中消失了。")
        self.ee_frame_id = self.model.getFrameId(end_effector_link)
        self.ee_frame_name = end_effector_link

        log.info(f"成功建立從 '{base_link}' 到 '{end_effector_link}' 的縮減模型。")
        log.info(f"縮減後的模型關節數 (nq): {self.model.nq}")
        log.debug(f"保留的關節: {[self.model.names[i] for i in range(1, self.model.njoints)]}")
        t_end = time.time()
        log.info(f"PinocchioKinematicsModel 初始化總耗時: {t_end - t_start:.3f} 秒")
    
    def fk(self, joint_positions: np.ndarray) -> np.ndarray:
        """為縮減後的模型計算正向運動學。"""
        if joint_positions.shape[0] != self.n_joints:
            raise ValueError(f"維度不匹配！期望的關節角度維度為 {self.n_joints}，但收到 {joint_positions.shape[0]}")
        pin.forwardKinematics(self.model, self.data, joint_positions)
        pin.updateFramePlacements(self.model, self.data)
        pose_se3 = self.data.oMf[self.ee_frame_id]
        return pose_se3.homogeneous

    def ik(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 5000,
            tol: float = 1e-6,
            step_size: float = 1.0
    ) -> np.ndarray:
        """
        Solve inverse kinematics using the Gauss-Newton method.

        The Gauss-Newton algorithm is similar to Newton's method but it approximates
        the Hessian as J^T * J, ignoring second-order terms.

        Args:
            target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
            seed: Initial guess for joint angles
            joint_limits: Tuple of (lower_limits, upper_limits) arrays
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            step_size: Step size factor (1.0 = full Gauss-Newton step)

        Returns:
            joint_angles: Solved joint angles
        """
        # Use provided joint limits or default to model limits
        if joint_limits is None:
            lower_limits = self.joint_lower_limit
            upper_limits = self.joint_upper_limit
        else:
            lower_limits, upper_limits = joint_limits

        # Initialize joint angles with seed or middle of joint range
        if seed is None:
            q = 0.5 * (lower_limits + upper_limits)
        else:
            q = seed.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize variables for the iterative solver
        converged = False

        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get current end-effector placement
            current_placement = self.data.oMf[self.ee_frame_id]

            # Compute the error in SE3 (log maps the difference to a spatial velocity)
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Check for convergence
            if np.linalg.norm(err_se3) < tol:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Pseudo-inverse of the Jacobian using the normal equations approach
            # For Gauss-Newton: delta_q = (J^T * J)^(-1) * J^T * err
            JtJ = J @ J.T

            try:
                # Solve the normal equations
                v = np.linalg.solve(JtJ, err_se3)
                delta_q = J.T @ v

                # Apply step size and update joint angles
                q = q + step_size * delta_q

                # Project back to joint limits
                q = np.clip(q, lower_limits, upper_limits)
            except np.linalg.LinAlgError:
                # If the matrix is singular, use a damped approach
                reg = 1e-3 * np.eye(JtJ.shape[0])
                v = np.linalg.solve(JtJ + reg, err_se3)
                delta_q = J.T @ v

                # Apply step size and update joint angles
                q = q + step_size * delta_q

                # Project back to joint limits
                q = np.clip(q, lower_limits, upper_limits)

        if not converged:
            print(
                f"Warning: Gauss-Newton IK did not converge after {max_iter} iterations. Best error: {np.linalg.norm(err_se3)}")

        return q
    