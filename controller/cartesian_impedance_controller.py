from __future__ import annotations

from controller.controller_base import ControllerBase
from motion.pin_model import RobotModel
import numpy as np
from hardware.base.utils import RobotJointState, convert_homo_2_7D_pose
import pinocchio as pin
from scipy.spatial.transform import Rotation as R, Slerp
from utils.array_utils import ensure_column_vector, ensure_flat_array

class CartesianImpedanceController(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        
        self._gravity_compensation = config.get("enable_gravity_compensation", True)
        
        self.translational_stiffness = config.get("translational_stiffness", 200.0)
        self.rotational_stiffness = config.get("rotational_stiffness", 20.0)
        self.translational_damping = config.get("translational_damping", 20.0)
        self.rotational_damping = config.get("rotational_damping", 5.0)
        
        self.translational_ki = config.get("translational_ki", 0.0)
        self.rotational_ki = config.get("rotational_ki", 0.0)
        
        self.nullspace_stiffness = config.get("nullspace_stiffness", 20.0)
        self.joint1_nullspace_stiffness = config.get("joint1_nullspace_stiffness", 20.0)
        
        self.arm_joint_idxes = config.get("arm_joint_idxes", None)
        
        # Match C++ default values for better recovery capability
        self.translational_clip_min = np.array(config.get("translational_clip_min", [-0.1, -0.1, -0.1]))
        self.translational_clip_max = np.array(config.get("translational_clip_max", [0.1, 0.1, 0.1]))
        self.rotational_clip_min = np.array(config.get("rotational_clip_min", [-0.3, -0.3, -0.3]))
        self.rotational_clip_max = np.array(config.get("rotational_clip_max", [0.3, 0.3, 0.3]))
        
        self.delta_tau_max = config.get("delta_tau_max", 1.0)
        
        self.filter_params = config.get("filter_params", 0.005)  # Increased default for better responsiveness
        
        self.saturation_values = config.get("saturation", None)
        
        self._setup_matrices()
        
        self.position_d = None
        self.orientation_d = None
        self.q_d_nullspace = None
        self.last_tau_d = None
        
        self.error_integral = np.zeros(6)
        
        self.initialized = False
    
    def _setup_matrices(self):
        self.cartesian_stiffness = np.zeros((6, 6))
        self.cartesian_stiffness[:3, :3] = self.translational_stiffness * np.eye(3)
        self.cartesian_stiffness[3:, 3:] = self.rotational_stiffness * np.eye(3)
        
        self.cartesian_damping = np.zeros((6, 6))
        self.cartesian_damping[:3, :3] = self.translational_damping * np.eye(3)
        self.cartesian_damping[3:, 3:] = self.rotational_damping * np.eye(3)
        
        self.Ki = np.zeros((6, 6))
        self.Ki[:3, :3] = self.translational_ki * np.eye(3)
        self.Ki[3:, 3:] = self.rotational_ki * np.eye(3)
    
    def _quaternion_error(self, q_current, q_desired):
        if q_current.dot(q_desired) < 0:
            q_current = -q_current
        
        q_error = q_desired * q_current.conjugate()
        
        return np.array([q_error.x, q_error.y, q_error.z])
    
    def _saturate_torque_rate(self, tau_d_calculated, tau_d_last):
        if tau_d_last is None:
            return tau_d_calculated
        
        tau_d_saturated = np.zeros_like(tau_d_calculated)
        for i in range(len(tau_d_calculated)):
            difference = tau_d_calculated[i] - tau_d_last[i]
            tau_d_saturated[i] = tau_d_last[i] + np.clip(difference, -self.delta_tau_max, self.delta_tau_max)
        
        return tau_d_saturated
    
    def compute_controller(self, target: list[dict[str, np.ndarray]], 
                           robot_state: RobotJointState | None = None):
        target = target[0]
        frame_name, target_pose = next(iter(target.items()))
        if len(target_pose) != 7:
            raise ValueError("target is not 7D pose containing [x,y,z,qx,qy,qz,qw]")
        
        # Ensure arrays are in correct format for pinocchio
        self._robot_model.update_kinematics(ensure_flat_array(robot_state._positions), 
                                           ensure_flat_array(robot_state._velocities), 
                                           ensure_flat_array(robot_state._accelerations))
        
        if not self.initialized:
            self._initialize_targets(frame_name, robot_state)
            self.initialized = True
        
        ee_pose_homo = self._robot_model.get_frame_pose(frame_name)
        ee_pose = convert_homo_2_7D_pose(ee_pose_homo)
        
        position_current = np.array(ee_pose[:3])
        
        position_target = np.array(target_pose[:3])
        orientation_target = R.from_quat(target_pose[3:])
        
        
        # Reset integral error unconditionally when new target is set (matching C++ equilibriumPoseCallback)
                # Reset integral error when target changes significantly (like C++ equilibriumPoseCallback)
        # if hasattr(self, '_last_target_pose'):
        #     position_change = np.linalg.norm(position_target - self._last_target_pose[:3])
        #     orientation_change = np.arccos(np.clip(
        #         np.abs(np.dot(orientation_target.as_quat(), R.from_quat(self._last_target_pose[3:]).as_quat())), 
        #         -1, 1))
        #     # Reset if position changes > 1cm or orientation changes > 5 degrees
        #     if position_change > 0.01 or orientation_change > np.deg2rad(5):
        #         self.error_integral = np.zeros(6)

        # C++ version: error_i.setZero() is called every time equilibriumPoseCallback is invoked
        self.error_integral = np.zeros(6)
        self._last_target_pose = target_pose.copy()
        
        self.position_d = self.filter_params * position_target + (1.0 - self.filter_params) * self.position_d
        
        # Use Slerp for quaternion interpolation with shortest path
        q_d_current = R.from_quat(self.orientation_d)
        q_d_target = orientation_target
        
        # Ensure shortest path rotation
        if np.dot(q_d_current.as_quat(), q_d_target.as_quat()) < 0:
            q_d_target = R.from_quat(-q_d_target.as_quat())
        
        # Create Slerp interpolator with correct key frames
        key_times = [0.0, 1.0]
        key_rots = R.from_quat(np.vstack([q_d_current.as_quat(), q_d_target.as_quat()]))
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate
        self.orientation_d = slerp(self.filter_params).as_quat()
        
        error = np.zeros(6)
        error[:3] = position_current - self.position_d
        
        for i in range(3):
            error[i] = np.clip(error[i], self.translational_clip_min[i], self.translational_clip_max[i])
        
        # Use log mapping for orientation error (consistent with LOCAL_WORLD_ALIGNED Jacobian)
        R_current = ee_pose_homo[:3, :3]
        R_desired = R.from_quat(self.orientation_d).as_matrix()
        
        # World-aligned orientation error using log map
        # error = log(R_des * R_cur^T) expressed in world frame
        error_rotation_matrix = R_desired @ R_current.T
        error[3:] = -pin.log3(error_rotation_matrix)
        
        for i in range(3):
            error[i+3] = np.clip(error[i+3], self.rotational_clip_min[i], self.rotational_clip_max[i])
        
        self.error_integral[:3] = np.clip(self.error_integral[:3] + error[:3], -0.1, 0.1)
        self.error_integral[3:] = np.clip(self.error_integral[3:] + error[3:], -0.3, 0.3)
        
        J = self._robot_model.get_jacobian(frame_name, robot_state._positions, 
                                           dim=self.arm_joint_idxes,
                                           reference_frame=pin.LOCAL_WORLD_ALIGNED)
        
        dq_full = robot_state._velocities
        if self.arm_joint_idxes is not None:
            dq_full = dq_full[self.arm_joint_idxes]
        dq_full = ensure_column_vector(dq_full)  # n_joints x 1 for matrix operations
        
        tau_task = J.T @ (-self.cartesian_stiffness @ error - 
                          self.cartesian_damping @ (J @ dq_full).flatten() - 
                          self.Ki @ self.error_integral)
        
        J_pinv = np.linalg.pinv(J)
        n_joints = len(robot_state._positions) if self.arm_joint_idxes is None else len(self.arm_joint_idxes)
        
        qe = self.q_d_nullspace - robot_state._positions
        if self.arm_joint_idxes is not None:
            qe = self.q_d_nullspace[self.arm_joint_idxes] - robot_state._positions[self.arm_joint_idxes]
        qe = ensure_flat_array(qe)
        
        qe[0] = qe[0] * self.joint1_nullspace_stiffness
        
        dqe = ensure_flat_array(dq_full)
        dqe[0] = dqe[0] * 2.0 * np.sqrt(self.joint1_nullspace_stiffness)
        
        tau_nullspace = (np.eye(n_joints) - J.T @ J_pinv.T) @ (
            self.nullspace_stiffness * qe - 
            2.0 * np.sqrt(self.nullspace_stiffness) * dqe
        )
        
        # Always compute Coriolis forces (similar to C++ version)
        C = self._robot_model.get_coriolis_matrix(robot_state._positions, robot_state._velocities, self.arm_joint_idxes)
        dq_for_coriolis = robot_state._velocities
        if self.arm_joint_idxes is not None:
            dq_for_coriolis = robot_state._velocities[self.arm_joint_idxes]
        dq_for_coriolis = ensure_column_vector(dq_for_coriolis)
        tau_coriolis = (C @ dq_for_coriolis).flatten()
        
        tau = np.zeros(self._robot_model.nv)
        if self._gravity_compensation:
            # Add gravity compensation
            tau_gravity = self._robot_model.get_gravity_vector(robot_state._positions, self.arm_joint_idxes)
            tau_compensation = tau_gravity + tau_coriolis
        else:
            # Only Coriolis compensation (like C++ version)
            tau_compensation = tau_coriolis
        
        if self.arm_joint_idxes is not None:
            tau[self.arm_joint_idxes] = tau_compensation + tau_task + tau_nullspace
        else:
            tau = tau_compensation + tau_task + tau_nullspace
        
        tau = self._saturate_torque_rate(tau, self.last_tau_d)
        self.last_tau_d = tau.copy()
        
        if self.saturation_values is not None:
            tau = np.clip(tau, self.saturation_values["min"], self.saturation_values["max"])
        
        return True, tau, 'torque'
    
    def _initialize_targets(self, frame_name, robot_state):
        ee_pose_homo = self._robot_model.get_frame_pose(frame_name)
        ee_pose = convert_homo_2_7D_pose(ee_pose_homo)
        
        self.position_d = np.array(ee_pose[:3])
        self.orientation_d = ee_pose[3:]
        self.q_d_nullspace = np.asarray(robot_state._positions).flatten().copy()
        
        # Initialize last_tau_d to compensation torques instead of zero
        C = self._robot_model.get_coriolis_matrix(robot_state._positions, robot_state._velocities, self.arm_joint_idxes)
        dq = robot_state._velocities
        if self.arm_joint_idxes is not None:
            dq = robot_state._velocities[self.arm_joint_idxes]
        dq = ensure_column_vector(dq)
        tau_coriolis = (C @ dq).flatten()
        
        self.last_tau_d = np.zeros(self._robot_model.nv)
        if self._gravity_compensation:
            tau_gravity = self._robot_model.get_gravity_vector(robot_state._positions, self.arm_joint_idxes)
            tau_compensation = tau_gravity + tau_coriolis
        else:
            tau_compensation = tau_coriolis
        
        if self.arm_joint_idxes is not None:
            self.last_tau_d[self.arm_joint_idxes] = tau_compensation
        else:
            self.last_tau_d = tau_compensation
    
    def set_stiffness(self, translational_stiffness=None, rotational_stiffness=None):
        if translational_stiffness is not None:
            self.translational_stiffness = translational_stiffness
        if rotational_stiffness is not None:
            self.rotational_stiffness = rotational_stiffness
        self._setup_matrices()
    
    def set_damping(self, translational_damping=None, rotational_damping=None):
        if translational_damping is not None:
            self.translational_damping = translational_damping
        if rotational_damping is not None:
            self.rotational_damping = rotational_damping
        self._setup_matrices()
    
    def reset_integral_error(self):
        self.error_integral = np.zeros(6)
    
    def reset(self, frame_name: str, robot_state: RobotJointState) -> None:
        """
        Reset controller internal state to current robot state
        
        Args:
            frame_name: End-effector frame name
            robot_state: Current robot joint state
        """
        # Update kinematics for current state
        self._robot_model.update_kinematics(
            ensure_flat_array(robot_state._positions),
            ensure_flat_array(robot_state._velocities),
            ensure_flat_array(robot_state._accelerations)
        )
        
        # Re-align desired pose to current measured pose
        ee_pose_homo = self._robot_model.get_frame_pose(frame_name)
        ee_pose = convert_homo_2_7D_pose(ee_pose_homo)
        self.position_d = np.array(ee_pose[:3])
        self.orientation_d = np.array(ee_pose[3:])
        
        # Reset nullspace target to current joint positions
        self.q_d_nullspace = np.asarray(robot_state._positions).flatten().copy()
        
        # Clear integral error
        self.reset_integral_error()
        
        # Initialize last_tau_d to compensation torques (not zero)
        C = self._robot_model.get_coriolis_matrix(robot_state._positions, robot_state._velocities, self.arm_joint_idxes)
        dq = robot_state._velocities
        if self.arm_joint_idxes is not None:
            dq = robot_state._velocities[self.arm_joint_idxes]
        dq = ensure_column_vector(dq)
        tau_coriolis = (C @ dq).flatten()
        
        self.last_tau_d = np.zeros(self._robot_model.nv)
        if self._gravity_compensation:
            tau_gravity = self._robot_model.get_gravity_vector(robot_state._positions, self.arm_joint_idxes)
            tau_compensation = tau_gravity + tau_coriolis
        else:
            tau_compensation = tau_coriolis
        
        if self.arm_joint_idxes is not None:
            self.last_tau_d[self.arm_joint_idxes] = tau_compensation
        else:
            self.last_tau_d = tau_compensation
        
        # Force re-initialization on next compute
        self.initialized = False
