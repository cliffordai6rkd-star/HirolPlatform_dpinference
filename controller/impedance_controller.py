from controller.controller_base import ControllerBase
from motion.pin_model import RobotModel
import numpy as np
from hardware.base.utils import RobotJointState, compute_pose_diff, convert_homo_2_7D_pose
import pinocchio as pin
from hardware.base.utils import matrix_sqrt

class ImpedanceController(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self._gravity_compensation = config.get("enable_gravity_compensation", True)
        self._stiffness = np.diag(config["stiffness"])
        self._damping = config.get("damping", None)
        self.arm_joint_idxes = config.get("arm_joint_idxes", None)
        self.saturation_values = config["saturation"]
        self.Kp_nullspace = np.array(config.get("kp_null", None)).astype(np.float64)
        self.q_des = config.get("q_des", None)
        self.dq_damping = config.get("dq_damping", 1.2)
    
    def compute_controller(self, target: dict, 
                           robot_state: RobotJointState | None = None):
        # print(f'target dict: {target}')
        frame_name, target = next(iter(target.items()))
        if len(target) != 7:
            raise ValueError("target is not 7D pose containing [x,y,z,qx,qy,qz,qw],"
                             "please check whether the ee_pose target is correctly set")
        
        self._robot_model.update_kinematics(robot_state._positions, robot_state._velocities, 
                                            robot_state._accelerations)
        ee_pose_homo = self._robot_model.get_frame_pose(frame_name)
        ee_pose = convert_homo_2_7D_pose(ee_pose_homo)
        pose_error = compute_pose_diff(target, ee_pose)
        # print(f'pose err: {pose_error}, norm: {np.linalg.norm(pose_error)}')
        cur_twist = self._robot_model.get_frame_twist(frame_name, 
                                                      reference_frame = pin.LOCAL_WORLD_ALIGNED)
        vel_error = np.zeros(6) - cur_twist
        # print(f'pose err: {np.linalg.norm(pose_error[3:])}, vel: {np.linalg.norm(vel_error)}')
        spatial_acc = self._robot_model.get_frame_acc(frame_name, 
                                                      reference_frame = pin.LOCAL_WORLD_ALIGNED)
        # compute kinemtics/dynamics parameters
        J = self._robot_model.get_jacobian(frame_name, robot_state._positions, 
                                           dim=self.arm_joint_idxes,
                                           reference_frame = pin.LOCAL_WORLD_ALIGNED)
        M = self._robot_model.get_inertial_matrix(robot_state._positions, 
                                                  self.arm_joint_idxes)
        M_inv = np.linalg.pinv(M)
        task_inertial_inv = J @ M_inv @ J.T  
        task_inertial = np.linalg.pinv(task_inertial_inv)

        # compute the target end-effector wrench
        if self._damping is None:
            kp_sqrt = np.sqrt(self._stiffness)
            # @TODO: check why the close loop damped system is not stable
            # task_inertial_sqrt = matrix_sqrt(task_inertial)
            # self._damping = task_inertial_sqrt @ kp_sqrt + kp_sqrt @ task_inertial_sqrt
            self._damping = 2 * kp_sqrt
        des_local_wrench = self._stiffness @ pose_error + self._damping @ vel_error
        des_local_wrench = task_inertial @ spatial_acc - des_local_wrench

        
        tau = np.zeros(self._robot_model.nv)
        if self._gravity_compensation:
            tau = self._robot_model.id(robot_state._positions, 
                            robot_state._velocities, robot_state._accelerations)
        # print(f'base tau: {tau}')
        tau_impedance = J.T @ des_local_wrench
        # print(f'impedance tau: {tau_impedance}')
        tau -= tau_impedance
                
        # nullspace 
        if self.Kp_nullspace is not None and self.q_des is not None:
            null_err = np.diag(self.Kp_nullspace) @ (self.q_des - robot_state._positions) \
                        - np.diag(self.dq_damping * np.sqrt(self.Kp_nullspace)) @ robot_state._velocities
            J_bar = M_inv @ J.T @ task_inertial
            tau_nullsapce = (np.eye(self._robot_model.nv) - J.T @ J_bar.T) @ null_err
            # print(f'nullspace tau: {tau_nullsapce}')
            tau += tau_nullsapce

        # saturation
        if not self.saturation_values is None:
            tau = np.clip(tau, self.saturation_values["min"], self.saturation_values["max"])
        return True, tau, 'torque'
    
    def set_damping(self, damping):
        self._damping = damping
        