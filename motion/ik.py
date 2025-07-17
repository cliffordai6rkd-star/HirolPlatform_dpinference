import pinocchio as pin
import numpy as np
import abc

class IkBase(abc.ABC, metaclass=abc.ABCMeta):
    def ik(self, pin_model: pin.Model, pin_data: pin.Data, target_pose_dict: dict, 
           curr_joint_positions: np.ndarray, tolerance: float, max_iter:float, 
           damping = None) -> tuple[bool, np.ndarray, str]:
        """
        Solve inverse kinematics
        Args:
            pin_model: pinocchio model
            pin_data: pinocchio data
            target_pose: Target pose dict, key: frame name, value: frame pose as a 4x4 homogeneous transformation matrix
            curr_joint_positions: current joint positions as initial guess 
            max_iter: Maximum number of iterations
            tolerance: Tolerance for convergence
            damping: damping term for singularity
        Returns:
            whether converged; joint_angles: Solved joint angles; str: position mode
        """
        raise NotImplementedError
    
class GaussianNetwon(IkBase):
    def ik(self, pin_model: pin.Model, pin_data: pin.Data, target_pose: dict, 
           curr_joint_positions: np.ndarray, tolerance: float, max_iter:float, 
           damping = 0.3) -> tuple[bool, np.ndarray, str]:
        # Use provided joint limits or default to model limits
        lower_limits = pin_model.lowerPositionLimit
        upper_limits = pin_model.upperPositionLimit

        # Initialize joint angles with seed or middle of joint range
        q = curr_joint_positions.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        frame_id = pin_model.getFrameId(next(iter(target_pose.keys())))
        target_pose = next(iter(target_pose.values()))
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize variables for the iterative solver
        converged = False
        
        step_size = 1.0
        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)

            # Get current end-effector placement
            current_placement = pin_data.oMf[frame_id]

            # Compute the error in SE3 (log maps the difference to a spatial velocity)
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Check for convergence
            if np.linalg.norm(err_se3) < tolerance:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(pin_model, pin_data, q)
            J = pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.LOCAL)

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

        return converged, q, "position"
    
class IK_DLS(IkBase):
    def ik(self, pin_model: pin.Model, pin_data: pin.Data, target_pose: dict, 
           curr_joint_positions: np.ndarray, tolerance: float, max_iter:float, 
           damping = 0.3) -> tuple[bool, np.ndarray, str]:
        # Use provided joint limits or default to model limits
        lower_limits = pin_model.lowerPositionLimit
        upper_limits = pin_model.upperPositionLimit
       
        # Initialize joint angles with seed or middle of joint range if not provided
        q = curr_joint_positions.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        frame_id = pin_model.getFrameId(next(iter(target_pose.keys())))
        target_pose = next(iter(target_pose.values()))
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize variables for the iterative solver
        converged = False

        lambda_reg = damping
        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)

            # Get current end-effector placement
            current_placement = pin_data.oMf[frame_id]

            # Compute the error in SE3 (log maps the difference to a spatial velocity)
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Check for convergence
            if np.linalg.norm(err_se3) < tolerance:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(pin_model, pin_data, q)
            J = pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.LOCAL)

            # Damped least squares method
            JJT = J @ J.T + lambda_reg * np.eye(6)
            delta_q = J.T @ np.linalg.solve(JJT, err_se3)

            # Update joint angles
            q = q + delta_q

            # Project back to joint limits
            q = np.clip(q, lower_limits, upper_limits)

        if not converged:
            # If the solver did not converge, return the best solution found
            print(f"Warning: IK did not converge after {max_iter} iterations. Best error: {np.linalg.norm(err_se3)}")

        return converged, q, "position"
    
class IK_LM(IkBase):
    def ik(self, pin_model: pin.Model, pin_data: pin.Data, target_pose: dict, 
           curr_joint_positions: np.ndarray, tolerance: float, max_iter:float, 
           damping = None) -> tuple[bool, np.ndarray, str]:
        # Use provided joint limits or default to model limits
        lower_limits = pin_model.lowerPositionLimit
        upper_limits = pin_model.upperPositionLimit
        n_joints = len(lower_limits)
       
        # Initialize joint angles with seed or middle of joint range if not provided
        q = curr_joint_positions.copy()
        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        frame_id = pin_model.getFrameId(next(iter(target_pose.keys())))
        target_pose = next(iter(target_pose.values()))
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize damping parameter
        lambda_init = 0.1
        lambda_k = lambda_init

        # Initialize variables for the iterative solver
        converged = False
        # last_error_norm = float('inf')
        
        lambda_min = 1e-6
        lambda_max = 1e-3
        lambda_factor = 10.0
        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)

            # Get current end-effector placement
            current_placement = pin_data.oMf[frame_id]

            # Compute the error in SE3
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector
            error_norm = np.linalg.norm(err_se3)

            # Check for convergence
            if error_norm < tolerance:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(pin_model, pin_data, q)
            J = pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.LOCAL)

            # Compute J^T * J and J^T * err
            JtJ = J.T @ J
            Jt_err = J.T @ err_se3

            # Add damping to the diagonal of J^T * J
            # (increase numerical stability and control step size)
            JtJ_damped = JtJ + lambda_k * np.eye(n_joints)

            # Solve the normal equations
            try:
                # v = np.linalg.solve(JtJ_damped, err_se3)
                # delta_q = J.T @ v
                delta_q = np.linalg.solve(JtJ_damped, Jt_err)

                # Try the update
                q_new = np.clip(q + delta_q, lower_limits, upper_limits)

                # Check if the new position reduces the error
                pin.forwardKinematics(pin_model, pin_data, q_new)
                pin.updateFramePlacements(pin_model, pin_data)
                new_err_se3 = pin.log(pin_data.oMf[frame_id].inverse() * target_placement).vector
                new_error_norm = np.linalg.norm(new_err_se3)

                if new_error_norm < error_norm:
                    # Accept the update and decrease lambda (more like Gauss-Newton)
                    q = q_new
                    lambda_k = max(lambda_min, lambda_k / lambda_factor)
                    # last_error_norm = new_error_norm
                else:
                    # Reject the update and increase lambda (more like gradient descent)
                    lambda_k = min(lambda_max, lambda_k * lambda_factor)
            except np.linalg.LinAlgError:
                # If matrix is singular, increase lambda and continue
                lambda_k = min(lambda_max, lambda_k * lambda_factor)
                continue

        if not converged:
            print(f"Warning: LM IK did not converge after {max_iter} iterations. Best error: {error_norm}")

        return converged, q, "position"
