import pinocchio as pin
import numpy as np
import abc
from typing import Tuple, Dict, Optional

try:
    import pink
    from pink.tasks import FrameTask
    import qpsolvers
    PINK_AVAILABLE = True
except ImportError:
    PINK_AVAILABLE = False

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
    
class GaussianNewton(IkBase):
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
    
class IK_PPINK(IkBase):
    def __init__(self):
        """Initialize PINK IK solver with cached components for performance."""
        if not PINK_AVAILABLE:
            raise ImportError(
                "PINK library not available. Please install 'pin-pink' and 'qpsolvers' packages."
            )
        
        # Cache QP solver selection (expensive operation)
        available_solvers = qpsolvers.available_solvers
        if "proxqp" in available_solvers:
            self._solver = "proxqp"
        elif "quadprog" in available_solvers:
            self._solver = "quadprog" 
        elif "osqp" in available_solvers:
            self._solver = "osqp"
        else:
            self._solver = available_solvers[0]
        
        # Cache constant parameters
        self._dt = 1e-2  # Integration timestep
        self._position_cost = 1.0  # [cost] / [m]
        self._orientation_cost = 1.0  # [cost] / [rad]
        self._lm_damping = 0.0  # Use QP solver damping instead
        self._gain = 1.0
        
        # Frame task cache - will be created/reused based on frame_name
        self._frame_task = None
        self._current_frame_name = None
    
    def ik(self, pin_model: pin.Model, pin_data: pin.Data, target_pose_dict: Dict[str, np.ndarray], 
           curr_joint_positions: np.ndarray, tolerance: float, max_iter: int, 
           damping: float = 1e-8) -> Tuple[bool, np.ndarray, str]:
        """
        Solve inverse kinematics using PINK (QP-based) solver.
        
        Args:
            pin_model: Pinocchio robot model
            pin_data: Pinocchio model data  
            target_pose_dict: Target poses {frame_name: 4x4_homogeneous_matrix}
            curr_joint_positions: Current joint positions as initial guess
            tolerance: Convergence tolerance for position and orientation errors
            max_iter: Maximum number of iterations
            damping: Tikhonov regularization damping parameter
            
        Returns:
            Tuple of (converged, joint_solution, "position")
            
        Raises:
            ValueError: If frame name not found or invalid target pose
        """
        # Use provided joint limits
        lower_limits = pin_model.lowerPositionLimit
        upper_limits = pin_model.upperPositionLimit
        
        # Initialize joint configuration 
        q = curr_joint_positions.copy()
        q = np.clip(q, lower_limits, upper_limits)
        
        # Extract frame name and target pose
        if not target_pose_dict:
            raise ValueError("target_pose_dict cannot be empty")
        
        frame_name = next(iter(target_pose_dict.keys()))
        target_pose = next(iter(target_pose_dict.values()))
        
        # Validate frame exists
        if not pin_model.existFrame(frame_name):
            raise ValueError(f"Frame '{frame_name}' not found in model")
        
        # Convert target pose to SE3
        if target_pose.shape != (4, 4):
            raise ValueError("Target pose must be a 4x4 homogeneous transformation matrix")
        
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        
        # Create PINK configuration
        configuration = pink.Configuration(pin_model, pin_data, q)
        
        # Create or reuse frame task (performance optimization)
        if self._frame_task is None or self._current_frame_name != frame_name:
            self._frame_task = FrameTask(
                frame_name,
                position_cost=self._position_cost,
                orientation_cost=self._orientation_cost,
                lm_damping=self._lm_damping,
                gain=self._gain
            )
            self._current_frame_name = frame_name
        
        # Update target (this is cheap operation)
        self._frame_task.set_target(target_placement)
        tasks = [self._frame_task]
        
        # Iterative solving
        converged = False
        for i in range(max_iter):
            # Compute current error
            error = self._frame_task.compute_error(configuration)
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < tolerance:
                converged = True
                break
                
            try:
                # Solve IK step using cached solver
                velocity = pink.solve_ik(
                    configuration,
                    tasks=tasks,
                    dt=self._dt,
                    damping=damping,
                    solver=self._solver
                )
                
                # Integrate velocity to get joint displacement
                q_new = pin.integrate(pin_model, configuration.q, velocity * self._dt)
                
                # Apply joint limits
                q_new = np.clip(q_new, lower_limits, upper_limits)
                
                # Update configuration
                configuration = pink.Configuration(pin_model, pin_data, q_new)
                
            except Exception as e:
                # If PINK solver fails, return current best solution
                print(f"Warning: PINK IK solver failed at iteration {i}: {str(e)}")
                break
        
        if not converged:
            print(f"Warning: PINK IK did not converge after {max_iter} iterations. Best error: {error_norm:.6f}")
        
        return converged, configuration.q, "position" 


class IK_PYROKI(IkBase):
    """Pyroki IK solver integration with HIROL platform."""
    
    def __init__(self, urdf_path: str = None, end_effector_link: str = None, **kwargs):
        """
        Initialize Pyroki IK solver with optional parameters for factory pattern support.
        
        Args:
            urdf_path: Path to robot URDF file (optional for factory pattern)
            end_effector_link: Name of end effector link (optional for factory pattern)
            **kwargs: Additional Pyroki configuration parameters
        """
        self._urdf_path = urdf_path
        self._ee_link = end_effector_link
        self._kwargs = kwargs
        self._adapter = None
        self._initialized = False
        
        # If parameters provided, initialize immediately (direct instantiation)
        if urdf_path is not None and end_effector_link is not None:
            self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize the Pyroki adapter with current parameters."""
        if self._initialized:
            return
            
        try:
            from .adapters import PyrokiAdapter, PYROKI_AVAILABLE
            if not PYROKI_AVAILABLE:
                raise ImportError("Pyroki dependencies not available")
            
            if self._urdf_path is None or self._ee_link is None:
                raise ValueError("URDF path and end effector link must be provided")
                
            self._adapter = PyrokiAdapter(self._urdf_path, self._ee_link, **self._kwargs)
            self._initialized = True
            
        except ImportError as e:
            raise ImportError(f"Failed to initialize Pyroki IK solver: {e}")
    
    def _configure_from_pinocchio(self, pin_model: pin.Model):
        """Extract configuration from Pinocchio model as fallback."""
        if self._urdf_path is None:
            # Try to get URDF path from pinocchio model if available
            # This is a fallback - ideally parameters should be provided
            import inspect
            frame = inspect.currentframe()
            try:
                # Walk up the call stack to find robot_model reference
                caller_locals = frame.f_back.f_back.f_locals if frame.f_back and frame.f_back.f_back else {}
                self_obj = caller_locals.get('self')
                if self_obj and hasattr(self_obj, '_robot_model'):
                    robot_model = self_obj._robot_model
                    if robot_model and hasattr(robot_model, 'urdf_path'):
                        self._urdf_path = robot_model.urdf_path
                    if robot_model and hasattr(robot_model, 'ee_link'):
                        self._ee_link = robot_model.ee_link
            finally:
                del frame
        
        if self._ee_link is None:
            # Use the first frame name from target as fallback
            self._ee_link = "gripper_link"  # Common default
    
    def ik(self, pin_model: pin.Model, pin_data: pin.Data, target_pose: dict, 
           curr_joint_positions: np.ndarray, tolerance: float, max_iter: float, 
           damping = None) -> tuple[bool, np.ndarray, str]:
        """
        Solve inverse kinematics using Pyroki.
        
        Args:
            pin_model: Pinocchio model (used for joint limits fallback)
            pin_data: Pinocchio data (not used by Pyroki)
            target_pose: Target pose dict, key: frame name, value: 4x4 homogeneous matrix
            curr_joint_positions: Initial joint positions (used as fallback seed)
            tolerance: Convergence tolerance
            max_iter: Maximum iterations (passed to adapter)
            damping: Damping parameter (not used by Pyroki)
            
        Returns:
            Tuple of (converged, joint_angles, "position")
        """
        # Lazy initialization: configure from context if not already done
        if not self._initialized:
            self._configure_from_pinocchio(pin_model)
            
            # Use target frame name if ee_link still not set
            if self._ee_link is None and target_pose:
                self._ee_link = next(iter(target_pose.keys()))
            
            self._initialize_adapter()
        
        if not target_pose:
            raise ValueError("Target pose dictionary cannot be empty")
        
        # Extract target pose (Pyroki handles single target)
        target_pose_matrix = next(iter(target_pose.values()))
        
        try:
            # Solve using Pyroki adapter
            converged, solution, solve_time = self._adapter.solve_single(
                target_pose_matrix, 
                curr_joint_positions,
                tolerance=tolerance,
                max_iterations=int(max_iter)
            )
            
            if converged and solution is not None:
                return True, solution, "position"
            else:
                # Fallback to middle of joint range if failed
                if hasattr(pin_model, 'lowerPositionLimit'):
                    fallback_solution = (pin_model.lowerPositionLimit + pin_model.upperPositionLimit) / 2
                else:
                    fallback_solution = curr_joint_positions
                return False, fallback_solution, "position"
                
        except Exception as e:
            print(f"Warning: Pyroki IK failed: {e}")
            # Return fallback solution
            if hasattr(pin_model, 'lowerPositionLimit'):
                fallback_solution = (pin_model.lowerPositionLimit + pin_model.upperPositionLimit) / 2
            else:
                fallback_solution = curr_joint_positions
            return False, fallback_solution, "position"


class IK_CUROBO(IkBase):
    """CuRobo IK solver integration with HIROL platform."""
    
    def __init__(self, urdf_path: str = None, end_effector_link: str = None, **kwargs):
        """
        Initialize CuRobo IK solver with optional parameters for factory pattern support.
        
        Args:
            urdf_path: Path to robot URDF file (optional for factory pattern)
            end_effector_link: Name of end effector link (optional for factory pattern) 
            **kwargs: Additional CuRobo configuration parameters
        """
        self._urdf_path = urdf_path
        self._ee_link = end_effector_link
        self._kwargs = kwargs
        self._adapter = None
        self._initialized = False
        
        # If parameters provided, initialize immediately (direct instantiation)
        if urdf_path is not None and end_effector_link is not None:
            self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize the CuRobo adapter with current parameters."""
        if self._initialized:
            return
            
        try:
            from .adapters import CuroboAdapter, CUROBO_AVAILABLE
            if not CUROBO_AVAILABLE:
                raise ImportError("CuRobo dependencies not available")
            
            if self._urdf_path is None or self._ee_link is None:
                raise ValueError("URDF path and end effector link must be provided")
                
            self._adapter = CuroboAdapter(self._urdf_path, self._ee_link, **self._kwargs)
            self._initialized = True
            
        except ImportError as e:
            raise ImportError(f"Failed to initialize CuRobo IK solver: {e}")
    
    def _configure_from_pinocchio(self, pin_model: pin.Model):
        """Extract configuration from Pinocchio model as fallback."""
        if self._urdf_path is None:
            # Try to get URDF path from pinocchio model if available
            import inspect
            frame = inspect.currentframe()
            try:
                # Walk up the call stack to find robot_model reference
                caller_locals = frame.f_back.f_back.f_locals if frame.f_back and frame.f_back.f_back else {}
                self_obj = caller_locals.get('self')
                if self_obj and hasattr(self_obj, '_robot_model'):
                    robot_model = self_obj._robot_model
                    if robot_model and hasattr(robot_model, 'urdf_path'):
                        self._urdf_path = robot_model.urdf_path
                    if robot_model and hasattr(robot_model, 'ee_link'):
                        self._ee_link = robot_model.ee_link
            finally:
                del frame
        
        if self._ee_link is None:
            # Use the first frame name from target as fallback
            self._ee_link = "gripper_link"  # Common default
    
    def ik(self, pin_model: pin.Model, pin_data: pin.Data, target_pose: dict, 
           curr_joint_positions: np.ndarray, tolerance: float, max_iter: float, 
           damping = None) -> tuple[bool, np.ndarray, str]:
        """
        Solve inverse kinematics using CuRobo.
        
        Args:
            pin_model: Pinocchio model (used for joint limits fallback)
            pin_data: Pinocchio data (not used by CuRobo)
            target_pose: Target pose dict, key: frame name, value: 4x4 homogeneous matrix
            curr_joint_positions: Initial joint positions (used as seed)
            tolerance: Convergence tolerance
            max_iter: Maximum iterations (passed to adapter)
            damping: Damping parameter (not used by CuRobo)
            
        Returns:
            Tuple of (converged, joint_angles, "position")
        """
        # Lazy initialization: configure from context if not already done
        if not self._initialized:
            self._configure_from_pinocchio(pin_model)
            
            # Use target frame name if ee_link still not set
            if self._ee_link is None and target_pose:
                self._ee_link = next(iter(target_pose.keys()))
            
            self._initialize_adapter()
        
        if not target_pose:
            raise ValueError("Target pose dictionary cannot be empty")
        
        # Extract target pose (CuRobo handles single target)
        target_pose_matrix = next(iter(target_pose.values()))
        
        try:
            # Solve using CuRobo adapter
            converged, solution, solve_time = self._adapter.solve_single(
                target_pose_matrix, 
                curr_joint_positions,
                tolerance=tolerance,
                max_iterations=int(max_iter)
            )
            
            if converged and solution is not None:
                return True, solution, "position"
            else:
                # Fallback to middle of joint range if failed
                if hasattr(pin_model, 'lowerPositionLimit'):
                    fallback_solution = (pin_model.lowerPositionLimit + pin_model.upperPositionLimit) / 2
                else:
                    fallback_solution = curr_joint_positions
                return False, fallback_solution, "position"
                
        except Exception as e:
            print(f"Warning: CuRobo IK failed: {e}")
            # Return fallback solution
            if hasattr(pin_model, 'lowerPositionLimit'):
                fallback_solution = (pin_model.lowerPositionLimit + pin_model.upperPositionLimit) / 2
            else:
                fallback_solution = curr_joint_positions
            return False, fallback_solution, "position"