"""
Simulation validator for IK benchmark testing.

Validates IK solutions in Mujoco simulation environment to ensure
solutions work correctly in practice, not just numerically.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import glog as log

from factory.components.robot_factory import RobotFactory
from simulation.base.sim_base import SimBase
from hardware.base.utils import RobotJointState


@dataclass
class SimValidationResult:
    """Results from simulation validation."""
    validated_solutions: int
    total_solutions: int
    sim_position_errors: List[float]
    sim_rotation_errors: List[float] 
    execution_success_rate: float
    collision_count: int
    joint_limit_violations: int
    average_execution_time: float
    failed_validations: List[Dict[str, Any]]


class SimDataRecorder:
    """Records simulation execution data for analysis."""
    
    def __init__(self):
        """Initialize data recorder."""
        self._execution_data = []
    
    def record_execution(self, target_pose: np.ndarray,
                        solution: np.ndarray, 
                        achieved_pose: Optional[np.ndarray],
                        execution_time: float,
                        success: bool,
                        error_info: Optional[str] = None):
        """
        Record single execution data.
        
        Args:
            target_pose: Target 4x4 transformation matrix
            solution: Joint angle solution
            achieved_pose: Actually achieved 4x4 pose in simulation
            execution_time: Execution time in seconds
            success: Whether execution was successful
            error_info: Error information if failed
        """
        record = {
            'timestamp': time.time(),
            'target_pose': target_pose,
            'solution': solution,
            'achieved_pose': achieved_pose,
            'execution_time': execution_time,
            'success': success,
            'error_info': error_info
        }
        self._execution_data.append(record)
    
    def export_execution_log(self, filepath: str):
        """
        Export execution log to file.
        
        Args:
            filepath: Output file path
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = []
        for record in self._execution_data:
            export_record = record.copy()
            for key, value in export_record.items():
                if isinstance(value, np.ndarray):
                    export_record[key] = value.tolist()
            export_data.append(export_record)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        log.info(f"Execution log exported to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self._execution_data:
            return {}
        
        successful_executions = [r for r in self._execution_data if r['success']]
        
        stats = {
            'total_executions': len(self._execution_data),
            'successful_executions': len(successful_executions),
            'success_rate': len(successful_executions) / len(self._execution_data),
            'average_execution_time': np.mean([r['execution_time'] for r in self._execution_data]),
            'max_execution_time': np.max([r['execution_time'] for r in self._execution_data]),
        }
        
        if successful_executions:
            execution_times = [r['execution_time'] for r in successful_executions]
            stats['successful_avg_time'] = np.mean(execution_times)
            stats['successful_std_time'] = np.std(execution_times)
        
        return stats


class SimValidator:
    """Validates IK solutions in simulation environment."""
    
    def __init__(self, robot_factory: RobotFactory, sim_config: Dict[str, Any]):
        """
        Initialize simulation validator.
        
        Args:
            robot_factory: Robot factory instance
            sim_config: Simulation configuration
        """
        self._robot_factory = robot_factory
        self._sim_config = sim_config
        self._simulation = None
        self._recorder = SimDataRecorder()
        
        # Initialize simulation
        self._initialize_simulation()
        
        log.info("SimValidator initialized")
    
    def validate_ik_solutions(self, 
                            solutions: List[np.ndarray],
                            target_poses: List[np.ndarray],
                            validation_tolerance: float = 1e-3) -> SimValidationResult:
        """
        Validate IK solutions in simulation.
        
        Args:
            solutions: List of joint angle solutions
            target_poses: Corresponding target poses
            validation_tolerance: Tolerance for pose validation
            
        Returns:
            SimValidationResult with validation metrics
        """
        assert len(solutions) == len(target_poses), \
            "Solutions and target poses must have same length"
        
        log.info(f"Validating {len(solutions)} IK solutions in simulation")
        
        validated_count = 0
        position_errors = []
        rotation_errors = []
        collision_count = 0
        joint_limit_violations = 0
        execution_times = []
        failed_validations = []
        
        for i, (solution, target_pose) in enumerate(zip(solutions, target_poses)):
            if i % (len(solutions) // 10) == 0 and i > 0:
                log.info(f"Validated {i}/{len(solutions)} solutions")
            
            start_time = time.time()
            
            try:
                # Execute joint command in simulation
                success, achieved_pose = self._execute_joint_command(solution)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                if success and achieved_pose is not None:
                    # Compute pose errors
                    pos_error, rot_error = self._compute_pose_error(achieved_pose, target_pose)
                    position_errors.append(pos_error)
                    rotation_errors.append(rot_error)
                    
                    # Check if within tolerance
                    if pos_error <= validation_tolerance and rot_error <= validation_tolerance:
                        validated_count += 1
                    
                    # Record execution
                    self._recorder.record_execution(
                        target_pose, solution, achieved_pose, execution_time, True
                    )
                else:
                    # Failed execution
                    position_errors.append(float('inf'))
                    rotation_errors.append(float('inf'))
                    
                    # Check failure reason
                    if self._check_collisions():
                        collision_count += 1
                        failure_reason = "collision"
                    elif self._check_joint_limits(solution):
                        joint_limit_violations += 1
                        failure_reason = "joint_limits"
                    else:
                        failure_reason = "execution_failed"
                    
                    failed_validations.append({
                        'index': i,
                        'reason': failure_reason,
                        'solution': solution,
                        'target_pose': target_pose
                    })
                    
                    self._recorder.record_execution(
                        target_pose, solution, None, execution_time, False, failure_reason
                    )
                    
            except Exception as e:
                log.warning(f"Simulation validation {i} failed: {e}")
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                position_errors.append(float('inf'))
                rotation_errors.append(float('inf'))
                
                failed_validations.append({
                    'index': i,
                    'reason': 'exception',
                    'error': str(e),
                    'solution': solution,
                    'target_pose': target_pose
                })
                
                self._recorder.record_execution(
                    target_pose, solution, None, execution_time, False, str(e)
                )
        
        # Calculate results
        total_solutions = len(solutions)
        execution_success_rate = validated_count / total_solutions if total_solutions > 0 else 0.0
        avg_execution_time = np.mean(execution_times) if execution_times else 0.0
        
        result = SimValidationResult(
            validated_solutions=validated_count,
            total_solutions=total_solutions,
            sim_position_errors=position_errors,
            sim_rotation_errors=rotation_errors,
            execution_success_rate=execution_success_rate,
            collision_count=collision_count,
            joint_limit_violations=joint_limit_violations,
            average_execution_time=avg_execution_time,
            failed_validations=failed_validations
        )
        
        log.info(f"Simulation validation complete: {validated_count}/{total_solutions} "
                f"validated ({execution_success_rate:.2%})")
        
        return result
    
    def _initialize_simulation(self):
        """Initialize simulation environment."""
        try:
            # Create simulation configuration
            sim_config = {
                'use_hardware': False,
                'use_simulation': True,
                'robot': self._sim_config.get('robot_type', 'fr3'),
                'simulation': 'mujoco'
            }
            
            # Get simulation from robot factory
            self._simulation = self._robot_factory._simulation
            
            if self._simulation is None:
                log.warning("No simulation available in robot factory, creating new one")
                # Could create a new simulation here if needed
                raise RuntimeError("Simulation not available")
            
            log.info("Simulation initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize simulation: {e}")
            raise
    
    def _execute_joint_command(self, joint_angles: np.ndarray, 
                              timeout: float = 5.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Execute joint command in simulation and get achieved pose.
        
        Args:
            joint_angles: Target joint angles
            timeout: Execution timeout in seconds
            
        Returns:
            Tuple of (success, achieved_pose)
        """
        try:
            # Check joint limits before execution
            if not self._check_joint_limits(joint_angles):
                log.debug("Joint angles outside limits")
                return False, None
            
            # Set joint command in simulation
            actuator_modes = ['position'] * len(joint_angles)  # Position control mode
            self._simulation.set_joint_command(actuator_modes, joint_angles)
            
            # Wait for simulation to reach target (simplified approach)
            start_time = time.time()
            tolerance = 0.01  # Joint angle tolerance
            
            while time.time() - start_time < timeout:
                # Get current joint states
                current_state = self._simulation.get_joint_states()
                
                # Check if close to target
                if np.allclose(current_state._positions, joint_angles, atol=tolerance):
                    # Get achieved TCP pose
                    achieved_pose = self._simulation.get_tcp_pose()
                    
                    if achieved_pose is not None:
                        return True, achieved_pose
                    else:
                        log.debug("Failed to get TCP pose from simulation")
                        return False, None
                
                # Small delay to avoid busy waiting
                time.sleep(0.01)
            
            log.debug(f"Joint command execution timed out after {timeout}s")
            return False, None
            
        except Exception as e:
            log.debug(f"Joint command execution failed: {e}")
            return False, None
    
    def _check_collisions(self) -> bool:
        """
        Check for collisions in simulation.
        
        Returns:
            True if collision detected
        """
        try:
            # This is a simplified collision check
            # In a real implementation, would query simulation collision status
            # For now, assume no collisions (placeholder)
            return False
            
        except Exception:
            return False
    
    def _check_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """
        Check if joint angles are within limits.
        
        Args:
            joint_angles: Joint angles to check
            
        Returns:
            True if within limits
        """
        try:
            robot_model = self._robot_factory._robot_model
            lower_limits = robot_model.model.lowerPositionLimit
            upper_limits = robot_model.model.upperPositionLimit
            
            return np.all(joint_angles >= lower_limits) and np.all(joint_angles <= upper_limits)
            
        except Exception as e:
            log.debug(f"Joint limit check failed: {e}")
            return False
    
    def _compute_pose_error(self, achieved: np.ndarray, 
                           target: np.ndarray) -> Tuple[float, float]:
        """
        Compute pose error between achieved and target poses.
        
        Args:
            achieved: Achieved 4x4 transformation matrix
            target: Target 4x4 transformation matrix
            
        Returns:
            Tuple of (position_error, rotation_error)
        """
        # Position error
        pos_error = np.linalg.norm(achieved[:3, 3] - target[:3, 3])
        
        # Rotation error (angle between rotation matrices)
        R_relative = achieved[:3, :3] @ target[:3, :3].T
        trace = np.clip(np.trace(R_relative), -1, 3)
        rot_error = np.arccos((trace - 1) / 2)
        
        return pos_error, abs(rot_error)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics from recorder.
        
        Returns:
            Dictionary with execution statistics
        """
        return self._recorder.get_statistics()
    
    def export_execution_log(self, filepath: str):
        """
        Export execution log.
        
        Args:
            filepath: Output file path
        """
        self._recorder.export_execution_log(filepath)
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        try:
            if self._simulation is not None:
                # Reset simulation (method depends on simulation implementation)
                # For now, assume simulation handles reset internally
                pass
        except Exception as e:
            log.warning(f"Failed to reset simulation: {e}")
    
    def cleanup(self):
        """Clean up simulation resources."""
        try:
            self._simulation = None
            log.info("SimValidator cleanup complete")
        except Exception as e:
            log.warning(f"SimValidator cleanup failed: {e}")