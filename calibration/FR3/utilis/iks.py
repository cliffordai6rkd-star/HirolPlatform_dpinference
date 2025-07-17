"""
部分IK的实现，用于计算逆运动学
"""

import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation as R
# import PyKDL as kdl
import panda_py
# import qpsolvers as qp
# import roboticstoolbox as rtb
# import spatialmath as sm
# from scipy.spatial.transform import Rotation as R

# 获取当前脚本所在目录
project_root = Path(__file__).parent.resolve()

def minkowski_ik(target_pose, model_path, reference_angles=None):
    """
    Solve IK using Minkowski optimization library.
    
    Args:
        target_pose: A 4x4 homogeneous transformation matrix
        model_path: Path to the Franka robot XML model
        reference_angles: Optional initial joint configuration
        
    Returns:
        Joint angles solution as numpy array
    """
    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    
    # Extract position and orientation from the pose
    target_pos = target_pose[:3, 3]
    target_rot_mat = target_pose[:3, :3]
    
    # Convert rotation matrix to quaternion (w, x, y, z format)
    target_quat_xyzw = R.from_matrix(target_rot_mat).as_quat()  # Returns x, y, z, w
    target_quat = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])  # Convert to w, x, y, z
    
    # Combine into the format expected by Minkowski
    wxyz_xyz = np.concatenate([target_quat, target_pos])
    T_target = mink.SE3(wxyz_xyz)
    
    # Set initial configuration
    if reference_angles is not None:
        q_init = np.zeros(model.nq)
        q_init[:min(len(reference_angles), 7)] = reference_angles[:min(len(reference_angles), 7)]
        configuration = mink.Configuration(model, q=q_init)
    else:
        # Default configuration if none provided
        configuration = mink.Configuration(model, q=np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]))
    
    # Create the IK task
    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",  # End effector frame name in the model
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    tasks = [end_effector_task]
    
    # Set target pose
    end_effector_task.set_target(T_target)
    
    # Add joint limit constraint
    config_limit = mink.ConfigurationLimit(model=model)
    limits = [config_limit]
    
    # Solver parameters
    max_iters = 500
    pos_threshold = 1e-4  # Position threshold in meters
    ori_threshold = 1e-3  # Orientation threshold in radians
    dt = 1e-3
    solver = "quadprog"
    
    # Run the IK solver
    for i in range(max_iters):
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, dt)
        
        err = end_effector_task.compute_error(configuration)
        pos_err = np.linalg.norm(err[:3])
        ori_err = np.linalg.norm(err[3:])
        
        if pos_err <= pos_threshold and ori_err <= ori_threshold:
            break
    
    # Return joint solution (first 7 joints for the Franka arm)
    return configuration.q[:7]

import numpy as np


def kdl_ik(target_pose, urdf_file, reference_angles=None):
    """
    Solve IK using KDL library.
    
    Args:
        target_pose: A 4x4 homogeneous transformation matrix
        urdf_file: Path to the robot URDF file
        reference_angles: Optional initial joint configuration
        
    Returns:
        Joint angles solution as numpy array
    """
    # Load the kinematic chain from URDF
    # Note: In a real implementation, you would parse the URDF to create the chain
    # This is a simplified version assuming the chain is already created
    chain = kdl.Chain()
    # ... (chain creation would go here in a real implementation)
    
    # You could use URDF parser like this:
    # from kdl_parser_py.urdf import treeFromFile
    # _, tree = treeFromFile(urdf_file)
    # chain = tree.getChain("panda_link0", "panda_link8")
    
    # For the example, we'll just use a predefined chain for Franka
    # In reality, you'd need to create a full chain with proper segments and joints
    
    # Create KDL solvers
    fk_solver = kdl.ChainFkSolverPos_recursive(chain)
    ik_vel_solver = kdl.ChainIkSolverVel_pinv(chain)
    ik_solver = kdl.ChainIkSolverPos_NR(chain, fk_solver, ik_vel_solver, maxiter=500, eps=1e-6)
    
    # Convert the homogeneous matrix to KDL frame
    rotation = kdl.Rotation(
        target_pose[0, 0], target_pose[0, 1], target_pose[0, 2],
        target_pose[1, 0], target_pose[1, 1], target_pose[1, 2],
        target_pose[2, 0], target_pose[2, 1], target_pose[2, 2]
    )
    translation = kdl.Vector(target_pose[0, 3], target_pose[1, 3], target_pose[2, 3])
    target_frame = kdl.Frame(rotation, translation)
    
    # Initial joint array
    num_joints = chain.getNrOfJoints()
    q_init = kdl.JntArray(num_joints)
    
    # Set initial guess if provided
    if reference_angles is not None:
        for i, val in enumerate(reference_angles[:num_joints]):
            q_init[i] = val
    
    # Output joint array
    q_out = kdl.JntArray(num_joints)
    
    # Solve IK
    ret = ik_solver.CartToJnt(q_init, target_frame, q_out)
    
    if ret >= 0:
        # Convert KDL JntArray to numpy array
        result = np.array([q_out[i] for i in range(num_joints)])
        return result
    else:
        # IK failed
        return None
    


def analytical_ik(target_pose, reference_angles=None):
    """
    Solve IK using the analytical solver from panda_py.
    
    Args:
        target_pose: A 4x4 homogeneous transformation matrix
        reference_angles: Optional initial joint configuration
        
    Returns:
        Joint angles solution as numpy array
    """
    # panda_py.ik is a direct wrapper to the analytical IK solver
    # It takes a homogeneous transformation matrix and returns joint angles
    
    if reference_angles is not None:
        q_solution = panda_py.ik(target_pose, q_init=reference_angles)
    else:
        q_solution = panda_py.ik(target_pose)
    
    return q_solution



def qp_ik(target_pose, current_joint_angles, max_iterations=100):
    """
    Solve IK using quadratic programming optimization.
    
    Args:
        target_pose: A 4x4 homogeneous transformation matrix
        current_joint_angles: Current robot joint angles
        max_iterations: Maximum number of iterations
        
    Returns:
        Joint angles solution as numpy array
    """
    # Create the Panda robot model from roboticstoolbox
    panda = rtb.models.Panda()
    
    # Number of joints
    n = 7
    
    # Convert target_pose to spatialmath SE3
    Tep = sm.SE3(target_pose, check=False)
    
    # Initial joint configuration
    q = np.array(current_joint_angles)
    
    # Parameters
    gain = 0.5
    threshold = 1e-3
    
    # Main iteration loop
    for i in range(max_iterations):
        # Get current end-effector pose
        Te = panda.fkine(q)
        
        # Compute the transformation from current to target
        eTep = Te.inv() * Tep
        
        # Compute error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
        
        # Check if we're close enough to the target
        if e <= threshold:
            break
        
        # Calculate the required end-effector spatial velocity
        v, _ = rtb.p_servo(Te, Tep, gain, threshold=threshold)
        
        # Gain term for control minimization
        Y = 0.01
        
        # Quadratic component of objective function
        Q = np.eye(n + 6)
        Q[:n, :n] *= Y
        Q[n:, n:] = (1 / e) * np.eye(6)
        
        # Equality constraints
        Aeq = np.c_[panda.jacobe(q), np.eye(6)]
        beq = v.reshape((6,))
        
        # Linear component: manipulability Jacobian
        c = np.r_[-panda.jacobm().reshape((n,)), np.zeros(6)]
        
        # Joint velocity and slack variable bounds
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
        ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]
        
        # Solve QP for joint velocities
        qd = qp.solve_qp(Q, c, None, None, Aeq, beq, lb=lb, ub=ub, solver='daqp')
        
        # Update joint positions
        q = q + qd[:n] * 0.1  # Small time step
    
    return q





def create_pose(position, orientation_euler):
    """Create a 4x4 homogeneous transformation matrix from position and Euler angles."""
    rot_matrix = R.from_euler('xyz', orientation_euler, degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rot_matrix
    pose[:3, 3] = position
    return pose

if __name__ == "__main__":
    # Create a target pose (position in meters, orientation in degrees)
    target_position = [0.5, 0.0, 0.5]  # x, y, z in meters
    target_orientation = [180, 0, 0]   # rx, ry, rz in degrees
    target_pose = create_pose(target_position, target_orientation)
    
    # Model paths (adjust as needed for your system)
    # xml_path = "/home/ryze/hirol/TicTacToe/franka_fr3/fr3_with_hand.xml"
    # urdf_file = "/home/ryze/hirol/TicTacToe/franka_fr3/fr3_franka_hand.urdf"
    xml_path = str(project_root / "assets" / "franka_fr3" / "fr3_with_hand.xml")
    urdf_file = str(project_root / "assets" / "franka_fr3" / "fr3_franka_hand.urdf")
    # Current joint configuration (for methods that need it)
    current_joints = [0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]
    import time
    ini = time.time()
    # Try each IK method
    print("Solving with Minkowski IK:")
    mink_solution = minkowski_ik(target_pose, xml_path, current_joints)
    print(f"Solution: {mink_solution.shape}")
    print(f"Target Pose: {panda_py.fk(mink_solution)}")
    
    print("\nSolving with KDL IK:")
    try:
        kdl_solution = kdl_ik(target_pose, urdf_file, current_joints)
        print(f"Solution: {kdl_solution}")
        print(f"Target Pose: {panda_py.fk(kdl_solution)}")
    except Exception as e:
        print(f"KDL IK failed: {e}")
    
    print("\nSolving with Analytical IK:")
    analytical_solution = analytical_ik(target_pose, current_joints)
    print(f"Solution: {analytical_solution}")
    print(f"Target Pose: {panda_py.fk(analytical_solution)}")
    
    print("\nSolving with QP IK:")
    qp_solution = qp_ik(target_pose, current_joints)
    print(f"Solution: {qp_solution}")
    print(f"Target Pose: {panda_py.fk(qp_solution)}")
    fin = time.time()
    print(f"Time: {fin-ini}")
    # 4个线性运行的时间在0.6-0.7s之间 如果插入10个轨迹点 在最极端的情况下是一条轨迹规划时间预计8s
