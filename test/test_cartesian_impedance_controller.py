#!/usr/bin/env python3
"""
Test script for Cartesian Impedance Controller
This script demonstrates how to use the CartesianImpedanceController
"""

from simulation.mujoco.mujoco_sim import MujocoSim
from controller.cartesian_impedance_controller import CartesianImpedanceController
from motion.pin_model import RobotModel
from hardware.base.utils import convert_homo_2_7D_pose
import time
import numpy as np
from cfg_handling import get_cfg

def main():
    # Load configurations
    cartesian_impedance_config = "controller/config/cartesian_impedance_fr3_cfg.yaml"
    cartesian_impedance_config = get_cfg(cartesian_impedance_config)
    
    model_config = "motion/config/robot_model_fr3_cfg.yaml"
    model_config = get_cfg(model_config)
    # Select the specific robot configuration (fr3_only or fr3_franka_hand)
    robot_config = model_config["fr3_only"]  # Use fr3_only configuration
    print(f'Robot config: {robot_config}')
    
    controller_config = cartesian_impedance_config["cartesian_impedance"]
    print(f'Controller config: {controller_config}')
    
    mujoco_config = "simulation/config/mujoco_fr3_scene.yaml"
    mujoco_config = get_cfg(mujoco_config)["mujoco"]
    print(f'Mujoco config: {mujoco_config}')
    
    # Initialize components
    model = RobotModel(robot_config)
    cartesian_impedance_controller = CartesianImpedanceController(controller_config, model)
    mujoco = MujocoSim(mujoco_config)
    
    # Setup target and TCP sites
    target_site = "target_site"
    tcp_site = "TCP_site"
    tcp_mocap = tcp_site.split('_')[0]
    
    # Print initial controller parameters
    print("\n=== Cartesian Impedance Controller Parameters ===")
    print(f"Translational stiffness: {cartesian_impedance_controller.translational_stiffness} N/m")
    print(f"Rotational stiffness: {cartesian_impedance_controller.rotational_stiffness} Nm/rad")
    print(f"Translational damping: {cartesian_impedance_controller.translational_damping} Ns/m")
    print(f"Rotational damping: {cartesian_impedance_controller.rotational_damping} Nms/rad")
    print(f"Nullspace stiffness: {cartesian_impedance_controller.nullspace_stiffness}")
    print(f"Gravity compensation: {cartesian_impedance_controller._gravity_compensation}")
    print("=" * 50 + "\n")
    
    target = {}
    iteration = 0
    start_time = time.time()
    
    try:
        while True:
            # Get mocap target pose
            target_value = mujoco.get_site_pose(target_site, "xyzw")
            target[model.ee_link] = target_value
            
            # Get current TCP pose and joint states
            # cur_tcp = mujoco.get_tcp_pose()
            joint_states = mujoco.get_joint_states()
            cur_tcp = model.get_frame_pose(model.ee_link, joint_states._positions)
            cur_tcp = convert_homo_2_7D_pose(cur_tcp)
            
            # Update visualization
            print(f'tcp: {tcp_mocap}')
            mujoco.set_target_mocap_pose(tcp_mocap, cur_tcp)
            
            # Compute control command
            success, torque_command, mode = cartesian_impedance_controller.compute_controller(
                target, joint_states
            )
            
            if not success:
                print(f"Warning: Controller computation failed at iteration {iteration}")
                continue
            
            # Apply control command
            mujoco.set_joint_command([mode] * len(torque_command), torque_command)
            
            # Log every 1000 iterations
            if iteration % 1000 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration}: {1000/elapsed:.1f} Hz")
                print(f"Target position: {target_value[:3]}")
                print(f"Current TCP position: {cur_tcp[:3]}")
                print(f"Position error norm: {np.linalg.norm(np.array(target_value[:3]) - np.array(cur_tcp[:3])):.4f} m")
                print(f"Max torque: {np.max(np.abs(torque_command)):.2f} Nm")
                print("-" * 50)
                start_time = time.time()
            
            iteration += 1
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nTest terminated by user")
        print(f"Total iterations: {iteration}")

def test_controller_features():
    """Test various features of the Cartesian Impedance Controller"""
    print("\n=== Testing Cartesian Impedance Controller Features ===")
    
    # Load configuration
    config_path = "controller/config/cartesian_impedance_fr3_cfg.yaml"
    config = get_cfg(config_path)["cartesian_impedance"]
    
    model_config = "motion/config/robot_model_fr3_cfg.yaml"
    model_config = get_cfg(model_config)
    robot_config = model_config["fr3_only"]  # Use fr3_only configuration
    
    # Create controller
    model = RobotModel(robot_config)
    controller = CartesianImpedanceController(config, model)
    
    # Test 1: Stiffness adjustment
    print("\n1. Testing stiffness adjustment:")
    print(f"   Initial translational stiffness: {controller.translational_stiffness}")
    controller.set_stiffness(translational_stiffness=300.0)
    print(f"   Updated translational stiffness: {controller.translational_stiffness}")
    
    # Test 2: Damping adjustment
    print("\n2. Testing damping adjustment:")
    print(f"   Initial rotational damping: {controller.rotational_damping}")
    controller.set_damping(rotational_damping=10.0)
    print(f"   Updated rotational damping: {controller.rotational_damping}")
    
    # Test 3: Integral error reset
    print("\n3. Testing integral error reset:")
    controller.error_integral = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(f"   Error integral before reset: {controller.error_integral}")
    controller.reset_integral_error()
    print(f"   Error integral after reset: {controller.error_integral}")
    
    print("\n=== Feature tests completed ===")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-features":
        test_controller_features()
    else:
        main()