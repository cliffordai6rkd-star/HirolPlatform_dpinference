#!/usr/bin/env python3
"""
Test script for AgiBot G1 dual-arm periodic motion.

This script demonstrates smooth periodic motion of both arms simultaneously.
"""

import numpy as np
import time
import sys
import os
import argparse
import glog as log

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from hardware.agibot_g1.agibot_g1 import AgibotG1


def create_sine_wave_trajectory(
    center_positions: np.ndarray,
    amplitudes: np.ndarray,
    frequency: float,
    duration: float,
    dt: float = 0.01
) -> np.ndarray:
    """
    Create sine wave trajectory for joints.
    
    Args:
        center_positions: Center positions for each joint (rad)
        amplitudes: Amplitude of oscillation for each joint (rad)
        frequency: Frequency of oscillation (Hz)
        duration: Total duration (s)
        dt: Time step (s)
        
    Returns:
        np.ndarray: Trajectory array of shape (num_timesteps, num_joints)
    """
    num_steps = int(duration / dt)
    times = np.linspace(0, duration, num_steps)
    trajectory = np.zeros((num_steps, len(center_positions)))
    
    for t_idx, t in enumerate(times):
        trajectory[t_idx] = center_positions + amplitudes * np.sin(2 * np.pi * frequency * t)
    
    return trajectory, times


def run_periodic_motion(
    robot: AgibotG1,
    frequency: float = 0.5,
    amplitude_deg: float = 10.0,
    duration: float = 10.0,
    control_rate: float = 50.0
):
    """
    Run periodic motion on robot arms.
    
    Args:
        robot: AgibotG1 robot instance
        frequency: Oscillation frequency (Hz)
        amplitude_deg: Amplitude in degrees
        duration: Total duration (s)
        control_rate: Control loop rate (Hz)
    """
    dt = 1.0 / control_rate
    amplitude_rad = np.deg2rad(amplitude_deg)
    
    # Get initial joint positions
    joint_states = robot.get_joint_states()
    initial_positions = joint_states._positions.copy()
    
    log.info(f"Starting periodic motion test")
    initial_deg = np.rad2deg(initial_positions[:14])
    log.info(f"Initial positions (first 14 joints): {initial_deg}")
    log.info(f"Frequency: {frequency} Hz, Amplitude: {amplitude_deg} deg, Duration: {duration} s")
    
    # Define which joints to move (all arm joints)
    # Left arm: indices 0-6, Right arm: indices 7-13
    num_arm_joints = 14
    
    # Create amplitude array - move shoulder and elbow joints more, wrist less
    amplitudes = np.zeros(robot._total_dof)
    for i in range(num_arm_joints):
        if i % 7 < 2:  # Shoulder joints
            amplitudes[i] = amplitude_rad
        elif i % 7 < 4:  # Elbow joints
            amplitudes[i] = amplitude_rad * 0.8
        else:  # Wrist joints
            amplitudes[i] = amplitude_rad * 0.5
    
    # Generate trajectory
    trajectory, times = create_sine_wave_trajectory(
        initial_positions,
        amplitudes,
        frequency,
        duration,
        dt
    )
    
    log.info(f"Generated trajectory with {len(trajectory)} points")
    log.info("Starting periodic motion...")
    
    # Execute trajectory
    start_time = time.time()
    for idx, target_position in enumerate(trajectory):
        loop_start = time.time()
        
        # Send joint command
        success = robot.set_joint_command(['position'], target_position)
        
        if not success:
            log.warning(f"Failed to send command at step {idx}")
        
        # Print progress every second
        if idx % int(control_rate) == 0:
            elapsed = time.time() - start_time
            progress = (idx / len(trajectory)) * 100
            log.info(f"Progress: {progress:.1f}%, Elapsed: {elapsed:.1f}s")
        
        # Maintain control rate
        loop_time = time.time() - loop_start
        if loop_time < dt:
            time.sleep(dt - loop_time)
        elif loop_time > 1.5 * dt:
            log.warning(f"Control loop slow: {loop_time:.3f}s > {dt:.3f}s")
    
    log.info("Periodic motion complete")
    
    # Return to initial position
    log.info("Returning to initial position...")
    return_trajectory = np.linspace(trajectory[-1], initial_positions, int(2.0 * control_rate))
    
    for target_position in return_trajectory:
        loop_start = time.time()
        robot.set_joint_command(['position'], target_position)
        
        loop_time = time.time() - loop_start
        if loop_time < dt:
            time.sleep(dt - loop_time)
    
    log.info("Returned to initial position")


def main():
    """Main function to run periodic motion test."""
    parser = argparse.ArgumentParser(description='AgiBot G1 Periodic Motion Test')
    parser.add_argument('--frequency', type=float, default=5,
                        help='Oscillation frequency in Hz (default: 0.5)')
    parser.add_argument('--amplitude', type=float, default=10.0,
                        help='Amplitude in degrees (default: 10)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration in seconds (default: 10)')
    parser.add_argument('--rate', type=float, default=50.0,
                        help='Control rate in Hz (default: 50)')
    parser.add_argument('--mock', action='store_true',
                        help='Force use of mock robot')
    parser.add_argument('--enable-head', action='store_true',
                        help='Enable head control')
    parser.add_argument('--enable-waist', action='store_true',
                        help='Enable waist control')
    args = parser.parse_args()
    
    # Force mock mode if requested
    if args.mock:
        os.environ['AGIBOT_USE_MOCK'] = '1'
    
    # Robot configuration
    config = {
        'dof': [7, 7],  # Dual 7-DOF arms
        'robot_name': 'AgiBot_G1_Test',
        'control_head': args.enable_head,
        'control_waist': args.enable_waist,
        'control_wheel': False,
        'control_gripper': False,
        'control_hand': False,
    }
    
    log.info("=" * 60)
    log.info("AgiBot G1 Periodic Motion Test")
    log.info("=" * 60)
    
    try:
        # Initialize robot
        log.info("Initializing AgiBot G1...")
        robot = AgibotG1(config)
        
        if not robot._is_initialized:
            log.error("Robot initialization failed")
            return 1
        
        log.info(f"Robot initialized with {robot._total_dof} DOF")
        log.info(f"Using {'mock' if robot._use_mock else 'real'} robot implementation")
        
        # Wait for user confirmation
        input("\nPress Enter to start periodic motion (Ctrl+C to abort)...")
        
        # Run periodic motion
        run_periodic_motion(
            robot,
            frequency=args.frequency,
            amplitude_deg=args.amplitude,
            duration=args.duration,
            control_rate=args.rate
        )
        
        log.info("Test completed successfully")
        
    except KeyboardInterrupt:
        log.info("\nTest interrupted by user")
        
    except Exception as e:
        log.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        if 'robot' in locals():
            log.info("Shutting down robot...")
            robot.close()
            
        # Clean up environment
        if 'AGIBOT_USE_MOCK' in os.environ and args.mock:
            del os.environ['AGIBOT_USE_MOCK']
    
    return 0


if __name__ == "__main__":
    sys.exit(main())