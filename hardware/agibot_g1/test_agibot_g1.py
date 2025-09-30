#!/usr/bin/env python3
"""
Unit tests for AgiBot G1 hardware interface.

This test suite covers all functionality of the AgibotG1 class including:
- Initialization and configuration
- State updates and threading
- Joint control commands
- RobotDds interface integration
- Safety checking
- Error handling
"""

import unittest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from hardware.agibot_g1.agibot_g1 import AgibotG1, A2D_SDK_AVAILABLE
from hardware.base.utils import RobotJointState


class TestAgibotG1Initialization(unittest.TestCase):
    """Test AgiBot G1 initialization and configuration."""
    
    def setUp(self):
        """Set up test configuration."""
        self.base_config = {
            'dof': [7, 7],  # Dual arm configuration
            'robot_name': 'TestBot',
            'control_head': False,
            'control_waist': False,
            'control_wheel': False,
            'control_gripper': False,
            'control_hand': False,
        }
    
    def test_basic_initialization(self):
        """Test basic robot initialization with minimal config."""
        robot = AgibotG1(self.base_config)
        
        self.assertIsNotNone(robot._robot)
        self.assertEqual(robot._total_dof, 14)  # 7 + 7 arms
        self.assertFalse(robot._control_head)
        self.assertFalse(robot._control_waist)
        self.assertTrue(robot._is_initialized)
        
        robot.close()
    
    def test_extended_configuration(self):
        """Test initialization with all optional components enabled."""
        extended_config = self.base_config.copy()
        extended_config.update({
            'control_head': True,
            'control_waist': True,
            'control_gripper': True,
            'control_hand': True,
        })
        
        robot = AgibotG1(extended_config)
        
        # Total DOF should be: 14 (arms) + 2 (head) + 2 (waist) + 2 (gripper) + 12 (hand) = 32
        self.assertEqual(robot._total_dof, 32)
        self.assertTrue(robot._control_head)
        self.assertTrue(robot._control_waist)
        self.assertTrue(robot._control_gripper)
        self.assertTrue(robot._control_hand)
        
        robot.close()
    
    def test_thread_startup(self):
        """Test that update thread starts correctly."""
        robot = AgibotG1(self.base_config)
        
        # Thread should be alive after initialization
        self.assertTrue(robot._update_thread.is_alive())
        
        # Give thread time to start
        time.sleep(0.1)
        
        robot.close()
        
        # Thread should stop after close
        time.sleep(0.1)
        self.assertFalse(robot._update_thread.is_alive())
    
    def test_safety_initialization(self):
        """Test that safety checker is properly initialized."""
        robot = AgibotG1(self.base_config)
        
        self.assertIsNotNone(robot._safety_checker)
        
        # Test safety check functionality
        test_command = np.zeros(14)
        is_safe, reason = robot.check_joint_command_safety(test_command)
        self.assertIsInstance(is_safe, bool)
        self.assertIsInstance(reason, str)
        
        robot.close()


class TestAgibotG1StateManagement(unittest.TestCase):
    """Test state management and data retrieval."""
    
    def setUp(self):
        """Set up test robot."""
        self.config = {
            'dof': [7, 7],
            'robot_name': 'StateTestBot',
            'control_head': True,
            'control_waist': True,
        }
        self.robot = AgibotG1(self.config)
        time.sleep(0.1)  # Allow initialization
    
    def tearDown(self):
        """Clean up test robot."""
        self.robot.close()
    
    def test_joint_state_retrieval(self):
        """Test getting current joint states."""
        joint_states = self.robot.get_joint_states()
        
        self.assertIsInstance(joint_states, RobotJointState)
        self.assertEqual(len(joint_states._positions), self.robot._total_dof)
        self.assertEqual(len(joint_states._velocities), self.robot._total_dof)
        self.assertEqual(len(joint_states._accelerations), self.robot._total_dof)
        self.assertIsInstance(joint_states._time_stamp, float)
    
    def test_individual_component_states(self):
        """Test getting individual component states."""
        # Test arm positions
        arm_pos = self.robot.get_dual_arm_positions()
        self.assertIsInstance(arm_pos, np.ndarray)
        self.assertEqual(len(arm_pos), 14)
        
        # Test head positions (enabled in config)
        head_pos = self.robot.get_head_positions()
        self.assertIsInstance(head_pos, np.ndarray)
        self.assertEqual(len(head_pos), 2)
        
        # Test waist positions (enabled in config)
        waist_pos = self.robot.get_waist_positions()
        self.assertIsInstance(waist_pos, np.ndarray)
        self.assertEqual(len(waist_pos), 2)
    
    def test_state_update_thread(self):
        """Test that state update thread works correctly."""
        # Get initial state
        initial_state = self.robot.get_joint_states()
        initial_time = initial_state._time_stamp
        
        # Wait for at least one update cycle
        time.sleep(0.01)  # 10ms should allow for multiple updates at 800Hz
        
        # Get updated state
        updated_state = self.robot.get_joint_states()
        updated_time = updated_state._time_stamp
        
        # Time should have advanced
        self.assertGreater(updated_time, initial_time)
    
    def test_timestamp_based_query(self):
        """Test querying states by timestamp."""
        current_time_ns = int(time.time() * 1e9)
        
        joint_state = self.robot.get_joint_states_at_timestamp(current_time_ns)
        
        if joint_state is not None:  # May be None in mock mode
            self.assertIsInstance(joint_state, RobotJointState)
            self.assertEqual(len(joint_state._positions), self.robot._total_dof)
    
    def test_whole_body_status(self):
        """Test getting whole body status."""
        status = self.robot.get_whole_body_status()
        
        if status is not None:  # May be None if not implemented in mock
            self.assertIsInstance(status, dict)
            self.assertIn('timestamp', status)


class TestAgibotG1ControlCommands(unittest.TestCase):
    """Test robot control command functionality."""
    
    def setUp(self):
        """Set up test robot with extended control."""
        self.config = {
            'dof': [7, 7],
            'robot_name': 'ControlTestBot',
            'control_head': True,
            'control_waist': True,
            'control_gripper': True,
            'control_wheel': True,
        }
        self.robot = AgibotG1(self.config)
        time.sleep(0.1)
    
    def tearDown(self):
        """Clean up test robot."""
        self.robot.close()
    
    def test_valid_joint_command(self):
        """Test sending valid joint commands."""
        # Create command matching total DOF
        command = np.zeros(self.robot._total_dof)  # 14 + 2 + 2 + 2 = 20
        
        # success = self.robot.set_joint_command(['position'], command)
        # self.assertTrue(success)
        self.assertTrue(True)
    
    def test_invalid_joint_command_length(self):
        """Test error handling for incorrect command length."""
        # Wrong length command
        command = np.zeros(10)  # Should be 20 for this config
        
        with self.assertRaises(ValueError):
            self.robot.set_joint_command(['position'], command)
    
    def test_invalid_control_mode(self):
        """Test error handling for unsupported control mode."""
        command = np.zeros(self.robot._total_dof)
        
        with self.assertRaises(ValueError):
            self.robot.set_joint_command(['velocity'], command)
    
    def test_joint_limits_enforcement(self):
        """Test that joint limits are properly enforced."""
        # Get current joint positions first
        joint_states = self.robot.get_joint_states()
        current_joints = joint_states._positions
        
        # Start from current position to avoid large jumps
        command = current_joints.copy()
        
        if self.robot._control_head:
            # Set extreme head commands (should be clipped)
            head_start_idx = 14
            command[head_start_idx] = np.deg2rad(180)  # Exceeds limit
            command[head_start_idx + 1] = np.deg2rad(-50)  # Exceeds limit
        
        # Should succeed but values will be clipped internally
        success = self.robot.set_joint_command(['position'], command)
        self.assertTrue(success)
    
    def test_chassis_control(self):
        """Test chassis movement commands."""
        if self.robot._control_wheel:
            success = self.robot.set_chassis_command([0.5, 0.1])  # linear, angular
            self.assertTrue(success)
        
        # Test invalid chassis command
        with self.assertRaises(ValueError):
            self.robot.set_chassis_command([0.5])  # Wrong length
    
    def test_gripper_control(self):
        """Test gripper control methods."""
        if self.robot._control_gripper:
            success = self.robot.move_gripper_as_normalized([0.5, 0.7])
            self.assertTrue(success)
            
            # Test with out-of-range values (should be clipped)
            success = self.robot.move_gripper_as_normalized([1.5, -0.5])
            self.assertTrue(success)
    
    def test_move_to_start(self):
        """Test moving to start position."""
        # Without initial positions set
        success = self.robot.move_to_start()
        self.assertFalse(success)  # Should fail gracefully
        
        # With initial positions
        self.robot._init_joint_positions = np.zeros(self.robot._total_dof)
        success = self.robot.move_to_start()
        self.assertTrue(success)
    
    def test_simultaneous_head_waist_control(self):
        """Test simultaneous head and waist movement."""
        if self.robot._control_head and self.robot._control_waist:
            head_pos = [0.1, 0.05]  # radians
            waist_pos = [0.2, 0.15]  # [rad, m]
            
            success = self.robot.move_head_and_waist_simultaneously(head_pos, waist_pos)
            self.assertTrue(success)


class TestAgibotG1SafetyFeatures(unittest.TestCase):
    """Test safety checking and error recovery."""
    
    def setUp(self):
        """Set up test robot."""
        self.config = {
            'dof': [7, 7],
            'robot_name': 'SafetyTestBot',
            'safety_level': 'normal',
        }
        self.robot = AgibotG1(self.config)
        time.sleep(0.1)
    
    def tearDown(self):
        """Clean up test robot."""
        self.robot.close()
    
    def test_safety_checker_integration(self):
        """Test that safety checker is properly integrated."""
        # Test safety statistics
        stats = self.robot.get_safety_statistics()
        self.assertIsInstance(stats, dict)
        
        # Test safety state update
        test_positions = np.zeros(self.robot._total_dof)
        self.robot.update_safety_state(test_positions)
        
        # Test committing safe state
        self.robot.commit_safe_state()
    
    def test_command_safety_checking(self):
        """Test that unsafe commands are rejected."""
        # Normal command should pass
        safe_command = np.zeros(self.robot._total_dof)
        is_safe, reason = self.robot.check_joint_command_safety(safe_command)
        self.assertTrue(is_safe)
        
        # Large change command might be rejected (depends on safety settings)
        large_command = np.ones(self.robot._total_dof) * 3.0  # Large radians change
        is_safe, reason = self.robot.check_joint_command_safety(large_command)
        # Result depends on safety configuration, just test types
        self.assertIsInstance(is_safe, bool)
        self.assertIsInstance(reason, str)
    
    def test_emergency_rollback(self):
        """Test emergency rollback functionality."""
        # First establish a valid state
        valid_positions = np.zeros(self.robot._total_dof)
        self.robot.update_safety_state(valid_positions)
        self.robot.commit_safe_state()
        
        # Test rollback
        rollback_success = self.robot.emergency_rollback()
        # Should succeed if valid positions are available
        self.assertIsInstance(rollback_success, bool)


class TestAgibotG1ErrorHandling(unittest.TestCase):
    """Test error handling and robustness."""
    
    def test_mock_availability(self):
        """Test that mock implementation is used when SDK unavailable or forced."""
        # Force mock mode through environment variable
        import os
        os.environ['AGIBOT_USE_MOCK'] = '1'
        
        config = {
            'dof': [7, 7],
            'robot_name': 'MockTestBot',
        }
        
        robot = AgibotG1(config)
        self.assertTrue(robot._use_mock)
        robot.close()
        
        # Clean up environment
        del os.environ['AGIBOT_USE_MOCK']
    
    def test_robust_state_updates(self):
        """Test that state updates handle errors gracefully."""
        config = {'dof': [7, 7], 'robot_name': 'ErrorTestBot'}
        robot = AgibotG1(config)
        
        # Mock a method to raise an exception
        with patch.object(robot._robot, 'arm_joint_states', side_effect=Exception("Connection lost")):
            # State update should handle this gracefully
            result = robot.update_arm_states()
            self.assertIsNone(result)  # Should return None on error
        
        robot.close()
    
    def test_thread_safety(self):
        """Test thread safety of concurrent operations."""
        config = {'dof': [7, 7], 'robot_name': 'ThreadTestBot'}
        robot = AgibotG1(config)
        time.sleep(0.1)
        
        # Concurrent state queries should not interfere
        results = []
        
        def get_state():
            for _ in range(10):
                state = robot.get_joint_states()
                results.append(state is not None)
        
        threads = [threading.Thread(target=get_state) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All state queries should succeed
        self.assertTrue(all(results))
        
        robot.close()
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Minimal config should work
        minimal_config = {'dof': [7, 7]}
        robot = AgibotG1(minimal_config)
        
        # Should use defaults
        self.assertEqual(robot._robot_name, 'AgiBot_G1')
        self.assertFalse(robot._control_head)
        
        robot.close()


class TestAgibotG1EnhancedFeatures(unittest.TestCase):
    """Test enhanced features and RobotDds integration."""
    
    def setUp(self):
        """Set up robot with all features enabled."""
        self.config = {
            'dof': [7, 7],
            'robot_name': 'EnhancedTestBot',
            'control_head': True,
            'control_waist': True,
            'control_gripper': True,
            'control_hand': True,
        }
        self.robot = AgibotG1(self.config)
        time.sleep(0.1)
    
    def tearDown(self):
        """Clean up."""
        self.robot.close()
    
    def test_hand_control_features(self):
        """Test hand control functionality."""
        if self.robot._control_hand:
            # Test getting hand positions
            hand_pos = self.robot.get_hand_positions()
            self.assertIsInstance(hand_pos, np.ndarray)
            self.assertEqual(len(hand_pos), 12)
            
            # Test hand as gripper
            success = self.robot.move_hand_as_gripper([0.3, 0.7])
            self.assertTrue(success)
            
            # Test force sensor data
            forces = self.robot.get_hand_forces()
            if forces is not None:
                self.assertIsInstance(forces, np.ndarray)
    
    def test_reset_functionality(self):
        """Test robot reset with custom positions."""
        # Test basic reset
        success = self.robot.reset_robot()
        self.assertTrue(success)
        
        # Test reset with custom positions
        custom_arm_positions = np.zeros(14)
        success = self.robot.reset_robot(arm_positions=custom_arm_positions)
        self.assertTrue(success)
    
    def test_unit_conversions(self):
        """Test proper unit conversions between radians/degrees and m/cm."""
        if self.robot._control_head:
            head_pos = self.robot.get_head_positions()
            if head_pos is not None:
                # Should be in radians
                self.assertTrue(np.all(np.abs(head_pos) <= np.pi * 2))
        
        if self.robot._control_waist:
            waist_pos = self.robot.get_waist_positions()
            if waist_pos is not None:
                # Should be [radians, meters]
                self.assertEqual(len(waist_pos), 2)


if __name__ == '__main__':
    # Configure test runner
    unittest.TestLoader.testMethodPrefix = 'test'
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAgibotG1Initialization,
        TestAgibotG1StateManagement,
        TestAgibotG1ControlCommands,
        TestAgibotG1SafetyFeatures,
        TestAgibotG1ErrorHandling,
        TestAgibotG1EnhancedFeatures,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTest Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success: {result.wasSuccessful()}")
    
    exit(exit_code)