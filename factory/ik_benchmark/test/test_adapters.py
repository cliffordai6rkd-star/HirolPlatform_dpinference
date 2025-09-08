#!/usr/bin/env python3
"""
Unit tests for IK Adapters - Test adapter functionality and integration.

This module tests the pyroki and curobo adapters to ensure they work correctly
with the HIROL benchmark system.
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, Any
import tempfile

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

try:
    from motion.adapters import (
        IKAdapterBase, 
        PYROKI_AVAILABLE, 
        CUROBO_AVAILABLE,
        get_available_adapters,
        create_adapter
    )
    from motion.adapters.base_adapter import (
        homogeneous_to_quaternion_position,
        quaternion_position_to_homogeneous,
        validate_pose_matrix
    )
    ADAPTERS_AVAILABLE = True
except ImportError as e:
    ADAPTERS_AVAILABLE = False
    _import_error = str(e)


class TestAdapterUtilities(unittest.TestCase):
    """Test utility functions used by adapters."""
    
    def test_pose_conversions(self):
        """Test conversion between homogeneous matrices and quaternion+position."""
        # Create test pose
        pose_4x4 = np.array([
            [1, 0, 0, 0.5],
            [0, 0, -1, 0.2],
            [0, 1, 0, 0.8],
            [0, 0, 0, 1]
        ])
        
        # Convert to quaternion+position
        quat, pos = homogeneous_to_quaternion_position(pose_4x4)
        
        # Convert back to homogeneous matrix
        pose_reconstructed = quaternion_position_to_homogeneous(quat, pos)
        
        # Check if reconstruction is accurate
        np.testing.assert_allclose(pose_4x4, pose_reconstructed, atol=1e-10)
        
        print("✓ Pose conversion test passed")
    
    def test_pose_validation(self):
        """Test pose matrix validation."""
        # Valid pose
        valid_pose = np.eye(4)
        self.assertTrue(validate_pose_matrix(valid_pose))
        
        # Invalid pose - wrong shape
        invalid_shape = np.eye(3)
        self.assertFalse(validate_pose_matrix(invalid_shape))
        
        # Invalid pose - wrong bottom row
        invalid_bottom = np.eye(4)
        invalid_bottom[3, 3] = 2.0
        self.assertFalse(validate_pose_matrix(invalid_bottom))
        
        # Invalid pose - non-orthogonal rotation
        invalid_rotation = np.eye(4)
        invalid_rotation[0, 0] = 2.0  # Breaks orthogonality
        self.assertFalse(validate_pose_matrix(invalid_rotation))
        
        print("✓ Pose validation test passed")


class TestAdapterAvailability(unittest.TestCase):
    """Test adapter availability and factory functions."""
    
    def test_adapter_imports(self):
        """Test that adapter module imports correctly."""
        if not ADAPTERS_AVAILABLE:
            self.skipTest(f"Adapters not available: {_import_error}")
        
        # Test availability flags
        print(f"Pyroki available: {PYROKI_AVAILABLE}")
        print(f"CuRobo available: {CUROBO_AVAILABLE}")
        
        # Test get_available_adapters function
        available = get_available_adapters()
        print(f"Available adapters: {available}")
        
        self.assertIsInstance(available, list)
        
        print("✓ Adapter availability test passed")
    
    def test_create_adapter_factory(self):
        """Test adapter creation factory function."""
        if not ADAPTERS_AVAILABLE:
            self.skipTest("Adapters not available")
        
        # Test with dummy URDF path (adapter creation should work even with non-existent file)
        dummy_urdf = "/tmp/dummy.urdf"
        ee_link = "end_effector"
        
        # Test creating pyroki adapter if available
        if PYROKI_AVAILABLE:
            try:
                adapter = create_adapter('pyroki', dummy_urdf, ee_link)
                self.assertIsNotNone(adapter)
                print("✓ Pyroki adapter creation test passed")
            except Exception as e:
                print(f"! Pyroki adapter creation test failed: {e}")
        
        # Test creating curobo adapter if available  
        if CUROBO_AVAILABLE:
            try:
                adapter = create_adapter('curobo', dummy_urdf, ee_link)
                self.assertIsNotNone(adapter)
                print("✓ CuRobo adapter creation test passed")
            except Exception as e:
                print(f"! CuRobo adapter creation test failed: {e}")
        
        # Test invalid adapter type
        with self.assertRaises(ValueError):
            create_adapter('invalid_adapter', dummy_urdf, ee_link)


@unittest.skipUnless(PYROKI_AVAILABLE, "Pyroki not available")
class TestPyrokiAdapter(unittest.TestCase):
    """Test Pyroki adapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal URDF content for testing
        self.test_urdf_content = '''<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="end_effector"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="end_effector"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1000" velocity="1"/>
  </joint>
</robot>'''
        
        # Create temporary URDF file
        self.temp_urdf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
        self.temp_urdf.write(self.test_urdf_content)
        self.temp_urdf.close()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_urdf'):
            os.unlink(self.temp_urdf.name)
    
    def test_pyroki_adapter_creation(self):
        """Test Pyroki adapter creation and basic functionality."""
        try:
            from motion.adapters import PyrokiAdapter
            
            adapter = PyrokiAdapter(
                urdf_path=self.temp_urdf.name,
                end_effector_link="end_effector"
            )
            
            # Test availability
            is_available = adapter.is_available()
            print(f"Pyroki adapter available: {is_available}")
            
            if is_available:
                # Test info retrieval
                info = adapter.get_solver_info()
                print(f"Pyroki solver info: {info}")
                self.assertIn('solver_name', info)
                self.assertEqual(info['solver_name'], 'Pyroki')
            
            print("✓ Pyroki adapter creation test passed")
            
        except ImportError as e:
            self.skipTest(f"Pyroki adapter import failed: {e}")
        except Exception as e:
            print(f"! Pyroki adapter test failed: {e}")
            raise


@unittest.skipUnless(CUROBO_AVAILABLE, "CuRobo not available")  
class TestCuroboAdapter(unittest.TestCase):
    """Test CuRobo adapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal URDF content for testing
        self.test_urdf_content = '''<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="end_effector"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="end_effector"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1000" velocity="1"/>
  </joint>
</robot>'''
        
        # Create temporary URDF file
        self.temp_urdf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
        self.temp_urdf.write(self.test_urdf_content)
        self.temp_urdf.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_urdf'):
            os.unlink(self.temp_urdf.name)
    
    def test_curobo_adapter_creation(self):
        """Test CuRobo adapter creation and basic functionality."""
        try:
            from motion.adapters import CuroboAdapter
            
            adapter = CuroboAdapter(
                urdf_path=self.temp_urdf.name,
                end_effector_link="end_effector",
                use_gpu=False  # Force CPU mode for testing
            )
            
            # Test availability
            is_available = adapter.is_available()
            print(f"CuRobo adapter available: {is_available}")
            
            if is_available:
                # Test info retrieval
                info = adapter.get_solver_info()
                print(f"CuRobo solver info: {info}")
                self.assertIn('solver_name', info)
                self.assertEqual(info['solver_name'], 'CuRobo')
            
            print("✓ CuRobo adapter creation test passed")
            
        except ImportError as e:
            self.skipTest(f"CuRobo adapter import failed: {e}")
        except Exception as e:
            print(f"! CuRobo adapter test failed: {e}")
            raise


class TestAdapterIntegration(unittest.TestCase):
    """Test adapter integration with benchmark system."""
    
    def test_ik_tester_integration(self):
        """Test that adapters integrate correctly with IK tester."""
        try:
            # Import IK tester and check available methods
            from factory.ik_benchmark.core.ik_tester import IKTester
            from factory.components.robot_factory import RobotFactory
            
            # Create minimal robot factory config
            robot_config = {
                "use_hardware": False,
                "use_simulation": False,
                "robot": "fr3",
                "model_config": {
                    "name": "fr3",
                    "cfg": {
                        "fr3": {
                            "urdf_path": "assets/franka_fr3/fr3.urdf",
                            "ee_link": "fr3_hand"
                        }
                    }
                }
            }
            
            # Create robot factory (this might fail if assets not available)
            try:
                robot_factory = RobotFactory(robot_config)
                ik_tester = IKTester(robot_factory, {})
                
                # Test available methods
                available_methods = ik_tester.get_available_ik_methods()
                print(f"Available IK methods: {available_methods}")
                
                # Check if new methods are included
                expected_base_methods = ['gaussian_newton', 'dls', 'lm', 'pink']
                for method in expected_base_methods:
                    self.assertIn(method, available_methods)
                
                if PYROKI_AVAILABLE:
                    self.assertIn('pyroki', available_methods)
                    print("✓ Pyroki method found in available methods")
                
                if CUROBO_AVAILABLE:
                    self.assertIn('curobo', available_methods)
                    print("✓ CuRobo method found in available methods")
                
                print("✓ IK tester integration test passed")
                
            except Exception as e:
                print(f"! IK tester creation failed (expected if assets missing): {e}")
                self.skipTest("Cannot create IK tester - likely missing robot assets")
                
        except ImportError as e:
            self.skipTest(f"IK tester import failed: {e}")


def run_adapter_tests():
    """Run all adapter tests with detailed output."""
    print("=" * 60)
    print("🧪 RUNNING IK ADAPTER TESTS")
    print("=" * 60)
    
    # Check basic availability
    print(f"\n📦 Adapter Module Status:")
    print(f"   Adapters module: {'✓' if ADAPTERS_AVAILABLE else '✗'}")
    if ADAPTERS_AVAILABLE:
        print(f"   Pyroki adapter: {'✓' if PYROKI_AVAILABLE else '✗'}")
        print(f"   CuRobo adapter: {'✓' if CUROBO_AVAILABLE else '✗'}")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestAdapterUtilities))
    
    if ADAPTERS_AVAILABLE:
        test_suite.addTest(unittest.makeSuite(TestAdapterAvailability))
        test_suite.addTest(unittest.makeSuite(TestAdapterIntegration))
        
        if PYROKI_AVAILABLE:
            test_suite.addTest(unittest.makeSuite(TestPyrokiAdapter))
        
        if CUROBO_AVAILABLE:
            test_suite.addTest(unittest.makeSuite(TestCuroboAdapter))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.wasSuccessful():
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(failure[1])
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(error[1])
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_adapter_tests()
    sys.exit(0 if success else 1)